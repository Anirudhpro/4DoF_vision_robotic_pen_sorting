# --- Imports ---
import cv2
from ultralytics import YOLO
import numpy as np
import json
import time
import sys
import threading
import serial
import json
import os
import shutil
from datetime import datetime

# -------------- Utility: timestamps --------------
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def ts_for_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# -------------- Session logging & files --------------
class TeeStdout:
    """Duplicate all prints to both the console and a file."""
    def __init__(self, original, file_handle):
        self.original = original
        self.file_handle = file_handle
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.original.write(data)
            fh = self.file_handle
            if fh and not getattr(fh, "closed", False):
                try:
                    fh.write(data)
                    fh.flush()
                except ValueError:
                    self.file_handle = None

    def flush(self):
        with self.lock:
            self.original.flush()
            fh = self.file_handle
            if fh and not getattr(fh, "closed", False):
                try:
                    fh.flush()
                except ValueError:
                    self.file_handle = None

    def detach_file(self):
        with self.lock:
            self.file_handle = None

# Prepare args: serial port + optional logs root
if len(sys.argv) < 2:
    print("Usage: python camera_stream.py <serial_port> [logs_root]")
    print("Example: python camera_stream.py /dev/tty.usbserial-xxx ResearchDataset")
    sys.exit(1)

serial_port = sys.argv[1]
logs_root = sys.argv[2] if len(sys.argv) >= 3 else "ResearchDataset"
os.makedirs(logs_root, exist_ok=True)

# Create a temp session dir now; will rename to 'log N' on exit.
_session_tag = ts_for_filename()
temp_session_dir = os.path.join(logs_root, f".session_tmp_{_session_tag}")
os.makedirs(temp_session_dir, exist_ok=True)

# Open log file and tee stdout
log_file_path = os.path.join(temp_session_dir, "log.txt")
_log_fh = open(log_file_path, "a", buffering=1, encoding="utf-8")
_tee = TeeStdout(sys.__stdout__, _log_fh)
sys.stdout = _tee

print(f"[{ts()}] Session started. Temp dir: {temp_session_dir}")
session_start_time = time.time()

# Video will be created after first frame (we need w/h/fps)
video_writer = None
video_path = os.path.join(temp_session_dir, "session_video.mp4")
_video_lock = threading.Lock()

def _clamp_fps(raw_fps):
    # Guard against 0, None, NaN, or silly values
    if raw_fps is None or raw_fps <= 0 or raw_fps > 240 or np.isnan(raw_fps):
        return 30.0
    return float(raw_fps)

def _init_video_writer(frame, cap):
    """Robustly initialize a VideoWriter for mp4; try multiple fourccs."""
    global video_writer
    h, w = frame.shape[:2]
    fps = _clamp_fps(cap.get(cv2.CAP_PROP_FPS))
    attempts = [("mp4v", video_path), ("avc1", video_path)]
    for fourcc_tag, path in attempts:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if vw.isOpened():
            video_writer = vw
            print(f"[{ts()}] Video writer initialized ({fourcc_tag}): {w}x{h}@{fps:.2f} -> {path}")
            return True
        else:
            try:
                vw.release()
            except:
                pass
            print(f"[{ts()}] WARN: Fourcc {fourcc_tag} failed to open writer.")
    print(f"[{ts()}] ERROR: Could not open any mp4 writer. Video will be unavailable.")
    video_writer = None
    return False

# Latest annotated frame cache (so the motion thread can snapshot)
latest_annotated = None
latest_lock = threading.Lock()

# Global robot thread handle so we can join it on shutdown
robot_thread = None

def save_snapshot(label):
    """Save a copy of the latest annotated frame with a timestamped filename; log it."""
    with latest_lock:
        if latest_annotated is None:
            print(f"[{ts()}] [Snapshot] Skipped ({label}) — no annotated frame yet.")
            return None
        img = latest_annotated.copy()
    fname = f"{ts_for_filename()}_{label}.png"
    fpath = os.path.join(temp_session_dir, fname)
    try:
        cv2.imwrite(fpath, img)
        print(f"[{ts()}] [Snapshot] Saved: {fname} ({label})")
        return fpath
    except Exception as e:
        print(f"[{ts()}] [Snapshot] ERROR saving {fname}: {e}")
        return None

def finalize_and_rename_session():
    """Close resources, compute duration, and rename temp folder to next 'log N' with gap filling."""
    # Close video safely
    global video_writer
    with _video_lock:
        if video_writer is not None:
            try:
                video_writer.release()
                print(f"[{ts()}] Video writer released.")
            except Exception as e:
                print(f"[{ts()}] ERROR releasing video writer: {e}")
            video_writer = None

    duration_sec = time.time() - session_start_time
    print(f"[{ts()}] Session duration: {duration_sec:.3f} sec")

    # Determine next log N (gap-filling)
    existing = []
    for name in os.listdir(logs_root):
        if name.lower().startswith("log "):
            try:
                n = int(name.split(" ", 1)[1])
                existing.append(n)
            except:
                pass
    existing_set = set(sorted(existing))
    n = 1
    while n in existing_set:
        n += 1
    final_dir = os.path.join(logs_root, f"log {n}")

    # Rename/move temp session dir to final
    try:
        shutil.move(temp_session_dir, final_dir)
        print(f"[{ts()}] Session artifacts moved to: {final_dir}")
    except Exception as e:
        print(f"[{ts()}] ERROR moving session dir to {final_dir}: {e}")
        print(f"[{ts()}] Leaving temp dir in place: {temp_session_dir}")

# --- Model and Calibration ---
model = YOLO("best.pt")
calib = np.load('calib_data.npz')
K = calib['K']
dist = calib['dist']
with open('Aruco/aruco_reference.json') as f:
    aruco_data = json.load(f)
rvec = np.array(aruco_data['rvec'][0])
tvec = np.array(aruco_data['tvec'][0])
R, _ = cv2.Rodrigues(rvec)

# --- Robot-to-tag transformation (update as needed) ---
robot_tag_xyz = np.array([300, 0, -57])  # mm - robot arm position relative to ArUco tag
robot_tag_theta = (3 * 3.14 / 2)  # radians

def tag_to_robot(point_tag):
    theta = robot_tag_theta
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    point_robot = Rz @ point_tag + robot_tag_xyz
    return point_robot

# ---- Color utilities (for OBB color classification) ----
ALLOWED_COLORS = ["blue", "red", "green", "grayscale"]

def _rotated_rect_corners(cx, cy, w, h, angle_rad):
    dx, dy = 0.5 * w, 0.5 * h
    local = np.array([[+dx,+dy],[+dx,-dy],[-dx,-dy],[-dx,+dy]], dtype=np.float32)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R2 = np.array([[c,-s],[s,c]], dtype=np.float32)
    rot = (local @ R2.T)
    rot[:,0] += cx; rot[:,1] += cy
    return rot.astype(np.int32)

def _mask_from_polygon(frame_shape_hw, polygon_pts, erode_px=3):
    h, w = frame_shape_hw[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts.astype(np.int32)], 255)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def _classify_color_in_polygon(frame_bgr, polygon_pts):
    mask = _mask_from_polygon(frame_bgr.shape, polygon_pts, erode_px=3)
    ys, xs = np.where(mask == 255)
    if len(xs) < 15:
        return "grayscale"
    patch = frame_bgr[ys, xs]
    hsv = cv2.cvtColor(patch.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    H, S, V = hsv[:,0].astype(np.float32), hsv[:,1].astype(np.float32), hsv[:,2].astype(np.float32)
    valid = ~((V > 225) & (S < 35))
    H, S, V = H[valid], S[valid], V[valid]
    if H.size == 0:
        return "grayscale"
    sat_ok = S > 35
    Hs = H[sat_ok]
    if Hs.size == 0:
        b_mean, g_mean, r_mean = patch[:,0].mean(), patch[:,1].mean(), patch[:,2].mean()
        b_dom = (b_mean - max(g_mean, r_mean)) > 18 and (b_mean / (g_mean + r_mean + 1e-3)) > 0.85
        g_dom = (g_mean - max(b_mean, r_mean)) > 18 and (g_mean / (b_mean + r_mean + 1e-3)) > 0.85
        r_dom = (r_mean - max(b_mean, g_mean)) > 18 and (r_mean / (b_mean + g_mean + 1e-3)) > 0.85
        if b_dom: return "blue"
        if g_dom: return "green"
        if r_dom: return "red"
        return "grayscale"
    def hue_frac(center, width=18):
        d = np.minimum(np.abs(Hs - center), 180 - np.abs(Hs - center))
        return float((d <= width).mean())
    frac_blue  = hue_frac(120, 20)
    frac_green = hue_frac(60, 18)
    frac_red   = max(hue_frac(0, 15), hue_frac(180, 15))
    fractions = {"blue": frac_blue, "green": frac_green, "red": frac_red}
    label = max(fractions, key=fractions.get)
    if fractions[label] >= 0.22:
        return label
    b_mean, g_mean, r_mean = patch[:,0].mean(), patch[:,1].mean(), patch[:,2].mean()
    if (b_mean - max(g_mean, r_mean)) > 20: return "blue"
    if (g_mean - max(b_mean, r_mean)) > 20: return "green"
    if (r_mean - max(b_mean, g_mean)) > 20: return "red"
    lab = cv2.cvtColor(patch.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    a, b = lab[:,1].astype(np.float32), lab[:,2].astype(np.float32)
    chroma = np.sqrt(a*a + b*b)
    if float(np.median(chroma)) < 8.0:
        return "grayscale"
    return label

# --- Helpers for projection ---
def pixel_to_robot(px_xy):
    u, v = float(px_xy[0]), float(px_xy[1])
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    ray_cam = np.linalg.inv(K) @ pixel_h
    n = R[:, 2]
    X0 = tvec.reshape(3)
    t_param = np.dot(n, X0) / np.dot(n, ray_cam)
    point_cam = t_param * ray_cam
    point_tag = R.T @ (point_cam - X0)
    point_robot = tag_to_robot(point_tag * 1000)
    return point_robot

def centers_of_short_edges(corners):
    p = corners.astype(np.float64)
    edges = [(0,1),(1,2),(2,3),(3,0)]
    lengths = [np.linalg.norm(p[a]-p[b]) for a,b in edges]
    k = int(np.argmin(lengths))
    a,b = edges[k]; c,d = edges[(k+2) % 4]
    mid1 = 0.5*(p[a] + p[b]); mid2 = 0.5*(p[c] + p[d])
    return mid1, mid2

# --- Robot Move Function ---
def send_json(ser, cmd):
    if 'x' in cmd and 'y' in cmd and 'z' in cmd:
        coords = [cmd['x'], cmd['y'], cmd['z']]
        if not validate_robot_coords(coords):
            radial_dist = np.sqrt(cmd['x']**2 + cmd['y']**2)
            print(f"[{ts()}] [Robot] CRITICAL ERROR: Command REJECTED - Outside workspace!")
            print(f"        Command: {json.dumps(cmd)}")
            print(f"        Radial distance: {radial_dist:.1f}mm (max: 500mm)")
            print(f"        Z: {cmd['z']}mm (range: -100 to 450mm)")
            return False
    ser.write(json.dumps(cmd).encode() + b'\n')
    print(f"[{ts()}] Sent: {json.dumps(cmd)}")
    return True

def validate_robot_coords(xyz):
    x, y, z = xyz
    radial_distance = np.sqrt(x**2 + y**2)
    if radial_distance > 500:
        return False
    if not (-100 <= z <= 450):
        return False
    if radial_distance < 80:
        return False
    return True

# --- small vector helpers ---
def _norm(v):
    return float(np.linalg.norm(v))

def _safe_unit(v):
    n = _norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0], dtype=float), 0.0
    return (v / n), n

def _angle_between_2d(u, v):
    """Angle between 2D vectors u and v in radians, safe and non-negative."""
    u = np.asarray(u, dtype=float)[:2]
    v = np.asarray(v, dtype=float)[:2]
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    u = u / nu; v = v / nv
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(dot))

def move_roArm(robot_targets, port):
    """
    robot_targets: list with keys 'robot_xyz','angle_deg','angle_rad','pixel',
    'pixel_dimensions','cam_xyz','color','width_mm','height_mm',
    'tip1_robot','tip2_robot'
    """
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        ser.setRTS(False)
        ser.setDTR(False)
        time.sleep(1)
    except Exception as e:
        print(f"[{ts()}] [Robot] Could not open serial port: {e}")
        return
    for i, target in enumerate(robot_targets):
        xyz = np.array(target['robot_xyz'], dtype=float)  # center of pen in ROBOT coords
        angle = target['angle_deg']
        angle_rad = float(target.get('angle_rad', np.radians(angle)))
        color_label = target.get('color', 'unknown')

        if not validate_robot_coords(xyz):
            radial_distance = np.sqrt(xyz[0]**2 + xyz[1]**2)
            print(f"[{ts()}] [Robot] Target {i+1} REJECTED - Outside workspace:")
            print(f"        Coordinates: X={xyz[0]:.1f}, Y={xyz[1]:.1f}, Z={xyz[2]:.1f} mm")
            print(f"        Radial distance: {radial_distance:.1f}mm (max: 500mm)")
            print(f"        Workspace: 1-meter diameter circle, Z: -100 to +450mm")
            continue

        print(f"[{ts()}] [Robot] Target {i+1}: PIXEL={target.get('pixel')}, CAM={target.get('cam_xyz')}, ROBOT={xyz}, ANGLE={angle:.2f}deg, COLOR={color_label}")

        # ---- Your new angle rule ----
        tip1 = np.array(target.get('tip1_robot', xyz), dtype=float)
        tip2 = np.array(target.get('tip2_robot', xyz), dtype=float)

        # Define local "down" direction from center toward origin (XY only)
        radial_dir_xy = -xyz[:2]
        rd_norm = np.linalg.norm(radial_dir_xy)
        radial_unit = np.array([1.0, 0.0], dtype=float) if rd_norm < 1e-9 else (radial_dir_xy / rd_norm)

        # Pick the tip closer toward origin relative to the center (bigger dot with radial_unit)
        d1 = float(np.dot(tip1[:2] - xyz[:2], radial_unit))
        d2 = float(np.dot(tip2[:2] - xyz[:2], radial_unit))
        chosen_tip = tip1 if d1 >= d2 else tip2
        chosen_tip_name = "tip1" if d1 >= d2 else "tip2"

        # Angle between: (tip -> center) and (center -> origin). Always non-negative.
        v_tip_to_center = (xyz[:2] - chosen_tip[:2])
        v_center_to_origin = -xyz[:2]
        penRadialAngle = abs(_angle_between_2d(v_tip_to_center, v_center_to_origin))

        print(f"[{ts()}] [Robot] angle_between(({chosen_tip_name}->center), (center->origin))="
              f"{np.degrees(penRadialAngle):.1f}deg | chosen_tip={chosen_tip_name}")

        # Decide motion type
        motion_type = "STANDARD" if penRadialAngle < (np.pi/4.0) else "COMPLEX"
        print(f"[{ts()}] [Motion] START {motion_type} for target {i+1}")
        save_snapshot(f"motion_start_{motion_type.lower()}")

        if motion_type == "STANDARD":
            # STANDARD MOTION PLANNING
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 400, "y": 0, "z": 200, "t": 2.0});  time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(xyz[0]), "y": int(xyz[1]), "z": 50, "t": 2.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(xyz[0]), "y": int(xyz[1]), "z": int(xyz[2]), "t": 2.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(xyz[0]), "y": int(xyz[1]), "z": int(xyz[2]), "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 350, "y": 0, "z": 200, "t": 2.95}); time.sleep(1.0)
            color_to_y = {"blue": 140, "red": 70, "green": -70, "grayscale": -140}
            dest_y = int(color_to_y.get(color_label, 0))
            print(f"[{ts()}] [Robot] Step 7: Routing color '{color_label}' to y={dest_y}")
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 100, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 80, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 80, "t": 2.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.8});  time.sleep(1.0)
        else:
            # ===== COMPLEX MOTION =====
            print(f"[{ts()}] [Robot] Complex motion planning (angle ≥ 45°)")
            center = xyz.copy()

            # Local axis x' = center -> origin, y' = 90° ccw
            radial_vec = np.array([-center[0], -center[1]], dtype=float)
            radial_norm = np.linalg.norm(radial_vec)
            xprime = np.array([1.0, 0.0], dtype=float) if radial_norm < 1e-6 else radial_vec / radial_norm

            # Use the SAME chosen tip we used for the angle decision
            lower_tip = chosen_tip

            # Approach point: 1/4 from center, 3/4 from chosen tip
            approach_pt = center * 0.25 + lower_tip * 0.75

            # Distance from center to approach point
            d_ca = _norm(approach_pt - center)

            # Move from center toward the origin by that distance to get 'dest'
            dir_to_origin_xy = np.array([xprime[0], xprime[1], 0.0], dtype=float)
            dest = center + dir_to_origin_xy * d_ca

            # Distance between approach point and dest
            d_ad = _norm(approach_pt - dest)

            # Vector from dest to approach point defines the sweep direction
            v_dest_to_ap = approach_pt - dest
            v_unit, _ = _safe_unit(v_dest_to_ap)
            ext_len = 2.0 * d_ad + 10.0
            init = dest + v_unit * ext_len

            total_len = _norm(dest - init)
            if total_len < 1e-6:
                path_pts = []
            else:
                u, _ = _safe_unit(dest - init)
                # Keep at least 10mm shy of dest
                closer_dest = dest - u * 10.0
                seg_len = _norm(closer_dest - init)
                steps = int(seg_len // 10.0)
                path_pts = [init + u * (10.0 * k) for k in range(1, steps + 1)]
                path_pts.append(closer_dest)

            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 400, "y": 0, "z": 200, "t": 2.0});  time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(init[0]), "y": int(init[1]), "z": 50, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(init[0]), "y": int(init[1]), "z": int(xyz[2]), "t": 2.95}); time.sleep(1.0)

            print(f"[{ts()}] [Robot] Complex path sweep: starting send_json loop with {len(path_pts)} segment(s)...")
            for p in path_pts:
                send_json(ser, {"T": 1041, "x": int(p[0]), "y": int(p[1]), "z": int(xyz[2]), "t": 2.95})
                time.sleep(0.15)
            last_xy = path_pts[-1] if len(path_pts) else init
            print(f"[{ts()}] [Robot] Complex path sweep: send_json loop complete. Last point used: ({int(last_xy[0])}, {int(last_xy[1])})")

            send_json(ser, {"T": 1041, "x": int(last_xy[0]), "y": int(last_xy[1]), "z": 50, "t": 2.95}); time.sleep(0.2)
            send_json(ser, {"T": 1041, "x": 380, "y": 0, "z": 200, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.8});  time.sleep(1.0)

        # MOTION END LOG + snapshot
        print(f"[{ts()}] [Motion] END {motion_type} for target {i+1}")
        save_snapshot(f"motion_end_{motion_type.lower()}")

    ser.close()
    print(f"[{ts()}] [Robot] Sequence complete.")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
if not cap.isOpened():
    print("Could not open webcam")
    if robot_thread and robot_thread.is_alive():
        print(f"[{ts()}] Waiting for robot thread to finish...")
        robot_thread.join(timeout=10.0)
    finalize_and_rename_session()
    _tee.detach_file()
    _log_fh.close()
    sys.exit(1)
cv2.namedWindow("Pen Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pen Detection", 640, 480)

# --- Detection Queue ---
robot_queue = []
robot_queue_lock = threading.Lock()

def on_spacebar(robot_targets, trigger_source="SPACE"):
    """Kick off motion thread and record a trigger snapshot + log."""
    global robot_thread
    if not robot_targets:
        print(f"[{ts()}] [Robot] No confident detections to send.")
        return
    if robot_thread is not None and robot_thread.is_alive():
        print(f"[{ts()}] [Robot] Robot is busy. Wait for previous sequence to finish.")
        return
    print(f"[{ts()}] [Trigger] {trigger_source} activated with {len(robot_targets)} target(s).")
    save_snapshot(f"trigger_{trigger_source.lower()}")
    robot_thread = threading.Thread(target=move_roArm, args=(robot_targets, serial_port), daemon=True)
    robot_thread.start()

# ---------- AUTO MODE UI (clean, no overlap) ----------
def _draw_pill_badge(img, label, enabled=True, margin=20, alpha=0.85):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad_x, pad_y = 16, 10
    pill_w, pill_h = text_w + pad_x * 2, text_h + pad_y * 2

    h, w = img.shape[:2]
    x2 = w - margin
    y1 = margin
    x1 = x2 - pill_w
    y2 = y1 + pill_h
    radius = pill_h // 2
    cy = (y1 + y2) // 2

    fill = (60, 170, 60) if enabled else (90, 90, 90)
    border = (40, 120, 40) if enabled else (60, 60, 60)
    text_color = (255, 255, 255)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), fill, cv2.FILLED, lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (x1 + radius, cy), (radius, radius), 0, 90, 270, fill, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(overlay, (x2 - radius, cy), (radius, radius), 0, -90, 90, fill, cv2.FILLED, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border, 2, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border, 2, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, cy), (radius, radius), 0, 90, 270, border, 2, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, cy), (radius, radius), 0, -90, 90, border, 2, cv2.LINE_AA)

    text_x = x1 + (pill_w - text_w) // 2
    text_y = y1 + (pill_h + text_h) // 2 - 2
    cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# Auto mode state
auto_mode = False
last_auto_trigger_time = 0.0
AUTO_COOLDOWNSEC = 2.0  # prevent rapid re-triggering

def maybe_auto_trigger(detections_for_this_frame):
    global last_auto_trigger_time
    if not auto_mode:
        return
    if not detections_for_this_frame:
        return
    if (robot_thread is not None) and robot_thread.is_alive():
        return
    now = time.time()
    if now - last_auto_trigger_time < AUTO_COOLDOWNSEC:
        return
    print(f"[{ts()}] [Auto] Triggering motion based on confident detection.")
    on_spacebar(detections_for_this_frame, trigger_source="AUTO")
    last_auto_trigger_time = now

# ----------------- Main Loop -----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{ts()}] Failed to grab frame")
            break

        h_frame, w_frame = frame.shape[:2]
        top_ignore = int(0.2 * h_frame)  # IGNORE TOP 20% (y < top_ignore)

        # Run YOLO (OBB)
        results = model(frame, task="obb", verbose=False)
        r0 = results[0]

        # Start annotated as a copy of the raw frame; we draw only what we keep
        annotated = frame.copy()

        # Shade the ignored region for clarity
        shade = annotated.copy()
        cv2.rectangle(shade, (0, 0), (w_frame, top_ignore), (50, 50, 50), thickness=cv2.FILLED)
        cv2.addWeighted(shade, 0.25, annotated, 0.75, 0, annotated)
        cv2.putText(annotated, "ignored", (10, max(15, top_ignore - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # --- Collect filtered detections (conf >= 0.7 and center outside top 20%) ---
        detections_for_this_frame = []
        if getattr(r0, "obb", None) is not None and r0.obb.xywhr is not None:
            xywhr = r0.obb.xywhr
            confs = r0.obb.conf if hasattr(r0.obb, "conf") else None
            clss  = r0.obb.cls  if hasattr(r0.obb, "cls")  else None
            names = r0.names if hasattr(r0, "names") else {}

            # Move to numpy for safety
            xywhr_np = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else xywhr
            confs_np = confs.cpu().numpy() if (confs is not None and hasattr(confs, "cpu")) else confs
            clss_np  = clss.cpu().numpy()  if (clss  is not None and hasattr(clss,  "cpu")) else clss

            for i, box in enumerate(xywhr_np):
                conf = float(confs_np[i]) if confs_np is not None else 1.0
                if conf < 0.7:
                    continue

                cx, cy, w, h, angle = box.tolist()
                cx_i, cy_i, w_i, h_i = int(cx), int(cy), int(w), int(h)

                # FILTER: skip anything whose center is in the top 20% of the frame
                if cy_i < top_ignore:
                    continue

                angle_deg = angle * (180.0 / np.pi)

                # Draw OBB outline (blue) and label like YOLO
                corners = _rotated_rect_corners(cx_i, cy_i, w_i, h_i, angle)
                cv2.polylines(annotated, [corners.reshape(-1, 2)], isClosed=True, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                # YOLO-style label box
                cls_name = names.get(int(clss_np[i]), "obj") if clss_np is not None else "obj"
                label_txt = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                txt_x = max(0, cx_i - tw // 2)
                txt_y = max(0, cy_i - h_i // 2 - 8)
                cv2.rectangle(annotated, (txt_x, txt_y - th - 6), (txt_x + tw + 6, txt_y), (255, 0, 0), thickness=cv2.FILLED)
                cv2.putText(annotated, label_txt, (txt_x + 3, txt_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Geometry & conversions
                pixel_homog = np.array([cx_i, cy_i, 1.0])
                ray_cam = np.linalg.inv(K) @ pixel_homog
                n = R[:, 2]
                X0 = tvec.reshape(3)
                t_param = np.dot(n, X0) / np.dot(n, ray_cam)
                point_cam = t_param * ray_cam
                point_tag = R.T @ (point_cam - X0)
                point_robot = tag_to_robot(point_tag * 1000)

                pixel_to_mm_scale = t_param * 1000
                width_mm = (w_i * pixel_to_mm_scale) / K[0, 0]
                height_mm = (h_i * pixel_to_mm_scale) / K[1, 1]

                # Short-edge midpoints
                tip_mid_px1, tip_mid_px2 = centers_of_short_edges(corners)
                cv2.circle(annotated, (int(tip_mid_px1[0]), int(tip_mid_px1[1])), 4, (0, 255, 255), -1)
                cv2.circle(annotated, (int(tip_mid_px2[0]), int(tip_mid_px2[1])), 4, (0, 255, 255), -1)

                # To robot
                tip1_robot = pixel_to_robot(tip_mid_px1)
                tip2_robot = pixel_to_robot(tip_mid_px2)

                # Color inside OBB
                color_label = _classify_color_in_polygon(frame, corners)

                # Two-line yellow info
                line1 = (f"Pix:({cx_i},{cy_i}) | Size:{width_mm:.1f}x{height_mm:.1f}mm "
                         f"| Angle:{angle:.2f}rad/{angle_deg:.1f}deg | Color:{color_label}")
                line2 = (f"Robot:({point_robot[0]:.0f},{point_robot[1]:.0f},{point_robot[2]:.0f})mm "
                         f"| TipsRobot:({tip1_robot[0]:.0f},{tip1_robot[1]:.0f},{tip1_robot[2]:.0f})/"
                         f"({tip2_robot[0]:.0f},{tip2_robot[1]:.0f},{tip2_robot[2]:.0f})mm")

                base_x, base_y = cx_i - 100, max(20, cy_i - 20)
                cv2.putText(annotated, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated, line2, (base_x, base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                detections_for_this_frame.append({
                    'robot_xyz': point_robot,
                    'angle_deg': float(angle_deg),
                    'angle_rad': float(angle),
                    'pixel': (cx_i, cy_i),
                    'pixel_dimensions': (w_i, h_i),
                    'cam_xyz': point_cam.copy(),
                    'color': color_label,
                    'width_mm': float(width_mm),
                    'height_mm': float(height_mm),
                    'tip1_robot': tip1_robot.astype(float),
                    'tip2_robot': tip2_robot.astype(float),
                })

        # --- Draw coordinate frame and info ---
        cv2.arrowedLine(annotated, (30, 30), (80, 30), (0, 0, 255), 2)  # X-axis (right)
        cv2.arrowedLine(annotated, (30, 30), (30, 80), (0, 255, 0), 2)  # Y-axis (down)
        cv2.putText(annotated, "X", (85, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated, "Y", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Resolution: {w_frame}x{h_frame}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, "0 rads @ axis", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Auto/Space badge ---
        _draw_pill_badge(annotated, "AUTO" if auto_mode else "SPACE", enabled=auto_mode, margin=20)

        # Update cached annotated (final UI frame) for snapshots
        with latest_lock:
            latest_annotated = annotated.copy()

        # Init video writer once (based on first final annotated frame)
        if video_writer is None:
            with _video_lock:
                if video_writer is None:
                    _init_video_writer(annotated, cap)

        # Append frame to video (the same one you see)
        with _video_lock:
            if video_writer is not None:
                try:
                    video_writer.write(annotated)
                except Exception as e:
                    print(f"[{ts()}] ERROR writing video frame: {e}")

        # Show UI
        cv2.imshow("Pen Detection", annotated)

        # Auto trigger if enabled
        maybe_auto_trigger(detections_for_this_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"[{ts()}] Quit key received.")
            break
        elif key == 32:  # Spacebar
            on_spacebar(detections_for_this_frame, trigger_source="SPACE")
        elif key in (ord('u'), ord('U')):  # Toggle Auto
            auto_mode = not auto_mode
            state = "ON" if auto_mode else "OFF"
            print(f"[{ts()}] [UI] Auto mode toggled {state}")

except KeyboardInterrupt:
    print(f"\n[{ts()}] KeyboardInterrupt received. Shutting down...")

finally:
    # 1) Close camera/UI
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass

    # 2) Wait for robot thread to finish so it won't print after log closes
    try:
        if robot_thread and robot_thread.is_alive():
            print(f"[{ts()}] Waiting for robot thread to finish...")
            robot_thread.join(timeout=20.0)
            if robot_thread.is_alive():
                print(f"[{ts()}] Robot thread still running; proceeding with shutdown.")
    except Exception as e:
        print(f"[{ts()}] Error while joining robot thread: {e}")

    # 3) Finalize session (release video & move folder)
    finalize_and_rename_session()

    # 4) Detach tee before closing file, then close file
    try:
        _tee.detach_file()
        _log_fh.close()
    except:
        pass
