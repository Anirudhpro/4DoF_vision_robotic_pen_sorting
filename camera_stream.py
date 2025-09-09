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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc  # for drawing angle arcs
plt.ion()  # enable interactive mode so plt.show() windows update live
import io

# Global visualization data containers
sent_robot_commands = []  # appended in send_json
_next_motion_id = 1
_active_motion_id = None

# Track indices of sent_robot_commands that belong to the active complex motion
_current_complex_indices = []
_complex_motion_active = False  # define before send_json references it

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
# Use a non-hidden directory name so it is visible in Finder immediately
temp_session_dir = os.path.join(logs_root, f"session_tmp_{_session_tag}")
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
# Load robot configuration
with open('config.json') as f:
    config = json.load(f)
robot_tag_xyz = np.array(config['robot_tag_xyz'])  # mm - robot arm position relative to ArUco tag
# Prefer explicit config; fall back to 270° using precise pi
_theta_rad_cfg = config.get('robot_tag_theta_rad')
_theta_deg_cfg = config.get('robot_tag_theta_deg')
if _theta_rad_cfg is not None:
    robot_tag_theta = float(_theta_rad_cfg)
elif _theta_deg_cfg is not None:
    robot_tag_theta = float(_theta_deg_cfg) * (np.pi / 180.0)
else:
    robot_tag_theta = 1.5 * np.pi

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
    # Log command for visualization
    try:
        if cmd.get('T') == 1041 and all(k in cmd for k in ('x','y','z')):
            # attach current active motion id (may be None)
            mid = _active_motion_id
            sent_robot_commands.append({
                'ts': time.time(),
                'x': float(cmd['x']),
                'y': float(cmd['y']),
                'z': float(cmd['z']),
                't': float(cmd.get('t', 0)),
                'raw': dict(cmd),
                'motion_id': mid,
                'hide': False
            })
            # Keep only recent N to bound memory
            if len(sent_robot_commands) > 500:
                del sent_robot_commands[:len(sent_robot_commands)-500]
    except Exception as _e:
        pass
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

def _pen_radial_angle(center_xy, tip_xy):
    """
    Angle (0..180°) between:
      a) center -> tip (vector a)
      b) center -> origin (vector r)
    Returns angle in radians (no supplemental 180° flipping).
    """
    c = np.asarray(center_xy, dtype=float)
    t = np.asarray(tip_xy, dtype=float)
    # Use center -> tip (intuitive direction for plotting)
    a = t - c              # center -> tip
    r = -c                 # center -> origin
    a_len = np.linalg.norm(a)
    r_len = np.linalg.norm(r)
    if a_len < 1e-9 or r_len < 1e-9:
        return 0.0
    cos_th = float(np.clip(np.dot(a / a_len, r / r_len), -1.0, 1.0))
    theta = float(np.arccos(cos_th))  # 0..pi
    return theta

# Helper: compute motion path points (2D XY robot coords) for visualization
def _compute_motion_path_points(center_xy, chosen_tip_xy, color_label, pen_radial_angle_rad, tip1_xy=None, tip2_xy=None):
    """Return a list of 2D points (robot-frame XY) representing the motion waypoints
    that move_roArm would use for this detection. This mirrors the logic in move_roArm
    but only for visualization (XY only)."""
    c = np.asarray(center_xy, dtype=float)
    tip = np.asarray(chosen_tip_xy, dtype=float)

    if pen_radial_angle_rad < (np.pi / 4.0):
        # STANDARD sequence: same waypoint logic as move_roArm
        # Compute a 10mm perpendicular offset from center to the line defined by the two tips
        # If tips are not provided, fall back to no offset
        if tip1_xy is not None and tip2_xy is not None:
            t1 = np.asarray(tip1_xy, dtype=float)[:2]
            t2 = np.asarray(tip2_xy, dtype=float)[:2]
            tip_vec = t2 - t1
            if np.linalg.norm(tip_vec) >= 1e-6:
                perp = np.array([-tip_vec[1], tip_vec[0]], dtype=float)
                perp_unit = perp / (np.linalg.norm(perp) + 1e-9)
                cand1 = c + perp_unit * 10.0
                cand2 = c - perp_unit * 10.0
                # Simple rule (same as viz & move_roArm): pick the candidate with larger Y
                new_center = cand1 if cand1[1] > cand2[1] else cand2
            else:
                new_center = c.copy()
        else:
            new_center = c.copy()

        pts = [
            np.array([120.0, 0.0]),
            np.array([400.0, 0.0]),
            new_center.copy(),
            np.array([350.0, 0.0])
        ]
        color_to_y = {"blue": 140, "red": 70, "green": -70, "grayscale": -140}
        dest_y = float(color_to_y.get(color_label, 0.0))
        pts.append(np.array([480.0, dest_y]))
        return [p.astype(float) for p in pts]
    else:
        # COMPLEX path: replicate the geometric construction from move_roArm
        originalCenter = np.array([c[0], c[1], 0.0], dtype=float)
        radial_vec = -originalCenter[:2]
        radial_norm = np.linalg.norm(radial_vec)
        xprime = np.array([1.0, 0.0], dtype=float) if radial_norm < 1e-6 else (radial_vec / radial_norm)

        lower_tip = np.array([tip[0], tip[1], 0.0], dtype=float)
        center3 = originalCenter.copy()
        # Use approach point closer to tip: 0.25*center + 0.75*chosen_tip
        approach_pt = center3 * 0.25 + lower_tip * 0.75
        d_ca = np.linalg.norm(approach_pt - center3)
        dir_to_origin_xy = np.array([xprime[0], xprime[1], 0.0], dtype=float)
        dest = center3 + dir_to_origin_xy * d_ca
        d_ad = np.linalg.norm(approach_pt - dest)
        v_dest_to_ap = approach_pt - dest
        v_unit = v_dest_to_ap / (np.linalg.norm(v_dest_to_ap) + 1e-9)
        ext_len = 2.0 * d_ad + 10.0
        init = dest + v_unit * ext_len

        total_len = np.linalg.norm(dest - init)
        path_pts = []
        if total_len >= 1e-6:
            u = (dest - init) / total_len
            closer_dest = dest - u * 10.0
            seg_len = np.linalg.norm(closer_dest - init)
            steps = int(seg_len // 10.0)
            for k in range(1, steps + 1):
                path_pts.append(init + u * (10.0 * k))
            path_pts.append(closer_dest)
            # If there are more than 7 path points, prune the last four to match robot/viz agreement
            if len(path_pts) > 7:
                path_pts = path_pts[:-4]

        pts3 = [init, dest] + path_pts
        pts2d = [np.array([p[0], p[1]], dtype=float) for p in pts3]
        return pts2d

# Helper: simple smoothing / densify of waypoint polyline for nicer curve
def _densify_and_smooth_polyline(pts, samples=80, smooth_window=5):
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 2:
        return pts
    xs = pts[:, 0]
    ys = pts[:, 1]
    # cumulative distance param
    d = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    t = np.concatenate(([0.0], np.cumsum(d)))
    if t[-1] == 0:
        return pts
    t_norm = t / t[-1]
    t_dense = np.linspace(0.0, 1.0, samples)
    x_dense = np.interp(t_dense, t_norm, xs)
    y_dense = np.interp(t_dense, t_norm, ys)
    # simple moving average smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / float(smooth_window)
        x_dense = np.convolve(x_dense, kernel, mode='same')
        y_dense = np.convolve(y_dense, kernel, mode='same')
    return np.vstack([x_dense, y_dense]).T

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
        # --- Fallback extraction to avoid KeyError ---
        raw_xyz = target.get('robot_xyz')
        if raw_xyz is None:
            if all(k in target for k in ('x','y','z')):
                raw_xyz = [target['x'], target['y'], target['z']]
            else:
                print(f"[{ts()}] [Robot] Target {i+1} missing robot_xyz; skipping. Keys={list(target.keys())}")
                continue
        try:
            xyz = np.array(raw_xyz, dtype=float)
        except Exception:
            print(f"[{ts()}] [Robot] Target {i+1} invalid robot_xyz={raw_xyz}; skipping.")
            continue
        angle = target.get('angle_deg')
        if angle is None:
            # fallback to generic 'angle' if present
            angle = target.get('angle', 0.0)
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

        # ---- Choose tip by perpendicular-to-radial rule (from previous step) ----
        tip1 = np.array(target.get('tip1_robot', xyz), dtype=float)
        tip2 = np.array(target.get('tip2_robot', xyz), dtype=float)

        radial_xy = -xyz[:2]
        rd_norm = np.linalg.norm(radial_xy)
        rhat = np.array([1.0, 0.0], dtype=float) if rd_norm < 1e-9 else (radial_xy / rd_norm)

        def foot_and_score(P):
            s = float(np.dot(P[:2] - xyz[:2], rhat))  # signed distance along radial from center
            F = xyz[:2] + s * rhat                    # perpendicular foot on radial line
            distF = float(np.linalg.norm(F))          # distance of that foot to origin
            prefer_penalty = 0 if s >= 0 else 1       # prefer feet "down" toward origin
            return F, s, distF, (prefer_penalty, distF, -s)

        F1, s1, dF1, score1 = foot_and_score(tip1)
        F2, s2, dF2, score2 = foot_and_score(tip2)
        chosen_tip = tip1 if score1 < score2 else tip2
        chosen_tip_name = "tip1" if score1 < score2 else "tip2"

        print(f"[{ts()}] [Robot] Tip selection: "
              f"tip1(s={s1:.1f}, |F|={dF1:.1f}) vs tip2(s={s2:.1f}, |F|={dF2:.1f}) -> {chosen_tip_name}")

        # ---- penRadialAngle exactly per diagram ----
        a_len = np.linalg.norm(xyz[:2] - chosen_tip[:2])
        penRadialAngle = _pen_radial_angle(xyz[:2], chosen_tip[:2])

        print(f"[{ts()}] [Robot] penRadialAngle={np.degrees(penRadialAngle):.1f}deg "
              f"(||tip->center||={a_len:.1f}mm; matched on radial to same length)")

        # --- Shift center by 10mm perpendicular to radial direction ---
        # Use the detected center directly; remove the 10mm perpendicular shift (no new_center)
        originalCenter = xyz.copy()
        radial_vec = -originalCenter[:2]
        radial_norm = np.linalg.norm(radial_vec)
        rhat = np.array([1.0, 0.0], dtype=float) if radial_norm < 1e-9 else (radial_vec / radial_norm)
        perp = np.array([-rhat[1], rhat[0]])  # Perpendicular direction in XY (kept for reference)

        # Decide motion type
        print(penRadialAngle)
        motion_type = "STANDARD" if penRadialAngle < (np.pi / 4.0) else "COMPLEX"
        print(f"[{ts()}] [Motion] START {motion_type} for target {i+1}")
        save_snapshot(f"motion_start_{motion_type.lower()}")

        if motion_type == "STANDARD":
            # STANDARD MOTION PLANNING
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.95}); time.sleep(0.5)
            send_json(ser, {"T": 1041, "x": 400, "y": 0, "z": 200, "t": 2.0});  time.sleep(0.9)
            # Compute a 10mm perpendicular offset from the center to the line defined by the two tips
            try:
                tip1_xy = np.asarray(tip1, dtype=float)[:2]
                tip2_xy = np.asarray(tip2, dtype=float)[:2]
                center_xy = originalCenter[:2]
                tip_vec = tip2_xy - tip1_xy
                tip_len = np.linalg.norm(tip_vec)
                if tip_len < 1e-6:
                    # degenerate tip line -> fallback to original center
                    chosen_xy = center_xy.copy()
                else:
                    # perpendicular direction to tip line
                    perp = np.array([-tip_vec[1], tip_vec[0]], dtype=float)
                    perp_unit = perp / (np.linalg.norm(perp) + 1e-9)
                    cand1 = center_xy + perp_unit * 10.0
                    cand2 = center_xy - perp_unit * 10.0
                    # Use same simple rule as the viz: pick the candidate with larger Y
                    chosen_xy = cand1 if cand1[1] > cand2[1] else cand2
                new_center = np.array([float(chosen_xy[0]), float(chosen_xy[1]), float(originalCenter[2])], dtype=float)
                print(f"[{ts()}] [Robot] STANDARD: using perpendicular offset point -> ({new_center[0]:.1f}, {new_center[1]:.1f}, {new_center[2]:.1f})")
            except Exception as _e:
                print(f"[{ts()}] [Robot] STANDARD: error computing perp offset, using original center: {_e}")
                new_center = originalCenter.copy()

            # Hover above the chosen pickup point, then descend
            send_json(ser, {"T": 1041, "x": int(new_center[0]), "y": int(new_center[1]), "z": 50, "t": 2.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(new_center[0]), "y": int(new_center[1]), "z": int(new_center[2]), "t": 2.0}); time.sleep(1.5)
            send_json(ser, {"T": 1041, "x": int(new_center[0]), "y": int(new_center[1]), "z": int(new_center[2]), "t": 3.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 350, "y": 0, "z": 200, "t": 2.95}); time.sleep(1.0)
            color_to_y = {"blue": 140, "red": 70, "green": -70, "grayscale": -140}
            dest_y = int(color_to_y.get(color_label, 0))
            print(f"[{ts()}] [Robot] Step 7: Routing color '{color_label}' to y={dest_y}")
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 100, "t": 3.0}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 60, "t": 3.0}); time.sleep(0.5)
            send_json(ser, {"T": 1041, "x": 480, "y": dest_y, "z": 60, "t": 2.0}); time.sleep(0.5)
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.8})
        else:
            global _complex_motion_active
            _complex_motion_active = True
            # ===== COMPLEX MOTION =====
            print(f"[{ts()}] [Robot] Complex motion planning (angle ≥ 45°)")
            center = originalCenter.copy()
            radial_vec = np.array([-center[0], -center[1]], dtype=float)
            radial_norm = np.linalg.norm(radial_vec)
            xprime = np.array([1.0, 0.0], dtype=float) if radial_norm < 1e-6 else radial_vec / radial_norm
            lower_tip = chosen_tip
            # Ensure lower_tip is 3D (some earlier metadata may hold 2D XY); promote if needed
            if lower_tip.shape[0] == 2:
                lower_tip = np.array([lower_tip[0], lower_tip[1], center[2]], dtype=float)
            # Approach point weighting (0.75 center + 0.25 tip) with consistent 3D shapes
            approach_pt = center * 0.25 + lower_tip * 0.75
            d_ca = _norm(approach_pt - center)
            dir_to_origin_xy = np.array([xprime[0], xprime[1], 0.0], dtype=float)
            dest = center + dir_to_origin_xy * d_ca
            d_ad = _norm(approach_pt - dest)
            v_dest_to_ap = approach_pt - dest
            v_unit, _ = _safe_unit(v_dest_to_ap)
            ext_len = 2.0 * d_ad + 10.0
            init = dest + v_unit * ext_len

            total_len = _norm(dest - init)
            path_pts = []
            if total_len >= 1e-6:
                u, _ = _safe_unit(dest - init)
                closer_dest = dest - u * 10.0
                seg_len = _norm(closer_dest - init)
                steps = int(seg_len // 10.0)
                for k in range(1, steps + 1):
                    path_pts.append(init + u * (10.0 * k))
                path_pts.append(closer_dest)
                # If there are more than 7 path points, prune the last four as requested
                if len(path_pts) > 7:
                    path_pts = path_pts[:-4]

            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.95}); time.sleep(0.5)
            send_json(ser, {"T": 1041, "x": 400, "y": 0, "z": 200, "t": 2.0});  time.sleep(0.9)
            send_json(ser, {"T": 1041, "x": int(init[0]), "y": int(init[1]), "z": 50, "t": 2.95}); time.sleep(1.0)
            send_json(ser, {"T": 1041, "x": int(init[0]), "y": int(init[1]), "z": int(center[2]), "t": 2.95}); time.sleep(1.0)
            print(f"[{ts()}] [Robot] Complex path sweep: starting send_json loop with {len(path_pts)} segment(s)...")
            for p in path_pts:
                send_json(ser, {"T": 1041, "x": int(p[0]), "y": int(p[1]), "z": int(center[2]), "t": 2.95}); time.sleep(0.40)
            last_xy = path_pts[-1] if len(path_pts) else init
            print(f"[{ts()}] [Robot] Complex path sweep: send_json loop complete. Last point used: ({int(last_xy[0]), int(last_xy[1])})")
            send_json(ser, {"T": 1041, "x": 380, "y": 0, "z": 200, "t": 2.95}); time.sleep(0.5)
            send_json(ser, {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.8});  time.sleep(0.5)

        # MOTION END LOG + snapshot
        print(f"[{ts()}] [Motion] END {motion_type} for target {i+1}")
        save_snapshot(f"motion_end_{motion_type.lower()}")

    ser.close()
    print(f"[{ts()}] [Robot] Sequence complete.")
    # Clear/hide sent command markers for this motion so they disappear from viz
    try:
        ended = _active_motion_id
        for c in sent_robot_commands:
            if c.get('motion_id') == ended:
                c['hide'] = True
    except Exception:
        pass
    # deactivate current motion id
    _active_motion_id = None

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
    global robot_thread, _next_motion_id, _active_motion_id
    if not robot_targets:
        print(f"[{ts()}] [Robot] No confident detections to send.")
        return
    if robot_thread is not None and robot_thread.is_alive():
        print(f"[{ts()}] [Robot] Robot is busy. Wait for previous sequence to finish.")
        return
    # assign a new motion id and mark active so sent commands are tagged
    motion_id = _next_motion_id
    _next_motion_id += 1
    _active_motion_id = motion_id
    print(f"[{ts()}] [Trigger] {trigger_source} activated with {len(robot_targets)} target(s). (motion_id={motion_id})")
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

def _create_robot_coord_plot(robot_coords, width=320, height=240):
    """Creates a 3D scatter plot of robot coordinates and returns it as an image buffer.
    NOTE: Use direct Y (no reversal) for horizontal axis per user request. Mapping used for plotting:
    plot_x = y, plot_y = x, plot_z = z
    """
    if not robot_coords:
        return None

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    coords = np.array(robot_coords)
    # Use Y directly (no negation) for horizontal axis; vertical = X
    xs = coords[:, 1]
    ys = coords[:, 0]
    zs = coords[:, 2]

    ax.scatter(xs, ys, zs, c='r', marker='o')

    ax.set_xlabel('Y (left +)', fontsize=8)
    ax.set_ylabel('X (up +)', fontsize=8)
    ax.set_zlabel('Z (mm)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_title('Robot Coordinates (visual frame)', fontsize=10)

    # Axis limits (Y range horizontally, X vertically)
    ax.set_xlim([-250, 250])
    ax.set_ylim([0, 500])
    ax.set_zlim([-100, 450])
    ax.view_init(elev=20., azim=-60)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plot_img = cv2.imdecode(img_arr, 1)

    return plot_img

# Interactive matplotlib viz globals and helpers
viz_initialized = False
viz_fig = None
viz_ax_xy = None
viz_scatter_det_xy = None
viz_radial_lines = []
viz_tip_markers = []
viz_txt_obj = None
viz_angle_artists = []
viz_motion_artists = []  # curve + moving marker + waypoints (cleared each frame)
last_detection_metadata = []  # ensure defined before any handler references


def init_interactive_viz():
    """Create interactive matplotlib figure and empty artists. Call once."""
    global viz_initialized, viz_fig, viz_ax_xy, viz_scatter_det_xy, viz_radial_lines, viz_txt_obj, viz_tip_markers, viz_angle_artists, viz_motion_artists
    viz_fig = plt.figure(figsize=(6, 6))
    viz_ax_xy = viz_fig.add_subplot(111)
    viz_scatter_det_xy = viz_ax_xy.scatter([], [], c=[], s=48, edgecolors='k')
    viz_radial_lines = []
    viz_tip_markers = []
    viz_angle_artists = []
    viz_motion_artists = []
    viz_ax_xy.set_title('XY (top-down) - visual frame')
    viz_ax_xy.set_xlim(250, -250)
    viz_ax_xy.set_ylim(0, 500)
    viz_ax_xy.set_xlabel('Y (left +)')
    viz_ax_xy.set_ylabel('X (up +)')
    viz_ax_xy.grid(alpha=0.3)
    viz_txt_obj = viz_fig.text(0.01, 0.01, '', va='bottom', ha='left', fontsize=8, family='monospace')
    viz_fig.tight_layout(rect=[0, 0.05, 1, 1])

    # connect key handler so space in this window triggers motion
    def _on_viz_key(event):
        # Allow both space (trigger) and 'u'/'U' (toggle auto) from the viz window
        try:
            global auto_mode
            if event.key == ' ':  # spacebar
                # Build robot_targets in expected structure
                robot_targets = []
                for d in last_detection_metadata:
                    if d.get('pen_radial_angle_rad') is None:
                        continue
                    rx = d.get('robot_xyz') or [d.get('x'), d.get('y'), d.get('z')]
                    if None in rx:
                        continue
                    tip1r = d.get('tip1_robot') or d.get('tip1')
                    tip2r = d.get('tip2_robot') or d.get('tip2')
                    robot_targets.append({
                        'robot_xyz': rx,
                        'angle_deg': d.get('angle_deg', d.get('angle', 0.0)),
                        'angle_rad': d.get('angle_rad', d.get('pen_radial_angle_rad')),
                        'pixel': d.get('pixel'),
                        'cam_xyz': d.get('cam_xyz'),
                        'color': d.get('color','unknown'),
                        'tip1_robot': tip1r,
                        'tip2_robot': tip2r,
                    })
                if not robot_targets:
                    print(f"[{ts()}] [VIZ_SPACE] No valid targets to trigger.")
                    return
                on_spacebar(robot_targets, trigger_source="VIZ_SPACE")
            elif event.key in ('u', 'U'):
                auto_mode = not auto_mode
                state = "ON" if auto_mode else "OFF"
                print(f"[{ts()}] [VIZ_UI] Auto mode toggled {state} from viz window")
        except Exception as e:
            print(f"[{ts()}] Viz key handler error: {e}")
    viz_fig.canvas.mpl_connect('key_press_event', _on_viz_key)

    plt.show(block=False)
    viz_initialized = True


def update_interactive_viz(detections_meta, command_log):
    """Update interactive XY plot. Only most recent complex path is shown (no buildup)."""
    global viz_initialized, viz_fig, viz_ax_xy, viz_scatter_det_xy, viz_radial_lines, viz_txt_obj, viz_tip_markers, viz_angle_artists, viz_motion_artists
    if not viz_initialized:
        init_interactive_viz()

    # Map coordinates for scatter (plot_x = y, plot_y = x)
    det_x = [d['y'] for d in detections_meta]
    det_y = [d['x'] for d in detections_meta]
    det_c = [d.get('color', 'grayscale') for d in detections_meta]

    # Defensive: remove any stray Line2D artists left on the axes (fixes persistent red Xs)
    try:
        for ln in list(viz_ax_xy.lines):
            try:
                ln.remove()
            except Exception:
                pass
    except Exception:
        pass

    base_rgb = {
        'blue': (0.0, 0.34, 0.95),
        'red': (0.9, 0.1, 0.1),
        'green': (0.0, 0.6, 0.0),
        'grayscale': (0.5, 0.5, 0.5)
    }
    occurrences = {}
    shaded_colors = []
    for c in det_c:
        occurrences[c] = occurrences.get(c, 0) + 1
        idx = occurrences[c] - 1
        base = base_rgb.get(c, (0.4, 0.4, 0.4))
        factor = 0.0 if idx == 0 else min(0.75, 0.25 + 0.15 * idx)
        r, g, b = base
        shade = (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)
        shaded_colors.append(shade)
    det_colors = shaded_colors

    if det_x:
        viz_scatter_det_xy.set_offsets(np.c_[det_x, det_y])
        viz_scatter_det_xy.set_color(det_colors)
    else:
        viz_scatter_det_xy.set_offsets(np.empty((0, 2)))
        viz_scatter_det_xy.set_color([])

    # Remove previous frame artists (complete flush)
    for coll in (viz_radial_lines, viz_tip_markers, viz_angle_artists, viz_motion_artists):
        for art in coll:
            try: art.remove()
            except Exception: pass
    viz_radial_lines = []
    viz_tip_markers = []
    viz_angle_artists = []
    viz_motion_artists = []

    # Radial lines
    for (x, y, c) in zip(det_x, det_y, det_colors):
        try:
            ln, = viz_ax_xy.plot([0.0, x], [0.0, y], linestyle='--', color=c, linewidth=1.0, alpha=0.9)
            viz_radial_lines.append(ln)
        except Exception:
            pass

    t_now = time.time()
    blink_alpha = 0.5 + 0.5 * np.sin(t_now * 6.0)

    # Tip markers
    for det, c in zip(detections_meta, det_colors):
        tip1 = det.get('tip1'); tip2 = det.get('tip2'); chosen = det.get('chosen_tip')
        if tip1 and tip2:
            t1x, t1y = tip1[1], tip1[0]
            t2x, t2y = tip2[1], tip2[0]
            alpha_norm = 0.95
            try:
                mk1, = viz_ax_xy.plot(t1x, t1y, '*', markersize=10, color=c, alpha=(blink_alpha if chosen == 'tip1' else alpha_norm), markeredgecolor='k')
                mk2, = viz_ax_xy.plot(t2x, t2y, '*', markersize=10, color=c, alpha=(blink_alpha if chosen == 'tip2' else alpha_norm), markeredgecolor='k')
                viz_tip_markers.extend([mk1, mk2])
            except Exception:
                pass

    drawn_complex_path = False

    for det, c in zip(detections_meta, det_colors):
        try:
            pra_rad = det.get('pen_radial_angle_rad')
            if pra_rad is None:
                pra_deg_fallback = det.get('pen_radial_angle_deg')
                if pra_deg_fallback is None:
                    continue
                pra_rad = np.radians(float(pra_deg_fallback))
            pra_rad = float(pra_rad)
            pra_deg = np.degrees(pra_rad)
            cx_r, cy_r = det['x'], det['y']
            cx_p, cy_p = cy_r, cx_r
            chosen = det.get('chosen_tip'); tip1 = det.get('tip1'); tip2 = det.get('tip2')
            if not tip1 or not tip2:
                continue
            chosen_tip_xy = tip1 if chosen == 'tip1' else tip2
            v_tip = np.array([chosen_tip_xy[0] - cx_r, chosen_tip_xy[1] - cy_r], float)
            v_orig = np.array([-cx_r, -cy_r], float)
            if np.linalg.norm(v_tip) < 1e-6 or np.linalg.norm(v_orig) < 1e-6:
                continue
            u1 = v_tip / np.linalg.norm(v_tip); u2 = v_orig / np.linalg.norm(v_orig)
            u1p = np.array([u1[1], u1[0]]); u2p = np.array([u2[1], u2[0]])
            ang1 = np.degrees(np.arctan2(u1p[1], u1p[0])); ang2 = np.degrees(np.arctan2(u2p[1], u2p[0]))
            diff = (ang2 - ang1 + 360) % 360
            if diff > 180:
                ang1, ang2 = ang2, ang1
            age = t_now - float(det.get('ts', t_now))
            decay = max(0.0, 1.0 - age / 1.0)
            LIVE_AGE_THRESHOLD = 0.25
            radius = min(35.0, max(18.0, 0.25 * np.linalg.norm([cx_r, cy_r])))
            arc_alpha = 0.6 * (decay if pra_rad >= (np.pi/4.0) else 1.0)
            arc = Arc((cx_p, cy_p), 2*radius, 2*radius, angle=0, theta1=ang1, theta2=ang2, color=c, lw=1.2, alpha=arc_alpha)
            viz_ax_xy.add_patch(arc); viz_angle_artists.append(arc)
            ray_alpha = 0.75 * (decay if pra_rad >= (np.pi/4.0) else 1.0)
            r1, = viz_ax_xy.plot([cx_p, cx_p + u1p[0]*radius], [cy_p, cy_p + u1p[1]*radius], color=c, lw=1.0, alpha=ray_alpha)
            r2, = viz_ax_xy.plot([cx_p, cx_p + u2p[0]*radius], [cy_p, cy_p + u2p[1]*radius], color=c, lw=1.0, alpha=ray_alpha)
            viz_angle_artists.extend([r1, r2])
            bis = u1 + u2
            if np.linalg.norm(bis) < 1e-6: bis = u1
            bis /= np.linalg.norm(bis); bis_p = np.array([bis[1], bis[0]])
            tx = cx_p + bis_p[0]*radius*0.75; ty = cy_p + bis_p[1]*radius*0.75
            txt = viz_ax_xy.text(tx, ty, f"{pra_deg:.0f}°", color=c, fontsize=8, ha='center', va='center',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.6))
            viz_angle_artists.append(txt)

            # Complex path (only one, only live)
            if (not drawn_complex_path) and pra_rad >= (np.pi/4.0) and age <= LIVE_AGE_THRESHOLD:
                motion_pts = _compute_motion_path_points(np.array([cx_r, cy_r]), chosen_tip_xy, det.get('color','grayscale'), pra_rad)
                if motion_pts:
                    # Removed smooth/densified curve drawing — draw only waypoint markers for clarity
                    # Waypoint markers with progressive alpha (start faint -> end opaque) to show direction
                    n_wp = len(motion_pts)
                    for wi, mp in enumerate(motion_pts):
                        mx, my = mp[0], mp[1]
                        progress = wi / (n_wp - 1) if n_wp > 1 else 1.0
                        # alpha scales with progress; endpoints still orange
                        if wi in (0, n_wp-1):
                            col = 'orange'; msize = 7; a = 0.95 * decay
                        else:
                            col = 'yellow'; msize = 5; a = (0.25 + 0.7 * progress) * decay  # fade-in along path
                        try:
                            wp, = viz_ax_xy.plot(my, mx, 'o', color=col, markersize=msize, alpha=a, markeredgecolor='k')
                            viz_motion_artists.append(wp)
                        except Exception:
                            pass
                    drawn_complex_path = True

            mode_label = 'COMPLEX' if pra_rad >= (np.pi/4.0) else 'STANDARD'
            # when motion will be STANDARD, compute the perpendicular 10mm offset used in move_roArm
            if mode_label == "STANDARD":
                try:
                    # get tip coordinates (fall back to detection tips or centers)
                    t1 = np.asarray(det.get('tip1_robot') or det.get('tip1') or [det.get('x'), det.get('y')], dtype=float)[:2]
                    t2 = np.asarray(det.get('tip2_robot') or det.get('tip2') or [det.get('x'), det.get('y')], dtype=float)[:2]
                    center_xy = np.array([cx_r, cy_r], dtype=float)
                    tip_vec = t2 - t1
                    if np.linalg.norm(tip_vec) < 1e-6:
                        chosen_xy = center_xy.copy()
                    else:
                        perp = np.array([-tip_vec[1], tip_vec[0]], dtype=float)
                        perp_unit = perp / (np.linalg.norm(perp) + 1e-9)
                        # use 10mm offset (robot-frame mm) to match move_roArm
                        cand1 = center_xy + perp_unit * 10.0
                        cand2 = center_xy - perp_unit * 10.0
                        # Prefer the candidate with the larger Y in robot coordinates
                        # (use robot-frame tip coordinates when available)
                        if cand1[1] > cand2[1]:
                            chosen_xy = cand1
                        else:
                            chosen_xy = cand2
                    # plot as a small cyan square (map plot_x=y, plot_y=x)
                    try:
                        sq, = viz_ax_xy.plot(chosen_xy[1], chosen_xy[0], marker='s', color='cyan', markersize=6, markeredgecolor='k')
                        viz_motion_artists.append(sq)
                    except Exception:
                        pass
                except Exception:
                    pass

            lbl_col = 'red' if mode_label == 'COMPLEX' else 'green'
            ml = viz_ax_xy.text(cx_p + 8, cy_p + 8, mode_label, color=lbl_col, fontsize=8, weight='bold')
            viz_angle_artists.append(ml)
        except Exception:
            continue

    # Optionally show last sent command as a red X (but only recent ones)
    if command_log:
        # show only recent, non-hidden commands and fade them out over 1s
        CMD_MARKER_LIFETIME = 1.0  # seconds
        recent_cmds = [c for c in command_log if not c.get('hide') and (t_now - float(c.get('ts', t_now)) <= CMD_MARKER_LIFETIME)]
        for c in recent_cmds:
            age = t_now - float(c.get('ts', t_now))
            alpha = max(0.0, 1.0 - (age / CMD_MARKER_LIFETIME))
            try:
                # store artist so it can be removed on next frame
                art, = viz_ax_xy.plot(c['y'], c['x'], marker='x', color=(1.0, 0.0, 0.0, alpha), markersize=8)
                viz_motion_artists.append(art)
            except Exception:
                try:
                    art, = viz_ax_xy.plot(c['y'], c['x'], marker='x', color='red', markersize=8)
                    viz_motion_artists.append(art)
                except Exception:
                    pass

    lines = []
    if detections_meta:
        lines.append(f"Detections: {len(detections_meta)}")
        for i, d in enumerate(detections_meta[:6]):
            lines.append(f"D{i+1}: X={d['x']:.0f} Y={d['y']:.0f} Z={d['z']:.0f} {d.get('color','')} PR={d.get('pen_radial_angle_deg','?')}°")
    if command_log:
        lines.append(f"Cmds sent: {len(command_log)} (last 1)")
        last = command_log[-1]
        lines.append(f"Last: ({last['x']:.0f},{last['y']:.0f},{last['z']:.0f}) t={last['t']:.2f}")
    viz_txt_obj.set_text('\n'.join(lines))

    try:
        viz_fig.canvas.draw_idle(); viz_fig.canvas.flush_events()
    except Exception:
        viz_initialized = False

# Replace the older OpenCV-based viz update with interactive updater in the main loop
# Auto mode state
auto_mode = False
show_plot = False
# NEW toggle + frame counter for separate viz window
show_viz_window = True
_frame_counter = 0
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
        robot_coords_for_plot = []
    # metadata list for visualization
        detection_metadata = []
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
                robot_coords_for_plot.append(point_robot)

                pixel_to_mm_scale = t_param * 1000
                width_mm = (w_i * pixel_to_mm_scale) / K[0, 0]
                height_mm = (h_i * pixel_to_mm_scale) / K[1, 1]

                # Short-edge midpoints
                tip_mid_px1, tip_mid_px2 = centers_of_short_edges(corners)
                tip1_robot = pixel_to_robot(tip_mid_px1)
                tip2_robot = pixel_to_robot(tip_mid_px2)

                # --- Compute penRadialAngle and chosen tip (same logic as move_roArm) ---
                center_xy = point_robot[:2]
                tip1_xy = tip1_robot[:2]
                tip2_xy = tip2_robot[:2]
                radial_xy = -center_xy
                rd_norm = np.linalg.norm(radial_xy)
                rhat = np.array([1.0, 0.0], dtype=float) if rd_norm < 1e-9 else (radial_xy / rd_norm)
                def foot_and_score(P):
                    s = float(np.dot(P - center_xy, rhat))
                    F = center_xy + s * rhat
                    distF = float(np.linalg.norm(F))
                    prefer_penalty = 0 if s >= 0 else 1
                    return F, s, distF, (prefer_penalty, distF, -s)
                F1, s1, dF1, score1 = foot_and_score(tip1_xy)
                F2, s2, dF2, score2 = foot_and_score(tip2_xy)
                chosen_tip_xy = tip1_xy if score1 < score2 else tip2_xy
                chosen_tip_robot = tip1_robot if score1 < score2 else tip2_robot
                chosen_tip_name = "tip1" if score1 < score2 else "tip2"
                penRadialAngle = _pen_radial_angle(center_xy, chosen_tip_xy)
                penRadialAngle_deg = np.degrees(penRadialAngle)

                # Draw tips: chosen tip in green, other in yellow
                tip1_color = (0,255,0) if chosen_tip_name == "tip1" else (0,255,255)
                tip2_color = (0,255,0) if chosen_tip_name == "tip2" else (0,255,255)
                cv2.circle(annotated, (int(tip_mid_px1[0]), int(tip_mid_px1[1])), 4, tip1_color, -1)
                cv2.circle(annotated, (int(tip_mid_px2[0]), int(tip_mid_px2[1])), 4, tip2_color, -1)

                # Color inside OBB
                color_label = _classify_color_in_polygon(frame, corners)

                # Two-line yellow info
                line1 = (f"Pix:({cx_i},{cy_i}) | Size:{width_mm:.1f}x{height_mm:.1f}mm "
                         f"| Angle:{angle:.2f}rad/{angle_deg:.1f}deg | Color:{color_label}")
                line2 = (f"Robot:({point_robot[0]:.0f},{point_robot[1]:.0f},{point_robot[2]:.0f})mm "
                         f"| TipsRobot:({tip1_robot[0]:.0f},{tip1_robot[1]:.0f},{tip1_robot[2]:.0f})/"
                         f"({tip2_robot[0]:.0f},{tip2_robot[1]:.0f},{tip2_robot[2]:.0f})mm")
                line3 = (f"penRadialAngle: {penRadialAngle:.2f} rad / {penRadialAngle_deg:.1f} deg (chosen: {chosen_tip_name})")

                base_x, base_y = cx_i - 100, max(20, cy_i - 20)
                cv2.putText(annotated, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated, line2, (base_x, base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated, line3, (base_x, base_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
                detection_metadata.append({
                    'x': float(point_robot[0]),
                    'y': float(point_robot[1]),
                    'z': float(point_robot[2]),
                    'color': color_label,
                    'angle': float(angle_deg),
                    'tip1': [float(tip1_robot[0]), float(tip1_robot[1])],
                    'tip2': [float(tip2_robot[0]), float(tip2_robot[1])],
                    'chosen_tip': chosen_tip_name,
                    'pen_radial_angle_rad': float(penRadialAngle),  # ensure radians stored
                    'pen_radial_angle_deg': float(penRadialAngle_deg),  # convenience (derived)
                    'ts': time.time(),  # timestamp for fade/cooldown
                })
    # update global last_detection_metadata
        last_detection_metadata = detection_metadata

        # --- Draw coordinate frame and info ---
        cv2.arrowedLine(annotated, (30, 30), (80, 30), (0, 0, 255), 2)  # X-axis (right)
        cv2.arrowedLine(annotated, (30, 30), (30, 80), (0, 255, 0), 2)  # Y-axis (down)
        cv2.putText(annotated, "X", (85, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated, "Y", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Resolution: {w_frame}x{h_frame}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, "0 rads @ axis", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Overlay plot if enabled ---
        if show_plot:
            plot_img = _create_robot_coord_plot(robot_coords_for_plot)
            if plot_img is not None:
                h_plot, w_plot = plot_img.shape[:2]
                # Position at bottom-right corner
                annotated[h_frame - h_plot:h_frame, w_frame - w_plot:w_frame] = plot_img

        # --- Auto/Space badge ---
        _draw_pill_badge(annotated, "AUTO" if auto_mode else "SPACE", enabled=auto_mode, margin=20, alpha=0.85)

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
        # Update the matplotlib viz every frame to match the video/frame rate
        if show_viz_window:
            try:
                update_interactive_viz(last_detection_metadata, sent_robot_commands)
            except Exception as e:
                print(f"[{ts()}] Viz update error: {e}")
        _frame_counter += 1

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
        elif key == ord('p'):
            show_plot = not show_plot
            state = "ON" if show_plot else "OFF"
            print(f"[{ts()}] [UI] Plot overlay toggled {state}")
        elif key == ord('v'):
            show_viz_window = not show_viz_window
            state = "ON" if show_viz_window else "OFF"
            print(f"[{ts()}] [UI] Separate viz window toggled {state}")

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
