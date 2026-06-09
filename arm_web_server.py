#!/usr/bin/env python3
"""
arm_web_server.py — Backend for the RoArm-M2-S web control panel.

Browsers can't open a serial port, so this small stdlib HTTP server holds the
serial connection, serves the React UI (arm_web/index.html), and exposes an API
that enforces the SAME workspace limits as camera_stream.py before sending
{"T":1041,...} to the arm.

    radial sqrt(x^2+y^2) in [80, 500] mm,  z in [-100, 450] mm,
    gripper t in [1.08 open .. 3.14 closed]

USAGE:
    python arm_web_server.py /dev/tty.usbserial-210      # live robot
    python arm_web_server.py --mock                      # no robot, prints only
    python arm_web_server.py                             # no port -> mock
then open  http://localhost:8765  (it tries to open automatically).
"""

import os, sys, json, time, threading, argparse, math, webbrowser, glob, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---- workspace limits (identical to camera_stream.py validate_robot_coords) ----
LIMITS = {"r_min": 80.0, "r_max": 500.0, "z_min": -100.0, "z_max": 450.0,
          "t_min": 1.08, "t_max": 3.14}
HOME = [120.0, 0.0, -20.0]
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


def validate(x, y, z):
    r = math.hypot(x, y)
    if r > LIMITS["r_max"]:
        return False, f"radial {r:.0f} > {LIMITS['r_max']:.0f}"
    if r < LIMITS["r_min"]:
        return False, f"radial {r:.0f} < {LIMITS['r_min']:.0f}"
    if not (LIMITS["z_min"] <= z <= LIMITS["z_max"]):
        return False, f"z {z:.0f} out of [{LIMITS['z_min']:.0f},{LIMITS['z_max']:.0f}]"
    return True, "ok"


class Robot:
    def __init__(self, port, mock):
        self.mock = mock or not port
        self.ser = None
        if not self.mock:
            import serial
            try:
                self.ser = serial.Serial(port, baudrate=115200, timeout=1, dsrdtr=None)
                self.ser.setRTS(False); self.ser.setDTR(False)
                time.sleep(1.0)
                threading.Thread(target=self._reader, daemon=True).start()
            except Exception as e:
                print(f"[robot] could not open {port}: {e}\n[robot] falling back to MOCK (robot not connected)")
                self.mock = True
                self.ser = None

    def _reader(self):
        while self.ser is not None:
            try:
                d = self.ser.readline().decode("utf-8", "ignore").strip()
                if d:
                    print(f"[robot] {d}")
            except Exception:
                break

    def send_raw(self, cmd):
        line = json.dumps(cmd)
        if self.mock:
            print(f"[MOCK] {line}")
        else:
            self.ser.write(line.encode() + b"\n")
            print(f"[sent] {line}")
        return cmd

    def move(self, x, y, z, t):
        return self.send_raw({"T": 1041, "x": round(x, 1), "y": round(y, 1),
                              "z": round(z, 1), "t": round(t, 3)})

    def close(self):
        if self.ser is not None:
            try: self.ser.close()
            except Exception: pass


def parse_and_send(robot, text):
    """'x y z t' or full JSON, like serial_simple_ctrl.py. T:1041 coordinate
    moves are limit-checked; other command types pass through raw."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "reason": "empty"}
    if text.startswith("{"):
        try:
            cmd = json.loads(text)
        except Exception as e:
            return {"ok": False, "reason": f"bad JSON: {e}"}
        if cmd.get("T") == 1041 and all(k in cmd for k in ("x", "y", "z")):
            ok, reason = validate(float(cmd["x"]), float(cmd["y"]), float(cmd["z"]))
            if not ok:
                return {"ok": False, "reason": reason}
        return {"ok": True, "sent": robot.send_raw(cmd)}
    parts = text.split()
    if len(parts) != 4:
        return {"ok": False, "reason": 'type "x y z t" or full JSON'}
    try:
        x, y, z, t = map(float, parts)
    except ValueError:
        return {"ok": False, "reason": "x y z t must be numbers"}
    ok, reason = validate(x, y, z)
    if not ok:
        return {"ok": False, "reason": reason}
    return {"ok": True, "sent": robot.move(x, y, z, t)}


ROBOT = None  # set in main
STATE = {"action": None}  # reposition session result: None -> "confirm" / "keep"

# ----------------------------- Camera (capture screen) -----------------------------
# One shared camera handle + a background reader thread that keeps the latest JPEG.
CAM = {"cap": None, "index": None, "jpg": None, "lock": threading.Lock(), "run": False}
ARUCO_DIR = os.path.join(HERE, "Aruco")


def cam_open(index):
    """Open camera `index` (release any previous). Returns True on success."""
    import cv2
    with CAM["lock"]:
        if CAM["cap"] is not None and CAM["index"] == index:
            return True
        if CAM["cap"] is not None:
            try: CAM["cap"].release()
            except Exception: pass
            CAM["cap"] = None
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            try: cap.release()
            except Exception: pass
            return False
        try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # fixed focus for stable detection
        except Exception: pass
        CAM["cap"] = cap
        CAM["index"] = index
        CAM["jpg"] = None
    if not CAM["run"]:
        CAM["run"] = True
        threading.Thread(target=_cam_loop, daemon=True).start()
    return True


def _cam_loop():
    import cv2
    while CAM["run"]:
        cap = CAM["cap"]
        if cap is None:
            time.sleep(0.05); continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with CAM["lock"]:
                CAM["jpg"] = buf.tobytes()
        time.sleep(0.03)


def cam_latest():
    with CAM["lock"]:
        return CAM["jpg"]


def list_cameras():
    """[{index, name}] in OpenCV order (via AVFoundation). Empty if unavailable."""
    try:
        import AVFoundation
        names = ['AVCaptureDeviceTypeBuiltInWideAngleCamera', 'AVCaptureDeviceTypeExternal',
                 'AVCaptureDeviceTypeExternalUnknown', 'AVCaptureDeviceTypeContinuityCamera']
        types = [getattr(AVFoundation, t) for t in names if hasattr(AVFoundation, t)]
        try:
            sess = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                types, AVFoundation.AVMediaTypeVideo, 0)
            devs = list(sess.devices())
        except Exception:
            devs = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
        return [{"index": i, "name": str(d.localizedName())} for i, d in enumerate(devs)]
    except Exception:
        return []


def next_capture_path():
    os.makedirs(ARUCO_DIR, exist_ok=True)
    used = set()
    for f in glob.glob(os.path.join(ARUCO_DIR, "*.jpg")):
        n = os.path.splitext(os.path.basename(f))[0]
        if n.isdigit():
            used.add(int(n))
    i = 1
    while i in used:
        i += 1
    return os.path.join(ARUCO_DIR, f"{i}.jpg")


def list_captures():
    os.makedirs(ARUCO_DIR, exist_ok=True)
    files = []
    for f in glob.glob(os.path.join(ARUCO_DIR, "*.jpg")):
        name = os.path.basename(f)
        n = os.path.splitext(name)[0]
        if n.isdigit():
            files.append(name)
    return sorted(files, key=lambda n: int(os.path.splitext(n)[0]))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet default logging
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]          # ignore query string for routing
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "arm_web", "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "index.html not found")
        elif path == "/api/config":
            tag = None
            try:
                with open(CONFIG_PATH) as f:
                    tag = json.load(f).get("robot_tag_xyz")
            except Exception:
                pass
            self._send(200, json.dumps({"mock": ROBOT.mock, "limits": LIMITS, "home": HOME, "tag": tag}))
        elif path == "/api/status":
            self._send(200, json.dumps({"action": STATE["action"]}))
        elif path == "/api/cameras":
            self._send(200, json.dumps({"cameras": list_cameras()}))
        elif path == "/api/captures":
            self._send(200, json.dumps({"captures": list_captures()}))
        elif path == "/api/capture_img":
            from urllib.parse import urlparse, parse_qs
            fn = parse_qs(urlparse(self.path).query).get("file", [""])[0]
            fp = os.path.join(ARUCO_DIR, os.path.basename(fn))
            if fp.endswith(".jpg") and os.path.exists(fp):
                with open(fp, "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            else:
                self._send(404, "not found")
        elif path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    jpg = cam_latest()
                    if jpg is None:
                        time.sleep(0.05); continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.04)
            except Exception:
                pass   # client disconnected
        else:
            self._send(404, "not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"ok": False, "reason": f"bad body: {e}"})); return
        if self.path == "/api/move":
            try:
                x, y, z, t = float(body["x"]), float(body["y"]), float(body["z"]), float(body["t"])
            except Exception as e:
                self._send(400, json.dumps({"ok": False, "reason": f"bad body: {e}"})); return
            ok, reason = validate(x, y, z)
            if not ok:
                self._send(200, json.dumps({"ok": False, "reason": reason})); return
            self._send(200, json.dumps({"ok": True, "sent": ROBOT.move(x, y, z, t)}))
        elif self.path == "/api/command":
            self._send(200, json.dumps(parse_and_send(ROBOT, body.get("text", ""))))
        elif self.path == "/api/keep_close":
            STATE["action"] = "keep"
            self._send(200, json.dumps({"ok": True, "kept": True}))
        elif self.path == "/api/home":
            self._send(200, json.dumps({"ok": True, "sent": ROBOT.move(HOME[0], HOME[1], HOME[2], 3.14)}))
        elif self.path in ("/api/save_tag", "/api/confirm"):
            try:
                x, y, z = float(body["x"]), float(body["y"]), float(body["z"])
                with open(CONFIG_PATH) as f:
                    cfg = json.load(f)
                cfg["robot_tag_xyz"] = [round(x), round(y), round(z)]
                with open(CONFIG_PATH, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"[config] robot_tag_xyz saved -> {cfg['robot_tag_xyz']}")
                if self.path == "/api/confirm":
                    STATE["action"] = "confirm"
                self._send(200, json.dumps({"ok": True, "saved": cfg["robot_tag_xyz"]}))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "reason": f"save failed: {e}"}))
        elif self.path == "/api/camera/open":
            idx = int(body.get("index", 0))
            self._send(200, json.dumps({"ok": cam_open(idx), "index": idx}))
        elif self.path == "/api/snap":
            jpg = cam_latest()
            if jpg is None:
                self._send(200, json.dumps({"ok": False, "reason": "no frame yet"})); return
            fp = next_capture_path()
            with open(fp, "wb") as f:
                f.write(jpg)
            self._send(200, json.dumps({"ok": True, "file": os.path.basename(fp)}))
        elif self.path == "/api/pick":
            fn = os.path.basename(body.get("file", ""))
            src = os.path.join(ARUCO_DIR, fn)
            if not (fn.endswith(".jpg") and os.path.exists(src)):
                self._send(200, json.dumps({"ok": False, "reason": "file not found"})); return
            shutil.copy(src, os.path.join(ARUCO_DIR, "aruco_calibration.jpg"))
            # remember which camera was used, so camera_stream opens the same one
            try:
                if CAM["index"] is not None:
                    cfg = json.load(open(CONFIG_PATH))
                    cfg["camera_index"] = CAM["index"]
                    json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)
            except Exception:
                pass
            STATE["action"] = "captured"
            self._send(200, json.dumps({"ok": True, "picked": fn}))
        else:
            self._send(404, json.dumps({"ok": False, "reason": "unknown endpoint"}))


def main():
    global ROBOT
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None, help="serial port (omit for --mock)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--http", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    ROBOT = Robot(args.port, args.mock)
    url = f"http://localhost:{args.http}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.http), Handler)
    mode = "MOCK (no robot)" if ROBOT.mock else f"LIVE on {args.port}"
    print(f"RoArm web control — {mode}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        srv.server_close()
        ROBOT.close() if hasattr(ROBOT, "close") else None


if __name__ == "__main__":
    main()
