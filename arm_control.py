#!/usr/bin/env python3
"""
arm_control.py — native pop-up joystick window for the RoArm-M2-S.

One command opens a desktop window (no browser, no separate server) with the
XY joystick + separate Z and gripper controls. The window talks straight to
Python, which holds the serial port and enforces the SAME limits as
camera_stream.py before sending {"T":1041,...}:

    radial sqrt(x^2+y^2) in [80, 500] mm,  z in [-100, 450] mm,
    gripper t in [1.08 open .. 3.14 closed]

USAGE:
    python arm_control.py /dev/tty.usbserial-210     # live robot
    python arm_control.py --mock                     # no robot, prints only
    python arm_control.py                             # no port -> mock
"""

import os, sys, json, time, threading, argparse, math
import webview

LIMITS = {"r_min": 80.0, "r_max": 500.0, "z_min": -100.0, "z_max": 450.0,
          "t_min": 1.08, "t_max": 3.14}
HOME = [120.0, 0.0, -20.0]
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "arm_web", "index.html")
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
            self.ser = serial.Serial(port, baudrate=115200, timeout=1, dsrdtr=None)
            self.ser.setRTS(False); self.ser.setDTR(False)
            time.sleep(1.0)
            threading.Thread(target=self._reader, daemon=True).start()

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
    """Accept 'x y z t' (4 numbers) or a full JSON command, like the original
    serial_simple_ctrl.py. Coordinate moves (T:1041 with x/y/z) are limit-checked;
    other command types (LED T:114, init T:100, etc.) pass through raw."""
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


class Api:
    """Exposed to the page as window.pywebview.api.*"""
    def __init__(self, robot):
        self.robot = robot
        self.window = None      # set in main() after the window is created

    def get_config(self):
        tag = None
        try:
            with open(CONFIG_PATH) as f:
                tag = json.load(f).get("robot_tag_xyz")
        except Exception:
            pass
        return {"mock": self.robot.mock, "limits": LIMITS, "home": HOME, "tag": tag}

    def save_tag(self, payload):
        """Write the current XYZ into config.json as robot_tag_xyz (persistent)."""
        try:
            x, y, z = float(payload["x"]), float(payload["y"]), float(payload["z"])
        except Exception as e:
            return {"ok": False, "reason": f"bad payload: {e}"}
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            cfg["robot_tag_xyz"] = [round(x), round(y), round(z)]
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[config] robot_tag_xyz saved -> {cfg['robot_tag_xyz']}")
            return {"ok": True, "saved": cfg["robot_tag_xyz"]}
        except Exception as e:
            return {"ok": False, "reason": f"write failed: {e}"}

    def move(self, payload):
        try:
            x, y, z, t = (float(payload["x"]), float(payload["y"]),
                          float(payload["z"]), float(payload["t"]))
        except Exception as e:
            return {"ok": False, "reason": f"bad payload: {e}"}
        ok, reason = validate(x, y, z)
        if not ok:
            return {"ok": False, "reason": reason}
        return {"ok": True, "sent": self.robot.move(x, y, z, t)}

    def command(self, text):
        return parse_and_send(self.robot, text)

    def _close_window(self):
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception as e:
                print(f"[window] close failed: {e}")

    def confirm(self, payload):
        """Save the current position as the ArUco tag, then close the window so
        full_run.py advances. Close is deferred 0.15s so this bridge call can
        return first (closing mid-call leaves the window stuck open)."""
        res = self.save_tag(payload)
        if res.get("ok"):
            threading.Timer(0.15, self._close_window).start()
        return res

    def keep_and_close(self):
        """Close WITHOUT saving — keep the existing ArUco tag in config."""
        threading.Timer(0.15, self._close_window).start()
        return {"ok": True, "kept": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None, help="serial port (omit for --mock)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)   # flush each line (live logging)
    except Exception:
        pass

    robot = Robot(args.port, args.mock)
    api = Api(robot)
    title = "RoArm-M2-S Control  " + ("[MOCK]" if robot.mock else "[LIVE]")
    print(f"Opening control window — {'MOCK (no robot)' if robot.mock else 'LIVE on '+args.port}")

    api.window = webview.create_window(title, url=INDEX, js_api=api,
                          width=1040, height=940, min_size=(900, 700),
                          background_color="#0e1014")
    webview.start()          # blocks until the window is closed
    robot.close()


if __name__ == "__main__":
    main()
