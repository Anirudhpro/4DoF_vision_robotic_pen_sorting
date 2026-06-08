#!/usr/bin/env python3
"""
arm_joystick.py — Simple two-stick joystick for the RoArm-M2-S over serial.

LEFT stick   -> jog X / Y (forward-back / left-right)
RIGHT stick  -> jog Z (up/down)  and gripper (left=open, right=close)

Push a stick to move; let go and it springs back to center and the arm stops.
Movement stays inside the SAME workspace limits camera_stream.py enforces:
    radial sqrt(x^2+y^2) in [80, 500] mm,   z in [-100, 450] mm.

Sends {"T":1041,"x":..,"y":..,"z":..,"t":..}  (t = gripper: 1.08 open .. 3.14 closed)

USAGE:
    python arm_joystick.py /dev/tty.usbserial-210     # live robot
    python arm_joystick.py --mock                     # UI only, no robot
    python arm_joystick.py                             # no port -> mock

KEYS:  H = home    ESC = quit
"""

import sys, json, time, threading, argparse, math
import pygame

# ---- workspace limits (identical to camera_stream.py validate_robot_coords) ----
R_MIN, R_MAX = 80.0, 500.0
Z_MIN, Z_MAX = -100.0, 450.0
T_MIN, T_MAX = 1.08, 3.14
HOME = (120.0, 0.0, -20.0)

# jog speeds at full stick deflection
SPEED_XY = 170.0   # mm/s
SPEED_Z  = 130.0   # mm/s
SPEED_T  = 1.6     # rad/s


def clamp_to_annulus(x, y):
    r = math.hypot(x, y)
    if r < 1e-6:
        return R_MIN, 0.0
    if r > R_MAX:
        return x * R_MAX / r, y * R_MAX / r
    if r < R_MIN:
        return x * R_MIN / r, y * R_MIN / r
    return x, y


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

    def move(self, x, y, z, t):
        cmd = {"T": 1041, "x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "t": round(t, 3)}
        line = json.dumps(cmd)
        if self.mock:
            print(f"[MOCK] {line}")
        else:
            try:
                self.ser.write(line.encode() + b"\n")
            except Exception as e:
                print(f"[robot] write error: {e}")

    def close(self):
        if self.ser is not None:
            try: self.ser.close()
            except Exception: pass


# ----------------------------- UI -----------------------------
W, H = 720, 460
L_CX, L_CY, STICK_R = 200, 210, 120     # left stick center + radius
R_CX, R_CY = 520, 210                    # right stick center
KNOB_R = 26

BG = (16, 18, 24); FG = (232, 232, 238); DIM = (110, 116, 128)
OK = (90, 200, 120); BAD = (235, 90, 90); RIM = (60, 66, 80)


class Stick:
    """A spring-return analog stick. Returns normalized (ax, ay) in [-1,1]."""
    def __init__(self, cx, cy):
        self.cx, self.cy = cx, cy
        self.kx, self.ky = cx, cy      # knob screen pos
        self.active = False

    def grab(self, mx, my):
        if math.hypot(mx - self.cx, my - self.cy) <= STICK_R + KNOB_R:
            self.active = True
            self._set(mx, my)
            return True
        return False

    def _set(self, mx, my):
        dx, dy = mx - self.cx, my - self.cy
        d = math.hypot(dx, dy)
        if d > STICK_R:
            dx, dy = dx * STICK_R / d, dy * STICK_R / d
        self.kx, self.ky = self.cx + dx, self.cy + dy

    def drag(self, mx, my):
        if self.active:
            self._set(mx, my)

    def release(self):
        self.active = False
        self.kx, self.ky = self.cx, self.cy     # spring back

    def axes(self):
        return (self.kx - self.cx) / STICK_R, (self.ky - self.cy) / STICK_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    robot = Robot(args.port, args.mock)
    live = not robot.mock

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("RoArm Joystick " + ("[LIVE]" if live else "[MOCK]"))
    font = pygame.font.SysFont("menlo,consolas,monospace", 16)
    big = pygame.font.SysFont("menlo,consolas,monospace", 22, bold=True)
    clock = pygame.time.Clock()

    left, right = Stick(L_CX, L_CY), Stick(R_CX, R_CY)
    x, y, z = HOME
    t = 3.14
    last_send = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_h:
                    x, y, z = HOME
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not left.grab(*ev.pos):
                    right.grab(*ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                left.release(); right.release()
            elif ev.type == pygame.MOUSEMOTION:
                left.drag(*ev.pos); right.drag(*ev.pos)

        # ---- integrate jog from stick deflection ----
        lx, ly = left.axes()      # left: x=strafe(Y), y=forward(X)
        rx, ry = right.axes()     # right: x=gripper, y=Z
        moving = left.active or right.active
        if left.active:
            x = x + (-ly) * SPEED_XY * dt     # push up -> +X forward
            y = y + (-lx) * SPEED_XY * dt     # push left -> +Y left
            x, y = clamp_to_annulus(x, y)
        if right.active:
            z = max(Z_MIN, min(Z_MAX, z + (-ry) * SPEED_Z * dt))   # up -> +Z
            t = max(T_MIN, min(T_MAX, t + (rx) * SPEED_T * dt))    # right -> close

        # ---- send (rate-limited) while a stick is held ----
        if moving and time.time() - last_send > 0.10:
            robot.move(x, y, z, t)
            last_send = time.time()

        # ------------------------------- draw -------------------------------
        screen.fill(BG)
        for s, label in [(left, "MOVE  X / Y"), (right, "Z  /  GRIP")]:
            pygame.draw.circle(screen, RIM, (s.cx, s.cy), STICK_R, 2)
            pygame.draw.circle(screen, (34, 38, 48), (s.cx, s.cy), 5)
            col = OK if s.active else DIM
            pygame.draw.circle(screen, col, (int(s.kx), int(s.ky)), KNOB_R)
            pygame.draw.circle(screen, FG, (int(s.kx), int(s.ky)), KNOB_R, 2)
            lab = font.render(label, True, DIM)
            screen.blit(lab, (s.cx - lab.get_width() // 2, s.cy + STICK_R + 10))

        # direction hints on the left stick
        screen.blit(font.render("fwd", True, RIM), (L_CX - 14, L_CY - STICK_R - 4))
        screen.blit(font.render("left", True, RIM), (L_CX - STICK_R - 34, L_CY - 8))
        # hints on right stick
        screen.blit(font.render("up", True, RIM), (R_CX - 8, R_CY - STICK_R - 4))
        screen.blit(font.render("open", True, RIM), (R_CX - STICK_R - 38, R_CY - 8))
        screen.blit(font.render("close", True, RIM), (R_CX + STICK_R + 6, R_CY - 8))

        # readout
        screen.blit(big.render("LIVE" if live else "MOCK", True, BAD if live else OK), (W - 90, 16))
        r = math.hypot(x, y)
        info = f"X{x:6.0f}  Y{y:6.0f}  Z{z:6.0f}  grip{t:5.2f}   r{r:5.0f}"
        screen.blit(font.render(info, True, FG), (24, H - 30))
        screen.blit(font.render("H = home    ESC = quit", True, DIM), (24, 16))

        pygame.display.flip()

    robot.close()
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit(); sys.exit(0)
