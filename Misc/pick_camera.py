#!/usr/bin/env python3
"""
pick_camera.py — visually choose the LifeCam and save its index to config.json.

OpenCV's camera index order on macOS is not reliable (Continuity Camera reshuffles
it), so the only sure way is to LOOK at each feed. Run this in YOUR Terminal
(needs camera permission). It opens each camera in turn:

    Y = "this is the one I want" (saves its index to config.json)
    N / any key = next camera
    Q = quit without saving
"""
import cv2
import json
import os

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')


def main():
    print("Cycling through cameras. Press Y on the one showing your WORKSPACE (the LifeCam).")
    chosen, quit_all = None, False
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # fixed focus
        except Exception: pass
        print(f"  index {idx}: showing…  (Y=select  N=next  Q=quit)")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"Index {idx}:  Y = use this   N = next   Q = quit",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Pick the LifeCam (your workspace view)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('y'), ord('Y')):
                chosen = idx; break
            if k in (ord('q'), ord('Q')):
                quit_all = True; break
            if k in (ord('n'), ord('N')):
                break
        cap.release()
        if chosen is not None or quit_all:
            break
    cv2.destroyAllWindows()

    if chosen is not None:
        try:
            cfg = json.load(open(CFG))
        except Exception:
            cfg = {}
        cfg['camera_index'] = chosen
        json.dump(cfg, open(CFG, 'w'), indent=2)
        print(f"\n  Saved camera_index = {chosen} to config.json  ✓")
        print("  full_run.py will now use this camera.")
    else:
        print("\n  No camera selected; config.json unchanged.")


if __name__ == "__main__":
    main()
