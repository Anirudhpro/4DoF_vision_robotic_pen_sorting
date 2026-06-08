#!/usr/bin/env python3
"""
camera_pick_and_capture.py — Step 2 of full_run: pick the camera, then capture.

Runs right after the position is confirmed. First you VISUALLY pick which camera
to use (no guessing indices), then you capture ArUco photos with that same camera.
The chosen index is saved to config.json so camera_stream uses it too.

USAGE:  python camera_pick_and_capture.py [output_folder]   (default: Aruco)

CONTROLS:
  Phase 1 (pick):     Y = use this camera   N = next camera   Q = quit
  Phase 2 (capture):  SPACE = save a photo  Q = done
"""
import cv2
import os
import sys
import glob
import json

OUT = sys.argv[1] if len(sys.argv) > 1 else "Aruco"
os.makedirs(OUT, exist_ok=True)
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
WIN = "Camera — pick then capture"


def label(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 140), 2, cv2.LINE_AA)


def pick_camera():
    """Cycle cameras; return (chosen_index, open_capture) or (None, None)."""
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release(); continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            label(frame, f"Index {idx}:  Y = use this camera    N = next    Q = quit")
            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('y'), ord('Y')):
                return idx, cap                      # keep this capture open for phase 2
            if k in (ord('n'), ord('N')):
                break
            if k in (ord('q'), ord('Q')):
                cap.release(); return None, None
        cap.release()
    return None, None


def next_filename():
    used = set()
    for f in glob.glob(os.path.join(OUT, '*.jpg')):
        name = os.path.splitext(os.path.basename(f))[0]
        if name.isdigit():
            used.add(int(name))
    n = 1
    while n in used:
        n += 1
    return os.path.join(OUT, f"{n}.jpg")


def main():
    print("Step 2 — pick your camera (Y on the workspace view), then capture the ArUco tag.")
    idx, cap = pick_camera()
    if idx is None:
        print("No camera selected.")
        cv2.destroyAllWindows()
        sys.exit(1)

    # save chosen index so camera_stream (and future runs) use the same camera
    try:
        cfg = json.load(open(CFG))
        cfg['camera_index'] = idx
        json.dump(cfg, open(CFG, 'w'), indent=2)
        print(f"Saved camera_index = {idx} to config.json")
    except Exception as e:
        print(f"(could not save camera_index: {e})")

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass

    print("Now capture: SPACE = save a photo of the ArUco tag, Q = done.")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame."); break
        label(frame, f"SPACE = capture ArUco photo   Q = done   (saved: {saved})")
        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            fn = next_filename()
            cv2.imwrite(fn, frame)
            saved += 1
            print(f"Saved {fn}")
        elif k in (ord('q'), ord('Q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    if saved == 0:
        print("No photos captured.")
        sys.exit(1)


if __name__ == "__main__":
    main()
