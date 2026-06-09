import cv2
import os
import sys
import glob
import json

def _camera_index():
    """Read camera_index from config.json (default 0)."""
    try:
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        return int(json.load(open(cfg)).get('camera_index', 0))
    except Exception:
        return 0

if len(sys.argv) < 2:
    print("Usage: python camera_capture.py <output_folder>")
    sys.exit(1)

output_folder = sys.argv[1]
os.makedirs(output_folder, exist_ok=True)

# Find the next available filename
existing = glob.glob(os.path.join(output_folder, '*.jpg'))
existing_nums = set()
for f in existing:
    base = os.path.basename(f)
    name, ext = os.path.splitext(base)
    if name.isdigit():
        existing_nums.add(int(name))

def get_next_filename():
    n = 1
    while n in existing_nums:
        n += 1
    existing_nums.add(n)
    return os.path.join(output_folder, f"{n}.jpg")

_ci = _camera_index()
print(f"Using camera index {_ci}")
cap = cv2.VideoCapture(_ci)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
if not cap.isOpened():
    print("Could not open webcam")
    sys.exit(1)

cv2.namedWindow("Camera Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Capture", 640, 480)

print("Press SPACE to capture and save an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        filename = get_next_filename()
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
