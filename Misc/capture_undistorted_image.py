import cv2
import numpy as np

# Load calibration data
with np.load('calib_data.npz') as X:
    K, dist = X['K'], X['dist']

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press SPACE to save undistorted image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    if frame is None:
        print("Warning: Frame is empty. Skipping.")
        continue

    # Undistort
    undistorted = cv2.undistort(frame, K, dist)

    # Show
    cv2.imshow('Undistorted Stream', undistorted)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        cv2.imwrite("aruco_calibration.jpg", undistorted)
        print("Saved as aruco_calibration.jpg")
    elif key == ord('w'):
        cv2.imwrite("snapshots.jpg", undistorted)
        print("Saved snapshot as snapshots.jpg")

cap.release()
cv2.destroyAllWindows()