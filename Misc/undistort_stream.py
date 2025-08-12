import cv2
import numpy as np

# Load calibration values from your file
data = np.load("calib_data.npz")
K = data['K']
dist = data['dist']

# Open your camera
cap = cv2.VideoCapture(0)  # Use correct camera index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply undistortion
    undistorted = cv2.undistort(frame, K, dist)

    # Show side-by-side
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()