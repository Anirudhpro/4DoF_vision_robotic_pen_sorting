import cv2
import numpy as np

data = np.load("calib_data.npz")
K = data['K']
dist = data['dist']

cap = cv2.VideoCapture(0)  # Use correct camera index if needed
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # fixed focus

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.undistort(frame, K, dist)

    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()