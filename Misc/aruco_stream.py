import cv2
import numpy as np

# Load camera calibration
calib = np.load('calib_data.npz')
camMatrix = calib['K']
distCoeffs = calib['dist']

# Parameters
markerLength = 0.2032  # in meters (8 inches)
estimatePose = True
showRejected = True

# Setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# Create object points for a single marker
objPoints = np.array([
    [-markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32)

# Open video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press ESC to quit")

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    imageCopy = image.copy()

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)

        if estimatePose:
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], markerLength, camMatrix, distCoeffs)
                cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec[0], tvec[0], markerLength * 1.5)

    if showRejected and rejected:
        cv2.aruco.drawDetectedMarkers(imageCopy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("ArUco Stream", imageCopy)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()