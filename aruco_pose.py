import cv2
import numpy as np
import json
import os

# Load calibration data
calib = np.load('calib_data.npz')
camera_matrix = calib['K']
dist_coeffs = calib['dist']

# Capture image from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

ret, image = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to capture image from webcam.")

# Undistort
h, w = image.shape[:2]
# new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
# undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

undistorted = image.copy()

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, _ = detector.detectMarkers(undistorted)

# Marker pose estimation
marker_length = 0.203  # meters (8 inches)
if ids is not None:
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(undistorted, corners)
        
        # Must reshape rvec and tvec to (3,1)
        rvec = rvecs[i].reshape((3, 1))
        tvec = tvecs[i].reshape((3, 1))
        cv2.drawFrameAxes(undistorted, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        print(f"Marker ID: {ids[i][0]}")
        print(f"Translation (mm): {tvec.T * 1000}")
        print(f"Rotation vector: {rvec.T}")

    # Save pose and calibration data to JSON
    pose_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rvec": rvecs[0].tolist(),
        "tvec": tvecs[0].tolist()
    }

    output_dir = "Aruco"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "aruco_reference.json"), "w") as f:
        json.dump(pose_data, f, indent=2)

    print("Saved ArUco pose and camera calibration to Aruco/aruco_reference.json")
else:
    print("No marker detected.")

# Show result
cv2.imshow("Pose Estimation", undistorted)
cv2.imwrite("aruco_tag_detection.jpg", undistorted)
cv2.waitKey(3000)  # waits 3 seconds
cv2.destroyAllWindows()