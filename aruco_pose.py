import cv2
import numpy as np
import json
import os

calib = np.load('calib_data.npz')
camera_matrix = calib['K']
dist_coeffs = calib['dist']

# Use the photo you captured + picked in the previous step (Aruco/aruco_calibration.jpg).
# Fall back to a fresh camera frame only if no captured photo exists.
img_path = os.path.join("Aruco", "aruco_calibration.jpg")
if os.path.exists(img_path):
    image = cv2.imread(img_path)
    print(f"Using captured ArUco photo: {img_path}")
else:
    try:
        _ci = int(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'))).get('camera_index', 0))
    except Exception:
        _ci = 0
    print(f"No captured photo found; grabbing a fresh frame from camera index {_ci}")
    cap = cv2.VideoCapture(_ci)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam and no captured photo found.")
    try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # fixed focus
    except Exception: pass
    ret, image = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from webcam.")
if image is None:
    raise RuntimeError(f"Could not load ArUco image ({img_path}).")

h, w = image.shape[:2]

undistorted = image.copy()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, _ = detector.detectMarkers(undistorted)

# Marker pose estimation
marker_length = 0.100  # meters (100 mm printed ArUco tag)
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

cv2.imshow("Pose Estimation", undistorted)
cv2.imwrite("aruco_tag_detection.jpg", undistorted)
cv2.waitKey(3000)  # waits 3 seconds
cv2.destroyAllWindows()