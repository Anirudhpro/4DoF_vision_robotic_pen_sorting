import cv2
import numpy as np
import glob
import os

# Settings
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 22  # mm meaured square size per ruler

# Object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# Paths
# input_folder = 'ExtIntVals/Data 4/CalibrationPictures'
# output_folder = 'ExtIntVals/Data 4/CalibratedLinePictures'
input_folder = 'CalibrationPictures'
output_folder = 'CalibratedLinePictures'
os.makedirs(output_folder, exist_ok=True)

# Grab all images
images = glob.glob(os.path.join(input_folder, '*.jpeg')) + glob.glob(os.path.join(input_folder, '*.jpg'))
print(f"Found {len(images)} images.")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # Save annotated image
        output_path = os.path.join(output_folder, f"calib_{idx+1:02d}.jpg")
        cv2.imwrite(output_path, img)

        # Show for 1.5 seconds
        cv2.imshow("Checkerboard Detection", img)
        cv2.waitKey(50)

cv2.destroyAllWindows()

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("calib_data.npz", K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Output summary
print("\nCalibration complete.")
print("Camera Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist.ravel())
print(f"Saved calibration to: calib_data.npz")
print(f"Saved drawn images to: {output_folder}")