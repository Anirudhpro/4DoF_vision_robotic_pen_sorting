import numpy as np
import cv2

# Load calibration data
data = np.load('calib_data.npz', allow_pickle=True)
objpoints = data['objpoints']
imgpoints = data['imgpoints']
camera_matrix = data['K']
dist_coeffs = data['dist']

total_error = 0
num_images = len(objpoints)

for i in range(num_images):
    # Project 3D points to image plane
    imgpoints2, _ = cv2.projectPoints(objpoints[i], np.zeros((3,1)), np.zeros((3,1)), camera_matrix, dist_coeffs)
    # Compute error
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print(f"Image {i+1}: Reprojection error = {error:.4f}")
    total_error += error

print(f"\nAverage reprojection error: {total_error / num_images:.4f}")