import numpy as np

# Load the calibration file
data = np.load("calib_data.npz")

# Print all keys in the file
print("Keys in .npz file:", data.files)

# Extract components
K = data["K"]
dist = data["dist"]
rvecs = data["rvecs"]
tvecs = data["tvecs"]

# Print camera matrix
print("\nCamera Matrix (K):")
print(K)

# Print distortion coefficients
print("\nDistortion Coefficients:")
print(dist.ravel())

# Analyze rotation and translation vectors
print("\nNumber of calibration images:", len(rvecs))

for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    print(f"\n--- Image {i + 1} ---")
    print(f"Rotation Vector (rvec): {rvec}")
    print(f"Translation Vector (tvec): {tvec}")