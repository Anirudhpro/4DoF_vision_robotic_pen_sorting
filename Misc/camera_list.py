import cv2

print("Scanning for available cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is active")
        breakpoint()
        cap.release()