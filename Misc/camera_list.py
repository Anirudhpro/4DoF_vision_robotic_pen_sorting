#!/usr/bin/env python3
"""List cameras with their indices so you can set "camera_index" in config.json.
The index shown here matches cv2.VideoCapture(index) on macOS (AVFoundation order)."""

def main():
    try:
        import AVFoundation
        names = ['AVCaptureDeviceTypeBuiltInWideAngleCamera', 'AVCaptureDeviceTypeExternal',
                 'AVCaptureDeviceTypeExternalUnknown', 'AVCaptureDeviceTypeContinuityCamera']
        types = [getattr(AVFoundation, t) for t in names if hasattr(AVFoundation, t)]
        try:
            sess = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                types, AVFoundation.AVMediaTypeVideo, 0)
            devs = list(sess.devices())
        except Exception:
            devs = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
        print("camera_index -> camera name  (put the right number in config.json):\n")
        for i, d in enumerate(devs):
            print(f"  {i}:  {d.localizedName()}")
        print('\nExample: if "LifeCam" is index 1, set  "camera_index": 1  in config.json')
    except Exception as e:
        print(f"(AVFoundation unavailable: {e}) — probing with OpenCV instead")
        import cv2
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  index {i}: available")
                cap.release()

if __name__ == "__main__":
    main()
