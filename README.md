<div align="center">

# 4DoF Vision-Guided Robotic Pen Sorting

![Python](https://img.shields.io/badge/Python-3.10+-2b5b84?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-00D9FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Vision-guided robotic manipulation with 4-DoF arms**

</div>

---

## Overview

Vision system for autonomous object manipulation with a 4-DoF RoArm-M2-S. Detects writing utensils using YOLOv8 OBB, transforms pixel coordinates to robot workspace, and executes adaptive pick-and-place with two grasp strategies based on object orientation.

**Pipeline:** Camera → YOLOv8 Detection → Color Classification → Coordinate Transform → Motion Planning → Robot Execution

---

## Features

- **YOLOv8 OBB Detection** - Custom model trained on 330 pen/pencil images
- **Dual Motion Strategies** - STANDARD (< 45°) and COMPLEX (≥ 45°) grasp modes
- **Camera Calibration** - Checkerboard intrinsic + ArUco extrinsic alignment
- **Color Sorting** - HSV/LAB classification into 4 bins (blue, red, green, grayscale)
- **Live Visualization** - OpenCV overlay + Matplotlib workspace plotting

---

## Installation

### Prerequisites

- Python 3.10+
- USB webcam
- RoArm-M2-S 4-DoF arm with serial interface
- Printed 8-inch ArUco marker (CV2 4X4_50, ID 0)
- 9×6 checkerboard pattern (22mm squares)

### Setup

```bash
# Clone repository
git clone https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting.git
cd 4DoF_vision_robotic_pen_sorting

# Install dependencies
pip install opencv-python numpy matplotlib ultralytics pyserial

# Create config.json
cat > config.json << EOF
{
  "serial_port": "/dev/tty.usbserial-210",
  "robot_tag_xyz": [300, 0, -57]
}
EOF
```

### Required: YOLOv8 Model

The trained YOLOv8 OBB model (`best.pt`, 6.3 MB) is not included in this repository. You need to either:
- Train your own model on pen/marker images using YOLOv8 OBB
- Contact the repository owner for the trained model file
- Place the model file as `best.pt` in the project root directory

### Required: RoArm Control Library

RoArm-M2-S control code is not included in this repository. Obtain the official control libraries from WaveShare:
- [WaveShare RoArm-M2-S Wiki](https://www.waveshare.com/wiki/RoArm-M2-S)
- Download the official Python SDK and place control scripts in a `RoArm/` directory
- Required: `serial_simple_ctrl.py` for serial JSON communication with the robot

---

## Quick Start

### 1. Camera Calibration

Capture 100+ checkerboard images:
```bash
mkdir -p CalibrationPictures
python camera_capture.py CalibrationPictures
# Press SPACE to capture, q to quit
```

Run calibration (outputs `calib_data.npz`):
```bash
python camera_calibrate.py
```

### 2. ArUco Calibration

Create directory and capture marker image:
```bash
mkdir -p Aruco
python camera_capture.py Aruco
# Press SPACE to capture, q to quit
```

Rename your best capture to `aruco_calibration.jpg`, then generate calibration:
```bash
mv Aruco/1.jpg Aruco/aruco_calibration.jpg  # or whichever image you want
python aruco_pose.py
```

**Note:** `aruco_pose.py` expects `Aruco/aruco_calibration.jpg` to exist. The `full_run.py` script automates this selection.

### 3. Run System

**Full pipeline (recommended):**
```bash
python full_run.py
```

**Direct detection:**
```bash
python camera_stream.py /dev/tty.usbserial-210 ResearchDataset
```

---

## Usage

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Trigger robot motion |
| `u` | Toggle auto-trigger mode |
| `p` | Toggle 3D plot overlay |
| `v` | Toggle Matplotlib window |
| `q` | Quit |

### Motion Planning

- **STANDARD** (angle < 45°): Perpendicular offset grasp → Color bin
- **COMPLEX** (angle ≥ 45°): Sweep reorientation → Color bin

### Color Routing

- Blue → Y=+140mm
- Red → Y=+70mm
- Green → Y=-70mm
- Grayscale → Y=-140mm

---

## Project Structure

```
.
├── camera_stream.py           # Main detection & control loop
├── full_run.py                # Complete pipeline orchestration
├── camera_calibrate.py        # Intrinsic calibration
├── aruco_pose.py              # Extrinsic calibration
├── camera_capture.py          # Image capture utility
├── check_calibration.py       # Verify calibration quality
├── RoArm/
│   ├── serial_simple_ctrl.py  # Serial robot control
│   └── http_simple_ctrl.py    # HTTP robot control (alternative)
├── Misc/                      # Utility scripts
│   ├── camera_list.py         # List available cameras
│   ├── aruco_stream.py        # Test ArUco detection
│   └── undistort_stream.py    # Test distortion correction
├── best.pt                    # YOLOv8 OBB model (NOT in repo - see Installation)
├── calib_data.npz             # Camera calibration (generated, gitignored)
├── config.json                # Configuration (you create, gitignored)
├── CalibrationPictures/       # Checkerboard images (you create)
├── CalibratedLinePictures/    # Annotated calibration (generated)
├── Aruco/
│   ├── aruco_calibration.jpg  # ArUco marker image (you provide)
│   └── aruco_reference.json   # ArUco calibration (generated, gitignored)
└── ResearchDataset/           # Session logs (generated, gitignored)
    └── log N/                 # Per-session recordings
```

**Gitignored files:** `*.npz`, `*.pt`, `*.jpg`, `*.png`, `*.mp4`, `config.json`, log folders

---

## Troubleshooting

**Missing directories:**
```bash
mkdir -p Aruco CalibrationPictures RoArm
```
The system expects these directories to exist before running calibration or detection scripts.

**Camera not found:**
```bash
python Misc/camera_list.py
```
This lists all available cameras on your system. If your camera isn't detected:
- Check USB connection and permissions
- Try a different camera index in the code
- Ensure no other application is using the camera

**Serial port not found:**
- macOS: `ls /dev/tty.usbserial-*`
- Linux: `ls /dev/ttyUSB* /dev/ttyACM*`
- Windows: Check Device Manager → Ports (COM & LPT)

If the robot isn't detected:
- Install USB-to-serial drivers (CP2102, CH340, or FTDI)
- Check the physical connection to the robot
- Verify power is connected to the RoArm-M2-S
- Update `serial_port` in `config.json` with the correct device path

**Calibration fails:**
- Ensure 100+ checkerboard images in `CalibrationPictures/` (more is better)
- Images should show the checkerboard from various angles and distances
- Verify ArUco image exists as `Aruco/aruco_calibration.jpg`
- The marker must be clearly visible and not blurred
- Run `check_calibration.py` to verify reprojection error < 0.5 pixels
- If error is high, recapture calibration images with better lighting

**Missing best.pt:**
- The YOLOv8 model file must be in the project root directory
- File size should be approximately 6.3 MB
- Train your own model using the provided dataset structure in `Pens.v2i.yolov8-obb/`
- Or contact the repository owner for the pre-trained model

**Missing RoArm control code:**
- Download from [WaveShare RoArm-M2-S Wiki](https://www.waveshare.com/wiki/RoArm-M2-S)
- Place Python SDK files in `RoArm/` directory
- Minimum required: `serial_simple_ctrl.py`
- Verify baud rate is set to 115200 in the control script

**Detection not working:**
- Verify `best.pt` exists in project root
- Check camera feed is clear and well-lit
- Pens/markers should be clearly visible against a plain background
- Lower confidence threshold in `camera_stream.py` (line ~125) if needed
- Ensure the top 20% of the frame is clear (this region is ignored)

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**Questions?** [Open an issue](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/issues)

</div>
