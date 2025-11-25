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

The trained YOLOv8 OBB model (`best.pt`) is not included. Train your own model or obtain separately and place in project root.

### Required: RoArm Control Library

RoArm-M2-S control code is not included. Download from [WaveShare RoArm-M2-S Wiki](https://www.waveshare.com/wiki/RoArm-M2-S) and place in `RoArm/` directory. Required file: `serial_simple_ctrl.py`

---

## Quick Start

### 1. Camera Calibration

Capture 100+ checkerboard images:
```bash
python camera_capture.py
# Press SPACE to capture, ESC to quit
# Images saved to CalibrationPictures/ (created automatically)
```

Run calibration (outputs `calib_data.npz`):
```bash
python camera_calibrate.py
```

### 2. ArUco Calibration

Capture ArUco marker image:
```bash
python camera_capture.py
# Press SPACE to capture marker image, ESC to quit
```

Move your best capture to Aruco directory and rename:
```bash
mkdir -p Aruco
mv CalibrationPictures/1.jpg Aruco/aruco_calibration.jpg
python aruco_pose.py
```

**Note:** `aruco_pose.py` expects `Aruco/aruco_calibration.jpg` to exist.

### 3. Run System

**Mock robot (testing without hardware):**
```bash
python camera_stream.py --mock-robot
```

**With real robot:**
```bash
python camera_stream.py
```

**Full pipeline with calibration:**
```bash
python full_run.py
```

---

## Usage

### Controls (camera_stream.py)

| Key | Action |
|-----|--------|
| `SPACE` | Trigger robot motion for detected object |
| `u` | Toggle auto-trigger mode |
| `p` | Toggle 3D plot overlay |
| `v` | Toggle Matplotlib workspace window |
| `ESC` | Quit |

### Motion Planning

- **STANDARD** (angle < 45°): Direct perpendicular grasp → Color bin
- **COMPLEX** (angle ≥ 45°): Sweep to reorient → Color bin

### Color Routing

- Blue → Y=+140mm
- Red → Y=+70mm  
- Green → Y=-70mm
- Grayscale → Y=-140mm

---

## Project Structure

```
.
├── camera_stream.py           # Main detection & robot control
├── full_run.py                # Complete pipeline with calibration
├── camera_calibrate.py        # Intrinsic camera calibration
├── aruco_pose.py              # Extrinsic ArUco calibration
├── camera_capture.py          # Image capture utility
├── test_pixel_conversion.py   # Unit tests for coordinate transform
├── test_coordinates.py        # Integration tests for calibration
├── RoArm/
│   └── serial_simple_ctrl.py  # Serial robot control (from WaveShare)
├── Misc/                      # Utility scripts
│   ├── camera_list.py         # List available cameras
│   ├── aruco_stream.py        # Test ArUco detection live
│   └── undistort_stream.py    # Test camera distortion correction
├── best.pt                    # YOLOv8 OBB model (NOT in repo)
├── calib_data.npz             # Camera calibration data (generated)
├── config.json                # Robot configuration (you create)
├── CalibrationPictures/       # Checkerboard images (auto-created)
├── CalibratedLinePictures/    # Annotated calibration (generated)
├── Aruco/
│   ├── aruco_calibration.jpg  # ArUco marker image (you provide)
│   └── aruco_reference.json   # ArUco calibration data (generated)
└── ResearchDataset/           # Session logs (auto-created)
    └── log_*/                 # Per-session data and videos
```

---

## Testing

Run unit tests:
```bash
python test_pixel_conversion.py
python test_coordinates.py
```

---

## Troubleshooting

**List available cameras:**
```bash
python Misc/camera_list.py
```

**Find serial port:**
- macOS: `ls /dev/tty.usbserial-*`
- Linux: `ls /dev/ttyUSB* /dev/ttyACM*`

**Test ArUco detection:**
```bash
python Misc/aruco_stream.py
```

**Test camera undistortion:**
```bash
python Misc/undistort_stream.py
```

---

## License

MIT License - see [LICENSE](LICENSE)
