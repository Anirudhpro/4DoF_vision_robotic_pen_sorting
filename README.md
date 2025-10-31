<div align="center">

# 4DoF Vision Robotic Pen Sorting

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#prerequisites)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](#dependencies)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-orange)](#object-detection)
[![ArUco](https://img.shields.io/badge/ArUco-Calibration-purple)](#calibration)
[![Last commit](https://img.shields.io/github/last-commit/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/commits/main)
[![Issues](https://img.shields.io/github/issues/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/issues)
[![Forks](https://img.shields.io/github/forks/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/network)
[![Stars](https://img.shields.io/github/stars/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/stargazers)

*Using Visual Intelligence and Motion Planning to Enable Complex Object Manipulation with a 4 DoF Robotics Arm*

</div>

---

## Project Description

This project demonstrates how cost-effective 4 DoF robotic arms can perform manipulation tasks typically requiring expensive 6 DoF systems by leveraging **visual intelligence** and **intelligent motion planning**. The system uses a custom-trained YOLOv8 Oriented Bounding Box (OBB) model to detect writing utensils, converts pixel coordinates into real-world robot coordinates through precise calibration, and executes sophisticated pick-and-place operations.

### Key Features & Technologies

**Computer Vision Pipeline**:
- Custom YOLOv8 OBB model trained on 330 annotated writing utensil images
- Real-time oriented bounding box detection with confidence thresholding (≥0.7)
- HSV/LAB color classification for sorting (blue, red, green, grayscale)
- Region-of-Interest filtering to eliminate clutter

**Calibration & Coordinate Transformation**:
- Checkerboard-based intrinsic calibration (Zhang's method) with 209 board captures
- ArUco-based extrinsic calibration for world alignment
- Precise pixel → camera → ArUco plane → robot coordinate transformation
- Automatic coordinate validation with safety gates

**Intelligent Motion Planning**:
- **STANDARD Motion**: Perpendicular left-offset grasp for low misalignment angles (<45°)
- **COMPLEX Motion**: Sweep-based reorientation for high misalignment angles (≥45°)
- Foam-assisted gripping with reduced closure to prevent object expulsion
- Color-based routing to designated drop-off zones

**Visualization & Debugging**:
- Dual UI: OpenCV real-time overlay + interactive Matplotlib workspace visualization
- Live coordinate display (pixel, camera-relative, robot coordinates)
- Motion preview with waypoint visualization
- Comprehensive session logging with MP4 recording and timestamped snapshots

<!-- Removed: 'Why These Technologies?' and 'Challenges Solved & Future Improvements' sections to keep README concise -->

---
## Table of Contents

- ![project](assets/icons/project.svg) [Project Description](#project-description)
- ![install](assets/icons/installation.svg) [Installation & Prerequisites](#system-requirements)
- ![calib](assets/icons/calibration.svg) [Calibration Workflow](#calibration-workflow)
- ![run](assets/icons/running.svg) [Running the System](#running-the-system)
- ![usage](assets/icons/project.svg) [Usage Instructions & Examples](#usage-instructions--examples)
- ![research](assets/icons/data.svg) [Research Methodology & Results](#research-methodology--results)
- ![citation](assets/icons/data.svg) [Citation](#citation)
- ![tests](assets/icons/tests.svg) [Tests](#tests)
### System Requirements

- **Python**: 3.10+ (recommended)
- **Operating System**: macOS/Linux (tested on macOS)
- **Hardware**: 
  - Camera accessible by OpenCV (USB webcam recommended)
  - RoArm-M2-S 4 DoF robotic arm with serial interface
  - Serial device path (e.g., `/dev/tty.usbserial-XXXX` on macOS)

### Dependencies

Install the complete environment:

```bash

<div align="center">

# 4DoF Robotic Pen Sorting — Vision‑Guided

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#requirements)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](#requirements)

</div>

Short: detects pens with a YOLOv8 OBB model, converts detections to robot coordinates via ArUco calibration, and runs pick/place motions on a 4‑DoF RoArm.

---

## Quick start

1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Create `config.json` (example)

```json
{
    "serial_port": "/dev/tty.usbserial-XXXX",
    "robot_tag_xyz": [300, 0, -57]
}
```

3) Calibrate camera (one-time)

- Put 100+ checkerboard images in `CalibrationPictures/`
- Run:

```bash
python camera_calibrate.py
```

4) Capture ArUco pose (one-time)

```bash
python aruco_pose.py
```

5) Run the system

```bash
python camera_stream.py <serial_port> [logs_directory]
# or full pipeline
python full_run.py
```

---

## Files you will use

- `camera_stream.py` — real-time detection + optional robot commands
- `camera_capture.py` — capture images for calibration/dataset
- `camera_calibrate.py` — intrinsic calibration (checkerboard)
- `aruco_pose.py` — ArUco extrinsic/world alignment
- `full_run.py` — run the full pipeline
- `RoArm/serial_simple_ctrl.py` — serial robot utility

---

## Configuration

- `serial_port`: path to robot serial port
- `robot_tag_xyz`: arm pose relative to the printed ArUco tag (pre-calibration), in mm

## Tests

- Unit tests:

```bash
python test_pixel_conversion.py
python test_coordinates.py
```

- Integration (no robot):

```bash
python camera_stream.py --mock-robot ResearchDataset
```

---

Notes:

- Icons are in `assets/icons/` and referenced relatively so they render on GitHub.
- This README focuses on practical usage; detailed research notes were moved out of the main README.

---

License: see `LICENSE` file.
**Output Format** (`aruco_reference.json`):
```json
{
## 4DoF Robotic Pen Sorting — Run & Usage

Short: this repository runs a vision pipeline that detects pens and (optionally) commands a 4‑DoF RoArm to pick and place.

Prerequisites
- Python 3.10+
- A webcam or USB camera supported by OpenCV
- (Optional) RoArm serial device for real robot runs

Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configuration
1. Copy `config.example.json` (or create `config.json`) and set `serial_port` and `robot_tag_xyz`.

Example `config.json`:
```json
{
  "serial_port": "/dev/tty.usbserial-XXXX",
  "robot_tag_xyz": [300, 0, -57]
}
```

Quick run
- Calibrate camera (one-time): `python camera_calibrate.py` (put checkerboard images in `CalibrationPictures/`)
- Capture ArUco pose (one-time): `python aruco_pose.py`
- Start detection (no robot): `python camera_stream.py --mock-robot ResearchDataset`
- Start detection + robot: `python camera_stream.py /dev/tty.usbserial-XXX ResearchDataset`
- Full pipeline: `python full_run.py`

Files you will use
- `camera_stream.py` — main real-time script
- `camera_calibrate.py` / `calib_data.npz` — intrinsics
- `aruco_pose.py` / `Aruco/aruco_reference.json` — extrinsics
- `camera_capture.py` — helper capture tool
- `RoArm/serial_simple_ctrl.py` — serial robot utility
- `full_run.py` — orchestrated pipeline

Tests
- Unit: `python test_pixel_conversion.py`, `python test_coordinates.py`
- Integration (no robot): `python camera_stream.py --mock-robot ResearchDataset`

Notes
- Icons are in `assets/icons/` and referenced relatively so they render on GitHub.
- License: MIT (see `LICENSE`)

---
   - Interactive capture interface
