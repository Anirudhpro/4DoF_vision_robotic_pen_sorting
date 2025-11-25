<div align="center">

# 4DoF Vision-Guided Robotic Pen Sorting

<img src="https://img.shields.io/badge/Python-3.10+-1e293b?style=flat-square&logo=python&logoColor=white&labelColor=1e293b" alt="Python"/>
<img src="https://img.shields.io/badge/OpenCV-4.x-1e293b?style=flat-square&logo=opencv&logoColor=white&labelColor=1e293b" alt="OpenCV"/>
<img src="https://img.shields.io/badge/YOLOv8-OBB-1e293b?style=flat-square&logo=yolo&logoColor=white&labelColor=1e293b" alt="YOLOv8"/>
<img src="https://img.shields.io/badge/License-MIT-1e293b?style=flat-square&labelColor=1e293b" alt="MIT"/>

### Autonomous robotic manipulation using 4-DoF arms through intelligent vision and motion planning

<br/>

<table>
<tr>
<td align="center" width="33%">

<div style="background: linear-gradient(145deg, #1e293b 0%, #475569 50%, #1e293b 100%); border-radius: 12px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
<img src="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/cpu.svg" width="40" height="40" style="filter: brightness(0) invert(1);"/>
</div>

**Computer Vision**
YOLOv8 OBB Detection
HSV/LAB Color Analysis
Real-time Processing

</td>
<td align="center" width="33%">

<div style="background: linear-gradient(145deg, #1e293b 0%, #475569 50%, #1e293b 100%); border-radius: 12px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
<img src="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/target.svg" width="40" height="40" style="filter: brightness(0) invert(1);"/>
</div>

**Calibration System**
Checkerboard Intrinsic
ArUco Extrinsic
Coordinate Transformation

</td>
<td align="center" width="33%">

<div style="background: linear-gradient(145deg, #1e293b 0%, #475569 50%, #1e293b 100%); border-radius: 12px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
<img src="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/move.svg" width="40" height="40" style="filter: brightness(0) invert(1);"/>
</div>

**Motion Planning**
Adaptive Grasp Strategies
Sweep Reorientation
Color-Based Routing

</td>
</tr>
</table>

</div>

---

## Overview

### What This Project Does

This project demonstrates **autonomous object manipulation** using a 4-DoF RoArm-M2-S robotic arm guided by computer vision. The system:

1. **Detects** writing utensils (pens, pencils, markers) using a custom YOLOv8 OBB model
2. **Analyzes** object position, orientation, and color
3. **Plans** adaptive grasp strategies based on object alignment
4. **Executes** precise pick-and-place operations
5. **Sorts** objects into color-coded bins

### Why This Project Is Useful

Traditional robotic pick-and-place systems typically require expensive 6-DoF arms with full rotational freedom. This project shows how a **budget-friendly 4-DoF arm** can accomplish similar tasks by:

- **Visual Intelligence**: Using YOLOv8 OBB to detect object position, orientation, and dimensions
- **Adaptive Motion Planning**: Implementing two distinct grasp strategies based on object alignment
- **Precise Calibration**: Multi-stage coordinate transformation (Pixel → Camera → ArUco → Robot)
- **Color-Based Sorting**: HSV/LAB color classification for automated routing

---

## Features

### Computer Vision Pipeline
- **YOLOv8 OBB Detection**: Custom model trained on 330 annotated images
- **Oriented Bounding Boxes**: Detects position, rotation, and dimensions of writing utensils
- **Color Classification**: HSV/LAB analysis to sort objects by color (blue, red, green, grayscale)
- **Smart Filtering**: Region-of-interest masking and confidence thresholding (≥0.7)

### Calibration System
- **Intrinsic Calibration**: Checkerboard-based camera calibration using Zhang's method
- **Extrinsic Calibration**: ArUco marker-based world frame alignment
- **Coordinate Transformation**: Precise pixel-to-robot coordinate mapping with validation
- **Multi-Stage Pipeline**: Camera frame → ArUco plane → Robot workspace

### Motion Planning
- **STANDARD Mode**: Perpendicular offset grasp for well-aligned objects (angle < 45°)
- **COMPLEX Mode**: Sweep-based reorientation for misaligned objects (angle ≥ 45°)
- **Safety Validation**: Workspace boundary checking (80-500mm radial, -100 to +450mm Z)
- **Color Routing**: Automated drop-off at color-coded bins

### Visualization & Logging
- **Dual UI**: Real-time OpenCV overlay + interactive Matplotlib workspace visualization
- **Session Recording**: MP4 video logging with timestamped snapshots
- **Live Debugging**: Coordinate display, motion preview, and status indicators
- **Interactive Control**: Manual triggering or auto-mode with configurable cooldown

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Hardware**:
  - USB webcam or built-in camera
  - RoArm-M2-S 4-DoF robotic arm
  - USB-to-serial adapter (e.g., CP2102, CH340)
  - Printed 8-inch ArUco marker (CV2 4X4_50 dictionary, ID 0)
  - 9×6 checkerboard calibration pattern (22mm squares)
- **Operating System**: macOS or Linux (tested on macOS 24.2.0)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting.git
   cd 4DoF_vision_robotic_pen_sorting
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   You'll also need to install OpenCV, NumPy, Matplotlib, and Ultralytics:
   ```bash
   pip install opencv-python numpy matplotlib ultralytics
   ```

4. **Configure the system**

   Create or update `config.json` in the project root:
   ```json
   {
     "serial_port": "/dev/tty.usbserial-210",
     "robot_tag_xyz": [300, 0, -57]
   }
   ```

   - `serial_port`: Path to your robot's serial device
     - macOS: `/dev/tty.usbserial-XXXX`
     - Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`
     - Windows: `COM3`, `COM4`, etc.
   - `robot_tag_xyz`: Robot base position relative to ArUco marker center (in mm)

---

## Quick Start

### Expected Directory Structure

After calibration, your project should have these folders and files:

```
4DoF_vision_robotic_pen_sorting/
├── CalibrationPictures/       # 100+ checkerboard images (you create)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── CalibratedLinePictures/    # Auto-generated by camera_calibrate.py
│   └── calib_*.jpg
├── Aruco/                     # ArUco marker image and calibration
│   ├── aruco_calibration.jpg  # Single image of ArUco marker (you provide)
│   └── aruco_reference.json   # Auto-generated by aruco_pose.py
├── calib_data.npz             # Auto-generated by camera_calibrate.py
├── config.json                # You create this
└── best.pt                    # YOLOv8 model (included in repo)
```

### One-Time Calibration

Before running the system, you need to calibrate your camera and establish the world coordinate frame:

#### 1. Camera Intrinsic Calibration

**Step 1a: Capture checkerboard images**

Run the capture script to save images to the `CalibrationPictures/` folder:
```bash
python camera_capture.py CalibrationPictures
```

- Press `SPACE` to capture each image (aim for 100+ images)
- Move the checkerboard to different positions, angles, and distances
- Press `q` when done

This creates a folder structure:
```
CalibrationPictures/
├── 1.jpg
├── 2.jpg
├── 3.jpg
└── ...
```

**Step 1b: Run calibration**

Process the captured images to compute camera intrinsics:
```bash
python camera_calibrate.py
```

This script:
- Reads all images from `CalibrationPictures/`
- Detects 9×6 checkerboard corners
- Saves annotated images to `CalibratedLinePictures/`
- Outputs `calib_data.npz` (camera matrix K and distortion coefficients)

#### 2. ArUco Extrinsic Calibration

**Step 2a: Prepare the workspace**

- Print an 8-inch ArUco marker (CV2 4X4_50 dictionary, ID 0)
- Place it in the robot's workspace where the camera can see it
- Ensure the marker is flat and well-lit

**Step 2b: Capture ArUco image**

You can either:
- Manually place a clear photo of the marker in `Aruco/aruco_calibration.jpg`, OR
- Use the full pipeline (`python full_run.py`) which captures it for you

**Step 2c: Generate calibration**

Run the ArUco pose estimation:
```bash
python aruco_pose.py
```

This script:
- Reads `Aruco/aruco_calibration.jpg`
- Detects the ArUco marker and computes its 6-DOF pose
- Outputs `Aruco/aruco_reference.json` (camera-to-marker transformation)

### Running the System

**Option 1: Full Pipeline (Recommended)**
```bash
python full_run.py
```

This orchestrates the complete workflow:
1. Tests serial connection and moves arm to reference position
2. Captures fresh ArUco images
3. User selects the best image
4. Regenerates ArUco calibration
5. Launches the main detection and control system

**Option 2: Detection Only (Skip Calibration)**
```bash
python full_run.py short
```

Jumps directly to the camera stream if you've already calibrated.

**Option 3: Manual Control**
```bash
python camera_stream.py /dev/tty.usbserial-210 ResearchDataset
```

Runs the detection system with manual triggering.

**Option 4: Mock Mode (No Robot Required)**
```bash
python camera_stream.py --mock-robot ResearchDataset
```

Test the vision pipeline without hardware.

---

## Usage

### Interactive Controls

When `camera_stream.py` is running:

| Key | Action |
|-----|--------|
| `SPACE` | Trigger robot motion on current detections |
| `u` or `U` | Toggle auto-trigger mode (2-second cooldown) |
| `p` | Toggle 3D coordinate plot overlay |
| `v` | Toggle separate Matplotlib visualization window |
| `q` | Quit the program |

### Understanding the Display

**OpenCV Window**:
- **Blue boxes**: Detected objects (OBB)
- **Green circles**: Selected grasp tip
- **Yellow circles**: Alternate tip
- **Text overlays**: Pixel coords, camera coords, robot coords, color, angle
- **Shaded region**: Top 20% (ignored to avoid clutter)
- **Status badge**: Shows AUTO or SPACE mode

**Matplotlib Window** (if enabled):
- **Scatter plot**: Pen locations in robot XY workspace
- **Stars**: Grasp tips (green = selected, yellow = alternate)
- **Red X**: Recently sent commands (fade after 1s)
- **Arc**: Pen radial angle visualization
- **Waypoints**: Motion path preview for COMPLEX mode

### Motion Planning Logic

The system chooses between two grasp strategies based on the **pen radial angle** (angle between the pen's chosen tip and the line from pen center to robot origin):

**STANDARD Mode** (angle < 45°):
- Perpendicular left-offset approach
- Sequence: Safe → Approach → Hover (10mm offset) → Descend → Grasp → Retract → Route to color bin → Drop → Home

**COMPLEX Mode** (angle ≥ 45°):
- Sweep-based reorientation with intermediate waypoints
- Constructs geometric path to align gripper with pen during approach
- Longer execution time (~2-3 seconds)

### Color-Based Routing

Detected pens are sorted into bins based on color:
- **Blue** → Y = +140mm
- **Red** → Y = +70mm
- **Green** → Y = -70mm
- **Grayscale** → Y = -140mm

All drop-offs occur at X = 480mm, Z = 60mm.

---

## Documentation

### Project Structure

```
.
├── camera_stream.py           # Main detection & control loop
├── full_run.py                # Complete pipeline orchestration
├── camera_capture.py          # Image capture utility
├── camera_calibrate.py        # Intrinsic calibration
├── aruco_pose.py              # Extrinsic calibration
├── RoArm/
│   ├── serial_simple_ctrl.py  # Serial robot control
│   └── http_simple_ctrl.py    # HTTP robot control (WiFi)
├── Misc/
│   ├── aruco_stream.py        # ArUco detection test
│   ├── camera_list.py         # List available cameras
│   └── undistort_stream.py    # Distortion correction test
├── best.pt                    # Trained YOLOv8 OBB model
├── calib_data.npz             # Camera calibration data
├── config.json                # Runtime configuration
└── requirements.txt           # Python dependencies
```

### Key Files

- [camera_stream.py](camera_stream.py): Main detection loop with YOLOv8 inference, coordinate transformation, and robot control
- [full_run.py](full_run.py): Orchestrates the 5-step pipeline from serial test to live control
- [aruco_pose.py](aruco_pose.py): Computes camera-to-ArUco transformation for world frame alignment
- [RoArm/serial_simple_ctrl.py](RoArm/serial_simple_ctrl.py): Low-level serial JSON command interface
- [best.pt](best.pt): Custom YOLOv8 OBB model trained on 330 pen/pencil images

### Calibration Data

- **`calib_data.npz`**: Camera matrix (K), distortion coefficients, and per-image poses
- **`Aruco/aruco_reference.json`**: ArUco marker pose (rvec, tvec) relative to camera

### Testing

Validate the coordinate transformation pipeline:
```bash
python test_pixel_conversion.py  # Test pixel → robot conversion
python test_coordinates.py       # Test coordinate math
python check_calibration.py      # Compute reprojection error
```

Run in mock mode to test without hardware:
```bash
python camera_stream.py --mock-robot ResearchDataset
```

---

## Troubleshooting

**Missing directories or files**:
```bash
# Create the Aruco folder if it doesn't exist
mkdir -p Aruco

# Create config.json if missing (update paths as needed)
echo '{"serial_port": "/dev/tty.usbserial-210", "robot_tag_xyz": [300, 0, -57]}' > config.json
```

**Camera not detected**:
```bash
python Misc/camera_list.py  # List available cameras
```

**Serial port not found**:
- macOS: Check `ls /dev/tty.usbserial-*`
- Linux: Check `ls /dev/ttyUSB* /dev/ttyACM*`
- Windows: Check Device Manager → Ports

**Calibration issues**:
- Ensure checkerboard is 9×6 with 22mm squares
- Capture images from diverse angles and distances
- Verify images are in `CalibrationPictures/` folder before running `camera_calibrate.py`
- Run `check_calibration.py` to verify reprojection error (<0.5 pixels is good)

**ArUco calibration fails**:
- Ensure `Aruco/aruco_calibration.jpg` exists and shows a clear view of the marker
- Verify the marker is 8 inches (203mm) and uses CV2 4X4_50 dictionary, ID 0
- Check that `calib_data.npz` exists (run camera calibration first)

**Robot not responding**:
- Verify serial port in `config.json`
- Test with `python RoArm/serial_simple_ctrl.py /dev/tty.usbserial-XXX`
- Check baud rate is 115200

**YOLOv8 not detecting pens**:
- Ensure `best.pt` is in the project root
- Lower confidence threshold in camera_stream.py (line ~125)
- Check GPU availability with `python Misc/checkGPU.py`

---

## Contributing

Contributions are welcome! Quick checklist:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this project in your research, please cite it using the following formats:

### BibTeX
```bibtex
@misc{4dof_pen_sorting,
  title={4DoF Vision-Guided Robotic Pen Sorting},
  author={Prabhakaran, Anirudh},
  year={2024},
  howpublished={\url{https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting}},
  note={A vision-guided robotic system demonstrating autonomous manipulation with 4-DoF arms}
}
```

### APA
Prabhakaran, A. (2024). *4DoF Vision-Guided Robotic Pen Sorting* [Computer software]. https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**What this means:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ⚠️ Warranty and liability limitations apply

---

## Support

### Getting Help

If you need help with this project:

1. **Check the documentation** - Read through this README and the [Troubleshooting](#troubleshooting) section
2. **Search existing issues** - [Browse issues](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/issues)
3. **Open a new issue** - Include your environment, steps to reproduce, and error messages

### Maintainers

This project is maintained by:
- **Anirudh Prabhakaran** ([@Anirudhpro](https://github.com/Anirudhpro))

### Acknowledgments

This project builds upon excellent open-source tools:
- **YOLOv8**: [Ultralytics](https://docs.ultralytics.com/) - Object detection framework
- **OpenCV**: [OpenCV.org](https://opencv.org/) - Computer vision library
- **RoArm-M2-S**: [Waveshare](https://www.waveshare.com/) - 4-DoF robotic arm
- **ArUco Markers**: OpenCV contrib module - Fiducial marker system

### Project Status

**Active Development** - This project is actively maintained and accepting contributions.

- 🐛 Bug fixes: Ongoing
- ✨ Feature additions: Welcome
- 📖 Documentation: Continuously improving
- 🧪 Testing: Expanding coverage

### Related Projects

If you're interested in this project, you might also like:
- [ROS (Robot Operating System)](https://www.ros.org/) - Comprehensive robotics framework
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html) - Learn computer vision

---

<div align="center">

**Questions?** Open an [issue](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/issues)

</div>
