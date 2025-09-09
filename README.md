# 4DoF Vision Robotic Pen Sorting

Computer-vision + robotics project for sorting pens with a 4-DOF robotic arm.  
This repository contains the detection, calibration, transformation and robot-control glue used to detect pens with a YOLO OBB model, convert pixel detections into robot coordinates via an ArUco reference, and execute pick / routing motions on a physical arm.

---

## Key features
- Real-time OBB detection using a YOLO model ([best.pt](best.pt)).
- Pixel → camera → ArUco tag → robot coordinate chain implemented in [pixel_to_robot](camera_stream.py#L120).
- Tip selection and radial-angle logic to decide between "STANDARD" and "COMPLEX" motion plans ([camera_stream.py](camera_stream.py)).
- Live annotated video with overlays (OpenCV) and an embedded workspace plot rendered by matplotlib.
- Motion sending and safety checks via serial commands to the robotic controller ([send_json](camera_stream.py#L160)).
- Tools for calibration, testing and visualization.

---

## Repo layout (important files)
- [camera_stream.py](camera_stream.py) — main capture, detection, visualization and motion dispatch.
- [camera_capture.py](camera_capture.py) — utilities to capture calibration images.
- [camera_calibrate.py](camera_calibrate.py) — generate `calib_data.npz` (camera intrinsics).
- [aruco_pose.py](aruco_pose.py) — ArUco-based pose estimation / tag reference utilities.
- [check_calibration.py](check_calibration.py) — sanity checks for calibration and transforms.
- [requirements.txt](requirements.txt) — Python dependencies.
- [test_coordinates.py](test_coordinates.py), [test_pixel_conversion.py](test_pixel_conversion.py) — small unit / sanity tests.
- [Aruco/aruco_reference.json](Aruco/aruco_reference.json) — saved ArUco rvec/tvec used by the pipeline.
- [best.pt](best.pt) — trained YOLO OBB model (used by [camera_stream.py](camera_stream.py)).
- `ResearchDataset/` — session logs and saved frames.

---

## Installation

1. Create & activate a Python 3.10+ virtualenv (recommended):
   ```sh
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   If your editor warns about missing `matplotlib` (or other libs), installing via the above will resolve it.

---

## Quick start — live stream + robot control

1. Ensure camera calibration file `calib_data.npz` and ArUco reference (`Aruco/aruco_reference.json`) exist. Use [camera_calibrate.py](camera_calibrate.py) and [aruco_pose.py](aruco_pose.py) if you need to (re)create them.

2. Connect robotic controller and find the serial device (example `/dev/tty.usbserial-xxx` on macOS).
# 4DoF Vision Robotic Pen Sorting

Computer-vision + robotics project for detecting pens and commanding a 4‑DOF arm to pick and route them. The repo contains detection, calibration, projection and serial control code that converts image detections into robot-space motions.

This README gives a concise, runnable overview for local use (environment, how to run the main programs and quick troubleshooting).

---

## Quick summary
- Main live program: `camera_stream.py` — captures frames, runs the YOLO OBB model, projects detections into robot coordinates, visualizes overlays, and sends motion JSON packets to the arm over serial.
- Orchestration helper: `full_run.py` — convenience script that runs parts of the pipeline (serial sanity, capture, ArUco management, pose estimation, then the live stream). It supports a `short` mode that runs only the live stream.

---

## Prerequisites
- Python 3.10+ recommended
- Camera connected and accessible by OpenCV
- Robot controller connected via serial (identify device path, e.g. `/dev/tty.usbserial-xxx`)
- A calibrated camera file (`calib_data.npz`) and an ArUco reference (`Aruco/aruco_reference.json`) — see Calibration section below

## Install
1. Create and activate a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
```
2. Install dependencies:
```sh
pip install -r requirements.txt
```

---

## Running the live system

1) Run `camera_stream.py` directly (typical):

```sh
python camera_stream.py <serial_port> [logs_root]
# example:
python camera_stream.py /dev/tty.usbserial-123 ResearchDataset
```

- Arguments:
  - `<serial_port>`: required. Serial device for the robot controller (e.g. `/dev/tty.usbserial-123`).
  - `[logs_root]`: optional. Folder where session data will be saved (default `ResearchDataset`).

- Behavior:
  - Opens an OpenCV window called "Pen Detection" showing annotated frames.
  - A separate Matplotlib XY viz window can be toggled with `v` and will be updated at the same rate as the frames.
  - Press Space to trigger the robot motion for the currently-detected confident detections.
  - Press `u` to toggle AUTO mode (auto-triggering), `p` to toggle the small plot overlay, `v` to toggle the separate viz window, and `q` to quit.

2) Or run the orchestrator `full_run.py`:

```sh
python full_run.py short   # runs camera_stream.py only (useful during development)
python full_run.py         # runs a multi-step process: serial test, capture, ArUco management, pose, then live stream
```

`full_run.py` automates common pre-steps (e.g. running the serial sanity check, capturing ArUco images and creating the reference pose). Use the `short` argument to skip directly to the live stream.

---

## Config
- `config.json` holds site-specific values (serial port key, robot_tag_xyz, etc.). If present, `full_run.py` and some helpers read it; `camera_stream.py` accepts the serial port as argument and will use the supplied logs folder.

---

## Calibration

- Create calibration images with `camera_capture.py` and run `camera_calibrate.py` to produce `calib_data.npz` (camera intrinsics).
- Produce or update the ArUco reference pose with `aruco_pose.py` — it writes `Aruco/aruco_reference.json` used at runtime.

---

## Important behaviors implemented in code
- Only detections with confidence >= 0.7 are considered for robot motion.
- Pixel → camera → ArUco plane → tag → robot transform is implemented in `pixel_to_robot` inside `camera_stream.py`.
- Motion commands are sent as JSON over serial (example format: `{ "T":1041, "x":..., "y":..., "z":..., "t":... }`).
- Workspace safety gate (`validate_robot_coords`) enforces radial distance and Z limits before sending a command; out-of-bounds commands are rejected and logged.

---

## Troubleshooting & notes
- If the Matplotlib viz seems slow when enabled, toggle it off with `v` to reduce CPU/GPU load.
- If the session folder contents do not appear in your file browser, check the printed session path — session folders are created under the chosen `logs_root`.
- If the robot doesn't move, verify the serial device path and that the controller is accepting JSON commands (monitor the printed `Sent: {...}` lines).

---

## Tests
- Run unit/sanity tests:
```sh
python test_pixel_conversion.py
python test_coordinates.py
```

---

## License
MIT — see [LICENSE](LICENSE)
