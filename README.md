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
  <div align="center">

  # 4DoF Vision Robotic Pen Sorting

  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#prerequisites)
  [![Last commit](https://img.shields.io/github/last-commit/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/commits/main)

  </div>

- If the session folder contents do not appear in your file browser, check the printed session path — session folders are created under the chosen `logs_root`.
  Vision + robotics system that detects pens in a video stream (YOLO OBB), reprojects detections into a robot coordinate frame using camera intrinsics and an ArUco reference, then commands a 4‑DOF arm to pick and route those pens. The project explores the full stack: calibration, perception, geometry, motion planning, visualization, and serial control.

  Research questions and goals (high level):
  - How reliably can oriented object detection (OBB) locate pen geometry for autonomous pick placement?
  - What reprojection accuracy can we achieve from pixel space into robot space with consumer hardware and a simple ArUco reference?
  - What minimal motion primitives are sufficient for robust picking, and how do visual previews help avoid mistakes?

  Key features:
  - Real-time detections with Ultralytics YOLO OBB.
  - Checkerboard-based camera calibration and ArUco-based world alignment.
  - Accurate pixel→camera→tag→robot transformations and safety-gated robot commands.
  - Dual UI: OpenCV overlay + interactive Matplotlib workspace viz with live previews and key bindings.

  > Note: A separate Polygence report (PDF) accompanies the repo and discusses motivation, background, and results. Place it in the repository if you want it versioned.

  ---

  ## Authors and contributions
  - Primary author: Anirudh (owner: `@Anirudhpro`) — system design, perception, calibration, visualization, and robot control code.
  - Mentors/Collaborators: [add names/roles here].

  Contact:
  - Please open a GitHub Issue for questions/bugs. You may add an email or ORCID here if desired.

  ---

  ## Table of contents
  - [Project description](#project-description)
  - [Authors and contributions](#authors-and-contributions)
  - [Data and file overview](#data-and-file-overview)
  - [Install and prerequisites](#install-and-prerequisites)
  - [Calibration workflow](#calibration-workflow)
    - [1) Camera intrinsics via checkerboard](#1-camera-intrinsics-via-checkerboard)
    - [2) ArUco tag pose (world alignment)](#2-aruco-tag-pose-world-alignment)
  - [Running the system](#running-the-system)
    - [Quick run (camera_stream)](#quick-run-camera_stream)
    - [Orchestrated run (full_run)](#orchestrated-run-full_run)
    - [Key bindings and UI](#key-bindings-and-ui)
  - [Usage notes and interoperability](#usage-notes-and-interoperability)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [Contributing](#contributing)
  - [Tests](#tests)

  ---

  ## Data and file overview
  Top-level files and purpose:
  - `camera_stream.py` — main real-time pipeline: capture → detect (YOLO OBB) → project to robot → visualize → send serial JSON.
  - `full_run.py` — orchestrator: serial sanity, capture, manage ArUco folder, ArUco pose, then camera stream; supports `short` mode.
  - `camera_calibrate.py` — calibrate camera intrinsics from a checkerboard image set. Saves `calib_data.npz` and annotated images.
  - `aruco_pose.py` — detect ArUco and estimate pose using `calib_data.npz`; saves `Aruco/aruco_reference.json` and `aruco_tag_detection.jpg`.
  - `camera_capture.py` — interactive image capture tool.
  - `requirements.txt` — Python dependencies.
  - `calib_data.npz` — camera matrix and distortion coefficients (generated).
  - `Aruco/aruco_reference.json` — ArUco pose w.r.t. camera; used to align into robot frame (generated).

  Important directories:
  - `CalibrationPictures/` — input images for checkerboard calibration (JPEG/JPG). Example images included.
  - `CalibratedLinePictures/` — annotated detection images saved during calibration.
  - `Aruco/` — ArUco assets and the generated `aruco_reference.json`.
  - `Pens.v1-roboflow-instant-1--eval-.yolov8-obb/` — dataset structure example for training/validation.
  - `yolo_init_model/`, `yolov8n.pt`, `yolov8n-obb.pt` — model weights/folders (be mindful of repository size).
  - `ResearchDataset/` — runtime session logs (created by `camera_stream.py`).
  - `runs/` — YOLO training outputs (if present).

  How data is produced/used:
  - `camera_calibrate.py` loads images from `CalibrationPictures/`, detects a 9×6 checkerboard (square size 22 mm), estimates intrinsics, and writes `calib_data.npz`.
  - `aruco_pose.py` loads `calib_data.npz`, detects a 4×4_50 ArUco tag from the webcam, assumes a physical marker length of 0.203 m (8 in), estimates pose, and writes `Aruco/aruco_reference.json`.
  - `camera_stream.py` reads `calib_data.npz` and `Aruco/aruco_reference.json` to convert pixel detections into robot coordinates and drive the arm.

  ---

  ## Install and prerequisites
  Requirements:
  - Python 3.10+
  - Mac/Linux recommended (tested on macOS)
  - Camera accessible by OpenCV
  - Robot controller connected via serial (macOS example: `/dev/tty.usbserial-XXXX`)

  Install steps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  Optional local config (`config.json` used by `full_run.py`):
  ```json
  {
    "serial_port": "/dev/tty.usbserial-XXXX",
    "robot_tag_xyz": [120, 0, -20]
  }
  ```

  ---

  ## Calibration workflow

  ### 1) Camera intrinsics via checkerboard
  Script: `camera_calibrate.py`
  - Expects a 9×6 checkerboard with square size 22 mm (edit `CHECKERBOARD` / `SQUARE_SIZE` if different).
  - Reads images from `CalibrationPictures/` with `.jpg` or `.jpeg` extensions.
  - Produces `calib_data.npz` and annotated images in `CalibratedLinePictures/`.

  Steps:
  ```bash
  # Put your calibration photos into CalibrationPictures/
  python camera_calibrate.py
  ```
  Output keys in `calib_data.npz`:
  - `K` — 3×3 camera matrix
  - `dist` — distortion coefficients
  - `rvecs`, `tvecs` — per-image extrinsics (diagnostic)

  Tips:
  - Use 20–30 images with varied orientations and coverage.
  - Ensure good lighting and sharp corners; avoid motion blur.

  ### 2) ArUco tag pose (world alignment)
  Script: `aruco_pose.py`
  - Uses `calib_data.npz`.
  - Detects DICT_4X4_50 tags from webcam 0.
  - Assumes marker length: 0.203 m (8 in). Adjust `marker_length` if your tag size differs.
  - Writes `Aruco/aruco_reference.json` and a visualization `aruco_tag_detection.jpg`.

  Steps:
  ```bash
  python aruco_pose.py
  ```
  If no marker is detected, verify that your tag is printed accurately, fully visible, and well lit.

  ---

  ## Running the system

  ### Quick run (`camera_stream`)
  Run directly if you already have `calib_data.npz` and `Aruco/aruco_reference.json`:
  ```bash
  python camera_stream.py <serial_port> [logs_root]
  # example
  python camera_stream.py /dev/tty.usbserial-123 ResearchDataset
  ```
  Arguments:
  - `<serial_port>` (required) — serial device for robot controller.
  - `[logs_root]` (optional) — folder for session artifacts (default: `ResearchDataset`).

  ### Orchestrated run (`full_run`)
  Convenience wrapper with two modes:
  ```bash
  python full_run.py short   # skip pre-steps; launches camera_stream only
  python full_run.py         # full pipeline (serial → capture → manage ArUco folder → pose → stream)
  ```
  What the full pipeline does:
  1. Serial control sanity (`RoArm/serial_simple_ctrl.py`) using `serial_port` from `config.json`.
  2. Capture images (`camera_capture.py`) into `Aruco/` with a simple UI (SPACE to save).
  3. Manage ArUco folder: ask which image to keep as `aruco_calibration.jpg`; deletes the rest.
  4. Estimate ArUco pose (`aruco_pose.py`) and write `Aruco/aruco_reference.json`.
  5. Launch the live `camera_stream.py`.

  ### Key bindings and UI
  In the live stream:
  - Space — trigger motion for current confident detections.
  - `u` — toggle AUTO mode (auto-trigger based on detections).
  - `p` — toggle small plot overlay in the OpenCV window.
  - `v` — toggle a separate Matplotlib workspace viz window.
  - `q` — quit.

  The Matplotlib viz shows:
  - Top-down XY axes (robot frame mapping), radial central lines, chosen-tip blink, pen radial-angle arc + label.
  - STANDARD vs COMPLEX motion previews (waypoint markers only), synchronized with the robot logic.

  ---

  ## Usage notes and interoperability
  - Only detections with confidence ≥ 0.7 are eligible for motion.
  - Coordinate transform chain: pixel → camera ray → intersect ArUco plane → tag frame → robot frame.
  - Safety: `validate_robot_coords` rejects commands outside a radial/Z envelope.
  - Session logs: artifacts saved under `ResearchDataset/<session>/`.
  - Models: large weights or `yolo_init_model/` can bloat the repo; consider adding them to `.gitignore` if not needed in version control.

  ---

  ## License
  MIT. If the `LICENSE` file is not present, consider adding one from https://choosealicense.com/licenses/mit/.

  ---

  ## Citation
  If you use this project in your work, please cite:

  Plaintext:
  > Anirudh. 4DoF Vision Robotic Pen Sorting (v1). GitHub repository: https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting. Accessed 2025-09-09.

  BibTeX:
  ```bibtex
  @misc{anirudh_pen_sorting_2025,
    author       = {Anirudh},
    title        = {4DoF Vision Robotic Pen Sorting},
    year         = {2025},
    howpublished = {GitHub},
    url          = {https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting},
    note         = {Accessed 2025-09-09}
  }
  ```

  ---

  ## Acknowledgments
  - Thanks to mentors and the Polygence program for guidance and feedback.
  - Open-source libraries: OpenCV, NumPy, Matplotlib, Ultralytics YOLO.
  - Please add any funding sources or institutional support here.

  ---

  ## Contributing
  Contributions are welcome via pull requests.
  - Open an issue to discuss substantial changes.
  - Keep changes focused and include brief rationale in the PR description.
  - If adding models or large files, prefer links or releases over committing to the main repo.

  ---

  ## Tests
  Run the sanity tests where available:
  ```bash
  python test_pixel_conversion.py
  python test_coordinates.py
  ```

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
