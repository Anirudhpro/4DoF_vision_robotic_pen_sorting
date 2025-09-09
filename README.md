<div align="center">

# 4DoF Vision-Guided Robotic Sorting of Cluttered Objects

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#3-installation-and-setup)
[![Last commit](https://img.shields.io/github/last-commit/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/commits/main)
[![Open Issues](https://img.shields.io/github/issues/Anirudhpro/4DoF_vision_robotic_pen_sorting)](https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting/issues)

</div>

---

## 1. Project Description

This repository contains the complete software pipeline for a computer vision-guided robotic arm system designed to sort pens from a cluttered environment. The project integrates real-time object detection, camera calibration, and 3D coordinate transformation to enable a 4-DOF robotic arm to intelligently perceive and interact with its workspace.

The core research objective was to investigate the feasibility and accuracy of using Oriented Bounding Box (OBB) detection models for robotic grasping tasks in unstructured settings. This involved overcoming challenges in camera-to-robot coordinate mapping, ensuring robust performance despite visual noise, and developing motion primitives suitable for picking elongated objects like pens.

**Key Technologies & Concepts Explored:**
- **Perception**: Ultralytics YOLOv8-OBB for detecting rotated bounding boxes of pens.
- **Calibration**: OpenCV for both intrinsic camera calibration (via checkerboards) and extrinsic world-frame alignment (via ArUco markers).
- **3D Geometry**: Transformation of 2D pixel coordinates into 3D robot-space coordinates for precise grasping.
- **Robotics & Control**: Serial communication for sending JSON-based motion commands to a 4-DOF robotic arm.
- **Software Engineering**: A modular Python pipeline with clear separation of concerns for calibration, detection, and control, including an interactive visualization dashboard.

*(Note: This project was developed as part of a Polygence research program. The accompanying PDF report provides a deeper dive into the background, methodology, and results.)*

---

## 2. Table of Contents
- [1. Project Description](#1-project-description)
- [2. Table of Contents](#2-table-of-contents)
- [3. Installation and Setup](#3-installation-and-setup)
- [4. How to Use the Project: A Step-by-Step Guide](#4-how-to-use-the-project-a-step-by-step-guide)
  - [Step 1: Intrinsic Camera Calibration](#step-1-intrinsic-camera-calibration)
  - [Step 2: World Frame Calibration (ArUco)](#step-2-world-frame-calibration-aruco)
  - [Step 3: Running the Live System](#step-3-running-the-live-system)
- [5. File and Data Overview](#5-file-and-data-overview)
- [6. Tests](#6-tests)
- [7. How to Contribute](#7-how-to-contribute)
- [8. Credits and Acknowledgments](#8-credits-and-acknowledgments)
- [9. License](#9-license)
- [10. Citation](#10-citation)

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10+
- A webcam compatible with OpenCV.
- A 4-DOF robotic arm controller connected via a serial port.
- Physical checkerboard and ArUco marker for calibration.

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting.git
    cd 4DoF_vision_robotic_pen_sorting
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Create a local configuration file:**
    The `full_run.py` script can use a `config.json` file for convenience. Create it in the root directory:
    ```json
    {
      "serial_port": "/dev/tty.usbserial-XXXX",
      "robot_tag_xyz": [120, 0, -20]
    }
    ```
    Replace `/dev/tty.usbserial-XXXX` with your robot's actual serial port.

---

## 4. How to Use the Project: A Step-by-Step Guide

This project requires a two-part calibration process before the main application can be run.

### Step 1: Intrinsic Camera Calibration
This step calculates the camera's internal parameters (focal length, optical center) and distortion coefficients.

**Script:** `camera_calibrate.py`

**How it Works:**
- It requires a set of images of a checkerboard pattern taken from various angles and distances.
- The script expects a `9x6` checkerboard with `22mm` squares by default (these values can be changed in the script).
- It processes images from the `CalibrationPictures/` directory.
- The output is `calib_data.npz`, a file containing the camera matrix and distortion coefficients. It also saves annotated images to `CalibratedLinePictures/` for verification.

**Instructions:**
1.  Place at least 15-20 sharp, well-lit photos of your checkerboard in the `CalibrationPictures/` folder.
2.  Run the script:
    ```bash
    python camera_calibrate.py
    ```
3.  Verify that `calib_data.npz` has been created and check the images in `CalibratedLinePictures/` to ensure the corners were detected correctly.

### Step 2: World Frame Calibration (ArUco)
This step establishes the spatial relationship between the camera and the robot's workspace using an ArUco marker as a fixed reference point.

**Script:** `aruco_pose.py`

**How it Works:**
- It loads the intrinsic parameters from `calib_data.npz`.
- It captures a single frame from the webcam to detect a specific ArUco marker (ID `4x4_50`).
- It calculates the marker's 3D position and orientation (pose) relative to the camera.
- This pose information is saved to `Aruco/aruco_reference.json`, which allows the main application to convert any point from the camera's view into the robot's coordinate system.

**Instructions:**
1.  Place the ArUco marker flat in the robot's workspace, in a known, fixed position.
2.  Run the script:
    ```bash
    python aruco_pose.py
    ```
3.  The script will display the detected marker with its axes drawn on it and save the output to `aruco_tag_detection.jpg`. Confirm that `Aruco/aruco_reference.json` has been created.

### Step 3: Running the Live System
With calibration complete, you can now run the main sorting application.

#### Option A: Direct Execution (Recommended for Development)
Run the main script directly, providing the serial port as an argument.

**Script:** `camera_stream.py`

**Instructions:**
```bash
python camera_stream.py <your_serial_port>
# Example:
python camera_stream.py /dev/tty.usbserial-123
```

#### Option B: Orchestrated Execution
Use the `full_run.py` script, which automates the entire pipeline, including calibration steps.

**Script:** `full_run.py`

**How it Works:**
- **Full Mode (`python full_run.py`):**
  1.  `serial_simple_ctrl.py`: Runs a quick sanity check of the serial connection.
  2.  `camera_capture.py`: Interactively captures images for ArUco calibration.
  3.  `manage_aruco_folder()`: Prompts you to select the best image and cleans up the folder.
  4.  `aruco_pose.py`: Runs the ArUco pose estimation.
  5.  `camera_stream.py`: Launches the main application.
- **Short Mode (`python full_run.py short`):**
  - Skips all calibration steps and jumps directly to launching `camera_stream.py`. Use this when your calibration is already up-to-date.

**Interactive Controls (in `camera_stream`):**
- **`Spacebar`**: Manually trigger the robot to pick up a detected pen.
- **`u`**: Toggle "Auto Mode," where the robot will automatically attempt to pick pens as they are detected.
- **`v`**: Toggle the detailed Matplotlib visualization window, which shows a top-down view of the workspace.
- **`p`**: Toggle a mini-plot overlay on the main camera feed.
- **`q`**: Quit the application.

---

## 5. File and Data Overview

-   **Core Scripts**:
    -   `camera_stream.py`: The main application logic.
    -   `camera_calibrate.py`: For intrinsic calibration.
    -   `aruco_pose.py`: For extrinsic calibration.
    -   `full_run.py`: Orchestrates the entire process.
-   **Calibration Data (Generated)**:
    -   `calib_data.npz`: Stores camera matrix and distortion coefficients.
    -   `Aruco/aruco_reference.json`: Stores the pose of the world-frame marker.
-   **Input Data**:
    -   `CalibrationPictures/`: Contains checkerboard images for `camera_calibrate.py`.
-   **Model Files**:
    -   `yolov8n-obb.pt`: The pre-trained Oriented Bounding Box model from Ultralytics.
-   **Output / Logs**:
    -   `ResearchDataset/`: Default directory where session artifacts (logs, images) are saved.
    -   `CalibratedLinePictures/`: Annotated checkerboard images for verification.
    -   `aruco_tag_detection.jpg`: A snapshot of the ArUco marker detection.

---

## 6. Tests
This project includes several scripts to test and validate the coordinate transformations. To ensure your calibration is accurate, run these tests:

```bash
# Test the pixel-to-robot coordinate conversion
python test_pixel_conversion.py

# Test specific hardcoded coordinates
python test_coordinates.py
```
These scripts will print out the results of the transformations, helping you debug any issues with your setup.

---

## 7. How to Contribute
Contributions are welcome! If you have an idea for an improvement or have found a bug, please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature or bugfix (`git checkout -b feature/my-new-feature`).
3.  Make your changes and commit them with a clear message.
4.  Push your branch to your fork (`git push origin feature/my-new-feature`).
5.  Open a Pull Request and describe the changes you've made.

Please adhere to the existing code style and ensure any new features are documented.

---

## 8. Credits and Acknowledgments
-   **Author**: Anirudh ([@Anirudhpro](https://github.com/Anirudhpro))
-   **Acknowledgments**: This project was developed under the mentorship of the Polygence program. Special thanks to the open-source community for providing the tools that made this work possible, including OpenCV, NumPy, and Ultralytics.

---

## 9. License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details. You are free to use, modify, and distribute this code for any purpose, provided you include the original copyright and license notice.

For help choosing a license for your own projects, visit [choosealicense.com](https://choosealicense.com/).

---

## 10. Citation
If you use this project in your research or work, please cite it as follows:

```
Anirudh. (2025). 4DoF Vision-Guided Robotic Sorting of Cluttered Objects (Version 1.0.0) [Computer software]. https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting
```

**BibTeX:**
```bibtex
@software{Anirudh_4DoF_Vision-Guided_Robotic_2025,
  author = {Anirudh},
  title = {{4DoF Vision-Guided Robotic Sorting of Cluttered Objects}},
  version = {1.0.0},
  year = {2025},
  url = {https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting}
}
```
