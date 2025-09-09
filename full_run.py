#!/usr/bin/env python3

import subprocess
import sys
import os
import json
import glob
import signal
import time

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def run_serial_simple_ctrl():
    """Run serial_simple_ctrl.py with the configured serial port"""
    config = load_config()
    robot_tag_xyz = config['robot_tag_xyz']
    serial_port = config['serial_port']
    
    print(f"Starting serial_simple_ctrl.py with port: {serial_port}")
    # Print the motion commands: first uses robot_tag_xyz for x,y,z, second is the static return command
    first_cmd = {"T": 1041, "x": int(robot_tag_xyz[0]), "y": int(robot_tag_xyz[1]), "z": int(robot_tag_xyz[2]), "t": 2.95}
    second_cmd = {"T": 1041, "x": 120, "y": 0, "z": -20, "t": 2.8}
    print(json.dumps(first_cmd))
    print(json.dumps(second_cmd))
    
    try:
        subprocess.run([
            sys.executable, 
            "RoArm/serial_simple_ctrl.py", 
            serial_port
        ], check=True)
    except KeyboardInterrupt:
        print("\nSerial control interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Serial control failed: {e}")

def run_camera_capture():
    """Run camera_capture.py to save images to Aruco folder"""
    print("\nStarting camera_capture.py - saving to Aruco folder")
    print("Press SPACE to capture images, 'q' to quit when done")
    
    try:
        subprocess.run([
            sys.executable,
            "camera_capture.py",
            "Aruco"
        ], check=True)
    except KeyboardInterrupt:
        print("\nCamera capture interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Camera capture failed: {e}")

def manage_aruco_folder():
    """Manage ArUco folder - get user input for image number and rename files"""
    print("\nManaging ArUco folder...")
    
    # Get image number from user
    while True:
        try:
            image_num = input("Enter the image number to use as aruco_calibration.jpg: ")
            image_num = int(image_num)
            break
        except ValueError:
            print("Please enter a valid number")
    
    aruco_folder = "Aruco"
    target_file = os.path.join(aruco_folder, f"{image_num}.jpg")
    calibration_file = os.path.join(aruco_folder, "aruco_calibration.jpg")
    
    # Check if the target file exists
    if not os.path.exists(target_file):
        print(f"Error: {target_file} does not exist!")
        return False
    
    # Delete existing aruco_calibration.jpg if it exists
    if os.path.exists(calibration_file):
        os.remove(calibration_file)
        print(f"Deleted existing {calibration_file}")
    
    # Rename the selected image to aruco_calibration.jpg
    os.rename(target_file, calibration_file)
    print(f"Renamed {target_file} to {calibration_file}")
    
    # Delete all other .jpg files in the folder
    jpg_files = glob.glob(os.path.join(aruco_folder, "*.jpg"))
    for jpg_file in jpg_files:
        if os.path.basename(jpg_file) != "aruco_calibration.jpg":
            os.remove(jpg_file)
            print(f"Deleted {jpg_file}")
    
    return True

def run_aruco_pose():
    """Run aruco_pose.py to generate ArUco reference"""
    print("\nStarting aruco_pose.py...")
    
    try:
        subprocess.run([
            sys.executable,
            "aruco_pose.py"
        ], check=True)
        print("ArUco pose estimation completed")
    except subprocess.CalledProcessError as e:
        print(f"ArUco pose estimation failed: {e}")
        return False
    return True

def run_camera_stream():
    """Run camera_stream.py with the configured serial port"""
    config = load_config()
    serial_port = config['serial_port']
    
    print(f"\nStarting camera_stream.py with port: {serial_port}")
    
    try:
        subprocess.run([
            sys.executable,
            "camera_stream.py",
            serial_port,
            "ResearchDataset"
        ], check=True)
    except KeyboardInterrupt:
        print("\nCamera stream interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Camera stream failed: {e}")

def main():
    """Main orchestration function
    Usage:
      python full_run.py          -> full pipeline
      python full_run.py short    -> skip to camera_stream only
    """
    mode = 'full'
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'short':
        mode = 'short'

    if mode == 'short':
        print("=== 4DoF Vision Robotic Pen Sorting - SHORT RUN ===")
        print("Skipping directly to camera_stream.py (steps 1-4 bypassed)\n")
        run_camera_stream()
        print("\n=== Short run completed ===")
        return

    print("=== 4DoF Vision Robotic Pen Sorting - Full Run ===")
    print("This script will run the complete pipeline:")
    print("1. Serial control")
    print("2. Camera capture") 
    print("3. ArUco folder management")
    print("4. ArUco pose estimation")
    print("5. Camera stream with robot control")
    print("\nPress Ctrl+C to interrupt any stage\n")
    
    # Step 1: Run serial_simple_ctrl.py
    print("STEP 1: Serial Control")
    run_serial_simple_ctrl()
    
    # Step 2: Run camera_capture.py
    print("\nSTEP 2: Camera Capture")
    run_camera_capture()
    
    # Step 3: Manage ArUco folder
    print("\nSTEP 3: ArUco Folder Management")
    if not manage_aruco_folder():
        print("ArUco folder management failed, exiting")
        return
    
    # Step 4: Run aruco_pose.py
    print("\nSTEP 4: ArUco Pose Estimation")
    if not run_aruco_pose():
        print("ArUco pose estimation failed, exiting")
        return
    
    # Step 5: Run camera_stream.py
    print("\nSTEP 5: Camera Stream with Robot Control")
    run_camera_stream()
    
    print("\n=== Full run completed ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFull run interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
