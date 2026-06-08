#!/usr/bin/env python3

import subprocess
import sys
import os
import json
import glob
import signal
import time
import urllib.request

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
    """Step 2: pick the camera, then capture ArUco photos (camera_pick_and_capture.py)."""
    print("\nStep 2 — pick the camera + capture the ArUco tag")
    print("  Pick: Y = use this camera, N = next.  Then: SPACE = photo, Q = done.")
    try:
        subprocess.run([
            sys.executable,
            "camera_pick_and_capture.py",
            "Aruco"
        ], check=True)
    except KeyboardInterrupt:
        print("\nCamera step interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Camera step failed: {e}")

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

def _http_json(url, post=False, timeout=4):
    data = b"{}" if post else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method=("POST" if post else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def reposition_aruco_prompt():
    """Step 1: open the control UI in a Chrome app-mode window (served by
    arm_web_server.py, which holds the serial link). The user picks Reposition or
    Keep current; on Confirm/Keep the server records the choice and writes
    robot_tag_xyz into config.json. We then home the robot out of the way so it
    doesn't block the ArUco scan."""
    config = load_config()
    serial_port = config['serial_port']
    base = "http://localhost:8765"
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    profile = "/tmp/roarm_ui_profile"

    # Robot must be connected to reposition (the arm has to move). If the serial
    # port isn't there, skip cleanly and keep the existing tag.
    if not os.path.exists(serial_port):
        print(f"Robot not connected at {serial_port} — skipping reposition, keeping tag {config.get('robot_tag_xyz')}.")
        print("(Plug in the robot's USB + 12V power if you want to reposition the ArUco tag.)")
        return

    # 0) kill any stale server holding port 8765 (else we'd read a cached choice)
    subprocess.run(["pkill", "-f", "arm_web_server.py"], capture_output=True)
    time.sleep(0.6)

    # 1) start the web + serial server
    server = subprocess.Popen([sys.executable, "arm_web_server.py", serial_port,
                               "--http", "8765", "--no-open"])
    for _ in range(40):                          # wait up to ~10s for it to come up
        try:
            _http_json(f"{base}/api/config"); break
        except Exception:
            time.sleep(0.25)
    else:
        print("Control server didn't start; keeping existing tag position.")
        server.terminate(); return

    # 2) open the UI as a Chrome app-mode window (isolated profile so we can close it)
    chrome_proc = None
    if os.path.exists(chrome):
        chrome_proc = subprocess.Popen(
            [chrome, f"--app={base}/?flow=reposition", f"--user-data-dir={profile}",
             "--no-first-run", "--no-default-browser-check", "--window-size=1080,980"])
    else:
        import webbrowser; webbrowser.open(f"{base}/?flow=reposition")
    print("Reposition window opened — choose Reposition or Keep current…")

    # 3) wait for the user's choice (Confirm or Keep)
    action = None
    try:
        while action is None:
            time.sleep(0.4)
            try:
                action = _http_json(f"{base}/api/status").get("action")
            except Exception:
                pass
    except KeyboardInterrupt:
        action = "keep"
    print(f"Reposition choice: {action}")

    # 4) home the robot out of the way for the ArUco scan
    try:
        _http_json(f"{base}/api/home", post=True)
        print("Homing robot for the ArUco scan…")
        time.sleep(2.0)
    except Exception as e:
        print(f"Home failed: {e}")

    # 5) close the UI + stop the server (frees the serial port for later steps)
    if chrome_proc:
        try:
            chrome_proc.terminate()
        except Exception:
            pass
    server.terminate()
    try:
        server.wait(timeout=5)
    except Exception:
        pass
    time.sleep(1.0)   # let the serial port fully release
    print(f"ArUco tag position is now: {load_config().get('robot_tag_xyz')}")


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
    print("1. ArUco tag positioning (reposition via joystick, optional)")
    print("2. Camera capture")
    print("3. ArUco folder management")
    print("4. ArUco pose estimation")
    print("5. Camera stream with robot control")
    print("\nPress Ctrl+C to interrupt any stage\n")

    # Step 1: Prompt to reposition the ArUco tag (joystick writes config.json)
    print("STEP 1: ArUco Tag Positioning")
    reposition_aruco_prompt()
    
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
