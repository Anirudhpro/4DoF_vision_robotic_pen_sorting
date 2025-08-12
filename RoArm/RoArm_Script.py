from roarm_sdk.roarm import roarm
import time

# Replace with your actual serial port
SERIAL_PORT = "/dev/tty.SLAB_USBtoUART"  # or COM3, /dev/ttyUSB0, etc.

# Create an instance of the robot arm
arm = roarm(
    roarm_type="roarm_m2",     # Or "roarm_m2_s" — either should work for the M2-S
    port=SERIAL_PORT,
    baudrate=115200            # Default baud rate
)

# Test: Move all joints
arm.set_servo_joint_angles([0.1, -0.2, 0.3, 0.0])
time.sleep(2)

# Test: Open/close gripper
arm.open_gripper()
time.sleep(1)
arm.close_gripper()

# Test: Get joint positions
angles = arm.get_joint_states()
print("Current joint angles:", angles)