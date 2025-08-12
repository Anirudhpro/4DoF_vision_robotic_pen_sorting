import serial
import argparse
import threading
import time
import json

ser = None

def read_serial():
    while True:
        try:
            data = ser.readline().decode('utf-8')
            if data:
                print(f"Received: {data}", end='')
        except Exception as e:
            print(f"Read error: {e}")
            break

def send_json(cmd):
    ser.write(json.dumps(cmd).encode() + b'\n')
    print(f"Sent: {json.dumps(cmd)}")

def main():
    global ser
    parser = argparse.ArgumentParser(description='Serial JSON Communication')
    parser.add_argument('port', type=str, help='Serial port name (e.g., COM1 or /dev/ttyUSB0)')
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, baudrate=115200, timeout=1, dsrdtr=None)
        ser.setRTS(False)
        ser.setDTR(False)
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    # Start background read thread
    serial_recv_thread = threading.Thread(target=read_serial, daemon=True)
    serial_recv_thread.start()

    time.sleep(1)  # let connection settle

    try:
        while True:
            command = input("Enter x y z t (or full JSON): ")

            if command.startswith('{'):
                # raw JSON passthrough
                ser.write(command.encode() + b'\n')
                continue

            parts = command.strip().split()
            if len(parts) != 4:
                print("Enter 4 values: x y z t")
                continue

            x, y, z, t = map(float, parts)

            # Activate LED
            send_json({"T": 114, "led": 255})

            # Send movement command
            send_json({
                "T": 1041,
                "x": x,
                "y": y,
                "z": z,
                "t": t
            })

            # Deactivate LED
            send_json({"T": 114, "led": 0})

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()