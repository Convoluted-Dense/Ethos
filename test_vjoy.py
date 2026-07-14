import time
import sys

def main():
    try:
        import pyvjoy
        joystick = pyvjoy.VJoyDevice(1)
        print("vJoy Device 1 initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize vJoy. {e}")
        print("Please make sure vJoy driver is installed in Windows and a device is enabled.")
        sys.exit(1)

    print("\n--- vJoy Axis X (Steering) Test ---")
    
    # 5-second countdown
    for i in range(5, 0, -1):
        print(f"Starting test in {i} seconds... (Switch to BeamNG now!)", end="\r")
        time.sleep(1.0)
    print("\n")

    # Axis Constants
    VJOY_AXIS_MIN = 0x1
    VJOY_AXIS_MAX = 0x8000
    VJOY_AXIS_MID = (VJOY_AXIS_MAX + VJOY_AXIS_MIN) // 2

    # Reset to center
    print("Centering X axis...")
    joystick.data.wAxisX = VJOY_AXIS_MID
    joystick.update()
    time.sleep(1.0)

    # Sweep from Mid to Min (Left)
    print("Sweeping Left...")
    steps = 50
    for i in range(steps):
        val = VJOY_AXIS_MID - int((VJOY_AXIS_MID - VJOY_AXIS_MIN) * (i / steps))
        joystick.data.wAxisX = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    # Sweep from Min to Max (Right)
    print("Sweeping Right...")
    for i in range(steps):
        val = VJOY_AXIS_MIN + int((VJOY_AXIS_MAX - VJOY_AXIS_MIN) * (i / steps))
        joystick.data.wAxisX = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    # Sweep back to Mid (Center)
    print("Sweeping back to Center...")
    for i in range(steps):
        val = VJOY_AXIS_MAX - int((VJOY_AXIS_MAX - VJOY_AXIS_MID) * (i / steps))
        joystick.data.wAxisX = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    print("Resetting to Center...")
    joystick.data.wAxisX = VJOY_AXIS_MID
    joystick.update()
    print("Done.")

if __name__ == "__main__":
    main()
