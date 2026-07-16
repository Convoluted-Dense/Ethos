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

    print("\n--- vJoy Axis Y (Throttle/Brake) Test ---")
    
    # 5-second countdown
    for i in range(5, 0, -1):
        print(f"Starting test in {i} seconds... (Switch to BeamNG now!)", end="\r")
        time.sleep(1.0)
    print("\n")

    # Axis Constants (Testing Negative Range)
    VJOY_AXIS_MIN = -32768
    VJOY_AXIS_MAX = 32767
    VJOY_AXIS_MID = 0

    # Reset to center
    print("Centering Y axis...")
    joystick.data.wAxisY = VJOY_AXIS_MID
    joystick.update()
    time.sleep(1.0)

    # Sweep from Mid to Min (e.g. Full Brake or Throttle depending on mapping)
    print("Sweeping Y Axis to Min...")
    steps = 50
    for i in range(steps):
        val = VJOY_AXIS_MID - int((VJOY_AXIS_MID - VJOY_AXIS_MIN) * (i / steps))
        joystick.data.wAxisY = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    # Sweep from Min to Max
    print("Sweeping Y Axis to Max...")
    for i in range(steps):
        val = VJOY_AXIS_MIN + int((VJOY_AXIS_MAX - VJOY_AXIS_MIN) * (i / steps))
        joystick.data.wAxisY = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    # Sweep back to Mid (Center)
    print("Sweeping Y Axis back to Center...")
    for i in range(steps):
        val = VJOY_AXIS_MAX - int((VJOY_AXIS_MAX - VJOY_AXIS_MID) * (i / steps))
        joystick.data.wAxisY = val
        joystick.update()
        time.sleep(0.02)
    time.sleep(1.0)

    print("Resetting Y Axis to Center...")
    joystick.data.wAxisY = VJOY_AXIS_MID
    joystick.update()
    print("Done.")

if __name__ == "__main__":
    main()
