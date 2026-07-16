import time
import sys
import ctypes

try:
    import pyvjoy
except ImportError:
    print("Error: pyvjoy is not installed.")
    sys.exit(1)

# Exact constants from test_cnn_v3.py
VJOY_SPEED_MIN = -32768
VJOY_SPEED_MAX = 32767
VJOY_SPEED_MID = 0

def map_to_vjoy(output: float) -> int:
    """Exactly the same math as test_cnn_v3.py"""
    if output >= 0:
        y_val = VJOY_SPEED_MID + int(output * (VJOY_SPEED_MAX - VJOY_SPEED_MID))
    else:
        y_val = VJOY_SPEED_MID - int(output * (VJOY_SPEED_MIN - VJOY_SPEED_MID))
        
    y_val = max(VJOY_SPEED_MIN, min(VJOY_SPEED_MAX, y_val))
    return y_val

def main():
    try:
        joystick = pyvjoy.VJoyDevice(1)
        joystick.data.wAxisY = VJOY_SPEED_MID
        joystick.update()
    except Exception as e:
        print(f"Failed to open vJoy device: {e}")
        return

    print("\n=== INTERACTIVE vJoy Cruise Control Mapping Test ===")
    print("You can tab into BeamNG! The keys will still work.")
    print("Controls:")
    print("  [SPACE] : Increase Throttle (+)")
    print("  [  P  ] : Decrease Throttle / Increase Brake (-)")
    print("  [ ESC ] : Quit Test\n")
    
    output = 0.0
    
    # Virtual Key Codes
    VK_SPACE = 0x20
    VK_P = 0x50
    VK_ESCAPE = 0x1B
    
    while True:
        # Check keys asynchronously (works globally even if window is unfocused)
        if ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            break
            
        if ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000:
            output += 0.01  # Ramp up slowly
        elif ctypes.windll.user32.GetAsyncKeyState(VK_P) & 0x8000:
            output -= 0.01  # Ramp down slowly
            
        # Clamp between -1.0 (Full Brake) and 1.0 (Full Throttle)
        output = max(-1.0, min(1.0, output))
        
        y_val = map_to_vjoy(output)
        joystick.data.wAxisY = y_val
        joystick.update()
        
        # Display
        if output >= 0:
            print(f"\rOutput: {output:+.2f} (Throttle: {int(output*100):3d}%) -> vJoy Y: {y_val:6d}    ", end="", flush=True)
        else:
            print(f"\rOutput: {output:+.2f} (Brake:    {int(abs(output)*100):3d}%) -> vJoy Y: {y_val:6d}    ", end="", flush=True)
            
        time.sleep(0.016) # ~60 Hz loop

    print("\n\nDone! Centering Y axis.")
    joystick.data.wAxisY = VJOY_SPEED_MID
    joystick.update()

if __name__ == "__main__":
    main()
