"""
sanjyit.py
==========
Tool to compare camera angles from real-life videos (placed in 'w_vids' folder)
with the active BeamNG.drive game window.

Controls:
- [Spacebar] : Toggle Play/Pause
- [A] or [Left Arrow] : Rewind 2 seconds
- [D] or [Right Arrow] : Fast Forward 2 seconds
- [R] : Restart video to the beginning
- [Q] or [ESC] : Quit the tool
"""

import os
import sys
import time
import ctypes
import cv2
import numpy as np
from ctypes import wintypes

# ---------------------------------------------------------------------------
# Window capture helpers (GDI32 PrintWindow)
# ---------------------------------------------------------------------------
def find_beamng_window():
    user32 = ctypes.windll.user32
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []

    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if "beamng.drive" in buf.value.lower():
                    found.append(hwnd)
                    return False
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    if found:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        return found[0]
    return None

def capture_printwindow(hwnd):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    hwndDC = user32.GetWindowDC(hwnd)
    mfcDC = gdi32.CreateCompatibleDC(hwndDC)
    saveBitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    gdi32.SelectObject(mfcDC, saveBitmap)
    user32.PrintWindow(hwnd, mfcDC, 2)  # PW_RENDERFULLCONTENT

    bmi = bytearray(40)
    bmi[0:4] = (40).to_bytes(4, "little")
    bmi[4:8] = w.to_bytes(4, "little", signed=True)
    bmi[8:12] = (-h).to_bytes(4, "little", signed=True)
    bmi[12:14] = (1).to_bytes(2, "little")
    bmi[14:16] = (32).to_bytes(2, "little")

    buf = bytearray(w * h * 4)
    gdi32.GetDIBits(mfcDC, saveBitmap, 0, h,
                    ctypes.byref(ctypes.c_char.from_buffer(buf)),
                    ctypes.byref(ctypes.c_char.from_buffer(bmi)), 0)

    gdi32.DeleteObject(saveBitmap)
    gdi32.DeleteDC(mfcDC)
    user32.ReleaseDC(hwnd, hwndDC)

    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    return img[:, :, :3].copy()  # BGRA -> BGR

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    folder_name = "w_vids"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder '{folder_name}'. Please place your reference videos in it.")
    
    # Supported video formats
    extensions = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
    video_files = [f for f in os.listdir(folder_name) if f.lower().endswith(extensions)]

    if not video_files:
        print(f"\n[Error] No video files found in the '{folder_name}' folder.")
        print(f"Please add some videos (e.g. .mp4, .avi) to 'C:\\Golden Buggy\\{folder_name}' and try again.\n")
        sys.exit(1)

    print("\n==============================================")
    print("Available Videos:")
    print("==============================================")
    for idx, f in enumerate(video_files):
        print(f"  {idx + 1}: {f}")
    print("==============================================")

    # Prompt user to choose video
    choice = None
    while True:
        try:
            val = input(f"Choose a video (1-{len(video_files)}): ").strip()
            choice = int(val) - 1
            if 0 <= choice < len(video_files):
                break
            else:
                print(f"Out of range. Enter a number between 1 and {len(video_files)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    selected_video = os.path.join(folder_name, video_files[choice])
    print(f"\n[OK] Selected: {selected_video}")
    print("Searching for BeamNG.drive window...")
    
    hwnd = find_beamng_window()
    if hwnd is None:
        print("[Warn] BeamNG.drive window not found. Blended view will show a placeholder/warning until BeamNG is started.")
    else:
        print(f"[OK] Bound to BeamNG HWND: {hwnd}")

    cap = cv2.VideoCapture(selected_video)
    if not cap.isOpened():
        print(f"[Error] Failed to open video file {selected_video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_delay_ms = int(1000 / fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_video = "Video Player"
    window_overlay = "Comparison Overlay (50-50 Blend)"
    window_beamng = "BeamNG Window Capture"
    cv2.namedWindow(window_video, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_overlay, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_beamng, cv2.WINDOW_NORMAL)

    # Resize to something reasonable for desktop view
    cv2.resizeWindow(window_video, 800, 450)
    cv2.resizeWindow(window_overlay, 800, 450)
    cv2.resizeWindow(window_beamng, 800, 450)

    paused = False
    current_frame_idx = 0

    print("\n----------------------------------------------")
    print("Controls:")
    print("  [Spacebar] : Play / Pause")
    print("  [A] or [Left Arrow] : Rewind 2 seconds")
    print("  [D] or [Right Arrow] : Fast Forward 2 seconds")
    print("  [R] : Restart from beginning")
    print("  [Q] or [ESC] : Quit")
    print("----------------------------------------------\n")

    while True:
        # If not paused, read next frame from video
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning when video ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_idx = 0
                continue
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # If paused, keep reading the same frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

        # Get BeamNG screen capture
        if hwnd is None:
            hwnd = find_beamng_window()  # Try to find it if it was missing earlier

        bng_frame = None
        if hwnd is not None:
            bng_frame = capture_printwindow(hwnd)

        # Build overlay frame
        h, w, c = frame.shape
        if bng_frame is not None:
            bng_resized = cv2.resize(bng_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            overlay_frame = cv2.addWeighted(frame, 0.5, bng_resized, 0.5, 0)
        else:
            # Draw warning overlay if BeamNG isn't running
            overlay_frame = frame.copy()
            cv2.rectangle(overlay_frame, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)
            cv2.putText(overlay_frame, "BeamNG Window NOT Found", (50, h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw frame number HUD on windows
        hud_frame = frame.copy()
        cv2.putText(hud_frame, f"Frame: {current_frame_idx}/{total_frames} | {'PAUSED' if paused else 'PLAYING'}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(overlay_frame, f"Frame: {current_frame_idx}/{total_frames} | {'PAUSED' if paused else 'PLAYING'}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Build raw BeamNG window frame for display
        if bng_frame is not None:
            bng_display = cv2.resize(bng_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bng_display = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(bng_display, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)
            cv2.putText(bng_display, "BeamNG Window NOT Found", (50, h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
        cv2.putText(bng_display, "BeamNG Raw Capture", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_video, hud_frame)
        cv2.imshow(window_overlay, overlay_frame)
        cv2.imshow(window_beamng, bng_display)

        # Poll keys
        # waitKey(delay) returns 32-bit integer. Mask with 0xFF for standard ASCII keys,
        # or compare directly for specialized keycodes (like Windows Arrow keys).
        key_raw = cv2.waitKey(frame_delay_ms if not paused else 100)
        key = key_raw & 0xFF

        # Play/Pause
        if key == 32:  # Spacebar
            paused = not paused
            print(f"Playback: {'PAUSED' if paused else 'PLAYING'}")

        # Quit
        elif key in (ord('q'), 27):  # ESC or Q
            print("Exiting comparison viewer.")
            break

        # Restart
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame_idx = 0
            print("Restarted video.")

        # Seek Backward / Rewind (2 seconds)
        elif key == ord('a') or key == ord('j') or key_raw == 2424832:
            skip_frames = int(fps * 2.0)
            target = max(0, current_frame_idx - skip_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current_frame_idx = target
            print(f"Rewound to frame {target}")

        # Seek Forward / Fast Forward (2 seconds)
        elif key == ord('d') or key == ord('l') or key_raw == 2555904:
            skip_frames = int(fps * 2.0)
            target = min(total_frames - 1, current_frame_idx + skip_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current_frame_idx = target
            print(f"Skipped to frame {target}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
