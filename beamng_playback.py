"""
beamng_playback.py
==================
Plays back a dataset folder recorded by beamng_collect.py.
Renders each saved frame with the same telemetry HUD as the live collector.

Usage
-----
    python beamng_playback.py                        # plays ./dataset at original FPS
    python beamng_playback.py --dataset mydata       # custom folder
    python beamng_playback.py --speed 0.5            # half speed
    python beamng_playback.py --speed 2.0            # double speed
    python beamng_playback.py --fps 30               # force specific display FPS

Controls
--------
    SPACE       pause / resume
    RIGHT (→)   step one frame forward  (while paused)
    LEFT  (←)   step one frame backward (while paused)
    q / ESC     quit
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HUD rendering  (identical style to beamng_collect.py)
# ---------------------------------------------------------------------------
_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_CLR_HUD = (0, 255, 180)
_CLR_TIM = (180, 180, 255)
_CLR_SHD = (0, 0, 0)
_CLR_NOD = (0, 120, 255)
_FS      = 0.60
_FT      = 2


def _put(img, text, x, y, color=_CLR_HUD):
    cv2.putText(img, text, (x+1, y+1), _FONT, _FS, _CLR_SHD, _FT+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), _FONT, _FS, color,     _FT,   cv2.LINE_AA)


def draw_hud(img: np.ndarray, telem: dict | None,
             frame_idx: int, total: int,
             elapsed_s: float, speed_mult: float,
             paused: bool, steering: float = 0.0,
             map_img: np.ndarray | None = None,
             offset: float = 0.0,
             steering_combined: float = 0.0) -> np.ndarray:
    vis = img.copy()
    ph, pw = 335, 500

    # semi-transparent dark panel
    overlay = vis[:ph, :pw].copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.55, vis[:ph, :pw], 0.45, 0, vis[:ph, :pw])

    y = 25
    # Frame counter + playback state
    state = "PAUSED" if paused else f"PLAY x{speed_mult:.2g}"
    _put(vis, f"Frame {frame_idx+1:>5} / {total}    [{state}]", 10, y)

    # Timeline bar
    progress = (frame_idx + 1) / max(total, 1)
    bar_x0, bar_y0 = 10, y + 14
    bar_w = pw - 20
    cv2.rectangle(vis, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + 5), (60, 60, 60), -1)
    cv2.rectangle(vis, (bar_x0, bar_y0), (bar_x0 + int(bar_w * progress), bar_y0 + 5), _CLR_HUD, -1)

    # Timestamp
    mm, ss = divmod(int(elapsed_s), 60)
    y += 30
    _put(vis, f"Time  {mm:02d}:{ss:02d}  ({elapsed_s:.2f} s)", 10, y, _CLR_TIM)
    
    y += 24
    # Draw visual steering bar (Raw)
    _put(vis, f"Steer (Raw) {steering:+.2f}", 10, y)
    sbar_w = 200
    sbar_x = 180
    cv2.rectangle(vis, (sbar_x, y-12), (sbar_x + sbar_w, y+2), (60, 60, 60), -1)
    mid_x = sbar_x + sbar_w // 2
    if steering > 0:
        cv2.rectangle(vis, (mid_x, y-12), (mid_x + int(steering * sbar_w/2), y+2), (0, 120, 255), -1)
    elif steering < 0:
        cv2.rectangle(vis, (mid_x + int(steering * sbar_w/2), y-12), (mid_x, y+2), (0, 120, 255), -1)
    cv2.line(vis, (mid_x, y-14), (mid_x, y+4), (255, 255, 255), 1)

    y += 22
    # Draw visual steering bar (Comb)
    _put(vis, f"Steer (Comb) {steering_combined:+.2f}", 10, y)
    cv2.rectangle(vis, (sbar_x, y-12), (sbar_x + sbar_w, y+2), (60, 60, 60), -1)
    mid_x = sbar_x + sbar_w // 2
    if steering_combined > 0:
        cv2.rectangle(vis, (mid_x, y-12), (mid_x + int(steering_combined * sbar_w/2), y+2), (0, 120, 255), -1)
    elif steering_combined < 0:
        cv2.rectangle(vis, (mid_x + int(steering_combined * sbar_w/2), y-12), (mid_x, y+2), (0, 120, 255), -1)
    cv2.line(vis, (mid_x, y-14), (mid_x, y+4), (255, 255, 255), 1)

    # Offset (the accumulated camera-shift label correction)
    y += 22
    off_color = (0, 200, 255) if offset > 0.01 else ((255, 150, 0) if offset < -0.01 else (0, 255, 120))
    _put(vis, f"Offset   {offset:+.4f}", 10, y, off_color)

    if telem is None:
        y += 28
        _put(vis, "No telemetry for this frame", 10, y, _CLR_NOD)
        return vis

    # Speed
    vx, vy, vz = telem.get("velX", 0), telem.get("velY", 0), telem.get("velZ", 0)
    speed_kmh = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6
    y += 26; _put(vis, f"Speed    {speed_kmh:8.2f} km/h", 10, y)

    # Acceleration
    ax, ay, az = telem.get("accX", 0), telem.get("accY", 0), telem.get("accZ", 0)
    acc_g = math.sqrt(ax**2 + ay**2 + az**2) / 9.81
    y += 26; _put(vis, f"Accel    {acc_g:8.3f} g   ({ax:.2f}, {ay:.2f}, {az:.2f} m/s^2)", 10, y)

    # Attitude
    roll  = math.degrees(telem.get("rollPos",  0))
    pitch = math.degrees(telem.get("pitchPos", 0))
    yaw   = math.degrees(telem.get("yawPos",   0))
    y += 26; _put(vis, f"Roll  {roll:+7.2f} deg   Pitch {pitch:+7.2f} deg   Yaw {yaw:+7.2f} deg", 10, y)

    # Angular velocity
    rv = telem.get("rollVel",  0)
    pv = telem.get("pitchVel", 0)
    yv = telem.get("yawVel",   0)
    y += 26; _put(vis, f"AngVel   r={rv:+.3f}  p={pv:+.3f}  y={yv:+.3f}  rad/s", 10, y)

    # Angular acceleration
    ra = telem.get("rollAcc",  0)
    pa = telem.get("pitchAcc", 0)
    ya = telem.get("yawAcc",   0)
    y += 26; _put(vis, f"AngAcc   r={ra:+.3f}  p={pa:+.3f}  y={ya:+.3f}  rad/s^2", 10, y)

    # Position
    px = telem.get("posX", 0); py2 = telem.get("posY", 0); pz = telem.get("posZ", 0)
    y += 26; _put(vis, f"Pos  ({px:.1f}, {py2:.1f}, {pz:.1f})", 10, y)

    # Display Map overlay if provided
    if map_img is not None:
        mh, mw = map_img.shape[:2]
        # Put it in the top right corner
        padding = 10
        if padding + mh <= vis.shape[0] and padding + mw <= vis.shape[1]:
            # Convert binary map to BGR for overlaying
            if len(map_img.shape) == 2:
                map_bgr = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
            else:
                map_bgr = map_img
            
            # Optional: Add a border around the map
            cv2.rectangle(vis, (vis.shape[1] - mw - padding - 2, padding - 2), 
                          (vis.shape[1] - padding + 2, padding + mh + 2), (0, 255, 0), 2)
            vis[padding:padding+mh, vis.shape[1]-mw-padding:vis.shape[1]-padding] = map_bgr

    # Controls reminder at bottom-right
    hint = "SPACE=pause  LEFT/RIGHT=step  Q=quit"
    cv2.putText(vis, hint, (10, vis.shape[0] - 10),
                _FONT, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir: Path):
    """
    Returns:
        frames  - sorted list of Path objects for each JPEG
        telem   - dict mapping frame filename -> telemetry dict (or None)
        fps     - estimated capture FPS inferred from timestamps (or default 10)
    """
    img_dir  = dataset_dir / "img"
    csv_path = dataset_dir / "telemetry.csv"

    if not img_dir.exists():
        print(f"ERROR: img/ directory not found in {dataset_dir}")
        sys.exit(1)

    frames = sorted(img_dir.glob("*.jpg"), key=lambda p: p.stem)
    if not frames:
        frames = sorted(img_dir.glob("*.png"), key=lambda p: p.stem)
    if not frames:
        print(f"ERROR: No JPEG/PNG images found in {img_dir}")
        sys.exit(1)

    print(f"[dataset] {len(frames)} frames in {img_dir}")

    # Infer FPS from timestamp filenames if they look numeric
    fps = 10.0
    try:
        t0 = float(frames[0].stem)
        t1 = float(frames[-1].stem)
        if t1 > t0:
            fps = (len(frames) - 1) / (t1 - t0)
            print(f"[dataset] Inferred FPS: {fps:.2f}")
    except ValueError:
        print(f"[dataset] Could not infer FPS from filenames, assuming {fps:.1f}")

    # Load CSV
    telem: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("frame", "")
                # Convert numeric fields from string to float, skip blanks
                parsed = {}
                for k, v in row.items():
                    if k in ("frame", "capture_time"):
                        parsed[k] = v
                    else:
                        try:
                            parsed[k] = float(v) if v != "" else None
                        except ValueError:
                            parsed[k] = None
                telem[name] = parsed
        print(f"[dataset] CSV loaded: {len(telem)} rows from {csv_path}")
    else:
        print(f"[dataset] No telemetry.csv found — frames will show without telemetry")

    return frames, telem, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(
        description="BeamNG dataset playback with telemetry HUD")
    p.add_argument("--dataset", default="dataset",
                   help="path to dataset folder (default: dataset)")
    p.add_argument("--speed",   type=float, default=1.0,
                   help="playback speed multiplier (default: 1.0)")
    p.add_argument("--fps",     type=float, default=None,
                   help="override display FPS (default: inferred from dataset)")
    p.add_argument("--loop",    action="store_true",
                   help="loop playback when it reaches the end")
    p.add_argument("--width",   type=int, default=960,
                   help="display window width (default: 960)")
    p.add_argument("--height",  type=int, default=540,
                   help="display window height (default: 540)")
    return p


def main():
    opt = make_parser().parse_args()

    dataset_dir = Path(opt.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    # Load
    frames, telem, inferred_fps = load_dataset(dataset_dir)
    fps         = opt.fps if opt.fps else inferred_fps
    frame_delay = 1.0 / (fps * opt.speed)
    total       = len(frames)

    print(f"[playback] {total} frames at {fps:.2f} FPS x{opt.speed} = {fps * opt.speed:.2f} FPS effective")
    print("[playback] Controls: SPACE=pause/resume  LEFT/RIGHT=step  Q/ESC=quit")
    print()

    # Playback state
    idx     = 0
    paused  = False
    t_start = time.perf_counter()
    t_last  = time.perf_counter()

    cv2.namedWindow("BeamNG Playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BeamNG Playback", opt.width, opt.height)

    while True:
        frame_path = frames[idx]
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[warn] Could not read {frame_path}")
            idx = (idx + 1) % total
            continue

        # Retrieve matching telemetry row
        row = telem.get(frame_path.name)
        # Only pass telem if at least one numeric field is populated
        t = None
        steering_val = 0.0
        offset_val   = 0.0
        combined_val = 0.0
        if row:
            steering_val = row.get("steering", 0.0)
            if steering_val is None:
                steering_val = 0.0
            offset_val = row.get("steering_offset", 0.0)
            if offset_val is None:
                offset_val = 0.0
            combined_val = row.get("steering_combined", steering_val + offset_val)
            if combined_val is None:
                combined_val = steering_val + offset_val
                
            has_data = any(
                v is not None and k not in ("frame", "capture_time", "steering", "steering_offset", "steering_combined")
                for k, v in row.items()
            )
            if has_data:
                t = row

        # Try to load map image if it exists
        map_img = None
        map_path = dataset_dir / "map" / frame_path.name
        if map_path.exists():
            map_img = cv2.imread(str(map_path))

        # Compute elapsed dataset time from the frame's own timestamp if available
        try:
            elapsed_s = float(frame_path.stem) - float(frames[0].stem)
        except ValueError:
            elapsed_s = idx / fps

        # Draw HUD
        vis = draw_hud(img, t, idx, total, elapsed_s, opt.speed, paused, steering_val, map_img, offset=offset_val, steering_combined=combined_val)

        # Fit to display window dimensions
        dh, dw = vis.shape[:2]
        if dw != opt.width or dh != opt.height:
            scale = min(opt.width / dw, opt.height / dh)
            vis   = cv2.resize(vis, (int(dw * scale), int(dh * scale)),
                               interpolation=cv2.INTER_LINEAR)

        cv2.imshow("BeamNG Playback", vis)

        # Key handling — waitKey(1) keeps UI responsive
        now    = time.perf_counter()
        budget = frame_delay - (now - t_last)
        key    = cv2.waitKey(max(1, int(budget * 1000))) & 0xFF

        if key in (ord("q"), 27):       # q or ESC
            break
        elif key == ord(" "):           # space  → toggle pause
            paused = not paused
        elif key == 83 or key == 0:     # RIGHT arrow  (83 on Windows, 0 on some)
            if paused:
                idx = min(idx + 1, total - 1)
        elif key == 81 or key == 255:   # LEFT arrow   (81 on Windows)
            if paused:
                idx = max(idx - 1, 0)

        # Advance frame only when playing
        if not paused:
            t_last = time.perf_counter()
            idx   += 1
            if idx >= total:
                if opt.loop:
                    idx     = 0
                    t_start = time.perf_counter()
                    t_last  = time.perf_counter()
                else:
                    print("\n[playback] End of dataset.")
                    # stay on last frame with paused=True so user can see it
                    idx    = total - 1
                    paused = True

    cv2.destroyAllWindows()
    print("Playback stopped.")


if __name__ == "__main__":
    main()
