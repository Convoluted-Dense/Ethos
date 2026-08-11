"""
active_learning_curator.py
==========================
Review, deduplicate, and approve/reject frames mined by the Active Learning
system in test_efficientnet.py.

Modes
-----
  1. **Interactive GUI** (default):
     Opens an OpenCV window for each pending frame.  The uncertainty score,
     model predictions, and reason for flagging are overlaid.
       A = Approve   R = Reject   S = Skip   Q = Quit

  2. **Auto-approve** (``--auto-approve``):
     Batch-approves all pending frames whose uncertainty (jitter) exceeds
     ``--min-uncertainty``.  No GUI needed.

After approval, a ``telemetry_al.csv`` is generated inside
``dataset/active_learning/approved/`` so that ``train_efficientnet.py``
can load the frames with ``--active-learning``.

Usage
-----
    python active_learning_curator.py                       # interactive
    python active_learning_curator.py --auto-approve        # auto-approve all
    python active_learning_curator.py --auto-approve --min-uncertainty 0.05
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PENDING_DIR  = Path("dataset/active_learning/pending")
APPROVED_DIR = Path("dataset/active_learning/approved")
REJECTED_DIR = Path("dataset/active_learning/rejected")
AL_CSV_NAME  = "telemetry_al.csv"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Perceptual hashing for duplicate detection  (average hash, 8×8)
# ---------------------------------------------------------------------------
def _avg_hash(img: np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64-bit average perceptual hash of a BGR image."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized > mean_val).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming_distance(h1: int, h2: int) -> int:
    """Number of differing bits between two hashes."""
    x = h1 ^ h2
    count = 0
    while x:
        count += 1
        x &= x - 1
    return count


def _is_duplicate(new_hash: int, existing_hashes: list[int],
                  max_distance: int = 5) -> bool:
    """Return True if new_hash is within max_distance bits of any existing hash."""
    for eh in existing_hashes:
        if _hamming_distance(new_hash, eh) <= max_distance:
            return True
    return False


# ---------------------------------------------------------------------------
# Load pending frames
# ---------------------------------------------------------------------------
def load_pending_frames(pending_dir: Path = None) -> list[dict]:
    """Find all .json sidecar files in pending_dir and return metadata dicts
    with added 'img_path' and 'meta_path' fields."""
    if pending_dir is None:
        pending_dir = PENDING_DIR
    frames = []
    if not pending_dir.exists():
        return frames

    for meta_path in sorted(pending_dir.glob("*.json")):
        stem = meta_path.stem
        img_path = pending_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["img_path"] = img_path
            meta["meta_path"] = meta_path
            frames.append(meta)
        except Exception:
            continue

    return frames


# ---------------------------------------------------------------------------
# Build approved hashes (for duplicate detection)
# ---------------------------------------------------------------------------
def load_approved_hashes() -> list[int]:
    """Compute perceptual hashes for all already-approved images."""
    hashes = []
    if not APPROVED_DIR.exists():
        return hashes

    for img_path in APPROVED_DIR.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            hashes.append(_avg_hash(img))

    return hashes


# ---------------------------------------------------------------------------
# Move frame between directories
# ---------------------------------------------------------------------------
def _move_frame(meta: dict, dest_dir: Path):
    """Move both .jpg and .json from pending to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    img_path: Path = meta["img_path"]
    meta_path: Path = meta["meta_path"]

    shutil.move(str(img_path), str(dest_dir / img_path.name))
    shutil.move(str(meta_path), str(dest_dir / meta_path.name))


# ---------------------------------------------------------------------------
# Generate telemetry_al.csv from approved frames
# ---------------------------------------------------------------------------
def generate_al_csv():
    """Create/overwrite telemetry_al.csv in APPROVED_DIR from the JSON sidecars.

    Columns mirror the main telemetry.csv format so that
    train_efficientnet.py can load them uniformly:
      frame, capture_time, steering, steering_offset, steering_combined,
      velX, velY, velZ  (plus placeholders for the rest)
    """
    csv_path = APPROVED_DIR / AL_CSV_NAME
    meta_files = sorted(APPROVED_DIR.glob("*.json"))

    if not meta_files:
        if csv_path.exists():
            csv_path.unlink()
        return 0

    # Full CSV header matching beamng_collect's format
    header = [
        "frame", "capture_time", "steering", "steering_offset",
        "steering_combined",
        "posX", "posY", "posZ",
        "velX", "velY", "velZ",
        "accX", "accY", "accZ",
        "upX", "upY", "upZ",
        "rollPos", "pitchPos", "yawPos",
        "rollVel", "pitchVel", "yawVel",
        "rollAcc", "pitchAcc", "yawAcc",
    ]

    rows = []
    for mf in meta_files:
        stem = mf.stem
        img_name = f"{stem}.jpg"
        if not (APPROVED_DIR / img_name).exists():
            continue
        try:
            with open(mf, "r") as f:
                meta = json.load(f)
        except Exception:
            continue

        steer = meta.get("target_steer", meta.get("true_steer", meta.get("raw_steer", 0.0)))
        speed_kmh = meta.get("speed_kmh", 0.0)
        # Convert speed_kmh back to approximate velX (m/s) for compatibility
        vel_ms = speed_kmh / 3.6

        row = {
            "frame": img_name,
            "capture_time": meta.get("timestamp", 0.0),
            "steering": steer,
            "steering_offset": 0.0,
            "steering_combined": steer,
            "posX": 0.0, "posY": 0.0, "posZ": 0.0,
            "velX": vel_ms, "velY": 0.0, "velZ": 0.0,
            "accX": 0.0, "accY": 0.0, "accZ": 0.0,
            "upX": 0.0, "upY": 0.0, "upZ": 1.0,
            "rollPos": 0.0, "pitchPos": 0.0, "yawPos": 0.0,
            "rollVel": 0.0, "pitchVel": 0.0, "yawVel": 0.0,
            "rollAcc": 0.0, "pitchAcc": 0.0, "yawAcc": 0.0,
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Interactive GUI curator
# ---------------------------------------------------------------------------
def interactive_review(frames: list[dict], approved_hashes: list[int]):
    """Show each pending frame in an OpenCV window for manual review."""
    approved = 0
    rejected = 0
    skipped = 0
    dup_skipped = 0

    WIN = "Active Learning Curator"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    total = len(frames)
    for idx, meta in enumerate(frames):
        img_path: Path = meta["img_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        # Check for duplicates
        img_hash = _avg_hash(img)
        if _is_duplicate(img_hash, approved_hashes):
            print(f"  [{idx+1}/{total}] DUPLICATE — auto-skipping {img_path.name}")
            _move_frame(meta, REJECTED_DIR)
            dup_skipped += 1
            continue

        # Initial target steer for this frame
        curr_target = float(meta.get("target_steer", meta.get("true_steer", meta.get("raw_steer", 0.0))))
        label_src   = meta.get("label_source", "HUD_ROI" if "true_steer" in meta else "MODEL_PRED")

        jitter = meta.get("jitter", 0.0)
        correction = meta.get("correction", 0.0)
        raw_s = meta.get("raw_steer", 0.0)
        smooth_s = meta.get("smoothed_steer", 0.0)
        speed = meta.get("speed_kmh", 0.0)
        reason = meta.get("reason", "unknown")

        while True:
            # Re-draw info overlay on a fresh copy
            display = img.copy()
            h, w = display.shape[:2]

            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            cv2.putText(display, f"[{idx+1}/{total}]  {img_path.name}",
                        (10, 22), FONT, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"Reason: {reason}   Jitter: {jitter:.4f}   Correction: {correction:.4f}",
                        (10, 44), FONT, 0.48, (0, 230, 130), 1, cv2.LINE_AA)
            cv2.putText(display, f"TARGET STEER: {curr_target:+.4f}  (Source: {label_src})",
                        (10, 68), FONT, 0.55, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(display, f"Model Raw: {raw_s:+.4f}   Smoothed: {smooth_s:+.4f}   Speed: {speed:.1f} km/h",
                        (10, 92), FONT, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, "A=Approve  R=Reject  S=Skip  <- / -> = Adjust Steer (+/-0.05)  Q=Quit",
                        (10, 118), FONT, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

            cv2.imshow(WIN, display)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("a"), ord("A")):
                # Update JSON metadata with final curated target steering
                meta["target_steer"] = round(curr_target, 6)
                meta["label_source"] = "CURATED_USER" if curr_target != meta.get("target_steer") else label_src
                with open(meta["meta_path"], "w") as f:
                    json.dump(meta, f, indent=2)

                _move_frame(meta, APPROVED_DIR)
                approved_hashes.append(img_hash)
                approved += 1
                print(f"  [{idx+1}/{total}] APPROVED  {img_path.name}  -> Target Steer: {curr_target:+.4f}")
                break
            elif key in (ord("r"), ord("R")):
                _move_frame(meta, REJECTED_DIR)
                rejected += 1
                print(f"  [{idx+1}/{total}] REJECTED  {img_path.name}")
                break
            elif key in (ord("s"), ord("S")):
                skipped += 1
                print(f"  [{idx+1}/{total}] SKIPPED   {img_path.name}")
                break
            elif key in (81, ord(","), ord("<")):  # Left arrow or comma: decrease steering
                curr_target = max(-1.0, curr_target - 0.05)
            elif key in (83, ord("."), ord(">")):  # Right arrow or period: increase steering
                curr_target = min(1.0, curr_target + 0.05)
            elif key in (ord("q"), 27):
                skipped += (total - idx)
                cv2.destroyAllWindows()
                return approved, rejected, skipped, dup_skipped

    cv2.destroyAllWindows()
    return approved, rejected, skipped, dup_skipped


# ---------------------------------------------------------------------------
# Auto-approve mode
# ---------------------------------------------------------------------------
def auto_approve(frames: list[dict], approved_hashes: list[int],
                 min_uncertainty: float = 0.0):
    """Batch-approve frames above the minimum uncertainty threshold."""
    approved = 0
    rejected = 0
    dup_skipped = 0

    for meta in frames:
        img_path: Path = meta["img_path"]

        jitter = meta.get("jitter", 0.0)
        reason = meta.get("reason", "")

        # Always approve manual flags regardless of threshold
        if reason == "manual" or jitter >= min_uncertainty:
            # Duplicate check
            img = cv2.imread(str(img_path))
            if img is not None:
                img_hash = _avg_hash(img)
                if _is_duplicate(img_hash, approved_hashes):
                    print(f"  DUPLICATE — rejecting {img_path.name}")
                    _move_frame(meta, REJECTED_DIR)
                    dup_skipped += 1
                    continue
                approved_hashes.append(img_hash)

            _move_frame(meta, APPROVED_DIR)
            approved += 1
            print(f"  APPROVED  {img_path.name}  (jitter={jitter:.4f}, reason={reason})")
        else:
            _move_frame(meta, REJECTED_DIR)
            rejected += 1
            print(f"  REJECTED  {img_path.name}  (jitter={jitter:.4f} < {min_uncertainty})")

    return approved, rejected, 0, dup_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Curate Active Learning mined frames for retraining")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Batch-approve all frames above --min-uncertainty")
    parser.add_argument("--min-uncertainty", type=float, default=0.02,
                        help="Minimum jitter threshold for auto-approve (default: 0.02)")
    parser.add_argument("--pending-dir", default=str(PENDING_DIR),
                        help=f"Directory with pending frames (default: {PENDING_DIR})")
    args = parser.parse_args()

    pending_dir = Path(args.pending_dir)

    print(f"\n{'='*60}")
    print(f"  Active Learning Curator")
    print(f"{'='*60}")
    print(f"  Pending dir  : {pending_dir}")
    print(f"  Approved dir : {APPROVED_DIR}")
    print(f"  Rejected dir : {REJECTED_DIR}")
    print(f"  Mode         : {'Auto-approve' if args.auto_approve else 'Interactive GUI'}")
    if args.auto_approve:
        print(f"  Min uncertainty: {args.min_uncertainty}")
    print()

    # Load pending frames
    frames = load_pending_frames(pending_dir)
    if not frames:
        print("No pending frames found. Run test_efficientnet.py with mining")
        print("enabled first (press M during inference).")
        sys.exit(0)

    print(f"Found {len(frames)} pending frame(s).\n")

    # Build hashes of already-approved frames for duplicate detection
    print("Building perceptual hashes of approved frames for duplicate detection...")
    approved_hashes = load_approved_hashes()
    print(f"  {len(approved_hashes)} existing approved frame(s) hashed.\n")

    # Process
    if args.auto_approve:
        approved, rejected, skipped, dup_skipped = auto_approve(
            frames, approved_hashes, args.min_uncertainty)
    else:
        approved, rejected, skipped, dup_skipped = interactive_review(
            frames, approved_hashes)

    # Generate telemetry CSV from approved frames
    print("\nGenerating telemetry_al.csv from approved frames...")
    n_csv = generate_al_csv()
    csv_path = APPROVED_DIR / AL_CSV_NAME

    # Summary
    print(f"\n{'='*60}")
    print(f"  Curation Summary")
    print(f"{'='*60}")
    print(f"  Approved         : {approved}")
    print(f"  Rejected         : {rejected}")
    print(f"  Duplicate-skipped: {dup_skipped}")
    print(f"  Skipped (later)  : {skipped}")
    print(f"  CSV rows written : {n_csv}  ({csv_path})")
    print(f"{'='*60}")
    if n_csv > 0:
        print(f"\n  Next step: python train_efficientnet.py --active-learning --epochs 10")
    print()


if __name__ == "__main__":
    main()
