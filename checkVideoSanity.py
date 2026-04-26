"""
check_videos.py
---------------
Checks frame width, height, and frame count for every video in the dataset.
Flags any videos that don't match the expected dimensions or frame count.

Usage:
    python check_videos.py --dataset /path/to/dataset
    python check_videos.py --dataset /path/to/dataset --expected-size 224 224 --expected-frames 150
    python check_videos.py --dataset /path/to/dataset --only-base     # skip augmented/prime files
    python check_videos.py --dataset /path/to/dataset --anomalies-only
"""

import cv2
import argparse
from pathlib import Path
from collections import defaultdict


def check_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Could not open file"}

    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return {"w": w, "h": h, "frames": frames, "fps": round(fps, 2)}


def is_augmented(name: str) -> bool:
    return "-zoom" in name


def is_prime_video(name: str) -> bool:
    return "-prime" in name and name.endswith(".mp4")


def check_dataset(
    dataset_root: str,
    expected_w: int = 224,
    expected_h: int = 224,
    expected_frames: int = 150,
    only_base: bool = False,
    anomalies_only: bool = False,
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs  = sorted([d for d in root.iterdir() if d.is_dir()])
    total       = 0
    anomalies   = 0
    size_counts = defaultdict(int)
    frame_counts = defaultdict(int)

    for class_dir in class_dirs:
        videos = sorted(class_dir.glob("*.mp4"))

        if only_base:
            videos = [v for v in videos if not is_augmented(v.name) and not is_prime_video(v.name)]

        if not videos:
            continue

        print(f"\n[{class_dir.name}] — {len(videos)} video(s)")

        for video in videos:
            info = check_video(video)
            total += 1

            if "error" in info:
                print(f"  ✗ {video.name:50s}  ERROR: {info['error']}")
                anomalies += 1
                continue

            w, h, frames, fps = info["w"], info["h"], info["frames"], info["fps"]
            size_counts[f"{w}x{h}"] += 1
            frame_counts[frames] += 1

            size_ok   = (w == expected_w and h == expected_h)
            frames_ok = (frames == expected_frames)
            is_anomaly = not size_ok or not frames_ok

            if is_anomaly:
                anomalies += 1

            if is_anomaly or not anomalies_only:
                flag = "✗" if is_anomaly else "✓"
                size_str   = f"{w}x{h}"   + ("" if size_ok   else f"  ← expected {expected_w}x{expected_h}")
                frames_str = f"{frames}f" + ("" if frames_ok else f"  ← expected {expected_frames}")
                print(f"  {flag} {video.name:50s}  {size_str:30s}  {frames_str}  @ {fps}fps")

    print("\n" + "─" * 70)
    print(f"Total videos checked : {total}")
    print(f"Anomalies found      : {anomalies}")
    print(f"\nSize distribution    : {dict(size_counts)}")
    print(f"Frame count dist.    : {dict(sorted(frame_counts.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video dimensions and frame counts in dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--expected-size", nargs=2, type=int, default=[224, 224],
                        metavar=("W", "H"), help="Expected frame size (default: 224 224)")
    parser.add_argument("--expected-frames", type=int, default=150,
                        help="Expected frame count per video (default: 150)")
    parser.add_argument("--only-base", action="store_true",
                        help="Only check base records (skip -prime and -zoom files)")
    parser.add_argument("--anomalies-only", action="store_true",
                        help="Only print videos that fail the expected size or frame count")
    args = parser.parse_args()

    check_dataset(
        dataset_root=args.dataset,
        expected_w=args.expected_size[0],
        expected_h=args.expected_size[1],
        expected_frames=args.expected_frames,
        only_base=args.only_base,
        anomalies_only=args.anomalies_only,
    )