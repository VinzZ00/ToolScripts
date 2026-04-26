"""
check_videos.py
---------------
Checks frame width, height, and frame count for every video in the dataset.
Flags any videos that don't match the expected dimensions or frame count.

File categories (auto-detected from filename):
  base      : {stem}.mp4                  — 224x224 cropped, used by model
  augmented : {stem}-zoom{n}.mp4          — zoom augmentations, should also be 224x224
  prime     : {stem}-prime.mp4            — raw uncropped, different size, skipped by default

Usage:
    python check_videos.py --dataset /path/to/dataset
    python check_videos.py --dataset /path/to/dataset --expected-size 224 224 --expected-frames 150
    python check_videos.py --dataset /path/to/dataset --only-base
    python check_videos.py --dataset /path/to/dataset --include-prime
    python check_videos.py --dataset /path/to/dataset --anomalies-only
"""

import cv2
import argparse
from pathlib import Path
from collections import defaultdict


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def categorize(name: str) -> str:
    """Return 'prime', 'augmented', or 'base'."""
    if "-prime" in name:
        return "prime"
    if "-zoom" in name or "-flipped" in name:
        return "augmented"
    return "base"


# ── Main ───────────────────────────────────────────────────────────────────────

def check_dataset(
    dataset_root: str,
    expected_w: int      = 224,
    expected_h: int      = 224,
    expected_frames: int = 150,
    only_base: bool      = False,
    include_prime: bool  = False,
    anomalies_only: bool = False,
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs   = sorted([d for d in root.iterdir() if d.is_dir()])
    total        = 0
    size_counts  = defaultdict(int)
    frame_counts = defaultdict(int)
    anomaly_list = []

    for class_dir in class_dirs:
        all_videos = sorted(class_dir.glob("*.mp4"))

        # Filter by category
        videos = []
        for v in all_videos:
            cat = categorize(v.name)
            if cat == "prime" and not include_prime:
                continue
            if cat == "augmented" and only_base:
                continue
            videos.append((v, cat))

        if not videos:
            continue

        base_count  = sum(1 for _, c in videos if c == "base")
        aug_count   = sum(1 for _, c in videos if c == "augmented")
        prime_count = sum(1 for _, c in videos if c == "prime")

        label_parts = [f"{base_count} base"]
        if aug_count:
            label_parts.append(f"{aug_count} augmented")
        if prime_count:
            label_parts.append(f"{prime_count} prime")

        print(f"\n[{class_dir.name}] — {', '.join(label_parts)}")

        for video, cat in videos:
            info = check_video(video)
            total += 1

            cat_label = f"[{cat}]"

            if "error" in info:
                print(f"  ✗ {cat_label:12s} {video.name:50s}  ERROR: {info['error']}")
                anomaly_list.append({
                    "class" : class_dir.name,
                    "cat"   : cat,
                    "file"  : video.name,
                    "reason": f"ERROR: {info['error']}",
                })
                continue

            w, h, frames, fps = info["w"], info["h"], info["frames"], info["fps"]

            # Prime videos have different expected size — just list them, don't flag
            if cat == "prime":
                size_counts[f"{w}x{h}"] += 1
                frame_counts[frames] += 1
                if not anomalies_only:
                    print(f"  - {cat_label:12s} {video.name:50s}  {w}x{h}  {frames}f  @ {fps}fps")
                continue

            size_counts[f"{w}x{h}"] += 1
            frame_counts[frames] += 1

            size_ok    = (w == expected_w and h == expected_h)
            frames_ok  = (frames == expected_frames)
            is_anomaly = not size_ok or not frames_ok

            if is_anomaly:
                reasons = []
                if not size_ok:
                    reasons.append(f"size {w}x{h} (expected {expected_w}x{expected_h})")
                if not frames_ok:
                    reasons.append(f"frames {frames} (expected {expected_frames})")
                anomaly_list.append({
                    "class" : class_dir.name,
                    "cat"   : cat,
                    "file"  : video.name,
                    "reason": " | ".join(reasons),
                })

            if is_anomaly or not anomalies_only:
                flag       = "✗" if is_anomaly else "✓"
                size_str   = f"{w}x{h}"   + ("" if size_ok   else f"  <- expected {expected_w}x{expected_h}")
                frames_str = f"{frames}f" + ("" if frames_ok else f"  <- expected {expected_frames}")
                print(f"  {flag} {cat_label:12s} {video.name:50s}  {size_str:35s}  {frames_str}  @ {fps}fps")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Total videos checked : {total}")
    print(f"  Anomalies found      : {len(anomaly_list)}")
    print(f"  Size distribution    : {dict(size_counts)}")
    print(f"  Frame count dist.    : {dict(sorted(frame_counts.items()))}")

    if anomaly_list:
        by_cat = defaultdict(list)
        for a in anomaly_list:
            by_cat[a["cat"]].append(a)

        for cat in ["base", "augmented", "prime"]:
            if cat not in by_cat:
                continue
            items = by_cat[cat]
            print(f"\n  ANOMALOUS [{cat.upper()}] FILES ({len(items)}):")
            print(f"  {'#':<4} {'Class':<12} {'File':<52} Reason")
            print("  " + "-" * 95)
            for i, a in enumerate(items, 1):
                print(f"  {i:<4} {a['class']:<12} {a['file']:<52} {a['reason']}")
    else:
        print("\n  All videos passed")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video dimensions and frame counts in dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--expected-size", nargs=2, type=int, default=[224, 224],
                        metavar=("W", "H"), help="Expected frame size for base+augmented (default: 224 224)")
    parser.add_argument("--expected-frames", type=int, default=150,
                        help="Expected frame count per video (default: 150)")
    parser.add_argument("--only-base", action="store_true",
                        help="Only check base records (skip augmented and prime files)")
    parser.add_argument("--include-prime", action="store_true",
                        help="Also list prime (raw uncropped) videos — shown but not size-checked")
    parser.add_argument("--anomalies-only", action="store_true",
                        help="Only print videos that fail the expected size or frame count")
    args = parser.parse_args()

    check_dataset(
        dataset_root=args.dataset,
        expected_w=args.expected_size[0],
        expected_h=args.expected_size[1],
        expected_frames=args.expected_frames,
        only_base=args.only_base,
        include_prime=args.include_prime,
        anomalies_only=args.anomalies_only,
    )