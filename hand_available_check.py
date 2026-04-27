"""
check_hands.py
--------------
Uses MediaPipe Hands (mp.solutions.hands) to verify every video in the dataset
contains a detectable hand. No model download required — the model is bundled
with mediapipe.

Compatible with mediapipe >= 0.9 (tested on 0.10.x).

Per video, reports:
  - Frames scanned     : how many frames were sampled
  - Frames with hand   : how many sampled frames had a hand detected
  - Detection rate     : percentage of sampled frames with hand
  - Status             : PASS / WARN / FAIL based on configurable thresholds

Status thresholds (configurable via CLI):
  PASS  : detection rate >= --pass-threshold  (default 80%)
  WARN  : detection rate >= --warn-threshold  (default 40%)
  FAIL  : detection rate <  --warn-threshold

By default checks only base records (not prime, not augmented zoom files).

Usage:
    python check_hands.py --dataset /path/to/dataset
    python check_hands.py --dataset /path/to/dataset --only-base
    python check_hands.py --dataset /path/to/dataset --include-augmented
    python check_hands.py --dataset /path/to/dataset --include-prime
    python check_hands.py --dataset /path/to/dataset --sample-rate 10
    python check_hands.py --dataset /path/to/dataset --all-frames
    python check_hands.py --dataset /path/to/dataset --anomalies-only
"""

import cv2
import argparse
import mediapipe as mp
from pathlib import Path
from collections import defaultdict


# ── Detector ───────────────────────────────────────────────────────────────────

def make_detector(min_confidence: float = 0.5):
    """
    Returns a MediaPipe Hands detector using the bundled model.
    No external model file needed.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    )


# ── Per-video check ────────────────────────────────────────────────────────────

def check_video_hands(
    path: Path,
    detector,
    sample_rate: int = 5,
    all_frames: bool = False,
) -> dict:
    """
    Run MediaPipe hand detection on a sampled subset of frames.

    Args:
        path:        Path to the video file.
        detector:    mp.solutions.hands.Hands instance (reused across calls).
        sample_rate: Check every Nth frame. Ignored if all_frames=True.
        all_frames:  Check every single frame (slow but thorough).

    Returns dict:
        total_frames, frames_scanned, frames_with_hand, detection_rate, error
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Could not open file"}

    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_scanned   = 0
    frames_with_hand = 0
    frame_idx        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not all_frames and frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        frames_scanned += 1
        if result.multi_hand_landmarks:
            frames_with_hand += 1

        frame_idx += 1

    cap.release()

    detection_rate = (frames_with_hand / frames_scanned * 100) if frames_scanned > 0 else 0.0

    return {
        "total_frames"    : total_frames,
        "frames_scanned"  : frames_scanned,
        "frames_with_hand": frames_with_hand,
        "detection_rate"  : round(detection_rate, 1),
        "error"           : None,
    }


# ── Categorization ─────────────────────────────────────────────────────────────

def categorize(name: str) -> str:
    if "-prime" in name:
        return "prime"
    if "-zoom" in name or "-flipped" in name:
        return "augmented"
    return "base"


# ── Main ───────────────────────────────────────────────────────────────────────

def check_dataset(
    dataset_root: str,
    sample_rate: int        = 5,
    all_frames: bool        = False,
    pass_threshold: float   = 80.0,
    warn_threshold: float   = 40.0,
    min_confidence: float   = 0.5,
    only_base: bool         = False,
    include_augmented: bool = False,
    include_prime: bool     = False,
    anomalies_only: bool    = False,
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs   = sorted([d for d in root.iterdir() if d.is_dir()])
    total        = 0
    anomaly_list = []
    rate_counts  = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}

    scan_label = "every frame" if all_frames else f"every {sample_rate}th frame"
    print(f"MediaPipe Hand Detection Sanity Check")
    print(f"Sampling   : {scan_label}")
    print(f"Thresholds : PASS >= {pass_threshold}%  |  WARN >= {warn_threshold}%  |  FAIL < {warn_threshold}%")
    print(f"Confidence : {min_confidence}")
    print()

    detector = make_detector(min_confidence)

    for class_dir in class_dirs:
        all_videos = sorted(class_dir.glob("*.mp4"))

        videos = []
        for v in all_videos:
            cat = categorize(v.name)
            if cat == "prime"     and not include_prime:     continue
            if cat == "augmented" and only_base:             continue
            if cat == "augmented" and not include_augmented: continue
            videos.append((v, cat))

        if not videos:
            continue

        counts = defaultdict(int)
        for _, c in videos:
            counts[c] += 1
        label_parts = [f"{counts['base']} base"]
        if counts["augmented"]: label_parts.append(f"{counts['augmented']} augmented")
        if counts["prime"]:     label_parts.append(f"{counts['prime']} prime")
        print(f"\n[{class_dir.name}] — {', '.join(label_parts)}")

        for video, cat in videos:
            report = check_video_hands(video, detector, sample_rate, all_frames)
            total += 1

            cat_label = f"[{cat}]"

            if report["error"]:
                print(f"  ERROR {cat_label:12s} {video.name:50s}  {report['error']}")
                rate_counts["ERROR"] += 1
                anomaly_list.append({
                    "class": class_dir.name, "cat": cat,
                    "file": video.name, "status": "ERROR",
                    "reason": report["error"],
                })
                continue

            scanned   = report["frames_scanned"]
            with_hand = report["frames_with_hand"]
            rate      = report["detection_rate"]

            if rate >= pass_threshold:
                status = "PASS"
            elif rate >= warn_threshold:
                status = "WARN"
            else:
                status = "FAIL"

            rate_counts[status] += 1
            is_anomaly = status in ("WARN", "FAIL")

            if is_anomaly:
                anomaly_list.append({
                    "class" : class_dir.name,
                    "cat"   : cat,
                    "file"  : video.name,
                    "status": status,
                    "reason": f"{rate}% hand detection ({with_hand}/{scanned} frames)",
                })

            if is_anomaly or not anomalies_only:
                icon     = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
                rate_str = f"{rate:5.1f}%  ({with_hand}/{scanned} frames)"
                print(f"  {icon} {cat_label:12s} {video.name:50s}  [{status}]  {rate_str}")

    detector.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Total videos checked : {total}")
    print(f"  PASS  (>= {pass_threshold}%)  : {rate_counts['PASS']}")
    print(f"  WARN  (>= {warn_threshold}%)  : {rate_counts['WARN']}")
    print(f"  FAIL  (<  {warn_threshold}%)  : {rate_counts['FAIL']}")
    print(f"  ERROR               : {rate_counts['ERROR']}")

    if anomaly_list:
        by_status = defaultdict(list)
        for a in anomaly_list:
            by_status[a["status"]].append(a)

        for status in ["FAIL", "WARN", "ERROR"]:
            if status not in by_status:
                continue
            items = by_status[status]
            print(f"\n  [{status}] FILES ({len(items)}):")
            print(f"  {'#':<4} {'Class':<12} {'Cat':<12} {'File':<45} Reason")
            print("  " + "-" * 100)
            for i, a in enumerate(items, 1):
                print(f"  {i:<4} {a['class']:<12} {a['cat']:<12} {a['file']:<45} {a['reason']}")
    else:
        print("\n  All videos passed ✓")
    print("=" * 75)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MediaPipe hand detection sanity check for sign language dataset"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Check every Nth frame (default: 5). Lower = slower but more accurate.")
    parser.add_argument("--all-frames", action="store_true",
                        help="Check every frame. Slow but thorough.")
    parser.add_argument("--pass-threshold", type=float, default=80.0,
                        help="Min detection rate %% to PASS (default: 80)")
    parser.add_argument("--warn-threshold", type=float, default=40.0,
                        help="Min detection rate %% to WARN instead of FAIL (default: 40)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="MediaPipe min hand detection confidence (default: 0.5)")
    parser.add_argument("--only-base", action="store_true",
                        help="Only check base records")
    parser.add_argument("--include-augmented", action="store_true",
                        help="Also check zoom-augmented videos (skipped by default)")
    parser.add_argument("--include-prime", action="store_true",
                        help="Also check prime (raw uncropped) videos (skipped by default)")
    parser.add_argument("--anomalies-only", action="store_true",
                        help="Only print WARN and FAIL videos")
    args = parser.parse_args()

    check_dataset(
        dataset_root=args.dataset,
        sample_rate=args.sample_rate,
        all_frames=args.all_frames,
        pass_threshold=args.pass_threshold,
        warn_threshold=args.warn_threshold,
        min_confidence=args.min_confidence,
        only_base=args.only_base,
        include_augmented=args.include_augmented,
        include_prime=args.include_prime,
        anomalies_only=args.anomalies_only,
    )