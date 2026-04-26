"""
check_csvs.py
-------------
Checks sanity of keypoint CSV files in the dataset.

Per file, validates:
  - Frame count       : number of rows == expected (default 150)
  - Keypoint count    : number of [x, y] pairs per row == expected (default 21)
  - Coordinate range  : all x, y values within [0.0, 1.0]
  - No empty rows     : no blank or unparseable lines

Usage:
    python check_csvs.py --dataset /path/to/dataset
    python check_csvs.py --dataset /path/to/dataset --expected-frames 150 --expected-keypoints 21
    python check_csvs.py --dataset /path/to/dataset --only-base
    python check_csvs.py --dataset /path/to/dataset --anomalies-only
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

COORD_PATTERN = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def check_csv(path: Path, expected_frames: int, expected_keypoints: int) -> dict:
    """
    Parse a keypoint CSV and return a report dict.
    Returns:
        {
            "frames"         : int,           actual row count
            "keypoint_counts": list[int],     keypoints found per row
            "bad_rows"       : list[dict],    rows with wrong keypoint count or out-of-range coords
            "error"          : str | None,    file-level error if unreadable
        }
    """
    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        return {"error": str(e)}

    # Strip empty lines for frame count
    non_empty = [l for l in lines if l.strip()]
    frames    = len(non_empty)

    keypoint_counts = []
    bad_rows        = []

    for row_idx, line in enumerate(non_empty):
        pairs = COORD_PATTERN.findall(line)
        kp_count = len(pairs)
        keypoint_counts.append(kp_count)

        row_issues = []

        if kp_count != expected_keypoints:
            row_issues.append(f"keypoints={kp_count} (expected {expected_keypoints})")

        for kp_idx, (x_str, y_str) in enumerate(pairs):
            x, y = float(x_str), float(y_str)
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                row_issues.append(f"kp{kp_idx} out of range ({x:.4f}, {y:.4f})")

        if row_issues:
            bad_rows.append({"row": row_idx, "issues": row_issues})

    return {
        "frames"         : frames,
        "keypoint_counts": keypoint_counts,
        "bad_rows"       : bad_rows,
        "error"          : None,
    }


def is_augmented(name: str) -> bool:
    return "-zoom" in name


def is_prime_video_csv(name: str) -> bool:
    # We only want -prime.csv files (the keypoint CSVs), exclude non-csv
    return not name.endswith(".csv")


def check_dataset(
    dataset_root: str,
    expected_frames: int    = 150,
    expected_keypoints: int = 21,
    only_base: bool         = False,
    anomalies_only: bool    = False,
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs   = sorted([d for d in root.iterdir() if d.is_dir()])
    total        = 0
    frame_counts = defaultdict(int)
    anomaly_list = []

    for class_dir in class_dirs:
        csvs = sorted(class_dir.glob("*-prime.csv"))

        if only_base:
            csvs = [c for c in csvs if not is_augmented(c.name)]

        if not csvs:
            continue

        print(f"\n[{class_dir.name}] — {len(csvs)} CSV(s)")

        for csv_path in csvs:
            report = check_csv(csv_path, expected_frames, expected_keypoints)
            total += 1

            if report["error"]:
                print(f"  ✗ {csv_path.name:55s}  ERROR: {report['error']}")
                anomaly_list.append({
                    "class" : class_dir.name,
                    "file"  : csv_path.name,
                    "reason": f"ERROR: {report['error']}",
                })
                continue

            frames    = report["frames"]
            bad_rows  = report["bad_rows"]
            kp_counts = report["keypoint_counts"]

            frame_counts[frames] += 1

            frames_ok = frames == expected_frames
            kp_ok     = all(k == expected_keypoints for k in kp_counts)
            coords_ok = len(bad_rows) == 0
            is_anomaly = not frames_ok or not kp_ok or not coords_ok

            if is_anomaly:
                reasons = []
                if not frames_ok:
                    reasons.append(f"frames={frames} (expected {expected_frames})")
                if not kp_ok:
                    bad_kp_rows = [i for i, k in enumerate(kp_counts) if k != expected_keypoints]
                    reasons.append(f"wrong keypoint count on rows {bad_kp_rows[:5]}"
                                   + ("..." if len(bad_kp_rows) > 5 else ""))
                if not coords_ok:
                    reasons.append(f"{len(bad_rows)} row(s) with out-of-range coords")
                anomaly_list.append({
                    "class"   : class_dir.name,
                    "file"    : csv_path.name,
                    "reason"  : " | ".join(reasons),
                    "bad_rows": bad_rows,
                })

            if is_anomaly or not anomalies_only:
                flag       = "✗" if is_anomaly else "✓"
                frame_str  = f"{frames}f"  + ("" if frames_ok else f"  ← expected {expected_frames}")
                kp_str     = f"{expected_keypoints}kp" if kp_ok else \
                             f"kp counts vary {set(kp_counts)}"
                coord_str  = "" if coords_ok else f"  {len(bad_rows)} coord issue(s)"
                print(f"  {flag} {csv_path.name:55s}  {frame_str:28s}  {kp_str}{coord_str}")

            # Print detail on bad rows (always, not just anomalies_only)
            if is_anomaly and bad_rows:
                for br in bad_rows[:5]:   # cap at 5 rows to avoid flooding
                    print(f"      row {br['row']:>3}: {'; '.join(br['issues'])}")
                if len(bad_rows) > 5:
                    print(f"      ... and {len(bad_rows) - 5} more row(s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Total CSVs checked   : {total}")
    print(f"  Anomalies found      : {len(anomaly_list)}")
    print(f"  Frame count dist.    : {dict(sorted(frame_counts.items()))}")

    if anomaly_list:
        print("\n  ANOMALOUS FILES:")
        print(f"  {'#':<4} {'Class':<12} {'File':<55} Reason")
        print("  " + "─" * 100)
        for i, a in enumerate(anomaly_list, 1):
            print(f"  {i:<4} {a['class']:<12} {a['file']:<55} {a['reason']}")
    else:
        print("\n  All CSV files passed ✓")
    print("═" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check keypoint CSV files in dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--expected-frames", type=int, default=150,
                        help="Expected number of frames (rows) per CSV (default: 150)")
    parser.add_argument("--expected-keypoints", type=int, default=21,
                        help="Expected number of [x, y] keypoints per row (default: 21)")
    parser.add_argument("--only-base", action="store_true",
                        help="Only check base records (skip -zoom augmented files)")
    parser.add_argument("--anomalies-only", action="store_true",
                        help="Only print CSVs that fail validation")
    args = parser.parse_args()

    check_dataset(
        dataset_root=args.dataset,
        expected_frames=args.expected_frames,
        expected_keypoints=args.expected_keypoints,
        only_base=args.only_base,
        anomalies_only=args.anomalies_only,
    )