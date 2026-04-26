"""
augment_zoom.py
---------------
Applies zoom augmentation to every base record in a dataset directory.

Directory structure expected:
    dataset_root/
        A/
            A_001.mp4
            A_001-prime.csv
            A_001-prime.mp4          ← skipped (raw uncropped)
            A_001-flipped.mp4        ← skipped (already augmented)
            A_001-flipped-prime.csv  ← skipped
            ...
        B/
            ...
        hello/
            ...

Outputs are saved alongside originals:
    A_001-zoom115.mp4
    A_001-zoom115-prime.csv
    A_001-zoom85.mp4
    A_001-zoom85-prime.csv

Usage:
    python augment_zoom.py --dataset /path/to/dataset --zoom 0.85 1.15
    python augment_zoom.py --dataset /path/to/dataset          # uses default zoom levels
    python augment_zoom.py --dataset /path/to/dataset --include-flipped
"""

import os
import ast
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ── Default zoom levels ────────────────────────────────────────────────────────
DEFAULT_ZOOM_FACTORS = [0.85, 0.90, 1.10, 1.15]


# ── Core functions ─────────────────────────────────────────────────────────────

def zoom_video(src_path: Path, dst_path: Path, s: float, prime_path: Path = None) -> None:
    """
    Zoom a video by factor s, centered on the frame.

    s > 1.0 -> zoom in:  crop int(W/s) x int(H/s) from center of src, resize to 224x224
    s < 1.0 -> zoom out: crop int(224/s) x int(224/s) from center of prime video, resize to 224x224
                         Falls back to black padding if prime_path is None or missing.
    s = 1.0 -> no-op copy from src
    """
    TARGET_W, TARGET_H = 224, 224

    if s < 1.0 and prime_path is not None and prime_path.exists():
        # Zoom out using prime video — extract a larger region to get real surrounding pixels
        cap = cv2.VideoCapture(str(prime_path))
        Wp  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hp  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Region size to crop from prime video: 224/s x 224/s, centered
        region_w = int(TARGET_W / s)
        region_h = int(TARGET_H / s)
        cx, cy   = Wp // 2, Hp // 2
        x1 = max(0, cx - region_w // 2)
        y1 = max(0, cy - region_h // 2)
        x2 = min(Wp, x1 + region_w)
        y2 = min(Hp, y1 + region_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (TARGET_W, TARGET_H))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y1:y2, x1:x2]
            resized  = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            writer.write(resized)

        cap.release()
        writer.release()
        return

    # Zoom in or fallback zoom out (black padding)
    cap = cv2.VideoCapture(str(src_path))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (W, H))

    if s > 1.0:
        crop_w = int(W / s)
        crop_h = int(H / s)
        x1 = (W - crop_w) // 2
        y1 = (H - crop_h) // 2

        def process_frame(frame):
            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    elif s < 1.0:
        # Fallback: shrink + black padding (prime video unavailable)
        small_w = int(W * s)
        small_h = int(H * s)
        pad_x = (W - small_w) // 2
        pad_y = (H - small_h) // 2

        def process_frame(frame):
            small  = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[pad_y:pad_y + small_h, pad_x:pad_x + small_w] = small
            return canvas

    else:
        def process_frame(frame):
            return frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process_frame(frame))

    cap.release()
    writer.release()



def zoom_keypoints(src_path: Path, dst_path: Path, s: float) -> None:
    """
    Zoom keypoints by factor s around center (0.5, 0.5).
    Reads line-by-line with regex to handle both quoted and unquoted [x, y] formats,
    since some CSVs in the dataset are written without field quoting.
    """
    import re
    pattern = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")

    out_lines = []
    with open(src_path, "r") as f:
        for line in f:
            pairs = pattern.findall(line)
            if not pairs:
                continue
            transformed = []
            for x_str, y_str in pairs:
                x, y = float(x_str), float(y_str)
                x2 = round(max(0.0, min(1.0, 0.5 + (x - 0.5) * s)), 4)
                y2 = round(max(0.0, min(1.0, 0.5 + (y - 0.5) * s)), 4)
                transformed.append(f"[{x2}, {y2}]")
            out_lines.append(",".join(transformed))

    with open(dst_path, "w") as f:
        f.write("\n".join(out_lines))


def augment_record(video_path: Path, zoom_factors: list, verbose: bool = True) -> None:
    """
    Augment a single base record (video + its paired CSV) with all zoom factors.
    Expects:  {stem}.mp4        — 224x224 cropped video (used by model, zoom in source)
              {stem}-prime.csv  — keypoints CSV (used by model)
              {stem}-prime.mp4  — raw uncropped video (zoom out source, NOT itself augmented)
    Outputs:  {stem}-zoom{factor}.mp4
              {stem}-zoom{factor}-prime.csv
    """
    stem      = video_path.stem
    folder    = video_path.parent
    csv_path  = folder / f"{stem}-prime.csv"
    prime_vid = folder / f"{stem}-prime.mp4"

    if not csv_path.exists():
        print(f"  [SKIP] No paired CSV for {video_path.name}")
        return

    has_prime = prime_vid.exists()
    if not has_prime and verbose:
        print(f"  [WARN] {stem}-prime.mp4 missing — zoom out will fall back to black padding")

    for s in zoom_factors:
        factor_tag = f"zoom{int(round(s * 100))}"

        out_video = folder / f"{stem}-{factor_tag}.mp4"
        out_csv   = folder / f"{stem}-{factor_tag}-prime.csv"

        if out_video.exists() and out_csv.exists():
            if verbose:
                print(f"  [EXISTS] {stem}-{factor_tag} — skipping")
            continue

        source = "prime" if s < 1.0 and has_prime else "cropped"
        if verbose:
            print(f"  -> {stem}-{factor_tag}  (s={s}, source={source})")

        zoom_video(video_path, out_video, s, prime_path=prime_vid if has_prime else None)
        zoom_keypoints(csv_path, out_csv, s)



def is_base_record(path: Path) -> bool:
    """
    True if the file is a base video record (not a derivative/augmented file).
    Base record: ends with .mp4 and stem contains no '-' separator variants.
    """
    name = path.name
    return (
        name.endswith(".mp4")
        and "-prime" not in name
        and "-flipped" not in name
        and not any(f"-zoom{i}" in name for i in range(50, 200))
    )


def is_flipped_record(path: Path) -> bool:
    """True if the file is the flipped variant of a base record."""
    name = path.name
    return (
        name.endswith(".mp4")
        and "-flipped" in name
        and "-prime" not in name
        and not any(f"-zoom{i}" in name for i in range(50, 200))
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def augment_dataset(
    dataset_root: str,
    zoom_factors: list[float] = DEFAULT_ZOOM_FACTORS,
    include_flipped: bool = False,
    verbose: bool = True,
) -> None:
    """
    Walk the dataset directory and apply zoom augmentation to every base record.

    Args:
        dataset_root:    Path to the root dataset directory.
        zoom_factors:    List of zoom scale factors to apply.
        include_flipped: Also zoom the flipped variants.
        verbose:         Print progress.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}\n")

    total_records = 0
    total_augmented = 0

    for class_dir in class_dirs:
        videos = sorted(class_dir.glob("*.mp4"))

        base_records    = [v for v in videos if is_base_record(v)]
        flipped_records = [v for v in videos if is_flipped_record(v)] if include_flipped else []

        targets = base_records + flipped_records
        print(f"[{class_dir.name}] {len(base_records)} base records"
              + (f" + {len(flipped_records)} flipped" if include_flipped else ""))

        for video_path in targets:
            if verbose:
                print(f"  Processing: {video_path.name}")
            augment_record(video_path, zoom_factors, verbose=verbose)
            total_records += 1
            total_augmented += len(zoom_factors)

        print()

    print("─" * 50)
    print(f"Done. {total_records} records processed.")
    print(f"      {total_augmented} augmented files generated.")
    print(f"      Each record now has {1 + len(zoom_factors)} versions (original + zooms).")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoom augmentation for sign language dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument(
        "--zoom", nargs="+", type=float,
        default=DEFAULT_ZOOM_FACTORS,
        help="Zoom factors to apply (default: 0.85 0.90 1.10 1.15)"
    )
    parser.add_argument(
        "--include-flipped", action="store_true",
        help="Also apply zoom to flipped variants"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    augment_dataset(
        dataset_root=args.dataset,
        zoom_factors=args.zoom,
        include_flipped=args.include_flipped,
        verbose=not args.quiet,
    )