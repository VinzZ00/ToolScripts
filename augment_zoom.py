"""
augment_zoom.py
---------------
Applies zoom augmentation to every base record in a dataset directory.

Directory structure expected:
    dataset_root/
        A/
            A_001.mp4
            A_001-prime.csv
            A_001-prime.mp4          <- skipped (raw uncropped, used as zoom out source)
            A_001-flipped.mp4        <- skipped unless --include-flipped
            A_001-flipped-prime.csv  <- skipped unless --include-flipped
            ...

Outputs are saved alongside originals:
    A_001-zoom115.mp4
    A_001-zoom115-prime.csv
    A_001-zoom85.mp4
    A_001-zoom85-prime.csv

Usage:
    python augment_zoom.py --dataset /path/to/dataset
    python augment_zoom.py --dataset /path/to/dataset --zoom 0.85 1.15
    python augment_zoom.py --dataset /path/to/dataset --include-flipped
    python augment_zoom.py --dataset /path/to/dataset --replace   # overwrite existing files
"""

import argparse
import re
import cv2
import numpy as np
from pathlib import Path


# ── Default zoom levels ────────────────────────────────────────────────────────
DEFAULT_ZOOM_FACTORS = [0.85, 0.90, 1.10, 1.15]


# ── Core functions ─────────────────────────────────────────────────────────────

def _find_crop_center(src_path: Path, prime_path: Path) -> tuple:
    """
    Use template matching on the first frame to find where the 224x224 crop
    sits inside the prime video. Returns (cx, cy) pixel center in prime-video space.
    Falls back to prime video center if matching fails.
    """
    cap_src   = cv2.VideoCapture(str(src_path))
    cap_prime = cv2.VideoCapture(str(prime_path))

    ret_s, src_frame   = cap_src.read()
    ret_p, prime_frame = cap_prime.read()

    cap_src.release()
    cap_prime.release()

    if not ret_s or not ret_p:
        Wp = int(cv2.VideoCapture(str(prime_path)).get(cv2.CAP_PROP_FRAME_WIDTH))
        Hp = int(cv2.VideoCapture(str(prime_path)).get(cv2.CAP_PROP_FRAME_HEIGHT))
        return Wp // 2, Hp // 2

    Hp, Wp = prime_frame.shape[:2]

    # Template matching only works if prime is larger than src
    if prime_frame.shape[0] < src_frame.shape[0] or prime_frame.shape[1] < src_frame.shape[1]:
        return Wp // 2, Hp // 2

    result = cv2.matchTemplate(prime_frame, src_frame, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is top-left corner of match; convert to center
    cx = max_loc[0] + src_frame.shape[1] // 2
    cy = max_loc[1] + src_frame.shape[0] // 2
    return cx, cy


def zoom_video(src_path: Path, dst_path: Path, s: float, prime_path: Path = None) -> None:
    """
    Zoom a video by factor s, centered on the frame.

    s > 1.0 -> zoom in:  crop int(W/s) x int(H/s) from center of src, resize to 224x224
    s < 1.0 -> zoom out: crop int(224/s) x int(224/s) from prime video, resize to 224x224
                         The crop center is found via template matching so the hand stays
                         correctly positioned regardless of where the 224x224 was cropped from.
                         Falls back to black padding if prime_path is None or missing.
    s = 1.0 -> no-op copy from src
    """
    TARGET_W, TARGET_H = 224, 224

    if s < 1.0 and prime_path is not None and prime_path.exists():
        cap_prime = cv2.VideoCapture(str(prime_path))
        Wp  = int(cap_prime.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hp  = int(cap_prime.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_prime.get(cv2.CAP_PROP_FPS)
        cap_prime.release()

        # Find where the 224x224 crop actually lives in the prime video
        cx, cy = _find_crop_center(src_path, prime_path)

        # Region to crop from prime: (224/s) x (224/s), centered on (cx, cy)
        region_w = int(TARGET_W / s)
        region_h = int(TARGET_H / s)

        x1 = cx - region_w // 2
        y1 = cy - region_h // 2
        x2 = x1 + region_w
        y2 = y1 + region_h

        # Shift the window if it goes out of bounds (keeps the region full-size)
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > Wp:
            x1 -= (x2 - Wp)
            x2 = Wp
        if y2 > Hp:
            y1 -= (y2 - Hp)
            y2 = Hp
        x1, y1 = max(0, x1), max(0, y1)

        cap = cv2.VideoCapture(str(prime_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (TARGET_W, TARGET_H))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
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
    Reads line-by-line with regex to handle both quoted and unquoted [x, y] formats.
    """
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


def augment_record(
    video_path: Path,
    zoom_factors: list,
    replace: bool = False,
    verbose: bool = True,
) -> None:
    """
    Augment a single base record (video + its paired CSV) with all zoom factors.

    Args:
        video_path:   Path to the base {stem}.mp4
        zoom_factors: List of zoom scale factors to apply
        replace:      If True, overwrite existing augmented files.
                      If False, skip factors where both output files already exist.
        verbose:      Print progress per file.
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
        out_video  = folder / f"{stem}-{factor_tag}.mp4"
        out_csv    = folder / f"{stem}-{factor_tag}-prime.csv"

        already_exists = out_video.exists() and out_csv.exists()

        if already_exists and not replace:
            if verbose:
                print(f"  [SKIP]   {stem}-{factor_tag} — already exists (use --replace to overwrite)")
            continue

        if already_exists and replace:
            out_video.unlink()
            out_csv.unlink()
            if verbose:
                print(f"  [REPLACE] {stem}-{factor_tag}  (s={s})")
        else:
            if verbose:
                source = "prime" if s < 1.0 and has_prime else "cropped"
                print(f"  [NEW]     {stem}-{factor_tag}  (s={s}, source={source})")

        zoom_video(video_path, out_video, s, prime_path=prime_vid if has_prime else None)
        zoom_keypoints(csv_path, out_csv, s)


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_base_record(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".mp4")
        and "-prime" not in name
        and "-flipped" not in name
        and not any(f"-zoom{i}" in name for i in range(50, 200))
    )


def is_flipped_record(path: Path) -> bool:
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
    zoom_factors: list = DEFAULT_ZOOM_FACTORS,
    include_flipped: bool = False,
    replace: bool = False,
    verbose: bool = True,
) -> None:
    """
    Walk the dataset directory and apply zoom augmentation to every base record.

    Args:
        dataset_root:    Path to the root dataset directory.
        zoom_factors:    List of zoom scale factors to apply.
        include_flipped: Also zoom the flipped variants.
        replace:         Overwrite existing augmented files instead of skipping.
        verbose:         Print progress.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    mode_label = "REPLACE mode" if replace else "SKIP mode"
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Zoom factors: {zoom_factors}  |  {mode_label}\n")

    total_records   = 0
    total_new       = 0
    total_replaced  = 0
    total_skipped   = 0

    for class_dir in class_dirs:
        videos = sorted(class_dir.glob("*.mp4"))

        base_records    = [v for v in videos if is_base_record(v)]
        flipped_records = [v for v in videos if is_flipped_record(v)] if include_flipped else []
        targets         = base_records + flipped_records

        print(f"[{class_dir.name}] {len(base_records)} base records"
              + (f" + {len(flipped_records)} flipped" if include_flipped else ""))

        for video_path in targets:
            if verbose:
                print(f"  Processing: {video_path.name}")

            # Count outcomes for summary
            stem   = video_path.stem
            folder = video_path.parent
            for s in zoom_factors:
                tag       = f"zoom{int(round(s * 100))}"
                out_video = folder / f"{stem}-{tag}.mp4"
                out_csv   = folder / f"{stem}-{tag}-prime.csv"
                exists    = out_video.exists() and out_csv.exists()
                if exists and replace:
                    total_replaced += 1
                elif exists:
                    total_skipped += 1
                else:
                    total_new += 1

            augment_record(video_path, zoom_factors, replace=replace, verbose=verbose)
            total_records += 1

        print()

    print("─" * 55)
    print(f"Done.  {total_records} records processed.")
    print(f"       {total_new} new files generated.")
    print(f"       {total_replaced} files replaced.")
    print(f"       {total_skipped} files skipped (already existed).")


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
    parser.add_argument(
        "--replace", action="store_true",
        help="Overwrite existing augmented files. Default behaviour is to skip them."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    augment_dataset(
        dataset_root=args.dataset,
        zoom_factors=args.zoom,
        include_flipped=args.include_flipped,
        replace=args.replace,
        verbose=not args.quiet,
    )