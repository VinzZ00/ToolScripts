"""
augment_zoom.py
---------------
Applies zoom augmentation to every base record in a dataset directory.

Directory structure expected:
    dataset_root/
        A/
            A_001.mp4
            A_001-prime.csv
            A_001-prime.mp4          <- used as zoom source for video
            A_001-flipped.mp4
            A_001-flipped-prime.csv
            ...

Outputs are saved alongside originals:
    A_001-zoom115.mp4
    A_001-zoom115-prime.csv
    A_001-zoom85.mp4
    A_001-zoom85-prime.csv

Video augmentation logic:
    Uses MediaPipe Hands on the first frame of the prime video to find the
    centroid of 21 hand landmarks as (cx, cy) in prime pixel space.
    Both zoom in and zoom out crop a region centered on (cx, cy) from the
    prime video, then resize to 224x224.
    Falls back to black padding on src if no hand is detected.

    Zoom IN  (s > 1.0): crop (224/s x 224/s) centered on (cx, cy) from prime
    Zoom OUT (s < 1.0): crop (224/s x 224/s) centered on (cx, cy) from prime

Keypoint augmentation logic (independent from video pipeline):
    Center is always (prime_w/2, prime_h/2) in prime pixel space.

    Zoom IN  (s > 1.0): crop window of (224/s x 224/s) centered at prime center.
                        Re-normalize keypoints relative to that crop window.
    Zoom OUT (s < 1.0): 224x224 crop centered at prime center, shrunk to (224*s x 224*s)
                        and padded with black borders to 224x224.
                        Re-normalize keypoints to account for shrink + padding.

Usage:
    python augment_zoom.py --dataset /path/to/dataset
    python augment_zoom.py --dataset /path/to/dataset --zoom 0.85 1.15
    python augment_zoom.py --dataset /path/to/dataset --include-flipped
    python augment_zoom.py --dataset /path/to/dataset --replace
"""

import argparse
import re
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_ZOOM_FACTORS = [0.85, 0.90, 1.10, 1.15]
TARGET_W, TARGET_H   = 224, 224


# ── MediaPipe hand center ──────────────────────────────────────────────────────

def _get_hand_center(prime_path: Path):
    """
    Run MediaPipe Hands on the first frame of the prime video.
    Returns (cx, cy) in prime pixel space as the centroid of all 21 landmarks.
    Returns None if no hand is detected.
    """
    cap = cv2.VideoCapture(str(prime_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    Hp, Wp = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands = mp.solutions.hands.Hands(max_num_hands=1)
    result = hands.process(rgb)
    hands.close()

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    cx = sum(lm.x for lm in landmarks) / 21.0 * Wp
    cy = sum(lm.y for lm in landmarks) / 21.0 * Hp

    return int(cx), int(cy)


# ── Video pipeline ─────────────────────────────────────────────────────────────

def _zoom_out_black_padding(src_path: Path, dst_path: Path, s: float) -> None:
    """Shrink the 224x224 src video and pad with black borders. Used as fallback."""
    cap    = cv2.VideoCapture(str(src_path))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (W, H))

    small_w = int(W * s)
    small_h = int(H * s)
    pad_x   = (W - small_w) // 2
    pad_y   = (H - small_h) // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small  = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[pad_y:pad_y+small_h, pad_x:pad_x+small_w] = small
        writer.write(canvas)

    cap.release()
    writer.release()


def zoom_video(
    src_path: Path,
    dst_path: Path,
    s: float,
    prime_path: Path = None,
    hand_center=None,
) -> None:
    """
    Zoom a video by factor s using MediaPipe-detected hand center.

    Both zoom in and zoom out crop a (224/s x 224/s) region from the prime video
    centered on hand_center (cx, cy), then resize to 224x224.

    Falls back to black padding on src if:
    - prime video is missing
    - hand_center is None (MediaPipe found no hand)
    """
    if prime_path is not None and prime_path.exists() and hand_center is not None:
        cx, cy = hand_center

        cap_prime = cv2.VideoCapture(str(prime_path))
        Wp  = int(cap_prime.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hp  = int(cap_prime.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_prime.get(cv2.CAP_PROP_FPS)
        cap_prime.release()

        region_w = int(TARGET_W / s)
        region_h = int(TARGET_H / s)
        x1 = cx - region_w // 2
        y1 = cy - region_h // 2
        x2 = x1 + region_w
        y2 = y1 + region_h

        # Shift window to stay within prime bounds
        if x1 < 0:   x2 -= x1;        x1 = 0
        if y1 < 0:   y2 -= y1;        y1 = 0
        if x2 > Wp:  x1 -= (x2 - Wp); x2 = Wp
        if y2 > Hp:  y1 -= (y2 - Hp); y2 = Hp
        x1, y1 = max(0, x1), max(0, y1)

        cap    = cv2.VideoCapture(str(prime_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (TARGET_W, TARGET_H))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(cv2.resize(frame[y1:y2, x1:x2], (TARGET_W, TARGET_H),
                                    interpolation=cv2.INTER_LINEAR))
        cap.release()
        writer.release()
        return

    # Fallback: no prime or no hand detected
    _zoom_out_black_padding(src_path, dst_path, s)


# ── Keypoint pipeline ──────────────────────────────────────────────────────────

def zoom_keypoints(
    src_path: Path,
    dst_path: Path,
    s: float,
    prime_w: int,
    prime_h: int,
) -> None:
    """
    Re-normalize keypoints to match the zoomed video frame.

    Keypoints in src_path are normalized to prime video space (prime_w x prime_h).
    Center is always (prime_w/2, prime_h/2) — independent of video pipeline.

    Zoom IN (s > 1.0):
        Crop window (TARGET_W/s x TARGET_H/s) centered at prime center.
        new_x = (x * prime_w - x1) / (TARGET_W/s)
        new_y = (y * prime_h - y1) / (TARGET_H/s)

    Zoom OUT (s < 1.0):
        224x224 crop centered at prime center, shrunk to 224*s x 224*s, padded to 224x224.
        crop_x = (x * prime_w - x1) / TARGET_W
        new_x  = crop_x * s + (1 - s) / 2
    """
    pattern = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")

    cx = prime_w / 2.0
    cy = prime_h / 2.0

    if s > 1.0:
        region_w = TARGET_W / s
        region_h = TARGET_H / s
        x1 = cx - region_w / 2.0
        y1 = cy - region_h / 2.0

        def transform(x, y):
            new_x = (x * prime_w - x1) / region_w
            new_y = (y * prime_h - y1) / region_h
            return (
                round(max(0.0, min(1.0, new_x)), 4),
                round(max(0.0, min(1.0, new_y)), 4),
            )

    elif s < 1.0:
        x1  = cx - TARGET_W / 2.0
        y1  = cy - TARGET_H / 2.0
        pad = (1.0 - s) / 2.0

        def transform(x, y):
            crop_x = (x * prime_w - x1) / TARGET_W
            crop_y = (y * prime_h - y1) / TARGET_H
            new_x  = crop_x * s + pad
            new_y  = crop_y * s + pad
            return (
                round(max(0.0, min(1.0, new_x)), 4),
                round(max(0.0, min(1.0, new_y)), 4),
            )

    else:
        def transform(x, y):
            return round(x, 4), round(y, 4)

    out_lines = []
    with open(src_path, "r") as f:
        for line in f:
            pairs = pattern.findall(line)
            if not pairs:
                continue
            transformed = []
            for x_str, y_str in pairs:
                new_x, new_y = transform(float(x_str), float(y_str))
                transformed.append(f"[{new_x}, {new_y}]")
            out_lines.append(",".join(transformed))

    with open(dst_path, "w") as f:
        f.write("\n".join(out_lines))


# ── Record augmentation ────────────────────────────────────────────────────────

def augment_record(
    video_path: Path,
    zoom_factors: list,
    replace: bool = False,
    verbose: bool = True,
) -> None:
    stem      = video_path.stem
    folder    = video_path.parent
    csv_path  = folder / f"{stem}-prime.csv"
    prime_vid = folder / f"{stem}-prime.mp4"

    if not csv_path.exists():
        print(f"  [SKIP] No paired CSV for {video_path.name}")
        return

    has_prime = prime_vid.exists()
    if not has_prime:
        print(f"  [WARN] {stem}-prime.mp4 missing — video will use black padding fallback")

    # Read prime dimensions once — used by keypoint pipeline for all zoom factors
    if has_prime:
        cap     = cv2.VideoCapture(str(prime_vid))
        prime_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prime_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Detect hand center once from prime video — reused for all zoom factors
        hand_center = _get_hand_center(prime_vid)
        if hand_center is None:
            print(f"  [WARN] No hand detected in {stem}-prime.mp4 — video will use black padding fallback")
        elif verbose:
            print(f"  [HAND]    center=({hand_center[0]}, {hand_center[1]}) in prime space")
    else:
        cap     = cv2.VideoCapture(str(video_path))
        prime_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prime_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        hand_center = None

    for s in zoom_factors:
        factor_tag = f"zoom{int(round(s * 100))}"
        out_video  = folder / f"{stem}-{factor_tag}.mp4"
        out_csv    = folder / f"{stem}-{factor_tag}-prime.csv"

        already_exists = out_video.exists() and out_csv.exists()

        if already_exists and not replace:
            if verbose:
                print(f"  [SKIP]    {stem}-{factor_tag} — already exists (use --replace to overwrite)")
            continue

        if already_exists and replace:
            out_video.unlink()
            out_csv.unlink()
            if verbose:
                print(f"  [REPLACE] {stem}-{factor_tag}  (s={s})")
        else:
            if verbose:
                src = "prime+mediapipe" if has_prime and hand_center else "black padding"
                print(f"  [NEW]     {stem}-{factor_tag}  (s={s}, source={src})")

        # Video: MediaPipe hand center for both zoom in and out
        zoom_video(
            video_path, out_video, s,
            prime_path=prime_vid if has_prime else None,
            hand_center=hand_center,
        )

        # Keypoints: always prime_w/2, prime_h/2 (independent of video pipeline)
        zoom_keypoints(csv_path, out_csv, s, prime_w=prime_w, prime_h=prime_h)


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
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    mode_label = "REPLACE mode" if replace else "SKIP mode"
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Zoom factors: {zoom_factors}  |  {mode_label}\n")

    total_records  = 0
    total_new      = 0
    total_replaced = 0
    total_skipped  = 0
    total_fallback = 0

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

            stem      = video_path.stem
            folder    = video_path.parent
            prime_vid = folder / f"{stem}-prime.mp4"

            # Check fallback count for summary
            if prime_vid.exists():
                hc = _get_hand_center(prime_vid)
                if hc is None:
                    total_fallback += 1

            for s in zoom_factors:
                tag     = f"zoom{int(round(s * 100))}"
                out_vid = folder / f"{stem}-{tag}.mp4"
                out_csv = folder / f"{stem}-{tag}-prime.csv"
                exists  = out_vid.exists() and out_csv.exists()
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
    print(f"       {total_fallback} records used black padding (no hand detected).")


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