"""
reindex_files.py
-----------------
Shifts already-indexed filenames from one starting index to another.
Handles both .mp4 and .csv (including -prime variants).

BEFORE (wrong start):
    Z/Z_720.mp4
    Z/Z_720-prime.csv
    Z/Z_721.mp4
    Z/Z_721-prime.csv

AFTER (correct start):
    Z/Z_920.mp4
    Z/Z_920-prime.csv
    Z/Z_921.mp4
    Z/Z_921-prime.csv

Usage:
    python reindex_files.py --dataset /path/to/dataset --from 720 --to 920
    python reindex_files.py --dataset /path/to/dataset --from 720 --to 920 --dry-run
"""

import argparse
import re
from pathlib import Path


def reindex_files(
    dataset_root: str,
    from_index: int,
    to_index: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    offset = to_index - from_index
    mode_label = "DRY RUN — no files will be renamed" if dry_run else "LIVE — files will be renamed"

    print(f"From index  : {from_index}")
    print(f"To index    : {to_index}")
    print(f"Offset      : {'+' if offset >= 0 else ''}{offset}")
    print(f"Mode        : {mode_label}\n")

    # Pattern: CLASS_INDEX.ext or CLASS_INDEX-prime.ext
    # e.g. Z_720.mp4, Z_720-prime.csv, A_720.mp4
    pattern = re.compile(r'^([A-Za-z]+)_(\d+)(-prime)?\.(mp4|csv)$')

    total_renamed = 0
    skipped = 0

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        cls = class_dir.name
        matches = []

        for f in sorted(class_dir.iterdir()):
            if not f.is_file():
                continue
            m = pattern.match(f.name)
            if not m:
                continue

            file_cls   = m.group(1)
            file_idx   = int(m.group(2))
            prime_tag  = m.group(3) or ''
            ext        = m.group(4)

            # Only touch files at or above from_index
            if file_idx < from_index:
                skipped += 1
                continue

            matches.append((f, file_cls, file_idx, prime_tag, ext))

        if not matches:
            continue

        print(f"[{cls}] {len(matches)} files to reindex")

        # Sort descending to avoid collisions when shifting up
        # (e.g. renaming 721→921 before 720→920 prevents overwrite)
        if offset > 0:
            matches.sort(key=lambda x: -x[2])
        else:
            matches.sort(key=lambda x: x[2])

        for f, file_cls, file_idx, prime_tag, ext in matches:
            new_idx  = file_idx + offset
            new_name = f"{file_cls}_{new_idx}{prime_tag}.{ext}"
            new_path = f.parent / new_name

            if verbose:
                action = "would rename" if dry_run else "renaming"
                print(f"  [{action}] {f.name:40s} → {new_name}")

            if not dry_run:
                f.rename(new_path)

            total_renamed += 1

    print("\n" + "─" * 55)
    print(f"Done.")
    print(f"  Renamed : {total_renamed} files{'  (dry run)' if dry_run else ''}")
    print(f"  Skipped : {skipped} files (index < {from_index})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shift indexed filenames from one starting index to another"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--from",    dest="from_index", type=int, required=True, help="Wrong starting index")
    parser.add_argument("--to",      dest="to_index",   type=int, required=True, help="Correct starting index")
    parser.add_argument("--dry-run", action="store_true", help="Preview without renaming")
    parser.add_argument("--quiet",   action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    reindex_files(
        dataset_root=args.dataset,
        from_index=args.from_index,
        to_index=args.to_index,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )