import cv2
import os
import glob
from typing import Union


def get_frame_count(video_path: str) -> int:
    """
    Returns the total number of frames in a video file.
    Falls back to manual counting if the fast method is unreliable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If fast method returns invalid result, count manually
    if total <= 0:
        total = 0
        while True:
            grabbed, _ = cap.read()
            if not grabbed:
                break
            total += 1

    cap.release()
    return total


def check_video_frame_count(
    directory: str,
    expected_frames: int,
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"),
    recursive: bool = False,
) -> dict:
    """
    Scans a directory for video files and checks if each has exactly
    `expected_frames` frames.

    Args:
        directory:       Path to the folder containing video files.
        expected_frames: The required number of frames each video must have.
        extensions:      Tuple of video file extensions to scan.
        recursive:       If True, scan subdirectories as well.

    Returns:
        A dict with:
            - "passed"  : list of filenames that meet the frame count
            - "failed"  : list of dicts {file, actual_frames} that do NOT meet it
            - "errors"  : list of dicts {file, error} for files that could not be read
    """
    results = {"passed": [], "failed": [], "errors": []}

    pattern = "**/*" if recursive else "*"
    all_files = glob.glob(os.path.join(directory, pattern), recursive=recursive)

    video_files = [
        f for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in extensions
    ]

    if not video_files:
        print(f"No video files found in: {directory}")
        return results

    print(f"Found {len(video_files)} video file(s). Checking frame counts...")
    print(f"Expected frame count: {expected_frames}\n")

    for video_path in sorted(video_files):
        filename = os.path.basename(video_path)
        try:
            actual = get_frame_count(video_path)
            if actual == expected_frames:
                results["passed"].append(filename)
                print(f"  [PASS] {filename} — {actual} frames")
            else:
                results["failed"].append({"file": filename, "actual_frames": actual})
                print(f"  [FAIL] {filename} — {actual} frames (expected {expected_frames})")
        except Exception as e:
            results["errors"].append({"file": filename, "error": str(e)})
            print(f"  [ERROR] {filename} — {e}")

    print("\n--- Summary ---")
    print(f"  Passed : {len(results['passed'])}")
    print(f"  Failed : {len(results['failed'])}")
    print(f"  Errors : {len(results['errors'])}")

    if results["failed"]:
        print("\nFiles that did NOT meet the expected frame count:")
        for item in results["failed"]:
            print(f"  - {item['file']} ({item['actual_frames']} frames)")

    if results["errors"]:
        print("\nFiles that could not be read:")
        for item in results["errors"]:
            print(f"  - {item['file']}: {item['error']}")

    return results

if __name__ == "__main__":
    check_video_frame_count("/Users/vinz/Documents/BINUS S2/SLR/handyTools/dataset-Elvin/O", expected_frames=150, extensions=".mp4", recursive=True)