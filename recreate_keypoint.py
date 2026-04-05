import os
import KeypointExtractor as kpExtract

DATASET_PATH = "dataset-Elvin"
LETTERS = ['O']  # adjust as needed

def extract_all_keypoints():
    for letter in LETTERS:
        folder = os.path.join(DATASET_PATH, letter)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        video_files = [
            f for f in os.listdir(folder)
            if f.endswith("-prime.mp4")  # matches both -prime and -flipped-prime
        ]

        if not video_files:
            print(f"No prime videos found in {folder}")
            continue

        print(f"\nFound {len(video_files)} videos in {folder}")

        for filename in sorted(video_files):
            video_path = os.path.join(folder, filename)
            csv_path = video_path.replace("-prime.mp4", "-prime.csv")

            # Skip if CSV already exists (resume-friendly)
            if os.path.exists(csv_path):
                print(f"  [SKIP] CSV already exists: {csv_path}")
                continue

            print(f"  [EXTRACT] {video_path}")
            try:
                kpExtract.extract_hand_keypoints(video_path, csv_path)
                print(f"  [DONE] -> {csv_path}")
            except Exception as e:
                print(f"  [ERROR] {video_path}: {e}")

if __name__ == "__main__":
    extract_all_keypoints()