import os
import sys

DATASET_DIR = "/Users/vinz/Documents/BINUS S2/SLR/handyTools/dataset-Elvin"

patterns = ["-zoom85", "-zoom90", "-zoom110", "-zoom115"]

to_delete = []

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        if any(p in fname for p in patterns):
            to_delete.append(os.path.join(class_path, fname))

print(f"Found {len(to_delete)} files to delete:\n")
for f in sorted(to_delete):
    print(f"  {f}")

print(f"\nTotal: {len(to_delete)} files")
confirm = input("\nProceed with deletion? (yes/no): ").strip().lower()

if confirm == "yes":
    for f in to_delete:
        os.remove(f)
    print("Done. All files deleted.")
else:
    print("Aborted. Nothing was deleted.")