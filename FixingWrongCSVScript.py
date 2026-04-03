import re
import os
import glob

# --- CONFIG ---
INPUT_FOLDER = "/Users/vinz/Documents/BINUS S2/SLR/handyTools/dataset-Elvin/B"       # folder containing your CSV files
OUTPUT_FOLDER = "/Users/vinz/Documents/BINUS S2/SLR/handyTools/dataset-Elvin/B" # where processed files will be saved
# --------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
print(f"Found {len(csv_files)} CSV file(s)")

for filepath in csv_files:
    filename = os.path.basename(filepath)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "r") as f:
        content = f.read()

    with open(filepath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Find all coordinate pairs in the row
        pairs = re.findall(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*\]', line)
        # Join them as quoted columns
        new_lines.append(",".join(f'"{p}"' for p in pairs))

    with open(output_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"  {filename}: {len(new_lines)} rows processed")

print("Done!")