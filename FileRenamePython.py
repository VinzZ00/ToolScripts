import os
import re


def transform_filename(filename: str) -> str | None:
    """
    Replace only the prefix 'Yang_' with 'C_'.
    Keeps the rest of the filename unchanged.
    """
    pattern = r"^Yang_(.+)$"
    replacement = r"C_\1"

    if re.match(pattern, filename):
        return re.sub(pattern, replacement, filename)

    return None


def rename_files(directory: str):
    """
    Rename files in a directory using regex transformation
    """
    for filename in os.listdir(directory):
        new_name = transform_filename(filename)

        if new_name:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            print(f"Renaming: {filename} → {new_name}")
            os.rename(old_path, new_path)


if __name__ == "__main__":
    folder_path = "/Users/vinz/Documents/BINUS S2/SLR/handyTools/dataset-Elvin/Yang"  # change this
    rename_files(folder_path)