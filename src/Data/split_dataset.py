"""
NIH ChestX-ray14 Dataset Splitting Script
- Split data based on Image Index from train_val_list.txt / test_list.txt
- Output folder structure:
    output_dir/
    |----- train_val/
        |----- descriptions/train_val.csv
        |----- images/*.png
    |----- test/
        |----- descriptions/test.csv
        |----- images/*.png
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm


# ==========================================================================================
# Step 1. Argument parsing
# ==========================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Dataset Splitter")

    parser.add_argument(
        "--image_dir",
        default="/storage/hjchoi/archive/image_file",
        help="Path to the folder containing original images"
    )
    parser.add_argument(
        "--csv_path",
        default="/storage/hjchoi/archive/Data_Entry_2017.csv",
        help="Path to the full dataset description CSV file"
    )
    parser.add_argument(
        "--train_val_list",
        default="/storage/hjchoi/archive/train_val_list.txt",
        help="Path to the txt file containing Image Index list for train/val split"
    )
    parser.add_argument(
        "--test_list",
        default="/storage/hjchoi/archive/test_list.txt",
        help="Path to the txt file containing Image Index list for test split"
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/hjchoi/archive/DATA",
        help="Root output folder where split data will be saved"
    )

    return parser.parse_args()


# ==========================================================================================
# Step 2. Load Image Index list from txt file
# ==========================================================================================
def load_image_index_list(txt_path: str) -> set:
    """
    Read Image Index list from a txt file and return it as a set.

    Why use set instead of list:
        - list requires O(n) time to check if a value exists
        - set allows O(1) lookup -> much faster for large-scale data
    """
    with open(txt_path, "r") as f:
        # Read each line, strip leading/trailing whitespace and newlines, convert to set
        index_set = set(line.strip() for line in f if line.strip())
    print(f"  -> Loaded {len(index_set)} Image Indices from {txt_path}")
    return index_set


# ==========================================================================================
# Step 3. Create output folder structure
# ==========================================================================================
def create_output_dirs(output_dir: str) -> dict:
    """
    Create the following folder structure:

    output_dir/
    |--- train_val/
        | ---- descriptions/
        | ---- images/
    |--- test/
        | ---- descriptions/
        | ---- images/
    """
    dirs = {
        "train_val_images":       os.path.join(output_dir, "train_val", "images"),
        "train_val_descriptions": os.path.join(output_dir, "train_val", "descriptions"),
        "test_images":            os.path.join(output_dir, "test", "images"),
        "test_descriptions":      os.path.join(output_dir, "test", "descriptions"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)  # Create directory; no error if it already exists
        print(f"  Directory created: {path}")

    return dirs


# ==========================================================================================
# Step 4. Split and save CSV
# ==========================================================================================
def split_and_save_csv(
    csv_path: str,
    train_val_set: set,
    test_set: set,
    dirs: dict
) -> tuple:
    """
    Split Data_Entry_2017.csv into train_val / test subsets and save each.

    Returns:
        train_val_df, test_df: split DataFrames
    """
    print("\n[Step 4] Splitting CSV file...")

    # Load the full CSV
    # pd.read_csv: reads a CSV file into a DataFrame (tabular structure)
    df = pd.read_csv(csv_path)

    # Drop any unnecessary unnamed columns if present
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print(f"  Total records: {len(df)}")

    # Filter rows by Image Index column
    # isin(): selects only rows whose value exists in the given set
    train_val_df = df[df["Image Index"].isin(train_val_set)].reset_index(drop=True)
    test_df      = df[df["Image Index"].isin(test_set)].reset_index(drop=True)

    print(f"  train_val records: {len(train_val_df)}")
    print(f"  test records:      {len(test_df)}")

    # Save each split as a CSV file
    train_val_csv_path = os.path.join(dirs["train_val_descriptions"], "train_val.csv")
    test_csv_path      = os.path.join(dirs["test_descriptions"],      "test.csv")

    train_val_df.to_csv(train_val_csv_path, index=False)
    test_df.to_csv(test_csv_path,           index=False)

    print(f"  Saved: {train_val_csv_path}")
    print(f"  Saved: {test_csv_path}")

    return train_val_df, test_df


# ������������������������������������������������������������������������������������������
# Step 5. Copy image files
# ������������������������������������������������������������������������������������������
def copy_images(
    image_dir: str,
    image_index_list: list,
    dest_dir: str,
    split_name: str
):
    """
    Copy images matching image_index_list from image_dir to dest_dir.

    Args:
        image_dir:        Source folder containing original images
        image_index_list: List of image filenames to copy (e.g. ['00000001_000.png', ...])
        dest_dir:         Destination folder to copy images into
        split_name:       Label for logging purposes ('train_val' or 'test')
    """
    print(f"\n[Step 5] Copying {split_name} images... (total: {len(image_index_list)})")

    missing = []  # Track images not found in the source folder

    # tqdm: wraps an iterable to automatically display a progress bar
    for img_name in tqdm(image_index_list, desc=f"  Copying {split_name}"):
        src_path  = os.path.join(image_dir, img_name)
        dest_path = os.path.join(dest_dir,  img_name)

        if os.path.exists(src_path):
            # shutil.copy: copies file from src to dest (without metadata)
            shutil.copy(src_path, dest_path)
        else:
            missing.append(img_name)

    print(f"  Copied: {len(image_index_list) - len(missing)} files")

    if missing:
        print(f"  Warning: {len(missing)} files not found: {missing[:5]} ...")  # Show up to 5 examples


# ==========================================================================================
# Main
# ==========================================================================================
def main():
    print("=" * 60)
    print("  NIH ChestX-ray14 Dataset Splitting Start")
    print("=" * 60)

    # Step 1. Parse arguments
    args = parse_args()

    # Step 2. Load Image Index lists from txt files
    print("\n[Step 2] Loading Image Index lists...")
    train_val_set = load_image_index_list(args.train_val_list)
    test_set      = load_image_index_list(args.test_list)

    # Check for any overlap between the two sets
    overlap = train_val_set & test_set  # set intersection
    if overlap:
        print(f"  Warning: {len(overlap)} overlapping Image Indices found between train_val and test!")

    # Step 3. Create output directories
    print("\n[Step 3] Creating output directories...")
    dirs = create_output_dirs(args.output_dir)

    # Step 4. Split and save CSV
    train_val_df, test_df = split_and_save_csv(
        csv_path      = args.csv_path,
        train_val_set = train_val_set,
        test_set      = test_set,
        dirs          = dirs
    )

    # Step 5. Copy images for each split
    copy_images(
        image_dir        = args.image_dir,
        image_index_list = train_val_df["Image Index"].tolist(),
        dest_dir         = dirs["train_val_images"],
        split_name       = "train_val"
    )
    copy_images(
        image_dir        = args.image_dir,
        image_index_list = test_df["Image Index"].tolist(),
        dest_dir         = dirs["test_images"],
        split_name       = "test"
    )

    # Step 6. Print final summary
    print("\n" + "=" * 60)
    print("  Dataset splitting complete!")
    print(f"  Output path: {args.output_dir}")
    print(f"  train_val -> images: {len(train_val_df)}, descriptions: train_val.csv")
    print(f"  test      -> images: {len(test_df)}, descriptions: test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()