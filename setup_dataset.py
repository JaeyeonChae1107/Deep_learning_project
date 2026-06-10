"""
Extract beef carcass grading dataset from zip files.

Output structure:
  dataset/images/
    train/
      grade_1pp/    <- Training/[원천]소도체_seg_1++.zip
      grade_1p/     <- Training/[원천]소도체_seg_1+.zip
      grade_1/      <- Training/[원천]소도체_seg_1.zip
      grade_2/      <- Training/[원천]소도체_seg_2.zip
      grade_3/      <- 70% of Validation/[원천]소도체_seg_3.zip
                       (Training source for grade 3 is missing from this dataset)
    val/
      grade_1pp/    <- first 15% of Validation/[원천]소도체_seg_1++.zip
      ...
    test/
      grade_1pp/    <- remaining 15% of Validation/[원천]소도체_seg_1++.zip
      ...

Run once before training:
  python setup_dataset.py
"""

import os
import zipfile
import random
import shutil
from pathlib import Path

SEED = 42
DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = DATASET_ROOT / "images"

# Map from zip keyword → folder name
GRADE_MAP = {
    "1++": "grade_1pp",
    "1+":  "grade_1p",
    "1":   "grade_1",
    "2":   "grade_2",
    "3":   "grade_3",
}
CLASS_NAMES = ["1++", "1+", "1", "2", "3"]

# Validation split: 70% train / 15% val / 15% test (for grades with no Training source)
# For regular grades: Training -> train, Validation -> 50% val / 50% test
VAL_SPLIT   = 0.50  # fraction of Validation that goes to val (regular grades)
GRADE3_TRAIN_SPLIT = 0.70  # fraction of grade-3 Validation used for training


def find_zip(directory: Path, tag: str, grade: str) -> Path | None:
    """Find a zip file matching [tag]소도체_seg_{grade}.zip in directory."""
    for f in directory.iterdir():
        if f.suffix == ".zip" and grade in f.name:
            # Check it contains the right tag (원천 or 라벨)
            if tag == "source" and "원천" in f.name:
                return f
            if tag == "label" and "라벨" in f.name:
                return f
    return None


def extract_files(zip_path: Path, filenames: list[str], dest_dir: Path):
    """Extract specific filenames from zip_path into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in filenames:
            dest_file = dest_dir / name
            if dest_file.exists():
                continue
            with zf.open(name) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)


def process_grade(grade: str, training_dir: Path, validation_dir: Path, rng: random.Random):
    grade_folder = GRADE_MAP[grade]
    train_src = find_zip(training_dir, "source", grade)
    val_src   = find_zip(validation_dir, "source", grade)

    train_out = OUTPUT_ROOT / "train" / grade_folder
    val_out   = OUTPUT_ROOT / "val"   / grade_folder
    test_out  = OUTPUT_ROOT / "test"  / grade_folder

    if grade != "3":
        # --- Regular grade: Training → train, Validation → val + test ---
        if train_src is None:
            print(f"  [SKIP] Training source zip not found for grade {grade}")
        else:
            all_files = sorted(zipfile.ZipFile(train_src).namelist())
            already = len(list(train_out.glob("*.jpg"))) if train_out.exists() else 0
            if already == len(all_files):
                print(f"  [OK]   train/grade_{grade_folder}: already extracted ({already} files)")
            else:
                print(f"  Extracting {len(all_files)} images → train/{grade_folder} ...")
                extract_files(train_src, all_files, train_out)

        if val_src is None:
            print(f"  [SKIP] Validation source zip not found for grade {grade}")
        else:
            all_files = sorted(zipfile.ZipFile(val_src).namelist())
            rng.shuffle(all_files)
            split_idx = int(len(all_files) * VAL_SPLIT)
            val_files  = all_files[:split_idx]
            test_files = all_files[split_idx:]

            for label, files, out in [("val", val_files, val_out), ("test", test_files, test_out)]:
                already = len(list(out.glob("*.jpg"))) if out.exists() else 0
                if already == len(files):
                    print(f"  [OK]   {label}/grade_{grade_folder}: already extracted ({already} files)")
                else:
                    print(f"  Extracting {len(files)} images → {label}/{grade_folder} ...")
                    extract_files(val_src, files, out)

    else:
        # --- Grade 3: Training source missing → use Validation with 70/15/15 split ---
        print(f"  NOTE: Training source zip for grade 3 is missing.")
        print(f"        Using Validation grade-3 images: {GRADE3_TRAIN_SPLIT*100:.0f}% train / "
              f"{(1-GRADE3_TRAIN_SPLIT)/2*100:.0f}% val / {(1-GRADE3_TRAIN_SPLIT)/2*100:.0f}% test")

        if val_src is None:
            print(f"  [SKIP] Validation source zip not found for grade 3")
            return

        all_files = sorted(zipfile.ZipFile(val_src).namelist())
        rng.shuffle(all_files)
        n = len(all_files)
        train_end = int(n * GRADE3_TRAIN_SPLIT)
        val_end   = train_end + int(n * (1 - GRADE3_TRAIN_SPLIT) / 2)

        splits = [
            ("train", all_files[:train_end],  train_out),
            ("val",   all_files[train_end:val_end], val_out),
            ("test",  all_files[val_end:],    test_out),
        ]
        for label, files, out in splits:
            already = len(list(out.glob("*.jpg"))) if out.exists() else 0
            if already == len(files):
                print(f"  [OK]   {label}/grade_{grade_folder}: already extracted ({already} files)")
            else:
                print(f"  Extracting {len(files)} images → {label}/{grade_folder} ...")
                extract_files(val_src, files, out)


def print_summary():
    print("\n=== Dataset Summary ===")
    total = 0
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_ROOT / split
        if not split_dir.exists():
            continue
        print(f"\n{split}/")
        split_total = 0
        for grade, folder in GRADE_MAP.items():
            d = split_dir / folder
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {folder:12s} ({grade:3s}): {n:6d} images")
            split_total += n
        print(f"  {'TOTAL':12s}      : {split_total:6d} images")
        total += split_total
    print(f"\nOverall total: {total} images")


def main():
    rng = random.Random(SEED)
    training_dir   = DATASET_ROOT / "Training"
    validation_dir = DATASET_ROOT / "Validation"

    if not training_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError(
            "Expected dataset/Training and dataset/Validation directories. "
            "Please ensure the zip files are in place."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for grade in CLASS_NAMES:
        print(f"\nProcessing grade {grade}:")
        process_grade(grade, training_dir, validation_dir, rng)

    print_summary()
    print("\nDone. You can now run: python src/train.py --model baseline")


if __name__ == "__main__":
    main()
