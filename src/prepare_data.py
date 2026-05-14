import random
import shutil
from pathlib import Path

from config import DATA_DIR

RAW_ROOT = DATA_DIR / "raw" / "lfw-deepfunneled"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_IMAGES_PER_PERSON = 2

random.seed(42)


def clear_split_dirs():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def get_image_files(person_dir: Path):
    valid_ext = {".jpg", ".jpeg", ".png"}
    return [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]


def split_dataset():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_ROOT}")

    clear_split_dirs()

    person_dirs = [p for p in RAW_ROOT.iterdir() if p.is_dir()]
    kept_classes = 0
    skipped_classes = 0
    total_images = 0

    for person_dir in person_dirs:
        images = get_image_files(person_dir)

        if len(images) < MIN_IMAGES_PER_PERSON:
            skipped_classes += 1
            continue

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
            skipped_classes += 1
            continue

        for img_path in train_imgs:
            dst = TRAIN_DIR / person_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst / img_path.name)

        for img_path in val_imgs:
            dst = VAL_DIR / person_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst / img_path.name)

        for img_path in test_imgs:
            dst = TEST_DIR / person_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst / img_path.name)

        kept_classes += 1
        total_images += total

    print("Dataset split complete")
    print(f"Classes kept: {kept_classes}")
    print(f"Classes skipped: {skipped_classes}")
    print(f"Total images used: {total_images}")


if __name__ == "__main__":
    split_dataset()