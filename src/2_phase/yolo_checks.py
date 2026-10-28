"""Integrity checks and visual inspection of the converted YOLO dataset."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import read_yolo_labels, yolo_to_xyxy
from common.plotting import show_image_with_boxes


def summarize_splits(out_dir) -> None:
    """Assert every split has one label file per image."""
    for split in ("train", "val", "test"):
        image_count = len(list((Path(out_dir) / "images" / split).iterdir()))
        label_count = len(list((Path(out_dir) / "labels" / split).iterdir()))
        print(f"{split:5s}: {image_count} images, {label_count} labels")
        assert image_count == label_count


def validate_labels(out_dir) -> None:
    """Assert every label is class 0 with coordinates in [0, 1]."""
    invalid_count = 0
    for label_path in (Path(out_dir) / "labels").rglob("*.txt"):
        for car_class, *coordinates in read_yolo_labels(label_path):
            if car_class != 0 or not all(0 <= value <= 1 for value in coordinates):
                invalid_count += 1
    print("out-of-range / wrong-class label lines:", invalid_count)
    assert invalid_count == 0


def pick_sample(out_dir, split="train") -> Path:
    return min((Path(out_dir) / "images" / split).iterdir())


def check_symlink(path) -> None:
    ok = Path(path).is_symlink() and Path(path).exists()
    print("image symlink ok:", ok)
    assert ok


def show_sample(out_dir, image_path, split="train"):
    """Overlay the YOLO labels (decoded back to pixels) on ``image_path``."""
    image_path = Path(image_path)
    label_path = Path(out_dir) / "labels" / split / f"{image_path.stem}.txt"
    labels = read_yolo_labels(label_path)
    with Image.open(image_path) as image:
        boxes = [
            yolo_to_xyxy(center_x, center_y, width, height, *image.size)
            for _, center_x, center_y, width, height in labels
        ]
        return show_image_with_boxes(
            image, boxes, title=f"{image_path.stem} — YOLO labels decoded back to pixels"
        )
