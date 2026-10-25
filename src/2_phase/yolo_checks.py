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
        n_img = len(list((Path(out_dir) / "images" / split).iterdir()))
        n_lbl = len(list((Path(out_dir) / "labels" / split).iterdir()))
        print(f"{split:5s}: {n_img} images, {n_lbl} labels")
        assert n_img == n_lbl


def validate_labels(out_dir) -> None:
    """Assert every label is class 0 with coordinates in [0, 1]."""
    bad = sum(cls != 0 or not all(0 <= v <= 1 for v in vals)
              for p in (Path(out_dir) / "labels").rglob("*.txt") for cls, *vals in read_yolo_labels(p))
    print("out-of-range / wrong-class label lines:", bad)
    assert bad == 0


def pick_sample(out_dir, split="train") -> Path:
    return min((Path(out_dir) / "images" / split).iterdir())


def check_symlink(path) -> None:
    ok = Path(path).is_symlink() and Path(path).exists()
    print("image symlink ok:", ok)
    assert ok


def show_sample(out_dir, image_path, split="train"):
    """Overlay the YOLO labels (decoded back to pixels) on ``image_path``."""
    img = Image.open(image_path)
    rows = read_yolo_labels(Path(out_dir) / "labels" / split / f"{Path(image_path).stem}.txt")
    boxes = [yolo_to_xyxy(*r[1:], *img.size) for r in rows]
    return show_image_with_boxes(img, boxes, title=f"{Path(image_path).stem} — YOLO labels decoded back to pixels")
