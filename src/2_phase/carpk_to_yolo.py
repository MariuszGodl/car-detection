"""Convert the combined devkit (CARPK + PUCPR+) to a YOLO dataset.

Devkit labels are ``x1 y1 x2 y2 class`` in pixels; YOLO labels are ``0 cx cy w h`` normalised
to [0, 1]. A deterministic val split is carved out of the official train split (the official
test split stays untouched). Images are symlinked into ``images/{train,val,test}``, labels go to
``labels/{train,val,test}`` and ``carpk.yaml`` describes the dataset for Ultralytics.
"""
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import xyxy_to_yolo
from common.config import DATA, YOLO_DIR
from common.data import find_image, link, load_split_ids, read_annotation, split_train_val


def _write_split(devkit, out_dir, split, ids) -> None:
    """Link the images and write one normalized, car-only label file per image."""
    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    for directory in (images_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    box_count = 0
    for image_id in ids:
        image_path = find_image(devkit / "Images", image_id)
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_lines = []
        annotations = read_annotation(devkit / "Annotations" / f"{image_id}.txt")
        for x1, y1, x2, y2, _ in annotations:
            box = xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height)
            if box is None:   # clipping can leave a box with no area
                continue
            center_x, center_y, width, height = box
            label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        (labels_dir / f"{image_id}.txt").write_text("".join(label_lines))
        link(image_path, images_dir / image_path.name)
        box_count += len(label_lines)
    print(f"{split:5s}: {len(ids):4d} images | {box_count:6d} boxes")


def build_dataset(devkit=DATA, out_dir=YOLO_DIR, val_fraction=0.1, seed=42, val_lots=()) -> Path:
    """Write the YOLO dataset under ``out_dir`` and return its ``carpk.yaml``.

    ``val_lots`` holds whole parking lots out as val (see ``common.data.split_train_val``); build such a
    dataset into its own ``out_dir`` so the default random-val dataset stays as it is."""
    devkit = Path(devkit)
    out_dir = Path(out_dir)
    official_train_ids = load_split_ids("train", devkit / "ImageSets")
    train_ids, val_ids = split_train_val(official_train_ids, val_fraction, seed, val_lots)
    test_ids = load_split_ids("test", devkit / "ImageSets")
    assert set(test_ids).isdisjoint(train_ids + val_ids), "test images leaked into train/val"
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        _write_split(devkit, out_dir, split, ids)
    yaml_path = out_dir / "carpk.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: car\n"
    )
    return yaml_path


if __name__ == "__main__":
    build_dataset()
