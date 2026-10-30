"""Build a one-class YOLO dataset (car) from the CARPK devkit into phase_2/yolo/.

Run from the repo root: ``python -m phase_2.dataset``.
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from phase_2.utils import ANN, get_image_path, load_split_ids, read_annotation

YOLO_DIR = Path("phase_2/yolo")


def _write_split(out_dir, split, ids) -> None:
    images_dir, labels_dir = out_dir / "images" / split, out_dir / "labels" / split
    for directory in (images_dir, labels_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    box_total = 0
    for img_id in ids:
        image_path = get_image_path(img_id)
        with Image.open(image_path) as im:
            img_w, img_h = im.size

        lines = []
        for x1, y1, x2, y2, _ in read_annotation(ANN / f"{img_id}.txt"):
            # Clip to the image; boxes that end up empty are dropped.
            x1, x2 = np.clip([x1, x2], 0, img_w)
            y1, y2 = np.clip([y1, y2], 0, img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}")

        (labels_dir / f"{img_id}.txt").write_text("".join(f"{line}\n" for line in lines))
        (images_dir / image_path.name).symlink_to(image_path.resolve())
        box_total += len(lines)
    print(f"{split:5s}: {len(ids):4d} images | {box_total:6d} boxes")


def write_yolo_dataset(out_dir=YOLO_DIR, val_ratio=0.2, seed=42) -> Path:
    """Write the dataset under out_dir and return the path of its carpk.yaml.

    The official train split is shuffled with a fixed seed; val_ratio of it becomes val.
    The official test split is used unchanged."""
    out_dir = Path(out_dir).resolve()
    train_ids = [str(i) for i in np.random.default_rng(seed).permutation(load_split_ids("train"))]
    n_val = int(val_ratio * len(train_ids))
    splits = {"train": train_ids[n_val:], "val": train_ids[:n_val], "test": load_split_ids("test")}
    for split, ids in splits.items():
        _write_split(out_dir, split, ids)

    yaml_path = out_dir / "carpk.yaml"
    yaml_path.write_text(
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: car\n"
    )
    return yaml_path


if __name__ == "__main__":
    print(write_yolo_dataset())
