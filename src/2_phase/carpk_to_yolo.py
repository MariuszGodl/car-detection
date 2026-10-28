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
    img_out, lbl_out = out_dir / "images" / split, out_dir / "labels" / split
    for d in (img_out, lbl_out):   # start clean so re-runs leave no stale files
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    n_boxes = 0
    for img_id in ids:
        img = find_image(devkit / "Images", img_id)
        with Image.open(img) as im:
            w, h = im.size
        boxes = [xyxy_to_yolo(*b[:4], w, h) for b in read_annotation(devkit / "Annotations" / f"{img_id}.txt")]
        lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in filter(None, boxes)]
        (lbl_out / f"{img_id}.txt").write_text("".join(lines))
        link(img, img_out / img.name)
        n_boxes += len(lines)
    print(f"{split:5s}: {len(ids):4d} images | {n_boxes:6d} boxes")


def build_dataset(devkit=DATA, out_dir=YOLO_DIR, val_fraction=0.1, seed=42, val_lots=()) -> Path:
    """Write the YOLO dataset under ``out_dir`` and return its ``carpk.yaml``.

    ``val_lots`` holds whole parking lots out as val (see ``common.data.split_train_val``); build such a
    dataset into its own ``out_dir`` so the default random-val dataset stays as it is."""
    devkit, out_dir = Path(devkit), Path(out_dir)
    train_ids, val_ids = split_train_val(load_split_ids("train", devkit / "ImageSets"), val_fraction, seed, val_lots)
    test_ids = load_split_ids("test", devkit / "ImageSets")
    assert set(test_ids).isdisjoint(train_ids + val_ids), "test images leaked into train/val"
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        _write_split(devkit, out_dir, split, ids)
    yaml_path = out_dir / "carpk.yaml"
    yaml_path.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                         "nc: 1\nnames:\n  0: car\n")
    return yaml_path


if __name__ == "__main__":
    build_dataset()
