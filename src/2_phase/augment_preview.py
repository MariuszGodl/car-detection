"""Before/after preview of the training augmentation, produced by the same Ultralytics dataset
class and default hyperparameters that ``model.train()`` uses (``augment=True`` for the train split)."""
import random
import sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import yolo_to_xyxy
from common.config import DATA_YAML, IMGSZ
from common.plotting import show_image_with_boxes


def train_dataset(data_yaml=DATA_YAML, **hyp_overrides) -> YOLODataset:
    """The augmented training dataset, with the same defaults as ``train_args`` (overridable)."""
    hyp = get_cfg(overrides=hyp_overrides)
    data = check_det_dataset(str(data_yaml))
    return YOLODataset(img_path=data["train"], imgsz=IMGSZ, augment=True, hyp=copy(hyp), data=data,
                       rect=False, batch_size=1, cache=False)


def _show_sample(ds, idx, ax, title):
    s = ds[idx]
    im = s["img"].permute(1, 2, 0).numpy()   # Format() returns RGB, CHW
    h, w = im.shape[:2]
    boxes = [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
             for x, y, bw, bh in s["bboxes"].tolist()]
    show_image_with_boxes(im, boxes, title=f"{title}\n{w}x{h} - {len(boxes)} target boxes", ax=ax)


def show_augmented(image_path, seed=0, warmup=64, **hyp_overrides):
    """Plot ``image_path`` with its labels (before) and two training samples built from it (after):
    one from the main phase (mosaic on) and one from the last ``close_mosaic`` epochs (mosaic off).

    Mosaic draws its 3 extra photos from a buffer filled while iterating the dataset, so ``warmup``
    random samples are loaded first, as happens during a real epoch.
    """
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    ds = train_dataset(**hyp_overrides)
    idx = next(i for i, f in enumerate(ds.im_files) if Path(f).name == Path(image_path).name)
    for i in np.random.choice(len(ds), min(warmup, len(ds)), replace=False):
        ds[int(i)]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    img = Image.open(image_path)
    gt = [yolo_to_xyxy(*b, *img.size) for b in ds.labels[idx]["bboxes"].tolist()]   # normalised cx cy w h
    show_image_with_boxes(img, gt, ax=axes[0],
                          title=f"BEFORE: original photo\n{img.size[0]}x{img.size[1]} - {len(gt)} labelled cars")
    _show_sample(ds, idx, axes[1], "AFTER #1: main epochs (mosaic + affine + HSV + flip)")
    ds.close_mosaic(hyp=get_cfg(overrides=hyp_overrides))   # what the trainer does for the last close_mosaic epochs
    _show_sample(ds, idx, axes[2], "AFTER #2: last 10 epochs (affine + HSV + flip, no mosaic)")
    fig.tight_layout()
    plt.show()
    return fig
