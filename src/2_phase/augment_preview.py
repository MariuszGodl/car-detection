"""Before/after preview of the training augmentation, produced by the same Ultralytics dataset
class and default hyperparameters that ``model.train()`` uses (``augment=True`` for the train split)."""
import random
import sys
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
    augmentation = get_cfg(overrides=hyp_overrides)
    data = check_det_dataset(str(data_yaml))
    return YOLODataset(img_path=data["train"], imgsz=IMGSZ, augment=True, hyp=augmentation, data=data,
                       rect=False, batch_size=1, cache=False)


def _show_sample(dataset, index, ax, title):
    sample = dataset[index]
    image = sample["img"].permute(1, 2, 0).numpy()   # RGB channels: CHW -> HWC for plotting
    height, width = image.shape[:2]
    boxes = [yolo_to_xyxy(*box, width, height) for box in sample["bboxes"].tolist()]
    show_image_with_boxes(
        image, boxes, title=f"{title}\n{width}x{height} - {len(boxes)} target boxes", ax=ax
    )


def show_augmented(image_path, seed=0, warmup=64, **hyp_overrides):
    """Plot ``image_path`` with its labels (before) and two training samples built from it (after):
    one from the main phase (mosaic on) and one from the last ``close_mosaic`` epochs (mosaic off).

    Mosaic draws its 3 extra photos from a buffer filled while iterating the dataset, so ``warmup``
    random samples are loaded first, as happens during a real epoch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = train_dataset(**hyp_overrides)
    image_names = [Path(path).name for path in dataset.im_files]
    image_index = image_names.index(Path(image_path).name)
    warmup_indices = np.random.choice(len(dataset), min(warmup, len(dataset)), replace=False)
    for index in warmup_indices:
        dataset[int(index)]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    with Image.open(image_path) as image:
        labels = dataset.labels[image_index]["bboxes"].tolist()
        boxes = [yolo_to_xyxy(*box, *image.size) for box in labels]
        show_image_with_boxes(
            image, boxes, ax=axes[0],
            title=f"BEFORE: original photo\n{image.width}x{image.height} - {len(boxes)} labelled cars",
        )
    _show_sample(dataset, image_index, axes[1], "AFTER #1: main epochs (mosaic + affine + HSV + flip)")
    dataset.close_mosaic(hyp=get_cfg(overrides=hyp_overrides))
    _show_sample(dataset, image_index, axes[2], "AFTER #2: last 10 epochs (affine + HSV + flip, no mosaic)")
    fig.tight_layout()
    plt.show()
    return fig
