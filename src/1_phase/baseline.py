"""Zero-shot YOLO11n baseline on the test split (Phase 1, section 2.1)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import gt_count, img_path, load_split_ids
from common.model import load_model, predict_cars  # noqa: F401  (load_model is used by the notebook)
from common.plotting import show_image_with_boxes


def load_test_ids():
    return load_split_ids("test")


def run_zeroshot_baseline(model, test_ids) -> dict:
    """Count cars with the un-fine-tuned model on every test image; score count MAE / RMSE."""
    gt = np.array([gt_count(i) for i in test_ids])
    pred = np.array([len(predict_cars(model, img_path(i)).boxes) for i in test_ids])
    err = pred - gt
    result = pd.DataFrame([["YOLO11n zero-shot (COCO-pretrained, no fine-tuning)", round(gt.mean(), 1),
                            round(pred.mean(), 1), round(np.abs(err).mean(), 2), round(np.sqrt((err ** 2).mean()), 2)]],
                          columns=["Model", "mean GT", "mean pred", "MAE", "RMSE"])
    return {"result": result, "gt": gt, "pred": pred}


def plot_zeroshot_examples(model, examples=("carpk__20160331_NTU_00050", "carpk__20161225_TPZ_00224",
                                            "carpk__20161225_TPZ_00250")) -> None:
    """A few test images with the un-fine-tuned model's boxes (red)."""
    _, axes = plt.subplots(1, len(examples), figsize=(18, 5))
    for ax, img_id in zip(axes, examples):
        img = Image.open(img_path(img_id))
        r = predict_cars(model, img)
        show_image_with_boxes(img, r.boxes.xyxy.cpu().numpy(), ax=ax, color="red",
                              title=f"{img_id}\nGT {gt_count(img_id)} cars  |  predicted {len(r.boxes)}")
    plt.suptitle("Zero-shot YOLO11n (COCO-pretrained, no CARPK fine-tuning) — "
                 "massively under-detects dense aerial cars", y=1.03)
    plt.tight_layout()
    plt.show()
