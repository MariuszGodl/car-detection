"""Zero-shot YOLO11n baseline on the CARPK test split (Phase 1, section 2.1)."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1])  # make `common` importable
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

from common.config import IMGSZ, CONF
from common.data import gt_count, img_path, load_split_ids
from common.metrics import count_errors
from common.model import load_model, predict_cars  # noqa: F401  (re-exported for the notebook)
from common.plotting import show_image_with_boxes


def load_test_ids():
    return load_split_ids("test")


def run_zeroshot_baseline(model, test_ids, verbose=False):
    """Run the un-fine-tuned model on every test image and score count MAE/RMSE.

    Returns a dict with the summary ``result`` DataFrame plus the raw ``gt`` /
    ``pred`` arrays for any further analysis.
    """
    gt, pred = [], []
    for k, img_id in enumerate(test_ids):
        r = predict_cars(model, img_path(img_id))
        gt.append(gt_count(img_id))
        pred.append(len(r.boxes))
        if k % 100 == 0:
            torch.cuda.empty_cache()

    gt, pred = np.array(gt), np.array(pred)
    err = count_errors(gt, pred)
    mae, rmse = err["mae"], err["rmse"]

    result = pd.DataFrame(
        [["YOLO11n zero-shot (COCO-pretrained, no fine-tuning)",
          round(gt.mean(), 1), round(pred.mean(), 1), round(mae, 2), round(rmse, 2)]],
        columns=["Model", "mean GT", "mean pred", "MAE", "RMSE"])

    if verbose:
        print(f"Zero-shot YOLO11n on {len(test_ids)} CARPK test images "
              f"(imgsz={IMGSZ}, conf={CONF}, class=car):")
        print(f"  mean GT count   : {gt.mean():.1f} cars/image")
        print(f"  mean pred count : {pred.mean():.1f} cars/image")
        print(f"  images with >=1 detection: {int((pred > 0).sum())}/{len(test_ids)}")
        print(f"  MAE = {mae:.2f}   RMSE = {rmse:.2f}")

    return {"result": result, "gt": gt, "pred": pred}


def plot_zeroshot_examples(model, examples=None):
    """Show a few test images with the un-fine-tuned model's boxes (red)."""
    if examples is None:
        examples = ["carpk__20160331_NTU_00050", "carpk__20161225_TPZ_00224",
                    "carpk__20161225_TPZ_00250"]

    fig, axes = plt.subplots(1, len(examples), figsize=(18, 5))
    for ax, img_id in zip(axes, examples):
        img = Image.open(img_path(img_id))
        r = predict_cars(model, img)
        show_image_with_boxes(
            img, r.boxes.xyxy.cpu().numpy(), ax=ax, color="red", linewidth=1.2,
            title=f"{img_id}\nGT {gt_count(img_id)} cars  |  predicted {len(r.boxes)}")
    plt.suptitle("Zero-shot YOLO11n (COCO-pretrained, no CARPK fine-tuning) — "
                 "massively under-detects dense aerial cars", y=1.03)
    plt.tight_layout()
    plt.show()
