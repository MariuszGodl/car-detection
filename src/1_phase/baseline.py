"""Zero-shot YOLO11n baseline on the test split (Phase 1, section 2.1)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import gt_count, img_path, load_split_ids
from common.metrics import count_metrics
from common.model import load_model, predict_cars  # noqa: F401  (load_model is used by the notebook)
from common.plotting import show_image_with_boxes


def load_test_ids() -> list:
    return load_split_ids("test")


def run_zeroshot_baseline(model, test_ids) -> dict:
    """Count detected cars, then compare with annotations. No training happens here."""
    true_counts = []
    predicted_counts = []
    for image_id in test_ids:
        prediction = predict_cars(model, img_path(image_id))
        true_counts.append(gt_count(image_id))
        predicted_counts.append(len(prediction.boxes))

    true_counts = np.array(true_counts)
    predicted_counts = np.array(predicted_counts)
    metrics = count_metrics(true_counts, predicted_counts)
    summary = pd.DataFrame([{
        "Model": "YOLO11n zero-shot (COCO-pretrained, no fine-tuning)",
        "mean GT": round(true_counts.mean(), 1),
        "mean pred": round(predicted_counts.mean(), 1),
        "MAE": round(metrics["MAE"], 2),
        "RMSE": round(metrics["RMSE"], 2),
    }])
    return {"result": summary, "gt": true_counts, "pred": predicted_counts}


def plot_zeroshot_examples(model, examples=("carpk__20160331_NTU_00050", "carpk__20161225_TPZ_00224",
                                            "carpk__20161225_TPZ_00250")) -> None:
    """A few test images with the un-fine-tuned model's boxes (red)."""
    _, axes = plt.subplots(1, len(examples), figsize=(18, 5), squeeze=False)
    for ax, image_id in zip(axes.flat, examples):
        with Image.open(img_path(image_id)) as image:
            prediction = predict_cars(model, image)
            boxes = prediction.boxes.xyxy.cpu().numpy()
            title = f"{image_id}\nGT {gt_count(image_id)} cars  |  predicted {len(boxes)}"
            show_image_with_boxes(image, boxes, ax=ax, color="red", title=title)
    plt.suptitle("Zero-shot YOLO11n (COCO-pretrained, no CARPK fine-tuning) — "
                 "massively under-detects dense aerial cars", y=1.03)
    plt.tight_layout()
    plt.show()
