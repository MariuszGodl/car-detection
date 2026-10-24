"""Zero-shot YOLO11n baseline on the CARPK test split (Phase 1, section 2.1)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import ANN, SETS, CAR_COCO_ID, IMGSZ, CONF, img_path

# Default Hugging Face source for the YOLO11n weights, used as a fallback when
# the weights file is missing locally.
HF_REPO_ID = "Ultralytics/YOLO11"
HF_FILENAME = "yolo11n.pt"

# In-memory cache of already-loaded YOLO models, keyed by weights path.
_MODEL_CACHE = {}


def _download_from_hf(weights):
    """Download ``weights`` from Hugging Face using the token in ``.env``."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=token)
    dest = Path(weights)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(Path(cached).read_bytes())
    return str(dest)


def load_model(weights="models/yolo11n.pt"):
    """Return the YOLO model, loading it only once.

    If the model is already in memory it is returned from the cache. Otherwise
    the local weights file is used; if that file is missing it is first
    downloaded from Hugging Face (``HF_REPO_ID``) using the ``HF_TOKEN`` in
    ``.env``.

    COCO-pretrained weights = exactly the model fine-tuned in Phase 2.
    """
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]

    if not Path(weights).exists():
        _download_from_hf(weights)

    model = YOLO(weights)
    _MODEL_CACHE[weights] = model
    return model


def load_test_ids():
    return SETS.joinpath("test.txt").read_text().split()


def gt_count(img_id):
    """Number of annotated cars for one image id."""
    lines = ANN.joinpath(f"{img_id}.txt").read_text().splitlines()
    return sum(1 for l in lines if l.strip())


def run_zeroshot_baseline(model, test_ids, verbose=False):
    """Run the un-fine-tuned model on every test image and score count MAE/RMSE.

    Returns a dict with the summary ``result`` DataFrame plus the raw ``gt`` /
    ``pred`` arrays for any further analysis.
    """
    gt, pred = [], []
    for k, img_id in enumerate(test_ids):
        r = model.predict(img_path(img_id), classes=[CAR_COCO_ID],
                          imgsz=IMGSZ, conf=CONF, verbose=False, device=0)
        gt.append(gt_count(img_id))
        pred.append(len(r[0].boxes))
        if k % 100 == 0:
            torch.cuda.empty_cache()

    gt, pred = np.array(gt), np.array(pred)
    mae = float(np.abs(pred - gt).mean())
    rmse = float(np.sqrt(((pred - gt) ** 2).mean()))

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
        r = model.predict(img, classes=[CAR_COCO_ID], imgsz=IMGSZ, conf=CONF,
                          verbose=False, device=0)[0]
        ax.imshow(img)
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor="red", linewidth=1.2))
        ax.set_title(f"{img_id}\nGT {gt_count(img_id)} cars  |  predicted {len(r.boxes)}")
        ax.axis("off")
    plt.suptitle("Zero-shot YOLO11n (COCO-pretrained, no CARPK fine-tuning) — "
                 "massively under-detects dense aerial cars", y=1.03)
    plt.tight_layout()
    plt.show()
