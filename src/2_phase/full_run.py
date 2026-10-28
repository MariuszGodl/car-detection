"""Run helpers: clearing the check runs, train vs val curves, train / val / test evaluation and inference
(a grid of random images, or one image or photo)."""
import math
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import train_eval
from common.boxes import read_yolo_labels
from common.config import CONF, DATA_YAML, IMGSZ, MAX_DET, ROOT, RUNS_DIR
from common.data import dataset_dir
from common.metrics import DETECTION_METRICS, count_metrics
from common.training import best_epoch, loss_columns, read_results

# Outputs of the Phase 2 checks only - never runs/train/, which holds the real training runs.
OLD_OUTPUTS = (RUNS_DIR / "checks",                 # 1 GPU vs DDP, interrupt + resume (section 7)
               RUNS_DIR / "sanity",                 # overfit-one-batch run (section 6)
               ROOT / "datasets" / "yolo_sanity",   # its 4-image dataset
               RUNS_DIR / "detect",                 # Ultralytics' default project (a train() without project=)
               ROOT / "datasets" / "coco8")         # downloaded when train() runs without data=
TRAIN, VAL, TRAIN_AUG = "tab:blue", "tab:orange", "lightgrey"


def clear_previous_runs(paths=OLD_OUTPUTS) -> list:
    """Delete earlier runs and their side datasets; returns what was removed."""
    removed = []
    for path in paths:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def plot_curves(run_dir):
    """Per epoch: P / R / mAP on the train subset vs val (top) and the three losses (bottom)."""
    validation = read_results(run_dir)
    train_subset = train_eval.read(run_dir)
    best = best_epoch(run_dir)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, metric in zip(axes[0], DETECTION_METRICS):
        ax.plot(train_subset["epoch"], train_subset[metric], color=TRAIN, label="train subset")
        ax.plot(validation["epoch"], validation[f"metrics/{metric}"], color=VAL, label="val")
        ax.set(title=metric, xlabel="epoch", ylim=(0, 1))
    losses = [column.split("/", 1)[1] for column in loss_columns(validation)]
    for ax, loss in zip(axes[1], losses):
        ax.plot(validation["epoch"], validation[f"train/{loss}"], color=TRAIN_AUG,
                label="train (augmented, during training)")
        ax.plot(train_subset["epoch"], train_subset[loss], color=TRAIN, label="train subset (no augmentation)")
        ax.plot(validation["epoch"], validation[f"val/{loss}"], color=VAL, label="val")
        ax.set(title=loss, xlabel="epoch")
    train_map = train_subset.set_index("epoch")["mAP50-95(B)"]
    validation_map = validation.set_index("epoch")["metrics/mAP50-95(B)"]
    gap = train_map - validation_map
    axes[1, 3].plot(gap.index, gap.values, color="tab:red")
    axes[1, 3].axhline(0, color="grey", lw=0.8)
    axes[1, 3].set(title="mAP50-95 gap (train subset − val)", xlabel="epoch")
    for ax in axes.flat:
        ax.axvline(best, color="grey", ls="--", lw=0.8)
    axes[0, 0].legend()
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"{Path(run_dir).name}: train vs val per epoch (dashed = best epoch {best})")
    fig.tight_layout()
    plt.show()
    return fig


def evaluate_splits(weights, splits=("train", "val", "test"), device=0, batch=16,
                    data=DATA_YAML) -> pd.DataFrame:
    """Score ``weights`` on every split; plots / PR curves go to ``<run>/eval_<split>/``."""
    weights = Path(weights)
    run_dir = weights.parents[1]
    rows = {}
    for split in splits:
        metrics = YOLO(str(weights)).val(
            data=str(ROOT / data), split=split, imgsz=IMGSZ, batch=batch, device=device,
            max_det=MAX_DET, project=str(run_dir), name=f"eval_{split}", exist_ok=True, verbose=False,
        )
        images = list((dataset_dir(data) / "images" / split).iterdir())
        rows[split] = {"images": len(images)}
        for metric in DETECTION_METRICS:
            rows[split][metric] = metrics.results_dict[f"metrics/{metric}"]
    summary = pd.DataFrame(rows).T
    summary["images"] = summary["images"].astype(int)
    return summary


def plot_splits(df):
    """Grouped bars: each metric on train / val / test."""
    ax = df[DETECTION_METRICS].T.plot.bar(figsize=(10, 4.5), rot=0, ylim=(0, 1), width=0.75)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set(title="best.pt on each split", ylabel="score")
    ax.legend(title="split", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return ax


def show_predictions(weights, n=10, split="test", seed=0, conf=CONF, device=0, cols=2,
                     data=DATA_YAML) -> pd.DataFrame:
    """Load ``weights`` from disk, predict ``n`` random ``split`` images and show them with
    predicted vs ground-truth car counts (count = number of boxes)."""
    model = YOLO(str(weights))
    dataset = dataset_dir(data)
    images = sorted((dataset / "images" / split).iterdir())
    selected_images = sorted(random.Random(seed).sample(images, min(n, len(images))))
    predictions = model.predict(
        [str(image) for image in selected_images], imgsz=IMGSZ, conf=conf,
        max_det=MAX_DET, device=device, verbose=False,
    )
    nrows = math.ceil(len(selected_images) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(8 * cols, 4.6 * nrows), squeeze=False)
    rows = []
    for ax, image, prediction in zip(axes.flat, selected_images, predictions):
        true_count = len(read_yolo_labels(dataset / "labels" / split / f"{image.stem}.txt"))
        predicted_count = len(prediction.boxes)
        error = predicted_count - true_count
        rows.append({"image": image.stem, "gt": true_count, "pred": predicted_count, "error": error})
        ax.imshow(prediction.plot(labels=False, conf=False, line_width=1)[..., ::-1])   # BGR -> RGB
        ax.set_title(f"{image.stem}\npred {predicted_count} | GT {true_count} | error {error:+d}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{Path(weights).name} on {len(selected_images)} {split} images (conf ≥ {conf})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    counts = pd.DataFrame(rows)
    metrics = count_metrics(counts["gt"], counts["pred"])
    print(f"count MAE {metrics['MAE']:.2f} | RMSE {metrics['RMSE']:.2f} over {len(counts)} images")
    return counts


def show_prediction(weights, image=None, split="test", conf=CONF, nms=None, device=0, seed=None,
                    data=DATA_YAML):
    """Predict one image with ``weights`` and show the boxes and the car count.

    ``image``: path to any photo; ``None`` picks a random ``split`` image of the dataset ``data``. If the image
    has a label file there, the true count and the error are shown too. ``nms=False`` uses YOLO26's
    NMS-free head; ``seed`` makes the random pick repeatable. Returns the Ultralytics result."""
    if image is None:
        image = random.Random(seed).choice(sorted((dataset_dir(data) / "images" / split).iterdir()))
    image = Path(image).expanduser()
    model = YOLO(str(weights))
    prediction = model.predict(
        str(image), imgsz=IMGSZ, conf=conf, max_det=MAX_DET, nms=nms, device=device, verbose=False
    )[0]
    predicted_count = len(prediction.boxes)
    title = f"{image.name}\npredicted {predicted_count} cars (conf ≥ {conf})"
    label = image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"   # <ds>/images/<split>/x.png
    if label.exists():
        true_count = len(read_yolo_labels(label))
        title += f"  |  true {true_count}  |  error {predicted_count - true_count:+d}"
    plt.figure(figsize=(14, 8))
    plt.imshow(prediction.plot(labels=False, conf=False, line_width=1)[..., ::-1])   # BGR -> RGB
    plt.title(title)
    plt.axis("off")
    plt.show()
    return prediction
