"""Sanity check: overfit one small batch (train == val, no augmentation / weight decay /
gradient accumulation). A correctly wired pipeline must drive the loss towards zero and
reach ~perfect mAP. DFL cannot reach 0 (its targets are soft), so the check uses box + cls.
"""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ROOT, RUNS_DIR, YOLO_DIR
from common.data import link
from common.training import LOSS_COLS, launch, read_results, train_args

SANITY_DIR = ROOT / "datasets" / "yolo_sanity"

OVERFIT_ARGS = dict(
    optimizer="AdamW", lr0=0.002, lrf=1.0, weight_decay=0.0, warmup_epochs=0,
    mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, fliplr=0.0, flipud=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, patience=0, workers=2, plots=False,
)


def make_overfit_dataset(n_images=4) -> Path:
    """Tiny dataset of ``n_images`` train images spread over the split (CARPK and PUCPR+)."""
    images = sorted((YOLO_DIR / "images" / "train").iterdir())
    step = len(images) // n_images
    selected_images = images[step // 2::step][:n_images]
    if SANITY_DIR.exists():
        shutil.rmtree(SANITY_DIR)
    (SANITY_DIR / "images").mkdir(parents=True)
    (SANITY_DIR / "labels").mkdir()
    for image in selected_images:
        link(image, SANITY_DIR / "images" / image.name)
        shutil.copy(YOLO_DIR / "labels" / "train" / f"{image.stem}.txt", SANITY_DIR / "labels")
    yaml_path = SANITY_DIR / "sanity.yaml"
    yaml_path.write_text(f"path: {SANITY_DIR}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: car\n")
    return yaml_path


def run_overfit(data_yaml, epochs=200, n_images=4) -> Path:
    """Train on the single batch on GPU 0 (``batch == nbs == n_images`` -> one step per epoch)."""
    args = train_args(
        data=data_yaml, epochs=epochs, batch_per_gpu=n_images, devices=[0],
        project=RUNS_DIR / "sanity", name="overfit_one_batch", nbs=n_images, **OVERFIT_ARGS,
    )
    return launch(args)


def check_overfit(run_dir, max_loss_ratio=0.1, min_map50=0.95) -> None:
    """Assert the box+cls loss collapsed and the batch is (almost) perfectly fit."""
    results = read_results(run_dir)
    box_cls = results["train/box_loss"] + results["train/cls_loss"]
    loss_ratio = box_cls.iloc[-1] / box_cls.iloc[0]
    map50 = results["metrics/mAP50(B)"].iloc[-1]
    print(f"total train loss: {results['train/total_loss'].iloc[0]:.3f} -> "
          f"{results['train/total_loss'].iloc[-1]:.3f} over {len(results)} epochs (1 step each)")
    print(f"box+cls loss    : {box_cls.iloc[0]:.3f} -> {box_cls.iloc[-1]:.3f} ({loss_ratio:.1%} of start); "
          f"DFL floor {results['train/dfl_loss'].iloc[-1]:.3f}")
    print(f"mAP50 on the memorised batch: {map50:.3f}")
    assert loss_ratio <= max_loss_ratio, "box+cls loss did not collapse"
    assert map50 >= min_map50, "model failed to memorise the batch"


def plot_overfit(run_dir) -> None:
    results = read_results(run_dir)
    _, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(13, 4))
    for column in LOSS_COLS + ["train/total_loss"]:
        loss_ax.plot(results["epoch"], results[column], label=column.split("/")[1])
    loss_ax.set(yscale="log", xlabel="epoch (= 1 step)", title="training loss")
    for column in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        metric_ax.plot(results["epoch"], results[column], label=column.split("/")[1])
    metric_ax.set(xlabel="epoch", title="val (= the same batch)")
    loss_ax.legend()
    metric_ax.legend()
    plt.tight_layout()
    plt.show()
