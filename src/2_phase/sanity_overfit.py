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
    shutil.rmtree(SANITY_DIR, ignore_errors=True)
    (SANITY_DIR / "images").mkdir(parents=True)
    (SANITY_DIR / "labels").mkdir()
    for img in images[step // 2::step][:n_images]:
        link(img, SANITY_DIR / "images" / img.name)
        shutil.copy(YOLO_DIR / "labels" / "train" / f"{img.stem}.txt", SANITY_DIR / "labels")
    yaml_path = SANITY_DIR / "sanity.yaml"
    yaml_path.write_text(f"path: {SANITY_DIR}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: car\n")
    return yaml_path


def run_overfit(data_yaml, epochs=200, n_images=4) -> Path:
    """Train on the single batch on GPU 0 (``batch == nbs == n_images`` -> one step per epoch)."""
    return launch(train_args(data_yaml, epochs, batch_per_gpu=n_images, devices=[0], project=RUNS_DIR / "sanity",
                             name="overfit_one_batch", nbs=n_images, **OVERFIT_ARGS))


def check_overfit(run_dir, max_loss_ratio=0.1, min_map50=0.95) -> None:
    """Assert the box+cls loss collapsed and the batch is (almost) perfectly fit."""
    df = read_results(run_dir)
    box_cls = df["train/box_loss"] + df["train/cls_loss"]
    ratio, map50 = box_cls.iloc[-1] / box_cls.iloc[0], df["metrics/mAP50(B)"].iloc[-1]
    print(f"total train loss: {df['train/total_loss'].iloc[0]:.3f} -> {df['train/total_loss'].iloc[-1]:.3f} "
          f"over {len(df)} epochs (1 step each)")
    print(f"box+cls loss    : {box_cls.iloc[0]:.3f} -> {box_cls.iloc[-1]:.3f} ({ratio:.1%} of start); "
          f"DFL floor {df['train/dfl_loss'].iloc[-1]:.3f}")
    print(f"mAP50 on the memorised batch: {map50:.3f}")
    assert ratio <= max_loss_ratio, "box+cls loss did not collapse"
    assert map50 >= min_map50, "model failed to memorise the batch"


def plot_overfit(run_dir) -> None:
    df = read_results(run_dir)
    _, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    for c in LOSS_COLS + ["train/total_loss"]:
        a.plot(df["epoch"], df[c], label=c.split("/")[1])
    a.set(yscale="log", xlabel="epoch (= 1 step)", title="training loss")
    for c in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        b.plot(df["epoch"], df[c], label=c.split("/")[1])
    b.set(xlabel="epoch", title="val (= the same batch)")
    a.legend(), b.legend()
    plt.tight_layout()
    plt.show()
