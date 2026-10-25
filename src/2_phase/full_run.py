"""Full training run: clean slate, train vs val curves, final train / val / test evaluation, demo inference."""
import math
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import train_eval
from common.boxes import read_yolo_labels
from common.config import CONF, DATA_YAML, IMGSZ, ROOT, RUNS_DIR, YOLO_DIR
from common.training import read_results

OLD_OUTPUTS = (RUNS_DIR,                            # checks/ (1 GPU, DDP, resume), sanity/, detect/ (COCO8 run)
               ROOT / "datasets" / "coco8",         # auto-downloaded by the accidental COCO8 run
               ROOT / "datasets" / "yolo_sanity")   # 4-image overfit set from section 6
METRICS = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
LOSSES = ["box_loss", "cls_loss", "dfl_loss"]
TRAIN, VAL, TRAIN_AUG = "tab:blue", "tab:orange", "lightgrey"


def clear_previous_runs(paths=OLD_OUTPUTS) -> list:
    """Delete earlier runs and their side datasets; returns what was removed."""
    removed = [Path(p) for p in paths if Path(p).exists()]
    for p in removed:
        shutil.rmtree(p)
    return removed


def best_epoch(run_dir) -> int:
    """Epoch whose weights are ``best.pt`` (max Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95)."""
    df = read_results(run_dir)
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return int(df.loc[fitness.idxmax(), "epoch"])


def plot_curves(run_dir):
    """Per epoch: P / R / mAP on the train subset vs val (top) and the three losses (bottom)."""
    val = read_results(run_dir)
    tr = train_eval.read(run_dir)
    best = best_epoch(run_dir)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, m in zip(axes[0], METRICS):
        ax.plot(tr["epoch"], tr[m], color=TRAIN, label="train subset")
        ax.plot(val["epoch"], val[f"metrics/{m}"], color=VAL, label="val")
        ax.set(title=m, xlabel="epoch", ylim=(0, 1))
    for ax, l in zip(axes[1], LOSSES):
        ax.plot(val["epoch"], val[f"train/{l}"], color=TRAIN_AUG, label="train (augmented, during training)")
        ax.plot(tr["epoch"], tr[l], color=TRAIN, label="train subset (no augmentation)")
        ax.plot(val["epoch"], val[f"val/{l}"], color=VAL, label="val")
        ax.set(title=l, xlabel="epoch")
    gap = tr.set_index("epoch")["mAP50-95(B)"] - val.set_index("epoch")["metrics/mAP50-95(B)"]
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


def evaluate_splits(weights, splits=("train", "val", "test"), device=0, batch=16) -> pd.DataFrame:
    """Score ``weights`` on every split; plots / PR curves go to ``<run>/eval_<split>/``."""
    from ultralytics import YOLO

    weights = Path(weights)
    run_dir = weights.parents[1]
    rows = {}
    for split in splits:
        m = YOLO(str(weights)).val(data=str(ROOT / DATA_YAML), split=split, imgsz=IMGSZ, batch=batch,
                                   device=device, project=str(run_dir), name=f"eval_{split}", exist_ok=True,
                                   verbose=False)
        rows[split] = {"images": len(list((ROOT / YOLO_DIR / "images" / split).iterdir())),
                       **{k: m.results_dict[f"metrics/{k}"] for k in METRICS}}
    df = pd.DataFrame(rows).T
    df["images"] = df["images"].astype(int)
    return df


def plot_splits(df):
    """Grouped bars: each metric on train / val / test."""
    ax = df[METRICS].T.plot.bar(figsize=(10, 4.5), rot=0, ylim=(0, 1), width=0.75)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.set(title="best.pt on each split", ylabel="score")
    ax.legend(title="split", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return ax


def show_predictions(weights, n=10, split="test", seed=0, conf=CONF, device=0, cols=2) -> pd.DataFrame:
    """Load ``weights`` from disk, predict ``n`` random ``split`` images and show them with
    predicted vs ground-truth car counts (count = number of boxes)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = sorted((ROOT / YOLO_DIR / "images" / split).iterdir())
    pick = sorted(random.Random(seed).sample(imgs, min(n, len(imgs))))
    results = model.predict([str(p) for p in pick], imgsz=IMGSZ, conf=conf, device=device, verbose=False)
    nrows = math.ceil(len(pick) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(8 * cols, 4.6 * nrows), squeeze=False)
    rows = []
    for ax, img, r in zip(axes.flat, pick, results):
        gt = len(read_yolo_labels(ROOT / YOLO_DIR / "labels" / split / f"{img.stem}.txt"))
        pred = len(r.boxes)
        rows.append({"image": img.stem, "gt": gt, "pred": pred, "error": pred - gt})
        ax.imshow(r.plot(labels=False, conf=False, line_width=1)[..., ::-1])   # BGR -> RGB
        ax.set_title(f"{img.stem}\npred {pred} | GT {gt} | error {pred - gt:+d}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{Path(weights).name} on {len(pick)} {split} images (conf ≥ {conf})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    df = pd.DataFrame(rows)
    print(f"count MAE {df['error'].abs().mean():.2f} | RMSE {(df['error'] ** 2).mean() ** 0.5:.2f} "
          f"over {len(df)} images")
    return df
