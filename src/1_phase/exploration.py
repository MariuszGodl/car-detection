"""Data loading, statistics and exploration of the combined devkit (Phase 1, section 3)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import xyxy_to_yolo
from common.config import ANN, IMG_H, IMG_W
from common.data import img_path, load_split_ids, read_annotation, split_train_val
from common.plotting import show_image_with_boxes


def load_splits():
    """Official (train_ids, test_ids, all_ids); asserts they are disjoint and cover every image."""
    train_ids, test_ids = load_split_ids("train"), load_split_ids("test")
    all_ids = sorted(p.stem for p in ANN.glob("*.txt"))
    print(f"#images total: {len(all_ids)} | #train: {len(train_ids)} | #test: {len(test_ids)}")
    assert set(train_ids).isdisjoint(test_ids), "train/test overlap!"
    assert set(train_ids) | set(test_ids) == set(all_ids), "splits do not cover all images"
    return train_ids, test_ids, all_ids


def check_sample_resolution(all_ids):
    size = Image.open(img_path(all_ids[0])).size
    print("Sample image:", all_ids[0], "size", size)
    assert size == (IMG_W, IMG_H)
    return size


def parse_annotations(all_ids, train_ids):
    """Per-box table, per-image counts and the degenerate (zero width/height) boxes left out."""
    train_set, rows = set(train_ids), []
    for img_id in all_ids:
        split = "train" if img_id in train_set else "test"
        rows += [dict(img_id=img_id, split=split, x1=x1, y1=y1, x2=x2, y2=y2, cls=c, w=x2 - x1, h=y2 - y1)
                 for x1, y1, x2, y2, c in read_annotation(ANN / f"{img_id}.txt")]
    df = pd.DataFrame(rows)
    valid = (df.w > 0) & (df.h > 0)
    boxes, degenerate = df[valid].copy(), df[~valid]
    boxes["area"], boxes["aspect"] = boxes.w * boxes.h, boxes.w / boxes.h
    per_image_counts = boxes.img_id.value_counts().reindex(all_ids, fill_value=0).to_dict()
    print(f"Total annotated cars: {len(boxes):,} | classes: {sorted(boxes.cls.unique())}")
    return boxes, per_image_counts, degenerate


def cars_per_image_summary(per_image_counts, train_ids, test_ids):
    counts = pd.Series(per_image_counts)
    return pd.DataFrame({name: dict(images=len(s), total_cars=s.sum(), min=s.min(), max=s.max(),
                                    mean=round(s.mean(), 2), median=s.median(), std=round(s.std(), 2))
                         for name, s in (("all", counts), ("train", counts[train_ids]),
                                         ("test", counts[test_ids]))}).T


def plot_cars_per_image(per_image_counts, train_ids, test_ids):
    counts = pd.Series(per_image_counts)
    _, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(counts, bins=40, color="#2b8cbe", edgecolor="white")
    ax[0].axvline(counts.mean(), color="red", ls="--", label=f"mean={counts.mean():.1f}")
    ax[0].set(title=f"Cars per image (all {len(counts)} images)", xlabel="# cars in image", ylabel="# images")
    ax[1].hist(counts[train_ids], bins=40, alpha=0.6, label="train", color="#2b8cbe")
    ax[1].hist(counts[test_ids], bins=40, alpha=0.6, label="test", color="#e34a33")
    ax[1].set(title="Cars per image — train vs test", xlabel="# cars in image", ylabel="# images")
    ax[0].legend(), ax[1].legend()
    plt.tight_layout()
    plt.show()


def _hists(series, titles, colors):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, s, t, c in zip(axes, series, titles, colors):
        ax.hist(s, bins=60, color=c, edgecolor="white")
        ax.set(title=t, ylabel="# boxes")
    return axes


def plot_box_wh(boxes):
    _hists([boxes.w, boxes.h], ["Box width (px)", "Box height (px)"], ["#3182bd", "#e6550d"])
    plt.tight_layout()
    plt.show()


def plot_box_sizes(boxes):
    axes = _hists([np.sqrt(boxes.area), boxes.aspect], ["Box size sqrt(area) (px)", "Aspect ratio (w/h)"],
                  ["#756bb1", "#31a354"])
    axes[1].axvline(boxes.aspect.median(), color="red", ls="--", label=f"median={boxes.aspect.median():.2f}")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_sample_gt(boxes, train_ids):
    sid = train_ids[0]
    sb = boxes[boxes.img_id == sid]
    show_image_with_boxes(Image.open(img_path(sid)), sb[["x1", "y1", "x2", "y2"]].to_numpy(),
                          title=f"{sid}  —  {len(sb)} cars (ground truth)")


def conversion_demo(boxes):
    """Devkit -> normalised YOLO conversion of one box."""
    r = boxes.iloc[0]
    cx, cy, w, h = xyxy_to_yolo(r.x1, r.y1, r.x2, r.y2, IMG_W, IMG_H)
    print(f"CARPK : {r.x1} {r.y1} {r.x2} {r.y2} {r.cls}")
    print(f"YOLO  : 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    assert all(0 <= v <= 1 for v in (cx, cy, w, h)), "normalized values out of range"
    return cx, cy, w, h


def train_val_partition(train_ids, test_ids, per_image_counts, val_frac=0.10, seed=42):
    """Same train/val carve as the Phase 2 YOLO dataset."""
    counts = pd.Series(per_image_counts)
    fit, val = split_train_val(train_ids, val_frac, seed)
    return pd.DataFrame({"split": ["train (fit)", "val", "test"],
                         "#images": [len(fit), len(val), len(test_ids)],
                         "#cars": [counts[ids].sum() for ids in (fit, val, test_ids)],
                         "source": ["official train (90%)", "official train (10%)", "official test (clean)"]})
