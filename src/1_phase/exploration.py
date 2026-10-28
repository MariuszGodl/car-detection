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
    train_ids = load_split_ids("train")
    test_ids = load_split_ids("test")
    assert len(train_ids) + len(test_ids) == len(sorted(p.stem for p in ANN.glob("*.txt")))
    all_ids = sorted(p.stem for p in ANN.glob("*.txt")) 
    print(f"#images total: {len(all_ids)} | #train: {len(train_ids)} | #test: {len(test_ids)}")
    assert set(train_ids).isdisjoint(test_ids), "train/test overlap!"
    assert set(train_ids) | set(test_ids) == set(all_ids), "splits do not cover all images"
    return train_ids, test_ids, all_ids


def check_sample_resolution(all_ids):
    with Image.open(img_path(all_ids[0])) as image:
        size = image.size
    print("Sample image:", all_ids[0], "size", size)
    assert size == (IMG_W, IMG_H)
    return size


def parse_annotations(all_ids, train_ids):
    """Build the box table, exclude invalid boxes, then count cars in each image."""
    train_set = set(train_ids)
    rows = []
    for image_id in all_ids:
        split = "train" if image_id in train_set else "test"
        for x1, y1, x2, y2, car_class in read_annotation(ANN / f"{image_id}.txt"):
            rows.append({
                "img_id": image_id,
                "split": split,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cls": car_class,
                "w": x2 - x1,
                "h": y2 - y1,
            })

    annotations = pd.DataFrame(rows)
    valid = (annotations["w"] > 0) & (annotations["h"] > 0)
    boxes = annotations[valid].copy()
    degenerate = annotations[~valid]
    boxes["area"] = boxes["w"] * boxes["h"]
    boxes["aspect"] = boxes["w"] / boxes["h"]
    per_image_counts = boxes["img_id"].value_counts().reindex(all_ids, fill_value=0).to_dict()
    print(f"Total annotated cars: {len(boxes):,} | classes: {sorted(boxes['cls'].unique())}")
    return boxes, per_image_counts, degenerate


def cars_per_image_summary(per_image_counts, train_ids, test_ids):
    counts = pd.Series(per_image_counts)
    splits = {"all": counts, "train": counts[train_ids], "test": counts[test_ids]}
    summary = {}
    for split, car_counts in splits.items():
        summary[split] = {
            "images": len(car_counts),
            "total_cars": car_counts.sum(),
            "min": car_counts.min(),
            "max": car_counts.max(),
            "mean": round(car_counts.mean(), 2),
            "median": car_counts.median(),
            "std": round(car_counts.std(), 2),
        }
    return pd.DataFrame(summary).T


def plot_cars_per_image(per_image_counts, train_ids, test_ids):
    counts = pd.Series(per_image_counts)
    _, (all_ax, split_ax) = plt.subplots(1, 2, figsize=(13, 4))
    all_ax.hist(counts, bins=40, color="#2b8cbe", edgecolor="white")
    all_ax.axvline(counts.mean(), color="red", ls="--", label=f"mean={counts.mean():.1f}")
    all_ax.set(title=f"Cars per image (all {len(counts)} images)", xlabel="# cars in image", ylabel="# images")
    split_ax.hist(counts[train_ids], bins=40, alpha=0.6, label="train", color="#2b8cbe")
    split_ax.hist(counts[test_ids], bins=40, alpha=0.6, label="test", color="#e34a33")
    split_ax.set(title="Cars per image — train vs test", xlabel="# cars in image", ylabel="# images")
    all_ax.legend()
    split_ax.legend()
    plt.tight_layout()
    plt.show()


def plot_box_wh(boxes):
    _, (width_ax, height_ax) = plt.subplots(1, 2, figsize=(12, 4))
    width_ax.hist(boxes["w"], bins=60, color="#3182bd", edgecolor="white")
    width_ax.set(title="Box width (px)", ylabel="# boxes")
    height_ax.hist(boxes["h"], bins=60, color="#e6550d", edgecolor="white")
    height_ax.set(title="Box height (px)", ylabel="# boxes")
    plt.tight_layout()
    plt.show()


def plot_box_sizes(boxes):
    _, (size_ax, aspect_ax) = plt.subplots(1, 2, figsize=(12, 4))
    size_ax.hist(np.sqrt(boxes["area"]), bins=60, color="#756bb1", edgecolor="white")
    size_ax.set(title="Box size sqrt(area) (px)", ylabel="# boxes")
    aspect_ax.hist(boxes["aspect"], bins=60, color="#31a354", edgecolor="white")
    aspect_ax.set(title="Aspect ratio (w/h)", ylabel="# boxes")
    median_aspect = boxes["aspect"].median()
    aspect_ax.axvline(median_aspect, color="red", ls="--", label=f"median={median_aspect:.2f}")
    aspect_ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_gt(boxes, train_ids):
    image_id = train_ids[0]
    image_boxes = boxes[boxes["img_id"] == image_id]
    coordinates = image_boxes[["x1", "y1", "x2", "y2"]].to_numpy()
    with Image.open(img_path(image_id)) as image:
        show_image_with_boxes(
            image, coordinates, title=f"{image_id}  —  {len(image_boxes)} cars (ground truth)"
        )


def conversion_demo(boxes):
    """Devkit -> normalised YOLO conversion of one box."""
    box = boxes.iloc[0]
    center_x, center_y, width, height = xyxy_to_yolo(
        box.x1, box.y1, box.x2, box.y2, IMG_W, IMG_H
    )
    print(f"CARPK : {box.x1} {box.y1} {box.x2} {box.y2} {box.cls}")
    print(f"YOLO  : 0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    normalized_box = (center_x, center_y, width, height)
    assert all(0 <= value <= 1 for value in normalized_box), "normalized values out of range"
    return normalized_box


def train_val_partition(train_ids, test_ids, per_image_counts, val_frac=0.10, seed=42):
    """Show the seeded 90/10 training split used by Phase 2; leave the official test split alone."""
    counts = pd.Series(per_image_counts)
    fit_ids, val_ids = split_train_val(train_ids, val_frac, seed)
    return pd.DataFrame({
        "split": ["train (fit)", "val", "test"],
        "#images": [len(fit_ids), len(val_ids), len(test_ids)],
        "#cars": [counts[fit_ids].sum(), counts[val_ids].sum(), counts[test_ids].sum()],
        "source": ["official train (90%)", "official train (10%)", "official test (clean)"],
    })
