"""CARPK data loading, statistics and exploration (Phase 1, section 3)."""
import numpy as np
import pandas as pd
from PIL import Image

from config import ANN, SETS, IMG_W, IMG_H, img_path


def load_splits():
    """Return the official (train_ids, test_ids, all_ids) and sanity-check them."""
    train_ids = SETS.joinpath("train.txt").read_text().split()
    test_ids = SETS.joinpath("test.txt").read_text().split()
    all_ids = sorted(p.stem for p in ANN.glob("*.txt"))

    print(f"#images total : {len(all_ids)}")
    print(f"#train (official): {len(train_ids)}")
    print(f"#test  (official): {len(test_ids)}")

    assert set(train_ids).isdisjoint(test_ids), "train/test overlap!"
    assert set(train_ids) | set(test_ids) == set(all_ids), "splits do not cover all images"
    print("Splits are disjoint and cover every image.")
    return train_ids, test_ids, all_ids


def check_sample_resolution(all_ids):
    """Confirm a sample image matches the expected native resolution."""
    sample = Image.open(img_path(all_ids[0]))
    print("Sample image:", all_ids[0], "size", sample.size, "mode", sample.mode)
    assert sample.size == (IMG_W, IMG_H)
    return sample.size


def parse_annotations(all_ids, train_ids):
    """Parse every annotation file into per-box tables and per-image counts.

    Each line: ``x1 y1 x2 y2 class`` (absolute pixels, class 1 = car).

    A handful of boxes in the PUCPR+ data are *degenerate* (``x1 == x2`` or
    ``y1 == y2``), i.e. they have zero width or height. These are not valid
    objects, cannot be normalised to a YOLO box, and would crash aspect-ratio
    math, so they are excluded from ``boxes``. They are returned separately as
    ``degenerate`` so the drop is explicit and auditable rather than silent.

    Returns
    -------
    boxes : DataFrame
        One row per valid box.
    per_image_counts : dict
        img_id -> number of valid boxes.
    degenerate : DataFrame
        The zero width/height boxes that were dropped (may be empty).
    """
    train_set = set(train_ids)
    records, degenerate, per_image_counts = [], [], {}

    for img_id in all_ids:
        n = 0
        for line in ANN.joinpath(f"{img_id}.txt").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            x1, y1, x2, y2, cls = map(int, line.split())
            w, h = x2 - x1, y2 - y1
            split = "train" if img_id in train_set else "test"
            records.append(dict(img_id=img_id, split=split,
                                x1=x1, y1=y1, x2=x2, y2=y2, cls=cls,
                                w=w, h=h, area=w * h, aspect=w / h))
            n += 1
        per_image_counts[img_id] = n

    boxes = pd.DataFrame.from_records(records)
    degenerate = pd.DataFrame.from_records(
        degenerate, columns=["img_id", "split", "x1", "y1", "x2", "y2", "cls", "w", "h"]
    )
    print(f"Total annotated cars: {len(boxes):,}")
    print("Classes present:", sorted(boxes.cls.unique()))
    return boxes, per_image_counts, degenerate


def cars_per_image_summary(per_image_counts, train_ids, test_ids):
    """Summary table of the cars-per-image distribution for all/train/test."""
    counts = pd.Series(per_image_counts)

    def describe(s):
        return dict(images=len(s), total_cars=int(s.sum()), min=int(s.min()),
                    max=int(s.max()), mean=round(s.mean(), 2),
                    median=float(s.median()), std=round(s.std(), 2))

    summary = pd.DataFrame({
        "all": describe(counts),
        "train": describe(counts[list(train_ids)]),
        "test": describe(counts[list(test_ids)]),
    }).T
    return summary


def plot_cars_per_image(per_image_counts, train_ids, test_ids):
    import matplotlib.pyplot as plt
    counts = pd.Series(per_image_counts)
    train_counts = counts[list(train_ids)]
    test_counts = counts[list(test_ids)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(counts.values, bins=40, color="#2b8cbe", edgecolor="white")
    ax[0].axvline(counts.mean(), color="red", ls="--", label=f"mean={counts.mean():.1f}")
    ax[0].set_title(f"Cars per image (all {len(counts)} images)")
    ax[0].set_xlabel("# cars in image"); ax[0].set_ylabel("# images"); ax[0].legend()

    ax[1].hist(train_counts.values, bins=40, alpha=0.6, label="train", color="#2b8cbe")
    ax[1].hist(test_counts.values, bins=40, alpha=0.6, label="test", color="#e34a33")
    ax[1].set_title("Cars per image — train vs test")
    ax[1].set_xlabel("# cars in image"); ax[1].set_ylabel("# images"); ax[1].legend()
    plt.tight_layout()
    plt.show()


def box_size_stats(boxes):
    """Box geometry stats and COCO small/medium/large size buckets."""
    img_area = IMG_W * IMG_H
    boxes["rel_area"] = boxes["area"] / img_area
    boxes["diag"] = np.sqrt(boxes["w"] ** 2 + boxes["h"] ** 2)

    size_stats = boxes[["w", "h", "area", "rel_area", "aspect"]].describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)

    small = (boxes["area"] < 32 * 32).mean()
    med = ((boxes["area"] >= 32 * 32) & (boxes["area"] < 96 * 96)).mean()
    large = (boxes["area"] >= 96 * 96).mean()
    print("COCO size buckets of CARPK cars:")
    print(f"  small  (<32x32):      {small:6.1%}")
    print(f"  medium (32x32-96x96): {med:6.1%}")
    print(f"  large  (>=96x96):     {large:6.1%}")
    print(f"\nBoxes occupy on average {boxes['rel_area'].mean():.4%} of the image area "
          f"-> small, dense objects.")
    return size_stats


def plot_box_wh(boxes):
    """Plot width and height distributions of the bounding boxes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(boxes["w"], bins=60, color="#3182bd", edgecolor="white")
    ax[0].set_title("Box width (px)")
    ax[0].set_xlabel("w"); ax[0].set_ylabel("# boxes")

    ax[1].hist(boxes["h"], bins=60, color="#e6550d", edgecolor="white")
    ax[1].set_title("Box height (px)")
    ax[1].set_xlabel("h"); ax[1].set_ylabel("# boxes")
    plt.tight_layout()
    plt.show()


def plot_box_sizes(boxes):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.sqrt(boxes["area"]), bins=60, color="#756bb1", edgecolor="white")
    ax[0].set_title("Box size  sqrt(area) (px)")
    ax[0].set_xlabel("sqrt(area)"); ax[0].set_ylabel("# boxes")

    ax[1].hist(boxes["aspect"], bins=60, color="#31a354", edgecolor="white")
    ax[1].axvline(boxes["aspect"].median(), color="red", ls="--",
                  label=f"median={boxes['aspect'].median():.2f}")
    ax[1].set_title("Aspect ratio (w/h)")
    ax[1].set_xlabel("w / h"); ax[1].set_ylabel("# boxes"); ax[1].legend()
    plt.tight_layout()
    plt.show()


def plot_sample_gt(boxes, train_ids):
    """Show one training image with its ground-truth boxes (green)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    sid = train_ids[0]
    img = Image.open(img_path(sid))
    sb = boxes[boxes.img_id == sid]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    for _, r in sb.iterrows():
        ax.add_patch(patches.Rectangle((r.x1, r.y1), r.w, r.h,
                     fill=False, edgecolor="lime", linewidth=1))
    ax.set_title(f"{sid}  —  {len(sb)} cars (ground truth)")
    ax.axis("off")
    plt.show()


def conversion_demo(boxes):
    """Demonstrate the CARPK -> normalized YOLO conversion on one box."""
    r = boxes.iloc[0]
    cx = (r.x1 + r.x2) / (2 * IMG_W)
    cy = (r.y1 + r.y2) / (2 * IMG_H)
    nw = (r.x2 - r.x1) / IMG_W
    nh = (r.y2 - r.y1) / IMG_H
    print(f"CARPK : {r.x1} {r.y1} {r.x2} {r.y2} {r.cls}")
    print(f"YOLO  : 0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    assert all(0 <= v <= 1 for v in (cx, cy, nw, nh)), "normalized values out of range"
    print("All normalized values are within [0, 1].")
    return cx, cy, nw, nh


def train_val_partition(train_ids, test_ids, per_image_counts, val_frac=0.10, seed=42):
    """Reproducible train/val carve from the official training ids."""
    counts = pd.Series(per_image_counts)
    rng = np.random.default_rng(seed)
    tr = np.array(sorted(train_ids))
    perm = rng.permutation(len(tr))
    val_n = round(val_frac * len(tr))
    val_ids = sorted(tr[perm[:val_n]])
    train_fit = sorted(tr[perm[val_n:]])

    partition = pd.DataFrame({
        "split": ["train (fit)", "val", "test"],
        "#images": [len(train_fit), len(val_ids), len(test_ids)],
        "#cars": [int(counts[train_fit].sum()), int(counts[val_ids].sum()),
                  int(counts[list(test_ids)].sum())],
        "source": ["official train (90%)", "official train (10%)", "official test (clean)"],
    })
    return partition
