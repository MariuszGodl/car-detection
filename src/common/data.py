"""Reading the combined devkit: image ids, annotations, splits, sources and parking lots.

Image ids look like ``<source>__<original id>``, e.g. ``carpk__20160331_NTU_00050`` (CARPK, lot NTU)
or ``pucpr__<id>`` (PUCPR+, a single fixed camera).
"""
import random
from pathlib import Path

from .config import ANN, DATA_YAML, IMAGES, ROOT, SETS


def find_image(images_dir, img_id) -> Path:
    return next(Path(images_dir).glob(f"{img_id}.*"))


def img_path(img_id) -> Path:
    return find_image(IMAGES, img_id)


def load_split_ids(split, sets_dir=SETS) -> list:
    return (Path(sets_dir) / f"{split}.txt").read_text().split()


def read_annotation(path) -> list:
    """(x1, y1, x2, y2, cls) pixel boxes of one annotation file."""
    return [tuple(map(int, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]


def gt_count(img_id) -> int:
    return len(read_annotation(ANN / f"{img_id}.txt"))


def source_of(img_id) -> str:
    """``carpk`` or ``pucpr``."""
    return img_id.split("__", 1)[0]


def lot_of(img_id) -> str:
    """Parking lot: ``carpk:<LOT>`` (e.g. ``carpk:NTU``) for CARPK, the source name otherwise."""
    source, _, rest = img_id.partition("__")
    parts = rest.split("_")   # <date>_<LOT>_<number>
    return f"carpk:{parts[1]}" if source == "carpk" and len(parts) >= 3 else source


def random_split(ids, val_fraction=0.1, seed=42):
    """(train, val): a seeded random ``val_fraction`` of ``ids`` (at least one image) goes to val."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def split_train_val(train_ids, val_fraction=0.1, seed=42, val_lots=()):
    """Carve (train, val) out of the official train ids.

    * default: a random ``val_fraction`` of all images,
    * ``val_lots``: every image of those CARPK lots, plus a random ``val_fraction`` of the images that
      belong to no CARPK lot (PUCPR+), so val still covers both sources. The other lots stay in train.
    """
    if not val_lots:
        return random_split(train_ids, val_fraction, seed)
    held_out = {i for i in train_ids if lot_of(i) in val_lots}
    assert held_out, f"no train ids in lots {val_lots}"
    no_lot = [i for i in train_ids if i not in held_out and not lot_of(i).startswith("carpk:")]
    val = held_out | set(random_split(no_lot, val_fraction, seed)[1] if no_lot else [])
    return sorted(set(train_ids) - val), sorted(val)


def dataset_dir(data=DATA_YAML) -> Path:
    """Folder of a YOLO dataset, given its ``carpk.yaml``."""
    return (ROOT / data).resolve().parent


def link(src, dst) -> None:
    """(Re)create ``dst`` as a symlink to ``src``."""
    dst = Path(dst)
    dst.unlink(missing_ok=True)
    dst.symlink_to(Path(src).resolve())
