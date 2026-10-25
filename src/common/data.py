"""Reading CARPK-style devkits: ``Images/<id>.<ext>``, ``Annotations/<id>.txt``
(one ``x1 y1 x2 y2 class`` per line) and ``ImageSets/{train,test}.txt``."""
import random
from pathlib import Path

from .config import ANN, IMAGES, SETS


def find_image(images_dir, img_id) -> Path:
    """Image file for ``img_id`` (CARPK is .png, PUCPR+ is .jpg)."""
    return next(Path(images_dir).glob(f"{img_id}.*"))


def img_path(img_id) -> Path:
    return find_image(IMAGES, img_id)


def load_split_ids(split, sets_dir=SETS) -> list:
    return (Path(sets_dir) / f"{split}.txt").read_text().split()


def read_annotation(path) -> list:
    """``(x1, y1, x2, y2, cls)`` tuples of one annotation file."""
    return [tuple(map(int, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]


def gt_count(img_id) -> int:
    return len(read_annotation(ANN / f"{img_id}.txt"))


def split_train_val(train_ids, val_fraction=0.1, seed=42):
    """Deterministic train/val carve of the official train ids (used by Phase 1 and 2)."""
    ids = sorted(train_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def link(src, dst) -> None:
    """Symlink ``dst`` -> ``src``, replacing any existing file."""
    dst = Path(dst)
    dst.unlink(missing_ok=True)
    dst.symlink_to(Path(src).resolve())
