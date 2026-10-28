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
    """(x1, y1, x2, y2, cls) tuples of one annotation file."""
    return [tuple(map(int, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]


def gt_count(img_id) -> int:
    return len(read_annotation(ANN / f"{img_id}.txt"))


def lot_of(img_id) -> str:
    source, _, rest = img_id.partition("__")
    parts = rest.split("_")
    return f"carpk:{parts[1]}" if source == "carpk" and len(parts) >= 3 else source


def split_train_val(train_ids, val_fraction=0.1, seed=42, val_lots=()):
    if val_lots:
        held = {i for i in train_ids if lot_of(i) in set(val_lots)}
        assert held, f"no train ids in lots {val_lots}"
        single = [i for i in train_ids if i not in held and not lot_of(i).startswith("carpk:")]
        val = held | set(split_train_val(single, val_fraction, seed)[1] if single else [])
        return sorted(set(train_ids) - val), sorted(val)
    ids = sorted(train_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def dataset_dir(data=DATA_YAML) -> Path:
    return (ROOT / data).resolve().parent


def link(src, dst) -> None:
    dst = Path(dst)
    dst.unlink(missing_ok=True)
    dst.symlink_to(Path(src).resolve())
