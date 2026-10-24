"""Combine the CARPK and PUCPR+ devkits into one final dataset (Phase 1, section 3).

Both source devkits share exactly the same layout and annotation format
(``x1 y1 x2 y2 class`` in absolute pixels, class ``1 = car``, native 1280x720),
so they can be merged into a single *final* devkit that the rest of the project
treats as one dataset.  Image ids are namespaced with a ``carpk__`` / ``pucpr__``
prefix so the two sources never clash and their origin stays traceable.
"""
import shutil
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1])  # make `common` importable
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd

from common.config import DATA as COMBINED
from common.data import (find_image, is_valid_box, load_split_ids,
                         parse_annotation_lines, place_file)

# Source devkits -> their data directory (same on-disk layout for both).
SOURCES = {
    "carpk": Path("datasets/CARPK_devkit/data"),
    "pucpr": Path("datasets/PUCPR+_devkit/data"),
}

# Splits we carry over.  Right now we simply reuse the dataset authors' official
# train/test split (their ``ImageSets/*.txt`` files) -- see ``author_split``.
SPLITS = ("train", "test")


def _clean_annotation(text):
    """Return ``(kept_text, n_dropped)`` for one annotation file's contents.

    Degenerate boxes (``x1 == x2`` or ``y1 == y2`` -> zero width/height) are not
    valid cars: they cannot be normalised to a YOLO box and break aspect-ratio
    math.  We drop them here, at dataset-build time, so the combined devkit that
    every downstream step reads is already clean rather than relying on each
    consumer to filter them out.
    """
    kept, n_dropped = [], 0
    for box in parse_annotation_lines(text):
        if is_valid_box(*box[:4]):
            kept.append(" ".join(map(str, box)))
        else:
            n_dropped += 1
    return ("\n".join(kept) + "\n" if kept else ""), n_dropped


def author_split(data_dir):
    """Return ``{split: [image_ids]}`` for one source devkit.

    This follows the **dataset authors' official** train/test split (their
    ``ImageSets/*.txt`` files).  It is deliberately the single place that
    decides which image goes to which split, so a different split (e.g. a
    custom random carve) can be plugged in here in the future without touching
    the merge logic.
    """
    return {split: load_split_ids(split, data_dir / "ImageSets") for split in SPLITS}


def _merge_source(name, data_dir, ann_dir, img_dir, merged, link_images):
    """Copy/link one source devkit into the combined dirs.

    Ids are namespaced with ``<name>__`` and appended to ``merged`` per split.
    Returns a list of per-split summary records for this source.
    """
    records = []
    n_dropped_total = 0
    for split, ids in author_split(data_dir).items():
        n_cars = 0
        for img_id in ids:
            new_id = f"{name}__{img_id}"

            ann_src = data_dir / "Annotations" / f"{img_id}.txt"
            cleaned, n_dropped = _clean_annotation(ann_src.read_text())
            (ann_dir / f"{new_id}.txt").write_text(cleaned)
            n_dropped_total += n_dropped
            n_cars += len([ln for ln in cleaned.splitlines() if ln.strip()])

            img_src = find_image(data_dir / "Images", img_id)
            place_file(img_src, img_dir / f"{new_id}{img_src.suffix}", link_images)

            merged.setdefault(split, []).append(new_id)

        records.append(dict(source=name, split=split,
                            images=len(ids), cars=n_cars))
    if n_dropped_total:
        print(f"  {name}: dropped {n_dropped_total} degenerate boxes "
              f"(zero width/height)")
    return records


def combine_datasets(link_images=True, overwrite=True):
    """Merge every source devkit into the single final ``combined_devkit``.

    Produces ``datasets/combined_devkit/data`` with the usual ``Annotations/``,
    ``Images/`` and ``ImageSets/{train,test}.txt`` sub-folders, where each id is
    prefixed by its source (e.g. ``carpk__20160331_NTU_00001``).

    Returns a per-source summary ``DataFrame`` (images / cars, by split).
    """

    ann_dir = COMBINED / "Annotations"
    img_dir = COMBINED / "Images"
    set_dir = COMBINED / "ImageSets"
    if overwrite and COMBINED.exists():
        shutil.rmtree(COMBINED)
    for d in (ann_dir, img_dir, set_dir):
        d.mkdir(parents=True, exist_ok=True)

    merged = {split: [] for split in SPLITS}
    records = []
    for name, data_dir in SOURCES.items():
        records += _merge_source(name, data_dir, ann_dir, img_dir,
                                 merged, link_images)

    for split, ids in merged.items():
        (set_dir / f"{split}.txt").write_text("\n".join(sorted(ids)) + "\n")

    summary = pd.DataFrame.from_records(records)
    total = summary.groupby("split")[["images", "cars"]].sum()
    print(f"Final combined devkit written to: {COMBINED}")
    print(f"  train: {total.loc['train', 'images']} images / "
          f"{total.loc['train', 'cars']} cars")
    print(f"  test : {total.loc['test', 'images']} images / "
          f"{total.loc['test', 'cars']} cars")
    return summary


if __name__ == "__main__":
    print(combine_datasets())
