import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ANN, DATA, IMAGES, SETS
from common.data import find_image, link, load_split_ids, read_annotation

SOURCES = {"carpk": Path("datasets/CARPK_devkit/data"), "pucpr": Path("datasets/PUCPR+_devkit/data")}


def combine_datasets() -> pd.DataFrame:
    """Write ``datasets/combined_devkit/data`` and return images / cars per source and split."""
    shutil.rmtree(DATA, ignore_errors=True)
    for d in (ANN, IMAGES, SETS):
        d.mkdir(parents=True)
    records, merged = [], {"train": [], "test": []}
    for name, src in SOURCES.items():
        n_dropped = 0
        for split in merged:
            ids, n_cars = load_split_ids(split, src / "ImageSets"), 0
            for img_id in ids:
                new_id = f"{name}__{img_id}"
                boxes = read_annotation(src / "Annotations" / f"{img_id}.txt")
                kept = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
                (ANN / f"{new_id}.txt").write_text("".join(" ".join(map(str, b)) + "\n" for b in kept))
                img = find_image(src / "Images", img_id)
                link(img, IMAGES / f"{new_id}{img.suffix}")
                merged[split].append(new_id)
                n_dropped += len(boxes) - len(kept)
                n_cars += len(kept)
            records.append(dict(source=name, split=split, images=len(ids), cars=n_cars))
        if n_dropped:
            print(f"  {name}: dropped {n_dropped} degenerate boxes (zero width/height)")
    for split, ids in merged.items():
        (SETS / f"{split}.txt").write_text("\n".join(sorted(ids)) + "\n")

    summary = pd.DataFrame(records)
    total = summary.groupby("split")[["images", "cars"]].sum()
    print(f"Final combined devkit written to: {DATA}")
    for split in merged:
        print(f"  {split:5s}: {total.loc[split, 'images']} images / {total.loc[split, 'cars']} cars")
    return summary


if __name__ == "__main__":
    print(combine_datasets())
