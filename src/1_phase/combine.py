"""Merge CARPK and PUCPR+ into one devkit (``datasets/combined_devkit/data``).

Both use the same layout and ``x1 y1 x2 y2 class`` pixel labels. Image ids get a ``<source>__`` prefix,
images are symlinked, zero width/height boxes are dropped and the official train / test splits are kept.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ANN, DATA, IMAGES, SETS
from common.data import find_image, link, load_split_ids, read_annotation

SOURCES = {"carpk": Path("datasets/CARPK_devkit/data"), "pucpr": Path("datasets/PUCPR+_devkit/data")}


def combine_datasets() -> pd.DataFrame:
    """Rebuild the combined devkit; return one summary row per source and official split."""
    if DATA.exists():
        shutil.rmtree(DATA)
    for directory in (ANN, IMAGES, SETS):
        directory.mkdir(parents=True)

    records = []
    combined_ids = {"train": [], "test": []}
    for source_name, source_dir in SOURCES.items():
        dropped_boxes = 0
        for split in ("train", "test"):
            image_ids = load_split_ids(split, source_dir / "ImageSets")
            car_count = 0
            for image_id in image_ids:
                combined_id = f"{source_name}__{image_id}"
                boxes = read_annotation(source_dir / "Annotations" / f"{image_id}.txt")
                label_lines = []
                for x1, y1, x2, y2, car_class in boxes:
                    if x2 <= x1 or y2 <= y1:
                        dropped_boxes += 1
                        continue
                    label_lines.append(f"{x1} {y1} {x2} {y2} {car_class}\n")

                (ANN / f"{combined_id}.txt").write_text("".join(label_lines))
                image = find_image(source_dir / "Images", image_id)
                link(image, IMAGES / f"{combined_id}{image.suffix}")
                combined_ids[split].append(combined_id)
                car_count += len(label_lines)

            records.append({
                "source": source_name,
                "split": split,
                "images": len(image_ids),
                "cars": car_count,
            })
        if dropped_boxes:
            print(f"  {source_name}: dropped {dropped_boxes} degenerate boxes (zero width/height)")

    for split, image_ids in combined_ids.items():
        (SETS / f"{split}.txt").write_text("\n".join(sorted(image_ids)) + "\n")

    summary = pd.DataFrame(records)
    total = summary.groupby("split")[["images", "cars"]].sum()
    print(f"Final combined devkit written to: {DATA}")
    for split in combined_ids:
        print(f"  {split:5s}: {total.loc[split, 'images']} images / {total.loc[split, 'cars']} cars")
    return summary


if __name__ == "__main__":
    print(combine_datasets())
