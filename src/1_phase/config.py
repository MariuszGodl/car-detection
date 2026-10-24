"""Shared paths and constants for the Phase 1 notebook."""
from pathlib import Path

# Final combined devkit (CARPK + PUCPR+), produced by src/1_phase/combine.py.
DATA = Path("datasets/combined_devkit/data")
IMAGES = DATA / "Images"
ANN = DATA / "Annotations"
SETS = DATA / "ImageSets"

IMG_W, IMG_H = 1280, 720   # native CARPK / PUCPR+ resolution

CAR_COCO_ID = 2            # 'car' in the COCO label set
IMGSZ, CONF = 1280, 0.25   # inference resolution / confidence threshold

# The combined devkit mixes image formats (CARPK .png + PUCPR+ .jpg), so image
# files must be resolved by id rather than assuming a single extension.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def img_path(img_id):
    """Return the image file for ``img_id`` in ``IMAGES`` (any known extension)."""
    for ext in _IMAGE_EXTS:
        p = IMAGES / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for id {img_id!r} in {IMAGES}")
