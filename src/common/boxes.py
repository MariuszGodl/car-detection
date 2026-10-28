"""Box conversions between devkit pixels ``x1 y1 x2 y2`` and YOLO's normalised ``cx cy w h``."""
from pathlib import Path


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Clamp a pixel box to the image and return normalised (cx, cy, w, h), or None if nothing is left."""
    x1, x2 = sorted((max(0, min(x1, img_w)), max(0, min(x2, img_w))))
    y1, y2 = sorted((max(0, min(y1, img_h)), max(0, min(y2, img_h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalised (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
    x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
    return x1, y1, x1 + w * img_w, y1 + h * img_h


def read_yolo_labels(path) -> list:
    """(cls, cx, cy, w, h) rows of a YOLO label file."""
    return [tuple(map(float, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]
