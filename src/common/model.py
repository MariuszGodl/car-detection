"""Pretrained checkpoints and the zero-shot (COCO 'car' class) predictor."""
import os
from functools import cache
from pathlib import Path

from ultralytics import YOLO

from .config import CAR_COCO_ID, CONF, DEFAULT_WEIGHTS, IMGSZ, MAX_DET


def ensure_weights(weights=DEFAULT_WEIGHTS) -> str:
    """Return ``weights`` as a path, downloading an official checkpoint first if it is missing
    (``yolo11n.pt`` from the Hugging Face hub with ``HF_TOKEN`` from ``.env``, others from Ultralytics' GitHub)."""
    path = Path(weights)
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "yolo11n.pt":
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download

        load_dotenv()
        cached = hf_hub_download("Ultralytics/YOLO11", "yolo11n.pt", token=os.getenv("HF_TOKEN"))
        path.write_bytes(Path(cached).read_bytes())
    else:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

        if path.name not in GITHUB_ASSETS_NAMES:
            raise FileNotFoundError(f"{path} does not exist and is not an official Ultralytics checkpoint")
        attempt_download_asset(path)
    return str(path)


@cache
def load_model(weights=DEFAULT_WEIGHTS) -> YOLO:
    """The model for ``weights``, loaded once per process."""
    return YOLO(ensure_weights(weights))


def predict_cars(model, source, device=0):
    """COCO 'car' detections of the un-fine-tuned model on one image."""
    return model.predict(source, classes=[CAR_COCO_ID], imgsz=IMGSZ, conf=CONF, max_det=MAX_DET,
                         verbose=False, device=device)[0]
