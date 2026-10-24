"""YOLO11n loading (downloaded from Hugging Face if missing) and zero-shot car prediction."""
import os
from pathlib import Path

from ultralytics import YOLO

from .config import CAR_COCO_ID, CONF, DEFAULT_WEIGHTS, IMGSZ


def ensure_weights(weights=DEFAULT_WEIGHTS) -> str:
    """Return ``weights``, downloading YOLO11n from Hugging Face (``HF_TOKEN`` in ``.env``) if missing."""
    if not Path(weights).exists():
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download

        load_dotenv()
        cached = hf_hub_download("Ultralytics/YOLO11", "yolo11n.pt", token=os.getenv("HF_TOKEN"))
        Path(weights).parent.mkdir(parents=True, exist_ok=True)
        Path(weights).write_bytes(Path(cached).read_bytes())
    return str(weights)


_models = {}


def load_model(weights=DEFAULT_WEIGHTS):
    """Cached YOLO model (loaded once per weights path)."""
    if str(weights) not in _models:
        _models[str(weights)] = YOLO(ensure_weights(weights))
    return _models[str(weights)]


def predict_cars(model, source):
    """COCO 'car' detections of the un-fine-tuned model on one image."""
    return model.predict(source, classes=[CAR_COCO_ID], imgsz=IMGSZ, conf=CONF, verbose=False, device=0)[0]
