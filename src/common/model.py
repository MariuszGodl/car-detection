"""Model loading (official checkpoints downloaded if missing) and zero-shot car prediction."""
import os
from pathlib import Path

from ultralytics import YOLO

from .config import CAR_COCO_ID, CONF, DEFAULT_WEIGHTS, IMGSZ, MAX_DET


def ensure_weights(weights=DEFAULT_WEIGHTS) -> str:
    """Return ``weights``; if the file is missing, download the official checkpoint of that name:
    ``yolo11n.pt`` from Hugging Face (``HF_TOKEN`` in ``.env``), any other (e.g. ``yolo26n.pt``) from the
    Ultralytics release assets. Anything else (a run's ``best.pt``) must already exist."""
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


_models = {}


def load_model(weights=DEFAULT_WEIGHTS):
    """Cached YOLO model (loaded once per weights path)."""
    if str(weights) not in _models:
        _models[str(weights)] = YOLO(ensure_weights(weights))
    return _models[str(weights)]


def predict_cars(model, source):
    """COCO 'car' detections of the un-fine-tuned model on one image."""
    return model.predict(source, classes=[CAR_COCO_ID], imgsz=IMGSZ, conf=CONF, max_det=MAX_DET,
                         verbose=False, device=0)[0]
