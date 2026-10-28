"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"   # absolute so DDP workers resolve the same dir

DATA = Path("datasets/combined_devkit/data")   # CARPK + PUCPR+, built by src/1_phase/combine.py
IMAGES, ANN, SETS = DATA / "Images", DATA / "Annotations", DATA / "ImageSets"

YOLO_DIR = Path("datasets/yolo")               # built by src/2_phase/carpk_to_yolo.py
DATA_YAML = YOLO_DIR / "carpk.yaml"
TRAIN_EVAL_DIR = YOLO_DIR / "train_eval"       # fixed train subset scored every epoch (common/train_eval.py)
TRAIN_EVAL_IMAGES = 200

DEFAULT_WEIGHTS = "models/yolo11n.pt"
YOLO26_WEIGHTS = "models/yolo26n.pt"   # NMS-free head: evaluate with nms=False
MODEL_CFG = "yolo11n.yaml"

IMG_W, IMG_H = 1280, 720   # native resolution
CAR_COCO_ID = 2            # 'car' in COCO
IMGSZ, CONF = 1280, 0.25
MAX_DET = 1000             # Ultralytics default is 300 < densest image (331 cars) -> would clip counts

DEVICES = (0, 1)           # 2x RTX 4080 SUPER -> DDP
BATCH_PER_GPU = 16         # ~11 GB VRAM per GPU at imgsz 1280
# PATIENCE = 5 is the baseline (stop after 5 epochs without a better val fitness). Phase 3 runs pass
# patience=0 for the full schedule: the whole lr decay and the last `close_mosaic` (10) epochs without mosaic.
EPOCHS, PATIENCE, SEED = 100, 5, 42
