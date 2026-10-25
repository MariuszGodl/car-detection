"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"   # absolute so DDP workers resolve the same dir

DATA = Path("datasets/combined_devkit/data")   # CARPK + PUCPR+, built by src/1_phase/combine.py
IMAGES, ANN, SETS = DATA / "Images", DATA / "Annotations", DATA / "ImageSets"

YOLO_DIR = Path("datasets/yolo")               # built by src/2_phase/carpk_to_yolo.py
DATA_YAML = YOLO_DIR / "carpk.yaml"

DEFAULT_WEIGHTS = "models/yolo11n.pt"
MODEL_CFG = "yolo11n.yaml"

IMG_W, IMG_H = 1280, 720   # native resolution
CAR_COCO_ID = 2            # 'car' in COCO
IMGSZ, CONF = 1280, 0.25

DEVICES = (0, 1)           # 2x RTX 4080 SUPER -> DDP
BATCH_PER_GPU = 16         # ~11 GB VRAM per GPU at imgsz 1280
EPOCHS, PATIENCE, SEED = 100, 5, 42
