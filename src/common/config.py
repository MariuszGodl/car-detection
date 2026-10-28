"""Paths and settings shared by every phase."""
from pathlib import Path

# --- paths (data paths are relative to the repo root, the notebooks' working directory) ---
ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"                       # absolute so DDP workers resolve the same dir

DATA = Path("datasets/combined_devkit/data")   # CARPK + PUCPR+ devkit, built by src/1_phase/combine.py
IMAGES, ANN, SETS = DATA / "Images", DATA / "Annotations", DATA / "ImageSets"

YOLO_DIR = Path("datasets/yolo")               # YOLO dataset, built by src/2_phase/carpk_to_yolo.py
DATA_YAML = YOLO_DIR / "carpk.yaml"
TRAIN_EVAL_IMAGES = 200                        # fixed train subset scored every epoch (common/train_eval.py)

# --- models ---
DEFAULT_WEIGHTS = "models/yolo11n.pt"          # COCO-pretrained YOLO11n
YOLO26_WEIGHTS = "models/yolo26n.pt"
MODEL_CFG = "yolo11n.yaml"
CAR_COCO_ID = 2                                # 'car' class of the COCO-pretrained model

# --- images and inference ---
IMG_W, IMG_H = 1280, 720                       # native resolution
IMGSZ, CONF = 1280, 0.25                       # input size; Ultralytics' default confidence threshold
MAX_DET = 1000                                 # Ultralytics' default (300) is too low for the densest images

# --- training ---
DEVICES = (0, 1)                               # 2x RTX 4080 SUPER -> DDP
BATCH_PER_GPU = 16                             # ~11 GB VRAM per GPU at imgsz 1280
EPOCHS, PATIENCE, SEED = 100, 5, 42
