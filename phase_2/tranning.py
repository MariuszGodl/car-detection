"""Fine-tune YOLO11n on the CARPK car dataset built by phase_2/dataset.py, using 2 GPUs.

Run from the repo root: ``python -m phase_2.tranning``.
"""
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO, settings

from phase_2.utils import CONF, IMGSZ, MAX_DET, count_metrics

DATA_YAML = Path("phase_2/yolo/carpk.yaml")
VAL_IMAGES = Path("phase_2/yolo/images/val")
VAL_LABELS = Path("phase_2/yolo/labels/val")

settings.update({"tensorboard": True})


def log_counting(trainer):
    """Log val-split MAE/RMSE of car counts to TensorBoard after each epoch (rank 0 only, also under DDP)."""
    images = sorted(VAL_IMAGES.iterdir())
    gt = [len((VAL_LABELS / f"{p.stem}.txt").read_text().splitlines()) for p in images]
    model = YOLO(trainer.last)
    pred = [len(model.predict(p, imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, verbose=False)[0].boxes)
            for p in images]
    m = count_metrics(gt, pred)
    with SummaryWriter(str(trainer.save_dir)) as w:
        w.add_scalar("count/MAE", m["MAE"], trainer.epoch + 1)
        w.add_scalar("count/RMSE", m["RMSE"], trainer.epoch + 1)


def train_yolo(model, model_name):
    project=str(Path.cwd() / "runs" / f"finetune_{model_name}")
    model.add_callback("on_model_save", log_counting)
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=1280,
        batch=16,
        device="0,1",
        project=project,
        name=model_name,
        verbose=False
    )
