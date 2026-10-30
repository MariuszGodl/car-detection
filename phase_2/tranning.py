"""Fine-tune YOLO11n on the CARPK car dataset built by phase_2/dataset.py, using 2 GPUs.

Run from the repo root: ``python -m phase_2.tranning``.
"""
import csv
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO, settings
from ultralytics.utils import YAML

from phase_2.console_logging import log_distributed_console, training_console
from phase_2.utils import CONF, IMGSZ, MAX_DET, count_metrics

DATA_YAML = Path("phase_2/yolo/carpk.yaml")
VAL_IMAGES = Path("phase_2/yolo/images/val")
VAL_LABELS = Path("phase_2/yolo/labels/val")
HPARAMS_LOG = Path("runs/hparams_log.jsonl")

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


def log_hparams(save_dir, model_name, hparams):
    with (Path(save_dir) / "results.csv").open(newline="") as f:
        results = list(csv.DictReader(f))
    best = max(results, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    metrics = {
        "hparam/mAP50": float(best["metrics/mAP50(B)"]),
        "hparam/mAP50-95": float(best["metrics/mAP50-95(B)"]),
    }

    with SummaryWriter(str(save_dir)) as writer:
        writer.add_hparams({"model": model_name, **hparams}, metrics)

    HPARAMS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HPARAMS_LOG.open("a") as f:
        f.write(json.dumps({
            "run": model_name,
            **hparams,
            "best_mAP50": metrics["hparam/mAP50"],
            "best_mAP50-95": metrics["hparam/mAP50-95"],
        }) + "\n")


def train_yolo(model, model_name, epochs=30, imgsz=1280, batch=16, continue_training=False, **hparams):
    """Train or resume, appending console output under tranning/."""
    with training_console(model_name):
        return _train_yolo(model, model_name, epochs, imgsz, batch, continue_training, **hparams)


def _train_yolo(model, model_name, epochs, imgsz, batch, continue_training, **hparams):
    project = str(Path.cwd() / "runs" / f"finetune_{model_name}")

    if continue_training:
        run_dir = Path(project) / model_name
        checkpoint = run_dir / "weights" / "last.pt"
        args_path = run_dir / "args.yaml"
        if not checkpoint.is_file() or not args_path.is_file():
            raise FileNotFoundError(f"Cannot resume {model_name}: expected {checkpoint} and {args_path}")

        saved_args = YAML.load(args_path)
        train_hparams = {
            key: value for key, value in saved_args.items()
            if isinstance(value, (str, int, float, bool))
        }
        model = YOLO(checkpoint)
        log_distributed_console(model, model_name)
        model.add_callback("on_model_save", log_counting)
        model.train(resume=True, verbose=False)
        log_hparams(model.trainer.save_dir, model_name, train_hparams)
        return

    train_hparams = {"epochs": epochs, "imgsz": imgsz, "batch": batch, **hparams}
    log_distributed_console(model, model_name)
    model.add_callback("on_model_save", log_counting)
    model.train(
        data=str(DATA_YAML),
        device="0,1",
        project=project,
        name=model_name,
        verbose=False,
        **train_hparams,
    )
    log_hparams(model.trainer.save_dir, model_name, train_hparams)
