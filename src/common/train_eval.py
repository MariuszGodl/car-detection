"""Accuracy on (unaugmented) training images, per epoch.

Ultralytics only scores val during training. :func:`on_fit_epoch_end` also runs its validator on a fixed
subset of train images (:func:`make_subset`) and appends the result to ``<run>/train_eval.csv`` and TensorBoard,
so train and val curves can be compared to spot over-fitting.
"""
import random
import shutil
from pathlib import Path

import pandas as pd

from .config import ROOT, SEED, TRAIN_EVAL_IMAGES, YOLO_DIR
from .data import dataset_dir

CSV = "train_eval.csv"


def make_subset(n=TRAIN_EVAL_IMAGES, seed=SEED, src=YOLO_DIR, dst=None) -> Path:
    """Symlink a fixed random sample of ``n`` train images (+ labels) of dataset ``src`` into
    ``dst`` (default ``<src>/train_eval``, where :func:`on_fit_epoch_end` looks for it)."""
    src = ROOT / src
    dst = ROOT / dst if dst else src / "train_eval"
    images = sorted((src / "images" / "train").iterdir())
    selected_images = sorted(random.Random(seed).sample(images, min(n, len(images))))
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    for image in selected_images:
        (dst / "images" / image.name).symlink_to(image.resolve())
        label = src / "labels" / "train" / f"{image.stem}.txt"
        if label.exists():
            (dst / "labels" / label.name).symlink_to(label.resolve())
    return dst


def _subset_images(trainer) -> Path:
    path = dataset_dir(trainer.args.data) / "train_eval" / "images"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} missing - run train_eval.make_subset(src=<dataset dir>) first")
    return path


def on_fit_epoch_end(trainer) -> None:
    """Reuse the validator on the fixed train subset, then restore it for validation."""
    from ultralytics.data import build_dataloader
    from ultralytics.utils import LOCAL_RANK, RANK
    from ultralytics.utils.callbacks import tensorboard as tb
    from ultralytics.utils.torch_utils import torch_distributed_zero_first

    validator = trainer.validator
    # final_eval re-fires this hook after validating best.pt (validator.training=False) -> skip on all ranks
    if validator is None or not validator.training:
        return
    if getattr(trainer, "_train_eval_loader", None) is None:
        batch_size = trainer.test_loader.batch_size
        with torch_distributed_zero_first(LOCAL_RANK):
            subset = _subset_images(trainer)
            dataset = trainer.build_dataset(str(subset), mode="val", batch=batch_size)
        trainer._train_eval_loader = build_dataloader(
            dataset, batch_size, workers=2, shuffle=False, rank=LOCAL_RANK, device=trainer.device
        )
    validation_loader = validator.dataloader
    validation_plots = validator.args.plots
    validator.dataloader = trainer._train_eval_loader
    validator.args.plots = False
    try:
        stats = validator(trainer)
    finally:
        validator.dataloader = validation_loader
        validator.args.plots = validation_plots
    if RANK not in {-1, 0}:
        return
    epoch = trainer.epoch + 1
    metrics = {}
    for name, value in stats.items():
        if name != "fitness":
            metrics[name.split("/", 1)[-1]] = float(value)
    csv_path = Path(trainer.save_dir) / CSV
    pd.DataFrame([{"epoch": epoch, **metrics}]).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )
    if tb.WRITER:
        for name, value in metrics.items():
            tb.WRITER.add_scalar(f"train_eval/{name}", value, epoch)


def register(model) -> None:
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def read(run_dir) -> pd.DataFrame:
    """``train_eval.csv`` (one row per epoch; a resumed epoch keeps its latest row)."""
    df = pd.read_csv(Path(run_dir) / CSV)
    return df.drop_duplicates("epoch", keep="last").reset_index(drop=True)
