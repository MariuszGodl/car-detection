"""Per-epoch accuracy on a fixed subset of the *training* images.

During training Ultralytics only scores the val split, so there is no train-set precision / recall /
mAP to compare against (the train losses are measured on augmented images). This module adds it:

* :func:`make_subset` builds ``datasets/yolo/train_eval/`` - symlinks to N train images + labels
  (own ``labels.cache``, so the train cache is untouched),
* :func:`on_fit_epoch_end` scores that subset after every epoch with the trainer's own validator
  (EMA weights, no augmentation - exactly like val) and writes ``<run>/train_eval.csv`` and
  TensorBoard scalars ``train_eval/*``.

Enable with ``train_args(..., train_eval=True)``; a resumed run re-enables it when
``train_eval.csv`` exists. In DDP the validator shards the images over the ranks and gathers on
rank 0, so the callback runs on every rank and only rank 0 writes.
"""
import random
import shutil
from pathlib import Path

import pandas as pd

from .config import ROOT, SEED, TRAIN_EVAL_IMAGES, YOLO_DIR

CSV = "train_eval.csv"


def make_subset(n=TRAIN_EVAL_IMAGES, seed=SEED, src=YOLO_DIR, dst=None) -> Path:
    """Symlink a fixed random sample of ``n`` train images (+ labels) of dataset ``src`` into
    ``dst`` (default ``<src>/train_eval``, where :func:`on_fit_epoch_end` looks for it)."""
    src = ROOT / src
    dst = ROOT / dst if dst else src / "train_eval"
    imgs = sorted((src / "images" / "train").iterdir())
    pick = sorted(random.Random(seed).sample(imgs, min(n, len(imgs))))
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    for img in pick:
        (dst / "images" / img.name).symlink_to(img.resolve())
        lbl = src / "labels" / "train" / f"{img.stem}.txt"
        if lbl.exists():
            (dst / "labels" / lbl.name).symlink_to(lbl.resolve())
    return dst


def _subset_images(trainer) -> Path:
    """``<dataset dir>/train_eval/images`` of the dataset this run trains on."""
    path = (ROOT / trainer.args.data).resolve().parent / "train_eval" / "images"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} missing - run train_eval.make_subset(src=<dataset dir>) first")
    return path


def on_fit_epoch_end(trainer) -> None:
    from ultralytics.data import build_dataloader
    from ultralytics.utils import LOCAL_RANK, RANK
    from ultralytics.utils.callbacks import tensorboard as tb
    from ultralytics.utils.torch_utils import torch_distributed_zero_first

    v = trainer.validator
    # final_eval re-fires this hook after validating best.pt (validator.training=False) -> skip on all ranks
    if v is None or not v.training:
        return
    if getattr(trainer, "_train_eval_loader", None) is None:
        bs = trainer.test_loader.batch_size
        with torch_distributed_zero_first(LOCAL_RANK):
            subset = _subset_images(trainer)
            ds = trainer.build_dataset(str(subset), mode="val", batch=bs)
        trainer._train_eval_loader = build_dataloader(ds, bs, workers=2, shuffle=False, rank=LOCAL_RANK,
                                                      device=trainer.device)
    saved = v.dataloader, v.args.plots
    v.dataloader, v.args.plots = trainer._train_eval_loader, False
    try:
        stats = v(trainer)
    finally:
        v.dataloader, v.args.plots = saved
    if RANK not in {-1, 0}:
        return
    epoch = trainer.epoch + 1
    row = {k.split("/", 1)[-1]: float(x) for k, x in stats.items() if k != "fitness"}   # precision(B), box_loss...
    csv = Path(trainer.save_dir) / CSV
    pd.DataFrame([{"epoch": epoch, **row}]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    if tb.WRITER:
        for k, x in row.items():
            tb.WRITER.add_scalar(f"train_eval/{k}", x, epoch)


def register(model) -> None:
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def read(run_dir) -> pd.DataFrame:
    """``train_eval.csv`` (one row per epoch; a resumed epoch keeps its latest row)."""
    df = pd.read_csv(Path(run_dir) / CSV)
    return df.drop_duplicates("epoch", keep="last").reset_index(drop=True)
