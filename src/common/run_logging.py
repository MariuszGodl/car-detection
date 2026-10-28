"""Extra training logs written on rank 0 via Ultralytics callbacks (on top of its own
``results.csv`` / ``args.yaml`` / TensorBoard scalars):

* ``train_log.txt`` - header, one greppable line per epoch and a final summary,
* ``hparams.yaml`` and TensorBoard HParams - the effective hyperparameters.

Callbacks are module-level so Ultralytics can pickle them into its DDP workers.
"""
import time
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import settings
from ultralytics.utils import RANK

TXT_LOG = "train_log.txt"


def enable_tensorboard() -> None:
    settings.update({"tensorboard": True})   # persisted, so DDP workers see it too


def _log(trainer, line) -> None:
    with open(Path(trainer.save_dir) / TXT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _fmt(d) -> str:
    return " ".join(f"{k}={float(v):.5g}" for k, v in d.items())


def collect_hparams(trainer) -> dict:
    """Ultralytics args plus the values it derives (per-GPU batch, accumulation, lr)."""
    hp = {k: v for k, v in vars(trainer.args).items() if isinstance(v, (bool, int, float, str))}
    world = max(trainer.world_size or 1, 1)
    hp.update(world_size=world, batch_per_gpu=trainer.batch_size // world,
              accumulate=trainer.accumulate, effective_batch=trainer.batch_size * trainer.accumulate,
              lr0_effective=trainer.optimizer.param_groups[0]["initial_lr"])
    return hp


def on_train_start(trainer) -> None:
    if RANK not in {-1, 0}:
        return
    trainer._hp, trainer._t_start = collect_hparams(trainer), time.time()
    (Path(trainer.save_dir) / "hparams.yaml").write_text(yaml.safe_dump(trainer._hp, sort_keys=False))
    if not trainer.start_epoch:   # fresh run reusing a name: start a new log
        (Path(trainer.save_dir) / TXT_LOG).unlink(missing_ok=True)
    resumed = f" (resumed from epoch {trainer.start_epoch})" if trainer.start_epoch else ""
    _log(trainer, f"# run {trainer.save_dir} started {time.strftime('%Y-%m-%d %H:%M:%S')}{resumed}")
    _log(trainer, "# hparams " + " ".join(f"{k}={v}" for k, v in trainer._hp.items()))


def on_fit_epoch_end(trainer) -> None:
    # After the last epoch (trainer.stop is set) final_eval re-fires this hook for best.pt -> skip it.
    if RANK not in {-1, 0} or getattr(trainer, "_stopped", False):
        return
    trainer._stopped = trainer.stop
    mem = torch.cuda.max_memory_reserved() / 1e9
    _log(trainer, f"epoch {trainer.epoch + 1}/{trainer.epochs} "
                  f"{_fmt(trainer.label_loss_items(trainer.tloss, prefix='train'))} {_fmt(trainer.metrics)} "
                  f"{_fmt(trainer.lr)} gpu_mem_GB={mem:.2f} time_s={time.time() - trainer._t_start:.1f}")


def on_train_end(trainer) -> None:
    metrics = {k: float(v) for k, v in trainer.metrics.items()}
    _log(trainer, f"final_best {_fmt(metrics)}")
    _log(trainer, f"# finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"best_epoch={trainer.stopper.best_epoch} best={trainer.best} last={trainer.last}")
    with SummaryWriter(str(trainer.save_dir)) as w:
        w.add_hparams(trainer._hp, {f"hparam/{k}": v for k, v in metrics.items()}, run_name="hparams")


def register_callbacks(model) -> None:
    for fn in (on_train_start, on_fit_epoch_end, on_train_end):
        model.add_callback(fn.__name__, fn)


def grep_log(run_dir, pattern="") -> list:
    return [ln for ln in (Path(run_dir) / TXT_LOG).read_text().splitlines() if pattern in ln]
