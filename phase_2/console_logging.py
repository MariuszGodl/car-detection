"""Append Ultralytics messages to tranning/tranning_log_[sanity_]<model_name>.txt."""
import logging
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from ultralytics.utils import LOGGER, RANK


@contextmanager
def training_console(model_name, sanity=False):
    prefix = "sanity_" if sanity else ""
    path = Path("tranning") / f"tranning_log_{prefix}{model_name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        LOGGER.info(f"\n--- Training: {model_name} ---")
        yield
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def _start_worker_log(trainer, model_name):
    if RANK == 0:
        trainer._console_log = training_console(model_name)
        trainer._console_log.__enter__()


def _close_worker_log(trainer):
    if RANK == 0 and hasattr(trainer, "_console_log"):
        trainer._console_log.__exit__(None, None, None)


def log_distributed_console(model, model_name):
    """Capture the rank-zero subprocess too when Ultralytics launches DDP."""
    model.add_callback("on_pretrain_routine_start", partial(_start_worker_log, model_name=model_name))
    model.add_callback("teardown", _close_worker_log)
