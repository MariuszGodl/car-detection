"""Fine-tuning helpers shared by Phase 2 and 3.

Training runs ``src/2_phase/train.py`` in a subprocess (:func:`launch`): DDP workers are
separate processes whose output never reaches a notebook cell, so it is streamed and
saved to ``<run>/console.txt``.

With ``device=[0, 1]`` Ultralytics' ``batch`` is the global batch (``batch // 2`` per GPU).
It does not scale ``lr0``; it accumulates gradients up to ``nbs=64`` and scales weight decay
accordingly, so 1-GPU and 2-GPU runs with the same global batch optimise the same way.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .config import (BATCH_PER_GPU, DATA_YAML, DEFAULT_WEIGHTS, DEVICES, EPOCHS, IMGSZ, MAX_DET, PATIENCE, ROOT,
                     RUNS_DIR, SEED)
from .metrics import fitness

TRAIN_SCRIPT = ROOT / "src" / "2_phase" / "train.py"
LOSS_COLS = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]


def train_args(data=DATA_YAML, epochs=EPOCHS, batch_per_gpu=BATCH_PER_GPU, devices=DEVICES,
               project=RUNS_DIR / "train", name="exp", train_eval=False, **overrides) -> dict:
    """Ultralytics ``train`` kwargs; ``batch`` is the global batch over ``devices``.

    ``train_eval=True`` also scores a fixed train subset every epoch (``common/train_eval.py``).
    """
    devices = list(devices)
    args = {
        "data": str(Path(data).resolve()),
        "epochs": epochs,
        "imgsz": IMGSZ,
        "batch": batch_per_gpu * len(devices),
        "device": devices if len(devices) > 1 else devices[0],
        "project": str(Path(project).resolve()),
        "name": name,
        "exist_ok": True,
        "seed": SEED,
        "deterministic": False,
        "patience": PATIENCE,
        "max_det": MAX_DET,
    }
    args.update(overrides)
    if train_eval:
        args["train_eval"] = True   # popped by src/2_phase/train.py
    return args


def _epochs_saved(run_dir, since) -> int:
    """Epochs whose ``last.pt`` was written after ``since``."""
    results_path = run_dir / "results.csv"
    checkpoint_path = run_dir / "weights" / "last.pt"
    try:
        results_time = results_path.stat().st_mtime
        checkpoint_time = checkpoint_path.stat().st_mtime
    except FileNotFoundError:
        return 0
    if results_time < since or checkpoint_time < results_time:
        return 0
    return len(results_path.read_text().splitlines()) - 1


def launch(args=None, weights=DEFAULT_WEIGHTS, resume_from=None, stop_after_epochs=None, echo=False) -> Path:
    """Train in a subprocess and return the run dir (output saved to ``<run>/console.txt``).

    ``args`` come from :func:`train_args`. ``resume_from`` is a run's ``last.pt``.
    ``stop_after_epochs`` kills the run once that many epochs are saved (simulated crash).
    """
    if resume_from:
        weights = resume_from
        args = {"resume": True}
        run_dir = Path(resume_from).resolve().parents[1]
    else:
        run_dir = Path(args["project"]) / args["name"]
    command = [sys.executable, "-u", str(TRAIN_SCRIPT), str(weights), json.dumps(args)]
    console_lines = []
    interrupted = False
    started_at = time.time()
    with subprocess.Popen(
        command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        for line in process.stdout:
            console_lines.append(line)
            if echo:
                print(line, end="", flush=True)
            if stop_after_epochs and not interrupted:
                saved_epochs = _epochs_saved(run_dir, started_at)
                if saved_epochs >= stop_after_epochs:
                    console_lines.append(f"\n### launch(): killing training after {stop_after_epochs} epochs\n")
                    os.killpg(process.pid, signal.SIGTERM)   # include the DDP workers
                    interrupted = True

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.txt"
    with log_path.open("a" if resume_from else "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n" + "".join(console_lines))
    if process.returncode and not interrupted:
        raise RuntimeError(f"training failed (exit {process.returncode}); see {log_path}")
    return run_dir


def finished(run_dir) -> bool:
    """Whether a run ended normally (``run_logging`` writes ``# finished`` to its ``train_log.txt``)."""
    log = Path(run_dir) / "train_log.txt"
    return log.exists() and "# finished" in log.read_text()


def launch_once(args, weights=DEFAULT_WEIGHTS, **kwargs) -> Path:
    """:func:`launch`, unless the run named in ``args`` already finished - so a notebook can be re-run top to
    bottom without retraining. An unfinished run starts again; continue it with ``launch(resume_from=...)``."""
    run_dir = Path(args["project"]) / args["name"]
    if finished(run_dir):
        print(f"{args['name']}: already trained -> {run_dir}")
        return run_dir
    return launch(args, weights, **kwargs)


def loss_columns(df, prefix="train/") -> list:
    """The ``<prefix>*_loss`` columns a run logged, in logging order."""
    return [
        column for column in df.columns
        if column.startswith(prefix) and column.endswith("_loss") and column != "train/total_loss"
    ]


def read_results(run_dir) -> pd.DataFrame:
    """A run's ``results.csv`` plus a ``train/total_loss`` column (sum of the logged train losses: box, cls and
    DFL for YOLO11; box, cls and L1 for YOLO26, which has no DFL)."""
    results = pd.read_csv(Path(run_dir) / "results.csv")
    results.columns = results.columns.str.strip()
    results["train/total_loss"] = results[loss_columns(results)].sum(axis=1)
    return results


def best_row(run_dir) -> pd.Series:
    """The ``results.csv`` row of the epoch saved as ``best.pt`` (highest val fitness)."""
    results = read_results(run_dir)
    return results.loc[fitness(results).idxmax()]


def best_epoch(run_dir) -> int:
    return int(best_row(run_dir)["epoch"])
