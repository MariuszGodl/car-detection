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

from .config import BATCH_PER_GPU, DATA_YAML, DEFAULT_WEIGHTS, DEVICES, EPOCHS, IMGSZ, PATIENCE, ROOT, RUNS_DIR, SEED

TRAIN_SCRIPT = ROOT / "src" / "2_phase" / "train.py"
LOSS_COLS = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]


def train_args(data=DATA_YAML, epochs=EPOCHS, batch_per_gpu=BATCH_PER_GPU, devices=DEVICES,
               project=RUNS_DIR / "train", name="exp", train_eval=False, **overrides) -> dict:
    """Ultralytics ``train`` kwargs; ``batch`` is the global batch over ``devices``.

    ``train_eval=True`` also scores a fixed train subset every epoch (``common/train_eval.py``).
    """
    devices = list(devices)
    args = {"data": str(Path(data).resolve()), "epochs": epochs, "imgsz": IMGSZ,
            "batch": batch_per_gpu * len(devices), "device": devices if len(devices) > 1 else devices[0],
            "project": str(Path(project).resolve()), "name": name, "exist_ok": True,
            "seed": SEED, "deterministic": False, "patience": PATIENCE, **overrides}
    if train_eval:
        args["train_eval"] = True   # popped by src/2_phase/train.py
    return args


def _epochs_saved(run_dir, since) -> int:
    """Epochs whose ``last.pt`` was written after ``since``."""
    csv, last = run_dir / "results.csv", run_dir / "weights" / "last.pt"
    try:
        t_csv, t_last = csv.stat().st_mtime, last.stat().st_mtime
    except FileNotFoundError:
        return 0
    if t_csv < since or t_last < t_csv:   # stale, or last.pt not yet written for this row
        return 0
    return len(csv.read_text().splitlines()) - 1


def launch(args=None, weights=DEFAULT_WEIGHTS, resume_from=None, stop_after_epochs=None, echo=False) -> Path:
    """Train in a subprocess and return the run dir (output saved to ``<run>/console.txt``).

    ``args`` come from :func:`train_args`. ``resume_from`` is a run's ``last.pt``.
    ``stop_after_epochs`` kills the run once that many epochs are saved (simulated crash).
    """
    if resume_from:
        weights, args = resume_from, {"resume": True}
        run_dir = Path(resume_from).resolve().parents[1]
    else:
        run_dir = Path(args["project"]) / args["name"]
    cmd = [sys.executable, "-u", str(TRAIN_SCRIPT), str(weights), json.dumps(args)]
    lines, killed, t0 = [], False, time.time()
    with subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True) as proc:
        for line in proc.stdout:
            lines.append(line)
            if echo:
                print(line, end="", flush=True)
            if stop_after_epochs and not killed and _epochs_saved(run_dir, t0) >= stop_after_epochs:
                lines.append(f"\n### launch(): killing training after {stop_after_epochs} epochs\n")
                os.killpg(proc.pid, signal.SIGTERM)
                killed = True
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "console.txt", "a" if resume_from else "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n" + "".join(lines))
    if proc.returncode and not killed:
        raise RuntimeError(f"training failed (exit {proc.returncode}); see {run_dir / 'console.txt'}")
    return run_dir


def read_results(run_dir) -> pd.DataFrame:
    """A run's ``results.csv`` plus a ``train/total_loss`` column."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = df.columns.str.strip()
    df["train/total_loss"] = df[LOSS_COLS].sum(axis=1)
    return df
