"""Training-infrastructure checks: 1 GPU vs DDP, interrupt + resume."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import RUNS_DIR
from common.training import LOSS_COLS, launch, read_results, train_args

SHORT_RUN = dict(epochs=3, fraction=0.5, workers=8, project=RUNS_DIR / "checks")
COLS = LOSS_COLS + ["train/total_loss"]


def ddp_vs_single(global_batch=16) -> dict:
    """Same short schedule, seed and global batch on 1 GPU and on 2 GPUs (DDP)."""
    single_args = train_args(name="single_gpu", devices=[0], batch_per_gpu=global_batch, **SHORT_RUN)
    ddp_args = train_args(name="ddp_2gpu", devices=[0, 1], batch_per_gpu=global_batch // 2, **SHORT_RUN)
    runs = {"1 GPU": launch(single_args), "2 GPU DDP": launch(ddp_args)}

    results = {}
    for label, run_dir in runs.items():
        results[label] = read_results(run_dir).set_index("epoch")[COLS + ["time"]]
    table = pd.concat(results, axis=1)
    single_time = table[("1 GPU", "time")].iloc[-1]
    ddp_time = table[("2 GPU DDP", "time")].iloc[-1]
    single_loss = table[("1 GPU", "train/total_loss")]
    ddp_loss = table[("2 GPU DDP", "train/total_loss")]
    return {
        "runs": runs,
        "table": table,
        "time_1gpu_s": single_time,
        "time_ddp_s": ddp_time,
        "speedup": single_time / ddp_time,
        "max_rel_loss_diff": ((single_loss - ddp_loss).abs() / single_loss).max(),
    }


def plot_ddp_vs_single(result) -> None:
    _, axes = plt.subplots(1, len(COLS), figsize=(16, 3.5))
    for ax, column in zip(axes, COLS):
        for label, style in (("1 GPU", "o-"), ("2 GPU DDP", "s--")):
            ax.plot(result["table"].index, result["table"][(label, column)], style, label=label)
        ax.set(title=column, xlabel="epoch")
    axes[0].legend()
    plt.suptitle(f"1 GPU vs 2-GPU DDP (same global batch) — speedup x{result['speedup']:.2f}")
    plt.tight_layout()
    plt.show()


def checkpoint_info(run_dir) -> dict:
    """Epoch / fitness / optimizer state stored in ``best.pt`` and ``last.pt``."""
    info = {}
    for name in ("best", "last"):
        checkpoint = torch.load(
            Path(run_dir) / "weights" / f"{name}.pt", map_location="cpu", weights_only=False
        )
        info[name] = {
            "epoch": checkpoint.get("epoch"),   # epoch -1 = finished
            "best_fitness": checkpoint.get("best_fitness"),
            "has_optimizer": checkpoint.get("optimizer") is not None,
            "date": checkpoint.get("date"),
        }
    return info


def interrupt_and_resume(epochs=4, stop_after=2) -> dict:
    """Kill a DDP run after ``stop_after`` epochs, resume it from ``last.pt`` and check continuity."""
    schedule = {**SHORT_RUN, "epochs": epochs}
    args = train_args(name="resume_check", devices=[0, 1], batch_per_gpu=8, **schedule)
    run_dir = launch(args, stop_after_epochs=stop_after)
    before = checkpoint_info(run_dir)
    epochs_before = read_results(run_dir)["epoch"].tolist()
    launch(resume_from=run_dir / "weights" / "last.pt")
    after = checkpoint_info(run_dir)
    epochs_after = read_results(run_dir)["epoch"].tolist()
    assert before["last"]["has_optimizer"], "interrupted last.pt must keep optimizer state"
    assert epochs_after == list(range(1, epochs + 1)), f"epochs not continuous: {epochs_after}"
    return {
        "run_dir": run_dir,
        "before": before,
        "after": after,
        "epochs_before": epochs_before,
        "epochs_after": epochs_after,
    }
