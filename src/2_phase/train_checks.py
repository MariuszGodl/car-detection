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
    runs = {"1 GPU": launch(train_args(name="single_gpu", devices=[0], batch_per_gpu=global_batch, **SHORT_RUN)),
            "2 GPU DDP": launch(train_args(name="ddp_2gpu", devices=[0, 1], batch_per_gpu=global_batch // 2,
                                           **SHORT_RUN))}
    table = pd.concat({k: read_results(d).set_index("epoch")[COLS + ["time"]] for k, d in runs.items()}, axis=1)
    t1, t2 = table[("1 GPU", "time")].iloc[-1], table[("2 GPU DDP", "time")].iloc[-1]
    loss1, loss2 = table[("1 GPU", "train/total_loss")], table[("2 GPU DDP", "train/total_loss")]
    return {"runs": runs, "table": table, "time_1gpu_s": t1, "time_ddp_s": t2, "speedup": t1 / t2,
            "max_rel_loss_diff": ((loss1 - loss2).abs() / loss1).max()}


def plot_ddp_vs_single(result) -> None:
    _, axes = plt.subplots(1, len(COLS), figsize=(16, 3.5))
    for ax, c in zip(axes, COLS):
        for label, style in (("1 GPU", "o-"), ("2 GPU DDP", "s--")):
            ax.plot(result["table"].index, result["table"][(label, c)], style, label=label)
        ax.set(title=c, xlabel="epoch")
    axes[0].legend()
    plt.suptitle(f"1 GPU vs 2-GPU DDP (same global batch) — speedup x{result['speedup']:.2f}")
    plt.tight_layout()
    plt.show()


def checkpoint_info(run_dir) -> dict:
    """Epoch / fitness / optimizer state stored in ``best.pt`` and ``last.pt``."""
    info = {}
    for name in ("best", "last"):
        ck = torch.load(Path(run_dir) / "weights" / f"{name}.pt", map_location="cpu", weights_only=False)
        info[name] = {"epoch": ck.get("epoch"), "best_fitness": ck.get("best_fitness"),   # epoch -1 = finished
                      "has_optimizer": ck.get("optimizer") is not None, "date": ck.get("date")}
    return info


def interrupt_and_resume(epochs=4, stop_after=2) -> dict:
    """Kill a DDP run after ``stop_after`` epochs, resume it from ``last.pt`` and check continuity."""
    args = train_args(name="resume_check", devices=[0, 1], batch_per_gpu=8, **{**SHORT_RUN, "epochs": epochs})
    run_dir = launch(args, stop_after_epochs=stop_after)
    before, epochs_before = checkpoint_info(run_dir), read_results(run_dir)["epoch"].tolist()
    launch(resume_from=run_dir / "weights" / "last.pt")
    after, epochs_after = checkpoint_info(run_dir), read_results(run_dir)["epoch"].tolist()
    assert before["last"]["has_optimizer"], "interrupted last.pt must keep optimizer state"
    assert epochs_after == list(range(1, epochs + 1)), f"epochs not continuous: {epochs_after}"
    return {"run_dir": run_dir, "before": before, "after": after,
            "epochs_before": epochs_before, "epochs_after": epochs_after}
