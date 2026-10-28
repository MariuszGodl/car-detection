"""Leave-one-lot-out cross-validation over the CARPK training lots (Phase 3).

The official test split is a parking lot (TPZ) that never appears in training, while the random val split
comes from the training lots GF1 / GF2 / NTU. Each fold here holds one whole training lot out as val, so
choices made on val are made on a lot the model never saw, like test. The official test split is not touched.

* :func:`build_folds` - one YOLO dataset per held-out lot, ``datasets/yolo_cv_<LOT>/`` (symlinks only),
* :func:`run_folds` - train the same recipe on every fold,
* :func:`fold_summary` - best epoch and held-out-lot metrics per fold,
* :func:`cv_threshold` - one count-MAE sweep pooled over every fold's held-out-lot images, each scored by
  the fold model that never saw them -> one ``conf`` for the recipe, chosen without looking at test.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_phase"))

import carpk_to_yolo
import compare
from common import train_eval, training
from common.config import CONF, SEED
from common.training import read_results

LOTS = ("carpk:GF1", "carpk:GF2", "carpk:NTU")


def short(lot) -> str:
    return lot.split(":")[-1]


def fold_dir(lot) -> Path:
    return Path("datasets") / f"yolo_cv_{short(lot)}"


def build_folds(lots=LOTS, val_fraction=0.1, seed=SEED) -> dict:
    """``{lot: carpk.yaml}``; fold ``lot`` has that lot (+ ``val_fraction`` of PUCPR+) as val, the rest as train."""
    yamls = {}
    for lot in lots:
        print(f"--- fold {short(lot)} (val = {lot} + {val_fraction:.0%} of PUCPR+)")
        yamls[lot] = carpk_to_yolo.build_dataset(out_dir=fold_dir(lot), val_fraction=val_fraction, seed=seed,
                                                 val_lots=[lot])
        train_eval.make_subset(src=fold_dir(lot))
    return yamls


def lots_table(yamls) -> pd.DataFrame:
    """#images per lot and split in every fold."""
    return pd.concat({short(lot): compare.lots_per_split(y) for lot, y in yamls.items()}, axis=1, names=["fold"])


def run_folds(yamls, prefix="cv", **recipe) -> dict:
    """Train ``recipe`` (``train_args`` overrides, e.g. ``flipud=0.5``) on every fold; ``{lot: run dir}``.
    A fold whose run already finished is skipped, so a crashed loop can simply be re-run."""
    return {lot: training.launch_once(training.train_args(data=yaml, name=f"{prefix}_{short(lot)}",
                                                          train_eval=True, **recipe))
            for lot, yaml in yamls.items()}


def fold_summary(runs) -> pd.DataFrame:
    """Per fold: best epoch (Ultralytics fitness on the held-out lot) and its val metrics."""
    rows = {}
    for lot, run_dir in runs.items():
        df = read_results(run_dir)
        fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
        best = df.loc[fit.idxmax()]
        rows[short(lot)] = {"epochs": len(df), "best epoch": int(best["epoch"]),
                            **{k: best[f"metrics/{k}"] for k in compare.DET}}
    return pd.DataFrame(rows).T


def cv_threshold(runs, yamls, device=0, batch=16) -> dict:
    """Pooled out-of-lot threshold: every fold's ``best.pt`` predicts its own held-out lot; one sweep over
    all of them picks ``conf``. Also returns per-fold sweeps and their minima, to show how stable it is."""
    frames = []
    for lot, run_dir in runs.items():
        df = compare.predict_scores(Path(run_dir) / "weights" / "best.pt", "val", device, batch, data=yamls[lot])
        frames.append(df[df["lot"] == lot].assign(fold=short(lot)))   # held-out lot only, not the PUCPR+ part
    pooled = pd.concat(frames, ignore_index=True)
    sweeps = {f: compare.conf_sweep(g) for f, g in pooled.groupby("fold")}
    sweeps["pooled"] = compare.conf_sweep(pooled)
    table = pd.DataFrame({f: {"images": int((pooled["fold"] == f).sum()) if f != "pooled" else len(pooled),
                              "best conf": compare.best_conf(sw), "MAE at best": sw["MAE"].min(),
                              f"MAE at {CONF}": float(sw.loc[(sw["conf"] - CONF).abs().idxmin(), "MAE"])}
                          for f, sw in sweeps.items()}).T
    return {"conf": compare.best_conf(sweeps["pooled"]), "sweeps": sweeps, "table": table, "pooled": pooled}


def plot_cv_threshold(cv):
    return compare.plot_sweeps(cv["sweeps"], title="Count MAE on the held-out lot vs threshold (lot cross-validation)")


def plot_fold_curves(runs):
    """Val mAP50-95 per epoch of every fold (each on its own held-out lot)."""
    _, ax = plt.subplots(figsize=(8, 4))
    for lot, run_dir in runs.items():
        df = read_results(run_dir)
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=f"val = {short(lot)}")
    ax.set(title="Held-out-lot mAP50-95 per epoch", xlabel="epoch", ylabel="mAP50-95")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax
