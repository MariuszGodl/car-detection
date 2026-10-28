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
from common.metrics import DETECTION_METRICS
from common.training import best_row, read_results

LOTS = ("carpk:GF1", "carpk:GF2", "carpk:NTU")


def short(lot) -> str:
    """``carpk:GF1`` -> ``GF1``."""
    return lot.split(":")[-1]


def fold_dir(lot) -> Path:
    """Dataset of the fold that holds ``lot`` out."""
    return Path("datasets") / f"yolo_cv_{short(lot)}"


def build_folds(lots=LOTS, val_fraction=0.1, seed=SEED) -> dict:
    """``{lot: carpk.yaml}``; fold ``lot`` has that lot (+ ``val_fraction`` of PUCPR+) as val, the rest as train."""
    datasets = {}
    for lot in lots:
        print(f"--- fold {short(lot)} (val = {lot} + {val_fraction:.0%} of PUCPR+)")
        output = fold_dir(lot)
        datasets[lot] = carpk_to_yolo.build_dataset(
            out_dir=output, val_fraction=val_fraction, seed=seed, val_lots=[lot]
        )
        train_eval.make_subset(src=output)
    return datasets


def lots_table(yamls) -> pd.DataFrame:
    """#images per lot and split in every fold."""
    tables = {}
    for lot, dataset in yamls.items():
        tables[short(lot)] = compare.lots_per_split(dataset)
    return pd.concat(tables, axis=1, names=["fold"])


def run_folds(yamls, prefix="cv", **recipe) -> dict:
    """Train ``recipe`` (``train_args`` overrides, e.g. ``flipud=0.5``) on every fold; ``{lot: run dir}``.
    A fold whose run already finished is skipped, so a crashed loop can simply be re-run."""
    runs = {}
    for lot, dataset in yamls.items():
        args = training.train_args(
            data=dataset, name=f"{prefix}_{short(lot)}", train_eval=True, **recipe
        )
        runs[lot] = training.launch_once(args)
    return runs


def fold_summary(runs) -> pd.DataFrame:
    """Per fold: best epoch (Ultralytics fitness on the held-out lot) and its val metrics."""
    rows = {}
    for lot, run_dir in runs.items():
        best = best_row(run_dir)
        row = {"epochs": len(read_results(run_dir)), "best epoch": int(best["epoch"])}
        for metric in DETECTION_METRICS:
            row[metric] = best[f"metrics/{metric}"]
        rows[short(lot)] = row
    return pd.DataFrame(rows).T


def cv_threshold(runs, yamls, device=0, batch=16) -> dict:
    """Pooled out-of-lot threshold: every fold's ``best.pt`` predicts its own held-out lot; one sweep over
    all of them picks ``conf``. Also returns per-fold sweeps and their minima, to show how stable it is."""
    held_out_predictions = []
    for lot, run_dir in runs.items():
        weights = Path(run_dir) / "weights" / "best.pt"
        predictions = compare.predict_scores(weights, "val", device, batch, data=yamls[lot])
        lot_predictions = predictions[predictions["lot"] == lot].copy()   # exclude the PUCPR+ part
        lot_predictions["fold"] = short(lot)
        held_out_predictions.append(lot_predictions)
    pooled = pd.concat(held_out_predictions, ignore_index=True)

    predictions_by_fold = {}
    for fold, predictions in pooled.groupby("fold"):
        predictions_by_fold[fold] = predictions
    predictions_by_fold["pooled"] = pooled

    sweeps = {}
    summary = {}
    for fold, predictions in predictions_by_fold.items():
        sweep = compare.conf_sweep(predictions)
        sweeps[fold] = sweep
        default_row = sweep.loc[(sweep["conf"] - CONF).abs().idxmin()]
        summary[fold] = {
            "images": len(predictions),
            "best conf": compare.best_conf(sweep),
            "MAE at best": sweep["MAE"].min(),
            f"MAE at {CONF}": float(default_row["MAE"]),
        }
    return {
        "conf": compare.best_conf(sweeps["pooled"]),
        "sweeps": sweeps,
        "table": pd.DataFrame(summary).T,
        "pooled": pooled,
    }


def plot_cv_threshold(cv):
    return compare.plot_sweeps(cv["sweeps"], title="Count MAE on the held-out lot vs threshold (lot cross-validation)")


def plot_fold_curves(runs):
    """Val mAP50-95 per epoch of every fold (each on its own held-out lot)."""
    _, ax = plt.subplots(figsize=(8, 4))
    for lot, run_dir in runs.items():
        results = read_results(run_dir)
        ax.plot(results["epoch"], results["metrics/mAP50-95(B)"], label=f"val = {short(lot)}")
    ax.set(title="Held-out-lot mAP50-95 per epoch", xlabel="epoch", ylabel="mAP50-95")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax
