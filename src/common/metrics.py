"""Scores used in every phase: detection (Ultralytics) and counting (count = number of boxes)."""
import numpy as np

# Ultralytics' detection metrics, as named in results.csv (prefixed "metrics/") and results_dict.
DETECTION_METRICS = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]


def fitness(results):
    """Ultralytics' model-selection score per epoch of a ``results.csv`` table; ``best.pt`` is its max."""
    return 0.1 * results["metrics/mAP50(B)"] + 0.9 * results["metrics/mAP50-95(B)"]


def count_metrics(gt, pred) -> dict:
    """Counting error over images: MAE, RMSE, bias (> 0 = overcounting) and total relative error."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    err = pred - gt
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean(),
            "rel_err_%": 100 * np.abs(err).sum() / gt.sum()}
