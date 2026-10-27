"""Compare fine-tuned checkpoints on the same held-out test split (Phase 3).

For every model (``{label: path/to/best.pt}``):

* **detection** - Ultralytics val on ``test`` (P / R / mAP50 / mAP50-95) with ``max_det`` raised,
* **counting** - count = #boxes on *every* test image: MAE / RMSE / bias / relative MAE, overall and
  per source (CARPK / PUCPR+), at the default ``conf=0.25`` and at the threshold that minimises
  count MAE on **val** (chosen without looking at test, so the test number stays honest),
* **head to head** - per test image, which model's count is closer to the ground truth.

Predictions are made once per split at a low threshold and the count at any ``conf >= min_conf`` is
read off the box scores. That is exact: greedy NMS keeps a box only if no *higher*-scored box overlaps
it, so the boxes scoring >= t after NMS are the boxes ``predict(conf=t)`` would return.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import read_yolo_labels
from common.config import CONF, DATA_YAML, IMGSZ, MAX_DET, ROOT, RUNS_DIR, YOLO_DIR

DET = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
SWEEP = np.round(np.arange(0.10, 0.701, 0.025), 3)
MIN_CONF = 0.05


def source_of(stem) -> str:
    return stem.split("__", 1)[0]


def lot_of(stem) -> str:
    """``carpk__20160331_NTU_00001`` -> ``carpk:NTU``; anything else -> its source (PUCPR+ is one lot)."""
    source, _, rest = stem.partition("__")
    parts = rest.split("_")
    return f"carpk:{parts[1]}" if source == "carpk" and len(parts) >= 3 else source


def lots_per_split(splits=("train", "val", "test")) -> pd.DataFrame:
    """#images per parking lot and split - shows whether the test lots were ever seen in training."""
    rows = [(lot_of(p.stem), s) for s in splits for p in (ROOT / YOLO_DIR / "images" / s).iterdir()]
    return pd.crosstab(pd.Series([r[0] for r in rows], name="lot"),
                       pd.Series([r[1] for r in rows], name="split"))[list(splits)]


def predict_scores(weights, split, device=0, batch=16, min_conf=MIN_CONF) -> pd.DataFrame:
    """One row per ``split`` image: source, GT count and the confidences of all post-NMS boxes."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = sorted((ROOT / YOLO_DIR / "images" / split).iterdir())
    rows = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        results = model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=min_conf, max_det=MAX_DET,
                                device=device, verbose=False)
        for img, r in zip(chunk, results):
            gt = len(read_yolo_labels(ROOT / YOLO_DIR / "labels" / split / f"{img.stem}.txt"))
            rows.append({"image": img.stem, "source": source_of(img.stem), "gt": gt,
                         "scores": r.boxes.conf.cpu().numpy()})
    df = pd.DataFrame(rows)
    capped = int((df["scores"].apply(len) >= MAX_DET).sum())
    if capped:
        print(f"WARNING: {capped} {split} images hit max_det={MAX_DET}; counts near conf={min_conf} are clipped")
    return df


def counts_at(df, conf) -> pd.Series:
    return df["scores"].apply(lambda s: int((s >= conf).sum()))


def count_metrics(gt, pred) -> dict:
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    err = pred - gt
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean(),
            "rel_err_%": 100 * np.abs(err).sum() / gt.sum()}


def conf_sweep(df, confs=SWEEP) -> pd.DataFrame:
    """Count metrics on ``df``'s images for every threshold in ``confs``."""
    return pd.DataFrame([{"conf": c, **count_metrics(df["gt"], counts_at(df, c))} for c in confs])


def detection_metrics(weights, label, split="test", device=0, batch=16) -> dict:
    """Ultralytics P / R / mAP on ``split``; plots go to ``runs/compare/<label>_<split>/``."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(ROOT / DATA_YAML), split=split, imgsz=IMGSZ, batch=batch, device=device,
                               max_det=MAX_DET, project=str(RUNS_DIR / "compare"), name=f"{label}_{split}",
                               exist_ok=True, verbose=False)
    return {k: m.results_dict[f"metrics/{k}"] for k in DET}


def compare(models, device=0, batch=16) -> dict:
    """Evaluate every ``{label: best.pt}`` on the same val / test images.

    Returns ``summary`` (one row per model), ``by_source`` (count metrics per source at the val-tuned
    threshold), ``sweeps`` (val MAE per threshold) and ``per_image`` (test counts per model).
    """
    summary, by_source, sweeps, per_image = {}, {}, {}, {}
    for label, weights in models.items():
        val_df = predict_scores(weights, "val", device, batch)
        test_df = predict_scores(weights, "test", device, batch)
        sweeps[label] = conf_sweep(val_df)
        best = float(sweeps[label].loc[sweeps[label]["MAE"].idxmin(), "conf"])

        row = detection_metrics(weights, label, "test", device, batch)
        row["tuned conf"] = best
        for tag, conf in ((f"@{CONF}", CONF), ("@tuned", best)):
            row.update({f"{k} {tag}": v for k, v in count_metrics(test_df["gt"], counts_at(test_df, conf)).items()})
        summary[label] = row

        test_df = test_df.assign(pred=counts_at(test_df, best)).drop(columns="scores")
        test_df["err"] = test_df["pred"] - test_df["gt"]
        per_image[label] = test_df
        by_source[label] = pd.DataFrame({src: {"images": len(g), **count_metrics(g["gt"], g["pred"])}
                                         for src, g in test_df.groupby("source")}).T

    summary = pd.DataFrame(summary).T
    by_source = pd.concat(by_source, names=["model", "source"])
    by_source["images"] = by_source["images"].astype(int)
    return {"summary": summary, "by_source": by_source, "sweeps": sweeps, "per_image": per_image}


def head_to_head(result, a, b) -> pd.DataFrame:
    """Per test image |error| of model ``a`` vs ``b`` (val-tuned thresholds) and who wins."""
    pa, pb = result["per_image"][a].set_index("image"), result["per_image"][b].set_index("image")
    df = pd.DataFrame({"source": pa["source"], "gt": pa["gt"], f"pred {a}": pa["pred"], f"pred {b}": pb["pred"],
                       f"|err| {a}": pa["err"].abs(), f"|err| {b}": pb["err"].abs()})
    diff = df[f"|err| {a}"] - df[f"|err| {b}"]
    df["closer"] = np.select([diff < 0, diff > 0], [a, b], "tie")
    print(df["closer"].value_counts().to_string())
    return df.sort_values(f"|err| {a}", ascending=False)


def plot_summary(result):
    """Detection metrics on test, grouped by model."""
    ax = result["summary"][DET].astype(float).T.plot.bar(figsize=(10, 4.5), rot=0, ylim=(0, 1), width=0.75)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.set(title="Detection on the test split", ylabel="score")
    ax.legend(title="model", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return ax


def plot_sweeps(result):
    """Val count MAE vs confidence threshold; the minimum is the threshold used on test."""
    _, ax = plt.subplots(figsize=(8, 4))
    for label, sw in result["sweeps"].items():
        line, = ax.plot(sw["conf"], sw["MAE"], label=label)
        best = sw.loc[sw["MAE"].idxmin()]
        ax.plot(best["conf"], best["MAE"], "o", color=line.get_color())
    ax.axvline(CONF, color="grey", ls="--", lw=0.8, label=f"default conf {CONF}")
    ax.set(title="Count MAE on val vs confidence threshold", xlabel="conf", ylabel="MAE (cars / image)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax


def plot_counts(result):
    """Predicted vs GT count on every test image (one panel per model) and the error distributions."""
    labels = list(result["per_image"])
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(5.5 * (len(labels) + 1), 5))
    hi = max(df[["gt", "pred"]].to_numpy().max() for df in result["per_image"].values()) * 1.05
    for ax, label in zip(axes, labels):
        df = result["per_image"][label]
        for src, g in df.groupby("source"):
            ax.scatter(g["gt"], g["pred"], s=10, alpha=0.6, label=src)
        ax.plot([0, hi], [0, hi], color="grey", lw=0.8)
        m = count_metrics(df["gt"], df["pred"])
        ax.set(title=f"{label}\nMAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | bias {m['bias']:+.2f}",
               xlabel="GT count", ylabel="predicted count", xlim=(0, hi), ylim=(0, hi))
        ax.legend()
    bins = np.arange(-40.5, 41.5, 2)
    for label in labels:
        axes[-1].hist(result["per_image"][label]["err"].clip(-40, 40), bins=bins, alpha=0.5, label=label)
    axes[-1].axvline(0, color="grey", lw=0.8)
    axes[-1].set(title="Count error per test image (clipped to ±40)", xlabel="pred − GT", ylabel="# images")
    axes[-1].legend()
    fig.tight_layout()
    plt.show()
    return fig
