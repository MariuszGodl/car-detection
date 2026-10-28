"""Compare fine-tuned checkpoints on the same held-out test split (Phase 3).

1. Choose a confidence threshold on validation images, or use one fixed beforehand.
2. Measure detection quality and car-count error on the test images.
3. Compare counts overall, by dataset source, and image by image.

Inference runs once per split at low confidence. Higher thresholds just filter the saved box scores,
so a threshold sweep does not rerun the model. Never choose a threshold using test images.

YOLO26 uses its one-to-many head + NMS by default; ``nms=False`` selects its NMS-free head.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import read_yolo_labels
from common.config import CONF, DATA_YAML, IMGSZ, MAX_DET, ROOT, RUNS_DIR
from common.data import dataset_dir, lot_of, source_of
from common.metrics import DETECTION_METRICS, count_metrics

SWEEP = np.round(np.arange(0.10, 0.701, 0.025), 3)   # thresholds tried when tuning conf
MIN_CONF = 0.05                                      # predictions keep every box above this


# ---------------------------------------------------------------- data

def lots_per_split(data=DATA_YAML, splits=("train", "val", "test")) -> pd.DataFrame:
    """#images per parking lot and split - shows whether the test lots were ever seen in training."""
    dataset = dataset_dir(data)
    rows = []
    for split in splits:
        for image in (dataset / "images" / split).iterdir():
            rows.append({"lot": lot_of(image.stem), "split": split})
    images = pd.DataFrame(rows)
    return pd.crosstab(images["lot"], images["split"]).reindex(columns=list(splits), fill_value=0)


# ---------------------------------------------------------------- predictions and counts

def predict_scores(weights, split, device=0, batch=16, min_conf=MIN_CONF, data=DATA_YAML, nms=None,
                   iou=0.7) -> pd.DataFrame:
    """One row per ``split`` image: source, lot, GT count and the confidences of all kept boxes.
    ``iou`` is the NMS overlap threshold (Ultralytics default 0.7); ``nms=False`` selects an NMS-free head."""
    model = YOLO(str(weights))
    dataset = dataset_dir(data)
    images = sorted((dataset / "images" / split).iterdir())
    rows = []
    for start in range(0, len(images), batch):
        image_batch = images[start:start + batch]
        predictions = model.predict(
            [str(image) for image in image_batch], imgsz=IMGSZ, conf=min_conf, max_det=MAX_DET,
            iou=iou, nms=nms, device=device, verbose=False,
        )
        for image, prediction in zip(image_batch, predictions):
            label_path = dataset / "labels" / split / f"{image.stem}.txt"
            rows.append({
                "image": image.stem,
                "source": source_of(image.stem),
                "lot": lot_of(image.stem),
                "gt": len(read_yolo_labels(label_path)),
                "scores": prediction.boxes.conf.cpu().numpy(),
            })

    scores = pd.DataFrame(rows)
    capped_images = int((scores["scores"].apply(len) >= MAX_DET).sum())
    if capped_images:
        print(f"WARNING: {capped_images} {split} images hit max_det={MAX_DET}; "
              f"counts near conf={min_conf} are clipped")
    return scores


def counts_at(df, conf) -> pd.Series:
    """Predicted count per image at threshold ``conf`` (boxes scoring >= conf)."""
    return df["scores"].apply(lambda scores: int((scores >= conf).sum()))


def conf_sweep(df, confs=SWEEP) -> pd.DataFrame:
    """Count metrics on ``df``'s images for every threshold in ``confs``."""
    rows = []
    for confidence in confs:
        predicted_counts = counts_at(df, confidence)
        metrics = count_metrics(df["gt"], predicted_counts)
        rows.append({"conf": confidence, **metrics})
    return pd.DataFrame(rows)


def best_conf(sweep) -> float:
    """Threshold with the lowest count MAE."""
    return float(sweep.loc[sweep["MAE"].idxmin(), "conf"])


def detection_metrics(weights, label, split="test", device=0, batch=16, data=DATA_YAML, nms=None) -> dict:
    """Ultralytics P / R / mAP on ``split``; plots go to ``runs/compare/<label>_<split>/``."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=str((ROOT / data).resolve()), split=split, imgsz=IMGSZ, batch=batch,
        device=device, max_det=MAX_DET, nms=nms, project=str(RUNS_DIR / "compare"),
        name=f"{label}_{split}", exist_ok=True, verbose=False,
    )
    return {name: metrics.results_dict[f"metrics/{name}"] for name in DETECTION_METRICS}


# ---------------------------------------------------------------- comparing models

def compare(models, device=0, batch=16, test_data=DATA_YAML) -> dict:
    """Evaluate every model of ``models`` (``{label: best.pt}`` or ``{label: {"weights", "val_data", "conf", "nms"}}``)
    on the same test images.

    Without ``conf`` a model's threshold is the one with the lowest count MAE on ``val_data``'s val split -
    which must be a split that model never trained on. With ``conf`` that threshold is used as given.
    Returns ``summary`` (one row per model), ``by_source`` (count metrics per source at the chosen
    threshold), ``sweeps`` (val MAE per threshold, tuned models only), ``per_image`` (test counts) and
    ``conf_source`` (how each threshold was chosen).
    """
    summary = {}
    by_source = {}
    sweeps = {}
    per_image = {}
    conf_source = {}
    for label, model in models.items():
        settings = {"weights": model} if isinstance(model, (str, Path)) else model
        weights = settings["weights"]
        nms = settings.get("nms")
        confidence = settings.get("conf")

        if confidence is None:
            validation_data = settings.get("val_data", DATA_YAML)
            validation = predict_scores(weights, "val", device, batch, data=validation_data, nms=nms)
            sweeps[label] = conf_sweep(validation)
            confidence = best_conf(sweeps[label])
            conf_source[label] = f"tuned on val of {Path(validation_data).parent.name}"
        else:
            confidence = float(confidence)
            conf_source[label] = "fixed"

        test = predict_scores(weights, "test", device, batch, data=test_data, nms=nms)
        row = detection_metrics(weights, label, "test", device, batch, data=test_data, nms=nms)
        row["tuned conf"] = confidence
        for tag, threshold in ((f"@{CONF}", CONF), ("@tuned", confidence)):
            metrics = count_metrics(test["gt"], counts_at(test, threshold))
            for name, value in metrics.items():
                row[f"{name} {tag}"] = value
        summary[label] = row

        counts = test.drop(columns="scores")
        counts["pred"] = counts_at(test, confidence)
        counts["err"] = counts["pred"] - counts["gt"]
        per_image[label] = counts

        source_metrics = {}
        for source, source_counts in counts.groupby("source"):
            source_metrics[source] = {
                "images": len(source_counts),
                **count_metrics(source_counts["gt"], source_counts["pred"]),
            }
        by_source[label] = pd.DataFrame(source_metrics).T

    by_source = pd.concat(by_source, names=["model", "source"])
    by_source["images"] = by_source["images"].astype(int)
    for label, source in conf_source.items():
        print(f"{label}: conf {summary[label]['tuned conf']:.3f} ({source})")
    return {
        "summary": pd.DataFrame(summary).T,
        "by_source": by_source,
        "sweeps": sweeps,
        "per_image": per_image,
        "conf_source": conf_source,
    }


def nms_check(weights, split="test", ious=(0.5, 0.7, 0.9), conf=CONF, device=0, batch=16,
              data=DATA_YAML) -> pd.DataFrame:
    """Count metrics at a fixed ``conf`` for several NMS IoU thresholds. If raising ``iou`` (= suppressing
    less) raises the count and lowers the error, NMS is removing real, heavily overlapping cars."""
    rows = {}
    for iou in ious:
        predictions = predict_scores(weights, split, device, batch, min_conf=conf, data=data, iou=iou)
        counts = counts_at(predictions, conf)
        rows[iou] = {
            "mean count": counts.mean(),
            "mean GT": predictions["gt"].mean(),
            **count_metrics(predictions["gt"], counts),
        }
    return pd.DataFrame(rows).T.rename_axis("NMS iou")


def head_to_head(result, a, b) -> pd.DataFrame:
    """Per test image |error| of model ``a`` vs ``b`` (chosen thresholds) and who wins."""
    counts_a = result["per_image"][a].set_index("image")
    counts_b = result["per_image"][b].set_index("image")
    comparison = pd.DataFrame({
        "source": counts_a["source"],
        "gt": counts_a["gt"],
        f"pred {a}": counts_a["pred"],
        f"pred {b}": counts_b["pred"],
        f"|err| {a}": counts_a["err"].abs(),
        f"|err| {b}": counts_b["err"].abs(),
    })
    error_difference = comparison[f"|err| {a}"] - comparison[f"|err| {b}"]
    comparison["closer"] = np.select([error_difference < 0, error_difference > 0], [a, b], "tie")
    print(comparison["closer"].value_counts().to_string())
    return comparison.sort_values(f"|err| {a}", ascending=False)


# ---------------------------------------------------------------- plots

def plot_summary(result):
    """Detection metrics on test, grouped by model."""
    metrics = result["summary"][DETECTION_METRICS].astype(float)
    ax = metrics.T.plot.bar(figsize=(10, 4.5), rot=0, ylim=(0, 1), width=0.75)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set(title="Detection on the test split", ylabel="score")
    ax.legend(title="model", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return ax


def plot_sweeps(sweeps, title="Count MAE on val vs confidence threshold"):
    """Val count MAE vs threshold per model; dot = minimum. ``sweeps`` = ``{label: sweep}`` or a
    :func:`compare` result."""
    sweeps = sweeps.get("sweeps", sweeps)
    _, ax = plt.subplots(figsize=(8, 4))
    for label, sweep in sweeps.items():
        line, = ax.plot(sweep["conf"], sweep["MAE"], label=label)
        best = sweep.loc[sweep["MAE"].idxmin()]
        ax.plot(best["conf"], best["MAE"], "o", color=line.get_color())
    ax.axvline(CONF, color="grey", ls="--", lw=0.8, label=f"default conf {CONF}")
    ax.set(title=title, xlabel="conf", ylabel="MAE (cars / image)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax


def plot_counts(result, labels=None):
    """Predicted vs GT count on every test image (one panel per model in ``labels``, default all) and
    the error distributions."""
    labels = list(labels or result["per_image"])
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(5.5 * (len(labels) + 1), 5))
    largest_count = max(result["per_image"][label][["gt", "pred"]].to_numpy().max() for label in labels)
    axis_limit = largest_count * 1.05
    for ax, label in zip(axes, labels):
        counts = result["per_image"][label]
        for source, source_counts in counts.groupby("source"):
            ax.scatter(source_counts["gt"], source_counts["pred"], s=10, alpha=0.6, label=source)
        ax.plot([0, axis_limit], [0, axis_limit], color="grey", lw=0.8)
        metrics = count_metrics(counts["gt"], counts["pred"])
        ax.set(
            title=f"{label}\nMAE {metrics['MAE']:.2f} | RMSE {metrics['RMSE']:.2f} | bias {metrics['bias']:+.2f}",
            xlabel="GT count", ylabel="predicted count", xlim=(0, axis_limit), ylim=(0, axis_limit),
        )
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
