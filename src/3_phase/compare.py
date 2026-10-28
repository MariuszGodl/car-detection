"""Compare fine-tuned checkpoints on the same held-out test split (Phase 3).

For every model:

* **detection** - Ultralytics val on ``test`` (P / R / mAP50 / mAP50-95) with ``max_det`` raised,
* **counting** - count = #boxes on *every* test image: MAE / RMSE / bias / relative error, overall and
  per source (CARPK / PUCPR+), at the default ``conf=0.25`` and at a chosen threshold. The threshold is
  either tuned on a val split the model never trained on, or fixed up front (e.g. the lot cross-validation
  threshold from ``lot_cv.cv_threshold``) - never picked on test,
* **head to head** - per test image, which model's count is closer to the ground truth.

Predictions are made once per split at a low threshold and the count at any ``conf >= min_conf`` is
read off the box scores. That is exact: greedy NMS keeps a box only if no *higher*-scored box overlaps
it, so the boxes scoring >= t after NMS are the boxes ``predict(conf=t)`` would return; an NMS-free head
(YOLO26 with ``nms=False``) keeps the top ``max_det`` boxes by score, which gives the same property.

YOLO26 checkpoints carry two heads. Ultralytics uses the one-to-many head + NMS unless ``nms=False`` is
passed, so a model spec can set ``"nms": False`` to score the NMS-free head.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import read_yolo_labels
from common.config import CONF, DATA_YAML, IMGSZ, MAX_DET, ROOT, RUNS_DIR
from common.data import lot_of

DET = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
SWEEP = np.round(np.arange(0.10, 0.701, 0.025), 3)
MIN_CONF = 0.05


def source_of(stem) -> str:
    return stem.split("__", 1)[0]


def dataset_dir(data=DATA_YAML) -> Path:
    """Directory of a dataset yaml (``images/`` and ``labels/`` live next to it)."""
    return (ROOT / data).resolve().parent


def lots_per_split(data=DATA_YAML, splits=("train", "val", "test")) -> pd.DataFrame:
    """#images per parking lot and split - shows whether the test lots were ever seen in training."""
    rows = [(lot_of(p.stem), s) for s in splits for p in (dataset_dir(data) / "images" / s).iterdir()]
    return pd.crosstab(pd.Series([r[0] for r in rows], name="lot"),
                       pd.Series([r[1] for r in rows], name="split")).reindex(columns=list(splits), fill_value=0)


def predict_scores(weights, split, device=0, batch=16, min_conf=MIN_CONF, data=DATA_YAML, nms=None,
                   iou=0.7) -> pd.DataFrame:
    """One row per ``split`` image: source, lot, GT count and the confidences of all kept boxes.
    ``iou`` is the NMS overlap threshold (Ultralytics default 0.7); ``nms=False`` selects an NMS-free head."""
    from ultralytics import YOLO

    model, ds = YOLO(str(weights)), dataset_dir(data)
    imgs = sorted((ds / "images" / split).iterdir())
    rows = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        results = model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=min_conf, max_det=MAX_DET,
                                iou=iou, nms=nms, device=device, verbose=False)
        for img, r in zip(chunk, results):
            gt = len(read_yolo_labels(ds / "labels" / split / f"{img.stem}.txt"))
            rows.append({"image": img.stem, "source": source_of(img.stem), "lot": lot_of(img.stem), "gt": gt,
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


def best_conf(sweep) -> float:
    return float(sweep.loc[sweep["MAE"].idxmin(), "conf"])


def detection_metrics(weights, label, split="test", device=0, batch=16, data=DATA_YAML, nms=None) -> dict:
    """Ultralytics P / R / mAP on ``split``; plots go to ``runs/compare/<label>_<split>/``."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str((ROOT / data).resolve()), split=split, imgsz=IMGSZ,
                               batch=batch, device=device, max_det=MAX_DET, nms=nms, project=str(RUNS_DIR / "compare"),
                               name=f"{label}_{split}", exist_ok=True, verbose=False)
    return {k: m.results_dict[f"metrics/{k}"] for k in DET}


def _spec(model) -> dict:
    """A model is a ``best.pt`` path, or ``{"weights", "val_data"?, "conf"?, "nms"?}``."""
    spec = {"weights": model} if isinstance(model, (str, Path)) else dict(model)
    spec.setdefault("val_data", DATA_YAML)
    spec.setdefault("conf", None)
    spec.setdefault("nms", None)
    return spec


def compare(models, device=0, batch=16, test_data=DATA_YAML) -> dict:
    """Evaluate every model of ``models`` (``{label: best.pt}`` or ``{label: {"weights", "val_data", "conf", "nms"}}``)
    on the same test images.

    Without ``conf`` a model's threshold is the one with the lowest count MAE on ``val_data``'s val split -
    which must be a split that model never trained on. With ``conf`` that threshold is used as given.
    Returns ``summary`` (one row per model), ``by_source`` (count metrics per source at the chosen
    threshold), ``sweeps`` (val MAE per threshold, tuned models only), ``per_image`` (test counts) and
    ``conf_source`` (how each threshold was chosen).
    """
    summary, by_source, sweeps, per_image, conf_source = {}, {}, {}, {}, {}
    for label, model in models.items():
        spec = _spec(model)
        if spec["conf"] is None:
            sweeps[label] = conf_sweep(predict_scores(spec["weights"], "val", device, batch, data=spec["val_data"],
                                                      nms=spec["nms"]))
            conf, conf_source[label] = best_conf(sweeps[label]), f"tuned on val of {Path(spec['val_data']).parent.name}"
        else:
            conf, conf_source[label] = float(spec["conf"]), "fixed"
        test_df = predict_scores(spec["weights"], "test", device, batch, data=test_data, nms=spec["nms"])

        row = detection_metrics(spec["weights"], label, "test", device, batch, data=test_data, nms=spec["nms"])
        row["tuned conf"] = conf
        for tag, c in ((f"@{CONF}", CONF), ("@tuned", conf)):
            row.update({f"{k} {tag}": v for k, v in count_metrics(test_df["gt"], counts_at(test_df, c)).items()})
        summary[label] = row

        test_df = test_df.assign(pred=counts_at(test_df, conf)).drop(columns="scores")
        test_df["err"] = test_df["pred"] - test_df["gt"]
        per_image[label] = test_df
        by_source[label] = pd.DataFrame({src: {"images": len(g), **count_metrics(g["gt"], g["pred"])}
                                         for src, g in test_df.groupby("source")}).T

    by_source = pd.concat(by_source, names=["model", "source"])
    by_source["images"] = by_source["images"].astype(int)
    print("\n".join(f"{k}: conf {summary[k]['tuned conf']:.3f} ({v})" for k, v in conf_source.items()))
    return {"summary": pd.DataFrame(summary).T, "by_source": by_source, "sweeps": sweeps,
            "per_image": per_image, "conf_source": conf_source}


def nms_check(weights, split="test", ious=(0.5, 0.7, 0.9), conf=CONF, device=0, batch=16, data=DATA_YAML) -> pd.DataFrame:
    """Count metrics at a fixed ``conf`` for several NMS IoU thresholds. If raising ``iou`` (= suppressing
    less) raises the count and lowers the error, NMS is removing real, heavily overlapping cars."""
    rows = {}
    for iou in ious:
        df = predict_scores(weights, split, device, batch, min_conf=conf, data=data, iou=iou)
        pred = counts_at(df, conf)
        rows[iou] = {"mean count": pred.mean(), "mean GT": df["gt"].mean(), **count_metrics(df["gt"], pred)}
    return pd.DataFrame(rows).T.rename_axis("NMS iou")


def head_to_head(result, a, b) -> pd.DataFrame:
    """Per test image |error| of model ``a`` vs ``b`` (chosen thresholds) and who wins."""
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


def plot_sweeps(sweeps, title="Count MAE on val vs confidence threshold"):
    """Val count MAE vs threshold per model; dot = minimum. ``sweeps`` = ``{label: sweep}`` or a
    :func:`compare` result."""
    sweeps = sweeps.get("sweeps", sweeps)
    _, ax = plt.subplots(figsize=(8, 4))
    for label, sw in sweeps.items():
        line, = ax.plot(sw["conf"], sw["MAE"], label=label)
        best = sw.loc[sw["MAE"].idxmin()]
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
    hi = max(result["per_image"][k][["gt", "pred"]].to_numpy().max() for k in labels) * 1.05
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


def show_prediction(weights, image=None, split="test", conf=0.25, nms=None, device=0, seed=None):
    """Predict one image with ``weights`` and show the boxes and the car count.

    ``image``: path to any photo; ``None`` picks a random image of ``split`` from ``datasets/yolo``.
    If the image has a label file there, the true count and the error are shown too.
    ``nms=False`` uses YOLO26's NMS-free head; ``seed`` makes the random pick repeatable."""
    if image is None:
        image = random.Random(seed).choice(sorted(Path("datasets/yolo/images", split).iterdir()))
    image = Path(image).expanduser()
    r = YOLO(str(weights)).predict(str(image), imgsz=IMGSZ, conf=conf, max_det=MAX_DET, nms=nms,
                                   device=device, verbose=False)[0]
    pred = len(r.boxes)
    title = f"{image.name}\npredicted {pred} cars (conf ≥ {conf})"
    label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
    if label.exists():
        gt = len(read_yolo_labels(label))
        title += f"  |  true {gt}  |  error {pred - gt:+d}"
    plt.figure(figsize=(14, 8))
    plt.imshow(r.plot(labels=False, conf=False, line_width=1)[..., ::-1])   # BGR -> RGB
    plt.title(title)
    plt.axis("off")
    plt.show()
    return r