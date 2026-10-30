from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
from matplotlib.patches import Rectangle
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

import phase_2.utils as p2u


def yolo_predictor(model, **kwargs):
    def predict(path, conf):
        r = p2u.predict_image(model, path, conf=conf, **kwargs)
        return r.orig_img[..., ::-1], r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    predict.model = model
    return predict


def frcnn_predictor(device=None, **kwargs):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001, box_detections_per_img=300, **kwargs).eval().to(device)
    preprocess, car = weights.transforms(), weights.meta["categories"].index("car")

    @torch.inference_mode()
    def predict(path, conf):
        img = read_image(str(path), mode=ImageReadMode.RGB)
        pred = model([preprocess(img).to(device)])[0]
        keep = (pred["labels"] == car) & (pred["scores"] >= conf)
        return img.permute(1, 2, 0).numpy(), pred["boxes"][keep].cpu().numpy(), pred["scores"][keep].cpu().numpy()
    predict.model = model
    return predict


all_results = {}

def results_df():
    return pd.DataFrame.from_dict(all_results, orient="index").rename_axis("model")

def model_size(model):
    params = sum(p.numel() for p in model.parameters())
    n_bytes = sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()])
    return params / 1e6, n_bytes / 1e6

def predict_and_plot(predict, image_dir="example_images", cap=4):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:cap]
    n_cols = 2
    n_rows = (len(image_paths) + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    for ax, image_path in zip(axes, image_paths):
        img, boxes, _ = predict(image_path, conf=p2u.CONF)
        n_gt = len(image_path.with_suffix(".txt").read_text().splitlines())
        ax.imshow(img)
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=1.5))
        ax.set_title(f"predicted: {len(boxes)} | ground truth: {n_gt}")
        ax.axis("off")
    plt.show()


def predict_and_count(predict, test_ids, model_name, warmup=3):
    true_counts, predicted_counts, tps, confs, latencies = [], [], [], [], []
    
    # Warmup for better latency results
    first_path = p2u.get_image_path(next(iter(test_ids)))
    for _ in range(warmup):
        predict(first_path, conf=p2u.MAP_CONF)
 
    for image_id in test_ids:
        t0 = time.perf_counter()
        _, boxes, scores = predict(p2u.get_image_path(image_id), conf=p2u.MAP_CONF)
        latencies.append((time.perf_counter() - t0) * 1000)
 
        true_counts.append(p2u.gt_count(image_id))
        predicted_counts.append(int((scores >= p2u.CONF).sum()))
        tps.append(p2u.match_detections(boxes, p2u.gt_boxes(image_id)))
        confs.append(scores)

    true_counts, predicted_counts = np.array(true_counts), np.array(predicted_counts)
    metrics = p2u.count_metrics(true_counts, predicted_counts)
    ap = p2u.map_metrics(np.concatenate(tps), np.concatenate(confs), true_counts.sum())

    params_m, size_mb = model_size(predict.model) if hasattr(predict, "model") else (np.nan, np.nan)

    all_results[model_name] = {
        "params (M)": params_m,
        "size (MB)": size_mb,
        "mean true_counts": true_counts.mean(),
        "mean predicted_counts": predicted_counts.mean(),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "mAP50": ap["mAP50"],
        "mAP50-95": ap["mAP50-95"],
        "latency ms (median)": np.median(latencies),
        "FPS": 1000 / np.mean(latencies),
    }
    return results_df().loc[[model_name]]