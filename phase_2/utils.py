from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from ultralytics.utils.metrics import ap_per_class, box_iou

CAR_COCO_ID = 2
IMG_W, IMG_H = 1280, 720
IMGSZ, CONF = 1280, 0.25
MAX_DET = 400
DATA = Path("datasets/CARPK_devkit/data")   # CARPK + PUCPR+ devkit, built by src/1_phase/combine.py
IMAGES, ANN, SETS = DATA / "Images", DATA / "Annotations", DATA / "ImageSets"
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MAP_CONF = 0.001   # mAP needs the full precision-recall curve, so keep low-confidence boxes (as model.val() does)


def load_split_ids(split, sets_dir=SETS) -> list:
    return (Path(sets_dir) / f"{split}.txt").read_text().split()

def get_image_path(img_id, images_dir=IMAGES) -> Path:
    return next(Path(images_dir).glob(f"{img_id}.*"))

def car_class_id(model) -> int:
    """'car' is class 2 in COCO-pretrained models but class 0 after fine-tuning on carpk.yaml."""
    return next(i for i, name in model.names.items() if name == "car")

def predict_image(model, image_path, conf=CONF, **kwargs):
    return model.predict(image_path, imgsz=IMGSZ, conf=conf, max_det=MAX_DET,
                         classes=[car_class_id(model)], verbose=False, **kwargs)[0]

def match_detections(pred_xyxy, gt_xyxy) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a true positive at IoU 0.50, 0.55, ..., 0.95 (Ultralytics' matching)."""
    tp = np.zeros((len(pred_xyxy), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return tp
    iou = box_iou(torch.as_tensor(np.asarray(gt_xyxy), dtype=torch.float32),
                  torch.as_tensor(np.asarray(pred_xyxy), dtype=torch.float32)).numpy()  # (n_gt, n_pred)
    for i, threshold in enumerate(IOU_THRESHOLDS):
        gt_idx, pred_idx = np.nonzero(iou >= threshold)
        if len(gt_idx):
            matches = np.stack([gt_idx, pred_idx], axis=1)
            matches = matches[iou[gt_idx, pred_idx].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # one gt per prediction
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # one prediction per gt
            tp[matches[:, 1], i] = True
    return tp

def map_metrics(tp, confidences, n_gt) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 for the single 'car' class from stacked match_detections() output."""
    ap = ap_per_class(tp, np.asarray(confidences), np.zeros(len(tp)), np.zeros(n_gt))[5]  # (1, 10)
    return {"mAP50": float(ap[0, 0]), "mAP50-95": float(ap[0].mean())}

def read_annotation(path) -> list:
    """(x1, y1, x2, y2, cls) pixel boxes of one annotation file."""
    return [tuple(map(int, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]

def gt_count(img_id) -> int:
    return len(read_annotation(ANN / f"{img_id}.txt"))

def count_metrics(gt, pred) -> dict:
    """Counting error over images: MAE, RMSE, bias (> 0 = overcounting) and total relative error."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    err = pred - gt
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean(),
            "rel_err_%": 100 * np.abs(err).sum() / gt.sum()}

def plot_prediction(image, prediction_count, gt_count, predicted_boxes):
    plt.figure(figsize=(10, 6))
    plt.imshow(image[..., ::-1])  # image is BGR
    for x1, y1, x2, y2 in predicted_boxes:
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=1.5))
    plt.title(f"predicted: {prediction_count} | ground truth: {gt_count}")
    plt.axis("off")
    plt.show()

def split_train(train_ids, val_ratio=0.2):
    """Split randomly shuffled train_ids into 80% train and 20% validation."""
    np.random.shuffle(train_ids)
    split_idx = int((1 - val_ratio) * len(train_ids))
    return train_ids[:split_idx], train_ids[split_idx:]

def parse_annotations(image_ids):
    """Build the box table, exclude invalid boxes, then count cars in each image."""
    rows = []
    for image_id in image_ids:
        for x1, y1, x2, y2, car_class in read_annotation(ANN / f"{image_id}.txt"):
            rows.append({
                "img_id": image_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cls": car_class,
                "w": x2 - x1,
                "h": y2 - y1,
            })
    return rows

def gt_boxes(img_id) -> list:
    """Return the ground truth boxes for a given image ID."""
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, cls in read_annotation(ANN / f"{img_id}.txt")]