"""Model + loss check: YOLO11n with a single-class head (one ``Detect`` head) and the standard
YOLO loss (box CIoU + cls BCE + DFL) running on a real labelled image."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.boxes import read_yolo_labels
from common.config import IMGSZ, MODEL_CFG
from common.model import load_model


def build_detector():
    """1-class YOLO11n with every COCO tensor whose shape still matches (as the trainer does)."""
    model = DetectionModel(MODEL_CFG, nc=1, verbose=False)
    model.load(load_model().model, verbose=False)
    model.names = {0: "car"}
    model.args = get_cfg()   # loss gains used by the criterion
    return model


def describe(model) -> dict:
    """Report the detection head, parameter count and the three YOLO11 loss gains."""
    heads = [layer for layer in model.modules() if isinstance(layer, Detect)]
    head = heads[-1]
    model_args = model.args if isinstance(model.args, dict) else vars(model.args)
    args = {**DEFAULT_CFG_DICT, **model_args}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "n_detect_heads": len(heads),
        "nc": head.nc,
        "outputs_per_anchor": head.no,
        "strides": [int(stride) for stride in head.stride],
        "params_M": round(parameter_count / 1e6, 2),
        "loss": type(model.init_criterion()).__name__,
        "loss_gains": {name: args[name] for name in ("box", "cls", "dfl")},
    }


def compare_with_coco() -> dict:
    """Build the 1-class detector and report what changed vs the 80-class COCO model."""
    coco = load_model().model
    car = build_detector()
    info = {"coco": describe(coco), "car": describe(car)}
    coco_weights = coco.state_dict()
    car_weights = car.state_dict()
    reinitialised = []
    for name, tensor in car_weights.items():
        if name not in coco_weights or coco_weights[name].shape != tensor.shape:
            reinitialised.append(name)

    for name, description in info.items():
        print(f"{name:4s}: {description}")
    print(f"transferred {len(car_weights) - len(reinitialised)}/{len(car_weights)} tensors from COCO; "
          "re-initialised (class-logit convs):")
    for name in reinitialised:
        print("   ", name, tuple(car_weights[name].shape))
    assert info["car"]["n_detect_heads"] == 1 and info["car"]["nc"] == 1
    info["reinitialised_tensors"] = reinitialised
    return info


def letterbox_batch(image_path, label_path) -> dict:
    """Resize without stretching, pad to a square, and move labels into the padded image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = IMGSZ / max(image.size)
    width = round(image.width * scale)
    height = round(image.height * scale)
    top = (IMGSZ - height) // 2
    left = (IMGSZ - width) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[top:top + height, left:left + width] = np.asarray(image.resize((width, height)))

    labels = np.array(read_yolo_labels(label_path), dtype=np.float32)
    boxes = labels[:, 1:] * [width, height, width, height] / IMGSZ
    boxes += [left / IMGSZ, top / IMGSZ, 0, 0]
    return {
        "img": torch.from_numpy(canvas).permute(2, 0, 1)[None].float() / 255,
        "cls": torch.from_numpy(labels[:, :1]),
        "bboxes": torch.from_numpy(boxes.astype(np.float32)),
        "batch_idx": torch.zeros(len(labels)),
    }


def loss_on_image(image_path, label_path, device="cuda:0") -> dict:
    """Forward + backward one labelled image through the fresh 1-class model."""
    model = build_detector().to(device).train()
    batch = letterbox_batch(image_path, label_path)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    loss, loss_items = model.loss(batch)
    loss.sum().backward()
    result = {name: float(loss_items[name]) for name in ("box_loss", "cls_loss", "dfl_loss")}
    result["grads_ok"] = all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"{Path(image_path).name}: {len(batch['cls'])} GT cars | {result}")
    assert result["grads_ok"] and np.isfinite(list(result.values())).all()
    return result
