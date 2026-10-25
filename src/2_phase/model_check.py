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
    heads = [m for m in model.modules() if isinstance(m, Detect)]
    args = {**DEFAULT_CFG_DICT, **(model.args if isinstance(model.args, dict) else vars(model.args))}
    return {"n_detect_heads": len(heads), "nc": heads[-1].nc, "outputs_per_anchor": heads[-1].no,
            "strides": [int(s) for s in heads[-1].stride], "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "loss": type(model.init_criterion()).__name__, "loss_gains": {k: args[k] for k in ("box", "cls", "dfl")}}


def compare_with_coco() -> dict:
    """Build the 1-class detector and report what changed vs the 80-class COCO model."""
    coco, car = load_model().model, build_detector()
    info = {"coco": describe(coco), "car": describe(car)}
    coco_sd, car_sd = coco.state_dict(), car.state_dict()
    reinit = [k for k, v in car_sd.items() if k not in coco_sd or coco_sd[k].shape != v.shape]
    for name, d in info.items():
        print(f"{name:4s}: {d}")
    print(f"transferred {len(car_sd) - len(reinit)}/{len(car_sd)} tensors from COCO; re-initialised (class-logit convs):")
    for k in reinit:
        print("   ", k, tuple(car_sd[k].shape))
    assert info["car"]["n_detect_heads"] == 1 and info["car"]["nc"] == 1
    info["reinitialised_tensors"] = reinit
    return info


def letterbox_batch(image_path, label_path) -> dict:
    """1-image Ultralytics training batch, letterboxed to ``IMGSZ``."""
    img = Image.open(image_path).convert("RGB")
    r = IMGSZ / max(img.size)
    nw, nh = round(img.width * r), round(img.height * r)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(img.resize((nw, nh)))

    rows = np.array(read_yolo_labels(label_path), dtype=np.float32)
    xywh = rows[:, 1:] * [nw, nh, nw, nh] / IMGSZ + [left / IMGSZ, top / IMGSZ, 0, 0]
    return {"img": torch.from_numpy(canvas).permute(2, 0, 1)[None].float() / 255,
            "cls": torch.from_numpy(rows[:, :1]), "bboxes": torch.from_numpy(xywh.astype(np.float32)),
            "batch_idx": torch.zeros(len(rows))}


def loss_on_image(image_path, label_path, device="cuda:0") -> dict:
    """Forward + backward one labelled image through the fresh 1-class model."""
    model = build_detector().to(device).train()
    batch = {k: v.to(device) for k, v in letterbox_batch(image_path, label_path).items()}
    loss, items = model.loss(batch)
    loss.sum().backward()
    out = {k: float(items[k]) for k in ("box_loss", "cls_loss", "dfl_loss")}
    out["grads_ok"] = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"{Path(image_path).name}: {len(batch['cls'])} GT cars | {out}")
    assert out["grads_ok"] and np.isfinite(list(out.values())).all()
    return out
