"""Sanity check: YOLO must overfit a few images (train == val). Run: ``python -m phase_2.sanity``."""
import shutil
from pathlib import Path

from ultralytics.utils import LOGGER

from phase_2.console_logging import training_console

SRC = Path("phase_2/yolo")
DST = Path("phase_2/yolo_sanity").resolve()


def dataset_sanity(n_images=4):
    """Copy the first ``n_images`` train images + labels into a tiny dataset; return its yaml."""
    for directory in (DST / "images", DST / "labels"):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    for img in sorted((SRC / "images/train").glob("*.png"))[:n_images]:
        shutil.copy(img, DST / "images")
        shutil.copy(SRC / "labels/train" / f"{img.stem}.txt", DST / "labels")
    yaml_path = DST / "data.yaml"
    yaml_path.write_text(f"path: {DST}\ntrain: images\nval: images\nnames:\n  0: car\n")
    return yaml_path


def run(model, epochs=200, n_images=4):
    """Run the sanity check and append its console output under tranning/."""
    model_name = Path(getattr(model, "ckpt_path", "")).stem or "overfit"
    with training_console(model_name, sanity=True):
        return _run(model, epochs, n_images)


def _run(model, epochs, n_images):
    """Train ``model`` (path or YOLO object) on ``n_images`` images and print mAP50 on them."""
    data = str(dataset_sanity(n_images))
    batch = min(n_images, 16)
    model_name = Path(getattr(model, "ckpt_path", "")).stem or "overfit"
    
    model.train(
        data=data, epochs=epochs, warmup_epochs=0, imgsz=640, batch=batch, device=0, workers=0,
        augment=False, mosaic=0, fliplr=0, scale=0, translate=0, hsv_h=0, hsv_s=0, hsv_v=0,
        plots=False, project=str(DST / "runs"), name=f"overfit_{model_name}", exist_ok=True, verbose=False
    )
    metrics = model.val(data=data, batch=batch, device=0, plots=False)
    LOGGER.info(f"mAP50 = {metrics.box.map50:.3f} (should be ~1.0 if overfitting works)")
    return metrics


if __name__ == "__main__":
    run()
