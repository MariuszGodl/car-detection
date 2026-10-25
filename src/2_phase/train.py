"""Training subprocess started by ``common.training.launch``:

    python src/2_phase/train.py <weights or last.pt> '<json Ultralytics train kwargs>'
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from common import train_eval
from common.model import ensure_weights
from common.run_logging import enable_tensorboard, register_callbacks

kwargs = json.loads(sys.argv[2])
use_train_eval = kwargs.pop("train_eval", False)   # not an Ultralytics arg
if kwargs.get("resume"):
    use_train_eval = (Path(sys.argv[1]).resolve().parents[1] / train_eval.CSV).exists()

enable_tensorboard()
model = YOLO(ensure_weights(sys.argv[1]))
register_callbacks(model)
if use_train_eval:
    train_eval.register(model)
model.train(**kwargs)
