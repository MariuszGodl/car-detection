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

def main():
    """Load the checkpoint, attach logging, then start or resume training."""
    weights = sys.argv[1]
    train_args = json.loads(sys.argv[2])
    evaluate_train_subset = train_args.pop("train_eval", False)   # our flag, not an Ultralytics arg
    if train_args.get("resume"):
        run_dir = Path(weights).resolve().parents[1]
        evaluate_train_subset = (run_dir / train_eval.CSV).exists()

    enable_tensorboard()
    model = YOLO(ensure_weights(weights))
    register_callbacks(model)
    if evaluate_train_subset:
        train_eval.register(model)
    model.train(**train_args)


if __name__ == "__main__":
    main()
