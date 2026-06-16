"""Fine-tune YOLOv8 on the NEU Steel Surface Defect dataset.

Usage:
    python src/train.py
    python src/train.py --model yolov8s --epochs 50 --batch 16
    python src/train.py --model yolov8m --epochs 100 --batch 8

Model size guide:
    yolov8n  ~3M params   fastest,  lower accuracy  (good for CPU-only machines)
    yolov8s  ~11M params  balanced  (recommended default)
    yolov8m  ~26M params  slowest,  highest accuracy

Training artifacts are written to runs/train/neu_det/.
The best checkpoint is also copied to models/best.pt for easy access.
"""

import argparse
import shutil
import sys
import types
from pathlib import Path

# Add repo root to sys.path so 'src' can be imported
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def ensure_torchvision_stub() -> None:
    """Provide a minimal torchvision stub when the local Python build lacks _bz2.

    Ultralytics imports SAM support transitively, which imports torchvision at module
    load time. In this environment, importing torchvision fails because the Python
    build is missing the _bz2 extension. Training YOLO detection does not need
    torchvision, so we install a tiny placeholder module only in that case.
    """
    try:
        import bz2  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "_bz2":
            raise
        if "torchvision" not in sys.modules:
            sys.modules["torchvision"] = types.ModuleType("torchvision")


def train(
    model_size: str = "yolov8s",
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    data_cfg: str = "configs/neu_det.yaml",
    output_dir: str = "runs/train",
    device: str = "",
) -> Path:
    ensure_torchvision_stub()
    from ultralytics import YOLO
    # shared constants
    from src.common import DEFAULT_WEIGHTS

    processed_dir = Path("data/processed/neu_det")
    if not processed_dir.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python src/prepare.py` first."
        )

    print(f"Loading pretrained {model_size} weights ...")
    model = YOLO(f"{model_size}.pt")

    print(f"Starting training — {epochs} epochs, batch={batch}, imgsz={img_size}")
    results = model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=15,        # stop early if no improvement for 15 epochs
        augment=True,       # Ultralytics built-in augmentations
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,    # disable mosaic in last 10 epochs for stability
        warmup_epochs=3,
        project=output_dir,
        name="neu_det",
        exist_ok=True,
        device=device or None,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = Path(DEFAULT_WEIGHTS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, dest)

    map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
    print("\nTraining complete.")
    print(f"  Best weights : {dest}")
    print(f"  mAP@0.5      : {map50:.4f}")
    print(f"  Full results : {results.save_dir}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET")
    parser.add_argument(
        "--model", default="yolov8s",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model size",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", default="configs/neu_det.yaml")
    parser.add_argument(
        "--device", default="",
        help="Device: '' for auto, 'cpu', '0' for GPU 0",
    )
    args = parser.parse_args()
    train(
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch=args.batch,
        data_cfg=args.data,
        device=args.device,
    )
