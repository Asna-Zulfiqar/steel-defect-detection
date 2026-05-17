"""Evaluate a trained YOLOv8 model on the NEU-DET test set.

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights models/best.pt --split test
"""

import argparse
from pathlib import Path


def evaluate(
    weights: str = "models/best.pt",
    data_cfg: str = "configs/neu_det.yaml",
    img_size: int = 640,
    split: str = "test",
    conf: float = 0.001,
) -> None:
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run `python src/train.py` first."
        )

    print(f"Loading {weights_path} ...")
    model = YOLO(str(weights_path))

    print(f"Running evaluation on '{split}' split ...")
    metrics = model.val(
        data=data_cfg,
        imgsz=img_size,
        split=split,
        conf=conf,
        verbose=False,
    )

    class_names = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches",
    ]

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")

    print("\n  Per-class AP@0.5:")
    for i, name in enumerate(class_names):
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
        print(f"    {name:<20s}: {ap:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on NEU-DET")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="configs/neu_det.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(
        weights=args.weights,
        data_cfg=args.data,
        img_size=args.imgsz,
        split=args.split,
    )
