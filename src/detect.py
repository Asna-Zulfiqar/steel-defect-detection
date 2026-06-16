"""Run inference on a single image and save the annotated result.

Usage:
    python src/detect.py --image path/to/image.jpg
    python src/detect.py --image path/to/image.jpg --weights models/best.pt --conf 0.25
"""

import sys
from pathlib import Path

# Add repo root to sys.path so 'src' can be imported
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import argparse
from src.common import CLASS_NAMES, DEFAULT_WEIGHTS


def detect(
    image_path: str,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    img_size: int = 640,
    save_dir: str = "runs/detect",
) -> None:
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run `python src/train.py` first."
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(weights_path))
    results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        imgsz=img_size,
        save=True,
        project=save_dir,
        name="result",
        exist_ok=True,
    )

    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No defects detected.")
        return

    print(f"\nDetected {len(boxes)} defect(s):")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        print(f"  {CLASS_NAMES[cls_id]:<20s}  conf={conf_val:.3f}  bbox=({x1},{y1},{x2},{y2})")

    out_path = Path(save_dir) / "result" / image_path.name
    print(f"\nAnnotated image saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect defects in a single image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-dir", default="runs/detect")
    args = parser.parse_args()
    detect(
        image_path=args.image,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
        save_dir=args.save_dir,
    )
