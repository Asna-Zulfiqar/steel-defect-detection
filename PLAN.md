# Steel Defect Detection System — Implementation Plan (Simplified)

## Context

A local, self-contained steel surface defect detection system. The user runs everything on their machine — no Docker, no notebooks, no REST APIs. A single Streamlit app serves as the interface with two modes: (1) upload an image for analysis, (2) live camera feed for real-time detection. Model is fine-tuned YOLOv8 on the NEU Steel Surface Defect dataset. Dependencies in a single `requirements.txt`, Python virtual environment.

---

## Recommended Dataset

**NEU Steel Surface Defect Database**
- URL: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- 6 classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- 1,800 grayscale images (300 per class), 200×200 px
- Download requires a free Kaggle account + API key

Secondary (optional, for more variety):
- GC10-DET: https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets

---

## Project Structure

```
/home/enigmatix/cv_project/
├── requirements.txt           # all dependencies in one file
├── PLAN.md                    # this file (copied here on approval)
├── README.md
├── .env.example               # KAGGLE_USERNAME, KAGGLE_KEY
├── .gitignore
│
├── data/
│   ├── raw/                   # downloaded dataset (gitignored)
│   │   └── neu_det/
│   └── processed/             # YOLO-format train/val/test (gitignored)
│       └── neu_det/
│           ├── images/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           └── labels/
│               ├── train/
│               ├── val/
│               └── test/
│
├── configs/
│   └── neu_det.yaml           # YOLO dataset config (paths + class names)
│
├── src/
│   ├── download.py            # download NEU-DET via Kaggle API
│   ├── prepare.py             # convert PASCAL VOC XML → YOLO TXT + splits
│   ├── train.py               # fine-tune YOLOv8 on NEU-DET
│   ├── evaluate.py            # print mAP, per-class AP, confusion matrix
│   └── detect.py              # run inference on a single image (CLI utility)
│
├── app/
│   └── main.py                # Streamlit app (image upload + live camera)
│
├── models/                    # saved model checkpoints (gitignored)
│   └── best.pt                # best YOLOv8 checkpoint after training
│
└── runs/                      # Ultralytics training output (gitignored)
```

---

## Environment Setup

**Recommendation: Python `venv`** — built-in, zero extra install, straightforward to activate. Pipenv adds a lock-file layer that isn't necessary here.

```bash
cd /home/enigmatix/cv_project
python3.12 -m venv .venv
source .venv/bin/activate        # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt`:**
```
# Core ML
torch>=2.3.1
torchvision>=0.18.1
ultralytics>=8.2.0              # YOLOv8 — includes training, eval, ONNX export

# Data handling
opencv-python>=4.9.0
albumentations>=1.4.0
numpy>=1.26
pandas>=2.0
Pillow>=10.3
scikit-learn>=1.5
pycocotools>=2.0.7              # COCO-style mAP evaluation

# Dataset download
kaggle>=1.6.12

# App interface
streamlit>=1.35
plotly>=5.22

# Utilities
tqdm>=4.66
python-dotenv>=1.0
```

---

## Step-by-Step Implementation

### Step 1 — Project Scaffold

Create the directory tree above and these base files:

**`.env.example`:**
```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

**`.gitignore`:**
```
.venv/
data/
models/
runs/
.env
__pycache__/
*.pyc
*.pt
```

**`configs/neu_det.yaml`:**
```yaml
path: data/processed/neu_det
train: images/train
val: images/val
test: images/test
nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

---

### Step 2 — Data Download (`src/download.py`)

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def download_neu_det(dest: Path = Path("data/raw/neu_det")) -> None:
    import kaggle
    dest.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "kaustubhdikshit/neu-surface-defect-database",
        path=dest,
        unzip=True,
    )
    print(f"Downloaded NEU-DET to {dest}")

if __name__ == "__main__":
    download_neu_det()
```

Run: `python src/download.py`

Users without a Kaggle account can download manually and unzip into `data/raw/neu_det/`.

---

### Step 3 — Dataset Preparation (`src/prepare.py`)

NEU-DET ships as images + PASCAL VOC `.xml` annotation files. This script:
1. Parses each `.xml` → converts bboxes to YOLO format (class cx cy w h, normalized)
2. Splits into train/val/test (70/15/15) with a fixed seed
3. Writes images and label `.txt` files to `data/processed/neu_det/`

Key functions:

```python
def parse_voc_xml(xml_path: Path) -> List[dict]:
    # Returns list of {class_name, xmin, ymin, xmax, ymax}

def voc_to_yolo(bbox: dict, img_w: int, img_h: int, class_map: dict) -> str:
    # Returns "class_id cx cy w h" string (normalized)

def prepare_neu_det(
    raw_dir: Path = Path("data/raw/neu_det"),
    out_dir: Path = Path("data/processed/neu_det"),
    split_ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> None:
    # Main entry: parse all XMLs, split, write images + labels

if __name__ == "__main__":
    prepare_neu_det()
    print("Dataset ready.")
```

Run: `python src/prepare.py`

---

### Step 4 — Model Training (`src/train.py`)

Fine-tune YOLOv8 on NEU-DET using the Ultralytics API.

```python
from ultralytics import YOLO
from pathlib import Path

def train(
    model_size: str = "yolov8s",    # n=fastest, s=good balance, m=higher accuracy
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    data_cfg: str = "configs/neu_det.yaml",
    output_dir: str = "runs/train",
) -> Path:
    model = YOLO(f"{model_size}.pt")   # loads ImageNet-pretrained weights
    results = model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        optimizer="AdamW",
        lr0=0.001,
        patience=15,              # early stopping
        augment=True,
        project=output_dir,
        name="neu_det",
        exist_ok=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    # Copy to models/best.pt for easy access
    (Path("models")).mkdir(exist_ok=True)
    import shutil
    shutil.copy(best_weights, "models/best.pt")
    print(f"Best model saved to models/best.pt  |  mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.3f}")
    return best_weights

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train(model_size=args.model, epochs=args.epochs, batch=args.batch)
```

Run: `python src/train.py --model yolov8s --epochs 50`

Training writes results (loss curves, mAP, confusion matrix) to `runs/train/neu_det/`.

**Model size guide:**
| Size | Params | Speed (CPU) | Accuracy |
|---|---|---|---|
| yolov8n | 3.2M | ~25ms | Lower |
| yolov8s | 11.2M | ~45ms | Good |
| yolov8m | 25.9M | ~90ms | Best local |

---

### Step 5 — Evaluation (`src/evaluate.py`)

```python
from ultralytics import YOLO

def evaluate(
    weights: str = "models/best.pt",
    data_cfg: str = "configs/neu_det.yaml",
    img_size: int = 640,
    split: str = "test",
) -> None:
    model = YOLO(weights)
    metrics = model.val(data=data_cfg, imgsz=img_size, split=split)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

if __name__ == "__main__":
    evaluate()
```

Run: `python src/evaluate.py`

**Target metrics:**
| Metric | Target |
|---|---|
| mAP@0.5 | > 0.80 |
| Precision | > 0.78 |
| Recall | > 0.75 |

---

### Step 6 — Streamlit App (`app/main.py`)

Single app with two tabs: **Image Upload** and **Live Camera**.

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_COLORS = {  # BGR for OpenCV
    "crazing": (255, 0, 0),
    "inclusion": (0, 255, 0),
    "patches": (0, 0, 255),
    "pitted_surface": (255, 165, 0),
    "rolled-in_scale": (128, 0, 128),
    "scratches": (0, 255, 255),
}

@st.cache_resource
def load_model(weights: str = "models/best.pt") -> YOLO:
    return YOLO(weights)

def draw_detections(image: np.ndarray, results) -> np.ndarray:
    annotated = image.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        color = CLASS_COLORS[label]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated

def run_inference(model: YOLO, image: np.ndarray, conf: float, iou: float):
    return model(image, conf=conf, iou=iou, verbose=False)

# ── App layout ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Steel Defect Detector", layout="wide")
st.title("Steel Defect Detection")
st.caption("Powered by YOLOv8 fine-tuned on NEU Steel Surface Defect Dataset")

# Sidebar controls
st.sidebar.header("Settings")
weights_path = st.sidebar.text_input("Model weights", value="models/best.pt")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.1, 0.95, 0.45, 0.05)

if not Path(weights_path).exists():
    st.error(f"Model not found at `{weights_path}`. Run `python src/train.py` first.")
    st.stop()

model = load_model(weights_path)

tab_upload, tab_camera = st.tabs(["Image Upload", "Live Camera"])

# ── Tab 1: Image Upload ──────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        image_np = np.array(pil_image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)

        with st.spinner("Running detection..."):
            results = run_inference(model, image_np, conf_threshold, iou_threshold)
            annotated = draw_detections(image_np, results)

        with col2:
            st.subheader("Detections")
            st.image(annotated, channels="RGB", use_container_width=True)

        # Detection table
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "Class": CLASS_NAMES[cls_id],
                "Confidence": f"{float(box.conf[0]):.3f}",
                "BBox (x1,y1,x2,y2)": [round(v) for v in box.xyxy[0].tolist()],
            })
        if detections:
            st.subheader(f"Found {len(detections)} defect(s)")
            st.dataframe(detections, use_container_width=True)
        else:
            st.success("No defects detected above the confidence threshold.")

# ── Tab 2: Live Camera ───────────────────────────────────────────────────────
with tab_camera:
    st.info("Click **Start Camera** to begin real-time detection from your webcam.")
    camera_index = st.number_input("Camera index (0 = default webcam)", 0, 10, 0, 1)

    run = st.checkbox("Start Camera")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_index}.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame.")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = run_inference(model, rgb_frame, conf_threshold, iou_threshold)
                annotated = draw_detections(rgb_frame, results)
                frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
                n_detections = len(results[0].boxes)
                stats_placeholder.metric("Defects in frame", n_detections)
                # Re-read checkbox state each iteration
                run = st.session_state.get("Start Camera", True)
            cap.release()
```

Run: `streamlit run app/main.py`

---

## How to Run End-to-End

```bash
# 1. Setup environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle credentials in .env)
cp .env.example .env   # fill in your KAGGLE_USERNAME and KAGGLE_KEY
python src/download.py

# 3. Prepare dataset (convert + split)
python src/prepare.py

# 4. Train model
python src/train.py --model yolov8s --epochs 50

# 5. Evaluate
python src/evaluate.py

# 6. Launch app
streamlit run app/main.py
```

---

## What's Kept vs. Removed

| Feature | Status | Reason |
|---|---|---|
| YOLOv8 fine-tuning | Kept | Core detection |
| Streamlit app (upload + camera) | Kept | Single local interface |
| Data download + preparation | Kept | Essential |
| Evaluation metrics (mAP) | Kept | Needed to validate model |
| Albumentations augmentations | Kept (via Ultralytics) | Auto-applied during training |
| Docker / CI/CD | Removed | Local-only project |
| Notebooks | Removed | Not needed |
| FastAPI REST service | Removed | No external serving |
| Segmentation branch (U-Net) | Removed | Adds complexity without UX benefit |
| Severity regression head | Removed | Over-engineered for local use |
| XAI (GradCAM/SHAP/LIME) | Removed | Can be added later if needed |
| CLIP few-shot learning | Removed | Overkill for initial system |
| Multi-file requirements | Merged into one `requirements.txt` | User preference |
| venv vs pipenv | venv chosen | Built-in, simpler |

---

## Critical Files

| File | Purpose |
|---|---|
| `requirements.txt` | All dependencies |
| `configs/neu_det.yaml` | YOLO dataset config — must match processed data paths exactly |
| `src/prepare.py` | VOC→YOLO conversion — correctness here determines model quality |
| `src/train.py` | Training entry point |
| `app/main.py` | Entire user-facing interface |
| `models/best.pt` | Trained checkpoint loaded by the app |
