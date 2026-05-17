# Steel Defect Detection System

Steel surface defect detection using YOLOv8 fine-tuned on the NEU Steel Surface Defect dataset. Run inference via image upload or live webcam in a Streamlit interface.

## Detected Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | crazing | Network of fine surface cracks |
| 1 | inclusion | Embedded foreign particles |
| 2 | patches | Irregular patch-like surface defects |
| 3 | pitted_surface | Small circular pits or holes |
| 4 | rolled-in_scale | Scale pressed in during rolling |
| 5 | scratches | Linear scratch marks |

## Quick Start

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset

Create a `.env` file from the template and fill in your Kaggle credentials:

```bash
cp .env.example .env
# Edit .env with your KAGGLE_USERNAME and KAGGLE_KEY
python src/download.py
```

> Alternatively, download manually from https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database and unzip into `data/raw/neu_det/`.

### 3. Prepare dataset

Converts PASCAL VOC XML annotations to YOLO format and splits into train/val/test:

```bash
python src/prepare.py
```

### 4. Train model

```bash
python src/train.py --model yolov8s --epochs 50 --batch 16
```

Model sizes: `yolov8n` (fastest) → `yolov8s` (balanced) → `yolov8m` (most accurate).

Trained weights are saved to `models/best.pt`.

### 5. Evaluate

```bash
python src/evaluate.py
```

Prints mAP@0.5, mAP@0.5:0.95, precision, and recall on the test split.

### 6. Launch the app

```bash
streamlit run app/main.py
```

Opens in your browser. Use the **Image Upload** tab to test images or the **Live Camera** tab for real-time detection.

## Project Structure

```
cv_project/
├── requirements.txt      # all dependencies
├── configs/
│   └── neu_det.yaml      # YOLO dataset config
├── src/
│   ├── download.py       # dataset download
│   ├── prepare.py        # VOC → YOLO conversion + splits
│   ├── train.py          # model training
│   ├── evaluate.py       # evaluation metrics
│   └── detect.py         # single-image CLI inference
├── app/
│   └── main.py           # Streamlit interface
├── data/                 # datasets (gitignored)
├── models/               # trained weights (gitignored)
└── runs/                 # training logs (gitignored)
```

## Target Metrics

| Metric | Target |
|--------|--------|
| mAP@0.5 | > 0.80 |
| Precision | > 0.78 |
| Recall | > 0.75 |
