"""Steel Defect Detection — Streamlit Interface.

Upload any steel surface image to get annotated defect detections.

Run:
    streamlit run app/main.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Shared constants
from src.common import CLASS_COLORS_RGB, CLASS_NAMES, DEFAULT_WEIGHTS, format_class_name


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def draw_detections(image_rgb: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image copy."""
    annotated = image_rgb.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = format_class_name(CLASS_NAMES[cls_id])
        color = CLASS_COLORS_RGB[CLASS_NAMES[cls_id]]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Filled background rectangle behind text
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def run_inference(model: YOLO, image_rgb: np.ndarray, conf: float, iou: float):
    return model(image_rgb, conf=conf, iou=iou, verbose=False)


def build_detection_table(results) -> list[dict]:
    rows = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        rows.append({
            "Class": format_class_name(class_name),
            "Confidence": f"{float(box.conf[0]):.3f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return rows


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Steel Defect Detector",
    page_icon="🔬",
    layout="wide",
)

st.title("Steel Defect Detection")
st.caption("YOLOv8 fine-tuned on the NEU Steel Surface Defect Dataset · 6 defect classes")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")
st.sidebar.info(f"**Model:** `{DEFAULT_WEIGHTS}`")
weights_path = DEFAULT_WEIGHTS
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold  = st.sidebar.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Defect classes**")
for name, color in CLASS_COLORS_RGB.items():
    hex_color = "#{:02X}{:02X}{:02X}".format(*color)
    display_name = format_class_name(name)
    st.sidebar.markdown(
        f'<span style="color:{hex_color}; font-weight:bold;">■</span> {display_name}',
        unsafe_allow_html=True,
    )

# Model guard
if not Path(weights_path).exists():
    st.error(
        f"Model weights not found at **{weights_path}**.\n\n"
        "Train the model first:\n```\npython src/train.py --model yolov8s --epochs 50\n```"
    )
    st.stop()

model = load_model(weights_path)

# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown("Upload a steel surface image to detect and classify defects.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    label_visibility="collapsed",
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_image)

    col_orig, col_det = st.columns(2, gap="medium")

    with col_orig:
        st.subheader("Original")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Detecting defects…"):
        results = run_inference(model, image_np, conf_threshold, iou_threshold)
        annotated = draw_detections(image_np, results)

    with col_det:
        st.subheader("Detections")
        st.image(annotated, channels="RGB", use_container_width=True)

    # Results summary
    rows = build_detection_table(results)
    st.divider()
    if rows:
        st.subheader(f"Found {len(rows)} defect(s)")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.success("No defects detected above the confidence threshold.")

