"""Steel Defect Detection — Streamlit Interface.

Tabs:
  1. Image Upload  — upload any steel surface image, get annotated detections
  2. Live Camera   — real-time detection from a webcam

Run:
    streamlit run app/main.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# RGB colours per class (for st.image which expects RGB)
CLASS_COLORS_RGB = {
    "crazing":        (220,  50,  50),
    "inclusion":      ( 50, 200,  50),
    "patches":        ( 50,  50, 220),
    "pitted_surface": (220, 140,  30),
    "rolled-in_scale":(160,  50, 200),
    "scratches":      ( 30, 200, 200),
}

DEFAULT_WEIGHTS = "models/best.pt"

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
        label = CLASS_NAMES[cls_id]
        color = CLASS_COLORS_RGB[label]

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
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        rows.append({
            "Class": CLASS_NAMES[cls_id],
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
weights_path = st.sidebar.text_input("Model weights path", value=DEFAULT_WEIGHTS)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold  = st.sidebar.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Defect classes**")
for name, color in CLASS_COLORS_RGB.items():
    hex_color = "#{:02X}{:02X}{:02X}".format(*color)
    st.sidebar.markdown(
        f'<span style="color:{hex_color}; font-weight:bold;">■</span> {name}',
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

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_upload, tab_camera = st.tabs(["📁  Image Upload", "📷  Live Camera"])


# ── Tab 1: Image Upload ───────────────────────────────────────────────────────

with tab_upload:
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


# ── Tab 2: Live Camera ────────────────────────────────────────────────────────

with tab_camera:
    st.markdown(
        "Enable the camera below for real-time defect detection on a live feed.\n\n"
        "> **Tip:** Position your camera over the steel surface, then check the box to start."
    )

    camera_index = st.number_input(
        "Camera index  (0 = built-in webcam, 1 = first external camera, …)",
        min_value=0, max_value=10, value=0, step=1,
    )

    run_camera = st.checkbox("▶  Start Camera", value=False)

    frame_slot  = st.empty()
    metric_slot = st.empty()
    status_slot = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(
                f"Could not open camera {int(camera_index)}. "
                "Check the camera index or try a different value."
            )
        else:
            status_slot.info("Camera running — uncheck the box above to stop.")
            frame_count = 0

            while True:
                # Re-evaluate checkbox on each iteration so unchecking stops the loop
                if not st.session_state.get(
                    "▶  Start Camera",
                    run_camera,   # fall back to the value at the start of this run
                ):
                    break

                ret, frame_bgr = cap.read()
                if not ret:
                    status_slot.warning("Failed to read from camera.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results   = run_inference(model, frame_rgb, conf_threshold, iou_threshold)
                annotated = draw_detections(frame_rgb, results)

                frame_slot.image(annotated, channels="RGB", use_container_width=True)
                n_det = len(results[0].boxes)
                metric_slot.metric("Defects in frame", n_det)
                frame_count += 1

            cap.release()
            status_slot.info("Camera stopped.")
    else:
        frame_slot.empty()
        metric_slot.empty()
