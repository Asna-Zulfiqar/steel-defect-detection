"""Shared constants and paths used across the project.

Keep class names and canonical paths here to avoid duplication.
"""
from pathlib import Path

# Defect classes (single source of truth)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Map name->id
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# RGB colors per class (used by the Streamlit UI)
CLASS_COLORS_RGB = {
    "crazing": (220, 50, 50),
    "inclusion": (50, 200, 50),
    "patches": (50, 50, 220),
    "pitted_surface": (220, 140, 30),
    "rolled-in_scale": (160, 50, 200),
    "scratches": (30, 200, 200),
}

# Canonical artifact & data paths
DEFAULT_WEIGHTS = "models/best.pt"
DATA_RAW = Path("data/raw/neu_det")
DATA_PROCESSED = Path("data/processed/neu_det")
CONFIG_PATH = Path("configs/neu_det.yaml")


def format_class_name(name: str) -> str:
    """Format class name for display: remove underscores/hyphens, title case.

    Examples:
        "crazing" → "Crazing"
        "pitted_surface" → "Pitted Surface"
        "rolled-in_scale" → "Rolled-In Scale"
    """
    # Replace underscores and hyphens with spaces, then title case
    formatted = name.replace("_", " ").replace("-", " ").title()
    return formatted


