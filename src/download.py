"""Download the NEU Steel Surface Defect dataset via the Kaggle API.

Usage:
    python src/download.py

Requires KAGGLE_USERNAME and KAGGLE_KEY set in a .env file (or environment).
Alternatively, download manually from:
    https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
and unzip into data/raw/neu_det/.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATASET_SLUG = "kaustubhdikshit/neu-surface-defect-database"
DEFAULT_DEST = Path("data/raw/neu_det")


def download_neu_det(dest: Path = DEFAULT_DEST) -> None:
    try:
        import kaggle
    except ImportError:
        raise SystemExit("kaggle package not found. Run: pip install kaggle")

    if dest.exists() and any(dest.iterdir()):
        print(f"Dataset already present at {dest}, skipping download.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NEU-DET to {dest} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_SLUG, path=dest, unzip=True)
    print("Download complete.")


if __name__ == "__main__":
    download_neu_det()
