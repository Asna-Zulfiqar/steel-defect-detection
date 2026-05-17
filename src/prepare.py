"""Convert NEU-DET PASCAL VOC annotations to YOLO format and split into train/val/test.

Usage:
    python src/prepare.py [--raw-dir data/raw/neu_det] [--out-dir data/processed/neu_det]

NEU-DET structure after download:
    data/raw/neu_det/
        IMAGES/          (or images/)  ← .jpg files
        ANNOTATIONS/     (or annotations/) ← .xml PASCAL VOC files

Output:
    data/processed/neu_det/
        images/{train,val,test}/
        labels/{train,val,test}/
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def find_subdirectory(root: Path, candidates: list[str]) -> Path:
    """Return the first matching subdirectory (case-insensitive)."""
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() in [c.lower() for c in candidates]:
            return entry
    raise FileNotFoundError(
        f"None of {candidates} found under {root}. "
        f"Available: {[e.name for e in root.iterdir() if e.is_dir()]}"
    )


def resolve_dataset_root(raw_dir: Path) -> Path:
    """Find the folder that directly contains image and annotation subdirectories."""
    candidates = [raw_dir]
    nested = raw_dir / "NEU-DET"
    if nested.is_dir():
        candidates.append(nested)

    for base in candidates:
        if not base.is_dir():
            continue
        try:
            find_subdirectory(base, ["images", "IMAGES", "JPEGImages"])
            find_subdirectory(base, ["annotations", "ANNOTATIONS", "Annotations"])
            return base
        except FileNotFoundError:
            pass

        for child in sorted((e for e in base.iterdir() if e.is_dir()), key=lambda p: p.name.lower()):
            try:
                find_subdirectory(child, ["images", "IMAGES", "JPEGImages"])
                find_subdirectory(child, ["annotations", "ANNOTATIONS", "Annotations"])
                return child
            except FileNotFoundError:
                continue

    raise FileNotFoundError(
        f"Could not locate a dataset folder with images and annotations under {raw_dir}."
    )


def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse a PASCAL VOC XML file and return a list of bounding box dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower().replace(" ", "_")
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        objects.append(
            {"name": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
             "img_w": img_w, "img_h": img_h}
        )
    return objects


def voc_to_yolo_line(obj: dict) -> str | None:
    """Convert one VOC bbox dict to a YOLO annotation line."""
    name = obj["name"]
    if name not in CLASS_MAP:
        return None
    cls_id = CLASS_MAP[name]
    img_w, img_h = obj["img_w"], obj["img_h"]
    cx = ((obj["xmin"] + obj["xmax"]) / 2) / img_w
    cy = ((obj["ymin"] + obj["ymax"]) / 2) / img_h
    w = (obj["xmax"] - obj["xmin"]) / img_w
    h = (obj["ymax"] - obj["ymin"]) / img_h
    # Clamp to [0, 1] to guard against annotation errors
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_image_file(img_dir: Path, stem: str) -> Path | None:
    """Find an image matching an annotation stem, searching nested class folders too."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        direct = img_dir / f"{stem}{ext}"
        if direct.exists():
            return direct

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        matches = list(img_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def prepare_neu_det(
    raw_dir: Path = Path("data/raw/neu_det"),
    out_dir: Path = Path("data/processed/neu_det"),
    split_ratios: tuple = SPLIT_RATIOS,
    seed: int = SEED,
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found. Run `python src/download.py` first, "
            "or manually unzip the dataset there."
        )

    # Locate the actual dataset root, then its image and annotation directories.
    dataset_root = resolve_dataset_root(raw_dir)
    img_dir = find_subdirectory(dataset_root, ["images", "IMAGES", "JPEGImages"])
    ann_dir = find_subdirectory(dataset_root, ["annotations", "ANNOTATIONS", "Annotations"])

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in {ann_dir}")

    print(f"Found {len(xml_files)} annotation files.")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(xml_files)
    n = len(xml_files)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train : n_train + n_val],
        "test": xml_files[n_train + n_val :],
    }

    # Create output directories
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    total_boxes = 0

    for split, files in splits.items():
        for xml_path in files:
            stem = xml_path.stem

            # Find the matching image (try common extensions, including nested class folders)
            img_path = find_image_file(img_dir, stem)

            if img_path is None:
                print(f"  [warn] No image found for {stem}, skipping.")
                skipped += 1
                continue

            objects = parse_voc_xml(xml_path)
            yolo_lines = [voc_to_yolo_line(o) for o in objects]
            yolo_lines = [l for l in yolo_lines if l is not None]

            if not yolo_lines:
                skipped += 1
                continue

            # Copy image
            shutil.copy(img_path, out_dir / "images" / split / img_path.name)

            # Write YOLO label file
            label_path = out_dir / "labels" / split / (stem + ".txt")
            label_path.write_text("\n".join(yolo_lines))
            total_boxes += len(yolo_lines)

        print(f"  {split:5s}: {len(files) - skipped} images")

    print(f"\nDataset ready at {out_dir}")
    print(f"  Total boxes: {total_boxes}  |  Skipped: {skipped}")
    print(f"  Train / Val / Test: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NEU-DET for YOLO training")
    parser.add_argument("--raw-dir", default="data/raw/neu_det", help="Raw dataset directory")
    parser.add_argument("--out-dir", default="data/processed/neu_det", help="Output directory")
    args = parser.parse_args()
    prepare_neu_det(raw_dir=Path(args.raw_dir), out_dir=Path(args.out_dir))
