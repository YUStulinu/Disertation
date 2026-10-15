"""Project paths, resolved from this file so every script works from any working directory."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../python_research
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = ROOT.parent / "photos"
SCREENSHOTS_DIR = ROOT.parent / "images" / "python_app"  # the S key of the live demos (the C++ app: qt_cpp_app/)


def mask_path_for(image_path: str | Path) -> Path:
    """Placed-object mask of a composite: X.jpg, X_harmonized.jpg or X_final.jpg -> X_mask.png."""
    p = Path(image_path)
    return p.with_name(re.sub(r"_(harmonized|final)$", "", p.stem) + "_mask.png")
