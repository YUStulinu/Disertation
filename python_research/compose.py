"""
Module 2 - Image Composition Pipeline - Step 4: basic composition.

Places the RGBA foreground (from extract_foreground.py) on a new background
with alpha blending - no colour correction yet, so it looks "pasted in",
which is exactly what harmonization fixes.

Saves <fg>_on_<bg>.jpg AND <fg>_on_<bg>_mask.png: the object's alpha mask at
its PLACED position in the composed image. Every later step (harmonize*.py,
add_shadow.py, metrics.py) reads that mask automatically.

Usage:
    python compose.py --fg data/foreground_results/photo_foreground.png --bg bg.jpg [--scale 0.6] [--pos 0.3 0.85]
"""

import argparse
from pathlib import Path

import cv2

from common.composition import place
from common.imaging import read_image, show, write_image
from common.paths import DATA_DIR, mask_path_for

OUTPUT_DIR = DATA_DIR / "composition_results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic composition - Module 2 Day 14")
    parser.add_argument("--fg", required=True, help="Foreground PNG with alpha (from extract_foreground.py)")
    parser.add_argument("--bg", required=True, help="Background image")
    parser.add_argument("--scale", type=float, default=0.5, help="Foreground height / background height")
    parser.add_argument("--pos", nargs=2, type=float, default=(0.5, 0.9), metavar=("CX", "CY"),
                        help="Bottom-centre of the foreground, as 0-1 fractions of the background")
    parser.add_argument("--px", nargs=2, type=int, metavar=("X", "Y"), help="Same anchor in pixels (overrides --pos)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    fg = cv2.imread(args.fg, cv2.IMREAD_UNCHANGED)
    if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
        raise ValueError(f"{args.fg} must be an RGBA PNG (as saved by extract_foreground.py).")
    composed, alpha = place(fg, read_image(args.bg), args.scale, tuple(args.pos), args.px and tuple(args.px))

    out = OUTPUT_DIR / f"{Path(args.fg).stem.replace('_foreground', '')}_on_{Path(args.bg).stem}.jpg"
    print("Saved:", write_image(out, composed))
    print("Saved:", write_image(mask_path_for(out), alpha))
    if not args.no_show:
        show("Composition result", composed)


if __name__ == "__main__":
    main()
