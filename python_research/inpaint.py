"""
Module 2 - Step 3: inpainting - fill the hole left in the source image.

  telea - OpenCV fast-marching method (Telea 2004), classical
  ns    - OpenCV Navier-Stokes method, classical
  lama  - LaMa (Suvorov et al., WACV 2022), Fast Fourier Convolutions, deep learning
          (pip install simple-lama-inpainting; weights download on first use)

Saves every result and a comparison figure. --eval measures quality against a
real ground truth: the object mask is moved onto an object-free part of the
same image, each method fills that artificial hole, and PSNR / SSIM are
computed against the true pixels there (the real hole cannot be scored -
nobody knows what was behind the object).

Usage:
    python inpaint.py --image photo.jpg --mask data/foreground_results/photo_mask.png [--methods telea ns lama] [--eval]
"""

import argparse
from pathlib import Path

import numpy as np

from common.composition import dilate, inpaint
from common.imaging import labeled_grid, read_image, read_mask, show, write_image
from common.paths import DATA_DIR
from common.quality import psnr, ssim

OUTPUT_DIR = DATA_DIR / "inpaint_results"
TITLES = {"telea": "Classical - TELEA", "ns": "Classical - Navier-Stokes", "lama": "DL - LaMa"}


def shifted_test_hole(mask: np.ndarray) -> np.ndarray | None:
    """The object mask moved (left/right/up/down, smallest shift first) to where it lies entirely
    inside the image without touching the object; None if there is no such position."""
    keep_out = dilate(mask, 15) > 0
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    shifts = [(0, dx) for dx in range(-xs.min(), w - xs.max())] + [(dy, 0) for dy in range(-ys.min(), h - ys.max())]
    for dy, dx in sorted(shifts, key=lambda d: abs(d[0]) + abs(d[1])):
        hole = np.zeros_like(mask)
        hole[ys + dy, xs + dx] = 255
        if not (keep_out & (hole > 0)).any():
            return hole
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inpainting: classical vs LaMa - Module 2 Day 13 / Week 6 Day 2")
    parser.add_argument("--image", required=True, help="Original image (object still in it)")
    parser.add_argument("--mask", required=True, help="Object mask from segment.py")
    parser.add_argument("--methods", nargs="+", choices=list(TITLES), default=["telea", "lama"])
    parser.add_argument("--eval", action="store_true", help="Score each method on a hole with known ground truth")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    image, stem = read_image(args.image), Path(args.image).stem
    mask = read_mask(args.mask, image.shape[:2])
    print(f"Hole: {100 * (mask > 0).mean():.1f}% of the image")
    panels = [(image * (mask[..., None] == 0), "Original (with hole)")]
    for method in args.methods:
        result = inpaint(image, mask, method)
        print("Saved:", write_image(OUTPUT_DIR / f"{stem}_{method}.jpg", result))
        panels.append((result, TITLES[method]))
    figure = labeled_grid(panels)
    print("Saved:", write_image(OUTPUT_DIR / f"{stem}_comparison.jpg", figure))

    if args.eval:
        hole = shifted_test_hole(mask)
        if hole is None:
            print("--eval: the object is too large to move onto an object-free area of this image.")
        else:
            print("Ground truth evaluation (object mask moved onto an object-free area):")
            region = dilate(hole, 3)  # the area actually filled
            for method in args.methods:
                filled = inpaint(image, hole, method)
                print(f"  {method:<6} PSNR={psnr(filled, image, region):6.2f} dB   SSIM={ssim(filled, image, region):.4f}")
    if not args.no_show:
        show("Inpainting", figure)


if __name__ == "__main__":
    main()
