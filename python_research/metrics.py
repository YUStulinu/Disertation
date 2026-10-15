"""
Module 2 - Week 4 Day 5: quantitative evaluation of harmonization (iHarmony4 protocol).

Real composites have no ground truth, so - like the iHarmony4 benchmark (Cong et
al., DoveNet, CVPR 2020) - test cases are built from REAL photos: an object is
segmented and randomly recoloured so it no longer fits its scene, then each
method has to repair it. The untouched photo is the ground truth:
  MSE / PSNR    whole image
  fMSE / fPSNR  object only (the region harmonization changes)
  fSSIM         structural similarity averaged inside the object

Usage:
    python metrics.py                                  # default photos, 5 recolourings each
    python metrics.py --images a.jpg b.jpg --trials 10 --methods none harmonizer
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from common.harmonization import METHOD_NAMES, METHODS, harmonize
from common.imaging import read_image
from common.paths import DATA_DIR
from common.plots import bar_chart
from common.quality import mse, psnr, ssim, synthetic_composite
from common.segmentation import describe, load_yolo, pick_detection, segment

OUTPUT_DIR = DATA_DIR / "metrics_results"
DEFAULT_PHOTOS = sorted((DATA_DIR / "benchmark_photos").glob("*.jpg"))  # real photos, object in its own scene
METRICS = ["mse", "fmse", "psnr", "fpsnr", "fssim"]


def write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonization benchmark - Week 4 Day 5")
    parser.add_argument("--images", nargs="+", type=Path, default=DEFAULT_PHOTOS, help="Real photos to recolour")
    parser.add_argument("--trials", type=int, default=5, help="Random recolourings per photo")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    args = parser.parse_args()

    yolo, cases = load_yolo("m"), []
    for i, path in enumerate(args.images):
        image = read_image(path)
        mask, result = segment(yolo, image)
        alpha = cv2.GaussianBlur(mask, (7, 7), 0)
        name = describe(result)[pick_detection(result)][0]
        cases += [(path.name, name, trial, image, mask, alpha, synthetic_composite(image, alpha, np.random.default_rng([i, trial])))
                  for trial in range(args.trials)]
    rows = []
    for method in args.methods:  # method by method: each heavy network stack is loaded only once
        for photo, name, trial, image, mask, alpha, composite in cases:
            out = harmonize(composite, alpha, method)
            rows.append([photo, name, trial, method, mse(out, image), mse(out, image, mask),
                         psnr(out, image), psnr(out, image, mask), ssim(out, image, mask)])
        print(f"  {METHOD_NAMES[method]:<26} fPSNR {np.mean([r[7] for r in rows[-len(cases):]]):6.2f} dB", flush=True)
    write_csv(OUTPUT_DIR / "benchmark_per_case.csv", [["image", "object", "trial", "method", *METRICS], *rows])

    print(f"\n{'Method':<26}" + "".join(f"{m:>9}" for m in METRICS) + f"   (n={len(cases)} cases)")
    summary = []
    for method in args.methods:
        vals = np.array([r[4:] for r in rows if r[3] == method], dtype=float)
        summary.append([method, *vals.mean(0).round(4), *vals.std(0).round(4)])
        print(f"{METHOD_NAMES[method]:<26}" + "".join(f"{v:>9.2f}" for v in vals.mean(0)))
    write_csv(OUTPUT_DIR / "benchmark_summary.csv", [["method", *METRICS, *(f"{m}_std" for m in METRICS)], *summary])

    labels = [METHOD_NAMES[s[0]] for s in summary]
    print("Saved:", bar_chart(labels, {"fPSNR": [s[4] for s in summary]}, "Object PSNR (higher is better)", "fPSNR (dB)", OUTPUT_DIR / "benchmark_fpsnr.png"))
    print("Saved:", bar_chart(labels, {"fMSE": [s[2] for s in summary]}, "Object MSE (lower is better)", "fMSE", OUTPUT_DIR / "benchmark_fmse.png", fmt="{:.0f}"))


if __name__ == "__main__":
    main()
