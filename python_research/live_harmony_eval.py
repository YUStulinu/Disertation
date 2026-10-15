"""
Modules 1 + 2: how much of the harmonization's colour change the live demo should give the vases.

The harmonization networks are trained (iHarmony4) to repair objects whose colours are wrong, so they
pull an object's colours towards the scene's. The museum photos' colours are right - only the light
differs. The live demo therefore keeps the networks' lightness change (exposure, contrast) whole and
gives the objects only a share of their colour change (common/handheld.keep_colours, COLOUR_SHARE).
For several shares, with exactly the live code (colour matrix fitted to each network's output, mean
of the three fits):
  benchmark   the iHarmony4-style test of metrics.py: real photos, the object randomly recoloured, the
              untouched photo is the ground truth (fMSE, fPSNR, fSSIM)
  vases       each catalog vase composited into the photos of museum halls and a town square: how much
              the result changes the vase's own colours (Lab dE, saturation ratio)
and a figure: vases as photographed, harmonized with all of the colour change, and as live now.

Usage:
    python live_harmony_eval.py [--shares 1 0.5 0.25 0]
"""

import argparse

import cv2
import numpy as np

from common.composition import blend
from common.handheld import COLOUR_SHARE, _paste, _resize_bgra, fit_colour_matrix, keep_colours, load_catalog
from common.harmonization import learned
from common.imaging import crop_object, labeled_grid, write_image
from common.paths import IMAGES_DIR
from common.quality import mse, psnr, ssim, synthetic_composite
from common.segmentation import load_yolo, segment
from metrics import DEFAULT_PHOTOS, OUTPUT_DIR, write_csv

NETWORKS = ("harmonizer", "pctnet", "pctnet_cnn")
SCENES = ("hall_kielce", "hall_met_sculpture_court", "hall_nelson_atkins", "hall_norton_simon", "outdoor_town_square")
FIGURE = (("outdoor_town_square", "amphora_red_figure", "Town square"), ("hall_met_sculpture_court", "jar_terracotta", "Museum court"),
          ("hall_nelson_atkins", "amphora_small_cypriot", "Warm hall"))


def live_result(composite: np.ndarray, outputs: list[np.ndarray], inside: np.ndarray, share: float) -> np.ndarray:
    """What the live demo draws: the mean of the matrices fitted to each network's output."""
    m = np.mean([fit_colour_matrix(composite, keep_colours(composite, out, share), inside) for out in outputs], axis=0)
    coloured = cv2.transform(composite.astype(np.float32), np.hstack([m[:3].T, 255 * m[3:].T]))
    return np.clip(coloured + 0.5, 0, 255).astype(np.uint8)


def vase_scene(scene: str, art) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """The vase standing in the scene photo (at the live harmonization's 512 px): composite, alpha
    (uint8), the object's inner pixels and the three networks' outputs."""
    bg = cv2.imread(str(IMAGES_DIR / f"{scene}.jpg"))
    bg = cv2.resize(bg, (512, round(bg.shape[0] * 512 / bg.shape[1])), interpolation=cv2.INTER_AREA)
    sprite = _resize_bgra(art.sprite, int(0.45 * bg.shape[0]))
    h, w = sprite.shape[:2]
    x, y = (bg.shape[1] - w) // 2, bg.shape[0] - h - 10
    a = sprite[..., 3:] / 255.0
    composite, alpha = bg.astype(np.float32), np.zeros(bg.shape[:2] + (1,), np.float32)
    _paste(composite, sprite[..., :3], a, x, y)
    _paste(alpha, np.ones_like(a), a, x, y)
    composite = np.clip(composite + 0.5, 0, 255).astype(np.uint8)
    alpha8 = np.clip(alpha[..., 0] * 255 + 0.5, 0, 255).astype(np.uint8)
    inside = cv2.erode((alpha[..., 0] > 0.9).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return composite, alpha8, inside, [learned(composite, alpha8, n) for n in NETWORKS]


def lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def benchmark(shares: list[float]) -> dict:
    yolo, scores = load_yolo("m"), {s: [] for s in shares}
    for i, path in enumerate(DEFAULT_PHOTOS):
        image = cv2.imread(str(path))
        mask, _ = segment(yolo, image)
        alpha = cv2.GaussianBlur(mask, (7, 7), 0)
        inside = cv2.erode((alpha > 229).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for trial in range(5):
            composite = synthetic_composite(image, alpha, np.random.default_rng([i, trial]))
            outputs = [learned(composite, alpha, n) for n in NETWORKS]
            for s in shares:
                out = blend(composite, live_result(composite, outputs, inside, s), alpha)
                scores[s].append((mse(out, image, mask), psnr(out, image, mask), ssim(out, image, mask)))
    return {s: np.mean(v, axis=0) for s, v in scores.items()}


def vases(shares: list[float]) -> dict:
    changes = {s: [] for s in shares}
    for scene in SCENES:
        for art in load_catalog():
            composite, _, inside, outputs = vase_scene(scene, art)
            before = lab(composite)[inside]
            chroma = np.hypot(before[:, 1], before[:, 2])
            for s in shares:
                after = lab(live_result(composite, outputs, inside, s))[inside]
                ratio = np.hypot(after[:, 1], after[:, 2])[chroma > 10].mean() / chroma[chroma > 10].mean()
                changes[s].append((np.linalg.norm(after - before, axis=1).mean(), ratio))
    return {s: np.array(v) for s, v in changes.items()}


def figure() -> None:
    catalog, panels = {a.name: a for a in load_catalog()}, []
    for scene, name, title in FIGURE:
        composite, alpha, inside, outputs = vase_scene(scene, catalog[name])
        crop = lambda image: crop_object(image, alpha, margin=0.35)
        panels += [(crop(composite), f"{title}: museum photo"),
                   (crop(blend(composite, live_result(composite, outputs, inside, 1.0), alpha)), "all colour change (before)"),
                   (crop(blend(composite, live_result(composite, outputs, inside, COLOUR_SHARE), alpha)),
                    f"{100 * COLOUR_SHARE:.0f}% colour change (live now)")]
    print("Saved:", write_image(OUTPUT_DIR / "live_harmony_colour.jpg", labeled_grid(panels, cols=3, height=420)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live harmonization: the share of the colour change given to the vases")
    parser.add_argument("--shares", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.0])
    args = parser.parse_args()
    bench, change = benchmark(args.shares), vases(args.shares)
    rows = [["colour_share", "fmse", "fpsnr", "fssim", "vase_dE", "vase_saturation_mean", "vase_saturation_min",
             "vase_saturation_max", "vases_within_15pct"]]
    print(f"{'share':>6} {'fMSE':>7} {'fPSNR':>7} {'fSSIM':>7} {'vase dE':>8} {'saturation (min-max)':>21} {'within 15%':>11}")
    for s in args.shares:
        (f_mse, f_psnr, f_ssim), v = bench[s], change[s]
        within = 100 * np.mean(np.abs(v[:, 1] - 1) <= 0.15)
        rows.append([s, *np.round([f_mse, f_psnr, f_ssim, v[:, 0].mean(), v[:, 1].mean(), v[:, 1].min(), v[:, 1].max(), within], 4)])
        print(f"{s:6.2f} {f_mse:7.1f} {f_psnr:7.2f} {f_ssim:7.4f} {v[:, 0].mean():8.1f} {v[:, 1].mean():9.2f} "
              f"({v[:, 1].min():.2f}-{v[:, 1].max():.2f}) {within:10.0f}%")
    write_csv(OUTPUT_DIR / "live_harmony_colour.csv", rows)
    figure()


if __name__ == "__main__":
    main()
