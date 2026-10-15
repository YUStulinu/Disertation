"""
Modules 1 + 2: how much of the museum photo's detail - the painting, the incised lines, the
inscriptions - a harmonized vase keeps in the live demo.

Before, a harmonized vase was blurred to the sharpness measured on the hands holding it: the hands'
smooth skin always reads blurred, so the blur sat at its cap (1.5 px per 480 rows, 3.4 px in Full
HD), and the cut-outs were at most 1080 px tall, so a large vase held close was enlarged. Now the
camera match leaves the sharpness alone (the webcam's sharpest edges measured sharper than the
cut-outs') and the cut-outs keep the photos' full resolution (common/handheld.py MAX_BLUR,
build_catalog.py MAX_HEIGHT).

For every catalog vase, drawn at the size it has in the Full HD demo (one hand ~500 px, two hands
~1000 px, a large vase held close ~1500 px) in the light of a warm museum hall (the colour matrix the
live harmonization fits there), compared with the museum photo in that same light:
  detail   high-frequency energy kept (standard deviation of the Laplacian, 100% = the photo's)
  SSIM     structural similarity of the lightness to the photo's
and a figure: details at 1:1 - the photo in the hall's light, harmonized before, harmonized now.

Usage:
    python live_harmony_detail.py
"""

import cv2
import numpy as np

from common.handheld import COLOUR_SHARE, HandheldScene, Layer, Track, _blend, _resize_bgra, fit_colour_matrix, keep_colours, load_catalog
from common.harmonization import _sharpness, learned
from common.imaging import labeled_grid, write_image
from common.quality import ssim
from live_harmony_eval import NETWORKS, vase_scene
from metrics import OUTPUT_DIR, write_csv

SCENE = "hall_nelson_atkins"     # a warm hall: the light changes the vase visibly
OLD_HEIGHT, OLD_BLUR = 1080, 1.5 * 1080 / 480   # the cut-outs' height and the blur (its cap, in Full HD) before
GREY = (118, 124, 128)
FIGURE = (("amphora_black_figure", 1000), ("amphora_red_figure", 1000), ("amphora_large_attic", 1500))


def size_in_demo(art) -> int:
    """Its height in the Full HD demo: one hand ~500 px, a two-hand vase ~1000 px, a large one held close ~1500 px."""
    return 500 if art.height_m <= 0.25 else 1500 if art.height_m >= 0.6 else 1000


def scene_light(art) -> np.ndarray:
    """The colour matrix the live harmonization fits for the vase in the hall (mean over the networks)."""
    composite, alpha, inside, _ = vase_scene(SCENE, art)
    fits = [fit_colour_matrix(composite, keep_colours(composite, learned(composite, alpha, n), COLOUR_SHARE), inside)
            for n in NETWORKS]
    return np.mean(fits, axis=0).astype(np.float32)


def resized(bgra: np.ndarray, w: int, h: int) -> np.ndarray:
    """Float BGRA at exactly w x h, resized with premultiplied alpha (as _resize_bgra)."""
    pm = bgra.astype(np.float32)
    pm[..., :3] *= pm[..., 3:] / 255
    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA if h < bgra.shape[0] else cv2.INTER_LINEAR)
    pm[..., :3] /= np.maximum(pm[..., 3:] / 255, 1e-3)
    return pm


def on_grey(colours: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.empty(colours.shape, np.uint8)
    out[:] = GREY
    _blend(out, colours, alpha, 0, 0)
    return out


def drawings(scene: HandheldScene, art, h: int, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(photo in the hall's light, harmonized before, harmonized now, the vase's inner mask), h px tall."""
    w = round(art.sprite.shape[1] * h / art.sprite.shape[0])
    transform = np.hstack([matrix[:3].T, 255 * matrix[3:].T])
    photo = resized(art.sprite, w, h)  # premultiplied INTER_AREA from the full-resolution cut-out
    alpha = photo[..., 3] / 255
    reference = on_grey(np.clip(cv2.transform(photo[..., :3], transform) + 0.5, 0, 255).astype(np.uint8), alpha)
    old = resized(np.clip(_resize_bgra(art.sprite, min(OLD_HEIGHT, art.sprite.shape[0])) + 0.5, 0, 255).astype(np.uint8), w, h)
    before = on_grey(np.clip(_sharpness(cv2.transform(old[..., :3], transform), OLD_BLUR) + 0.5, 0, 255).astype(np.uint8),
                     old[..., 3] / 255)
    tr = Track(0, ("p", "pair"), [], 0.0)
    tr.artefact, tr.opacity = art, 1.0
    scene.tracks, scene.worker.version = [tr], scene.worker.version + 1
    colours, weight, _ = scene._sprite(Layer(tr, w, h, 0, 0, "sides"), (0, 0, w, h), (matrix, 0.0))  # exactly as live
    now = on_grey(colours, weight)
    inner = cv2.erode((alpha > 0.98).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return reference, before, now, inner


def detail(image: np.ndarray, inner: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)[inner].std())


def lightness(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


class _Version:
    version = 0


def main() -> None:
    catalog = load_catalog()
    scene = HandheldScene(catalog, "none")
    scene.worker = _Version()
    rows = [["vase", "height_px", "detail_before_pct", "detail_now_pct", "ssim_before", "ssim_now"]]
    crops = {}
    print(f"{'vase':26s} {'px':>5s} {'detail before':>14s} {'detail now':>11s} {'SSIM before':>12s} {'SSIM now':>9s}")
    for art in catalog:
        h = size_in_demo(art)
        reference, before, now, inner = drawings(scene, art, h, scene_light(art))
        ref = detail(reference, inner)
        mask = inner.astype(np.uint8) * 255
        row = [art.name, h, 100 * detail(before, inner) / ref, 100 * detail(now, inner) / ref,
               ssim(lightness(before), lightness(reference), mask), ssim(lightness(now), lightness(reference), mask)]
        rows.append([row[0], row[1], *np.round(row[2:], 3)])
        print(f"{row[0]:26s} {h:5d} {row[2]:13.0f}% {row[3]:10.0f}% {row[4]:12.3f} {row[5]:9.3f}")
        crops[art.name] = (reference, before, now)
    write_csv(OUTPUT_DIR / "live_harmony_detail.csv", rows)
    panels = []
    for name, h in FIGURE:
        reference, before, now = crops[name]
        cy, cx = int(0.55 * reference.shape[0]), reference.shape[1] // 2  # the painted belly, at 1:1
        part = lambda image: image[cy - 190:cy + 190, cx - 190:cx + 190]
        title = name.replace("_", " ").title()
        panels += [(part(reference), f"{title}: museum photo, hall light"), (part(before), "harmonized before (blur, 1080 px)"),
                   (part(now), "harmonized now")]
    print("Saved:", write_image(OUTPUT_DIR / "live_harmony_detail.jpg", labeled_grid(panels, cols=3, height=380)))


if __name__ == "__main__":
    main()
