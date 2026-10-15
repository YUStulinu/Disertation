"""Image quality metrics (PSNR, fMSE, SSIM) and synthetic test cases with known ground truth."""

import cv2
import numpy as np

from common.composition import blend


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    return float(err[mask > 127].mean() if mask is not None else err.mean())


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    e = mse(a, b, mask)
    return float("inf") if e == 0 else 10 * np.log10(255.0 ** 2 / e)


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5) over the image or inside a mask.

    The SSIM map is computed on the full 2-D images, so the local windows are
    spatially meaningful; the mask only selects which map values are averaged.
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    maps = []
    for ch in range(a.shape[2] if a.ndim == 3 else 1):
        x = (a[..., ch] if a.ndim == 3 else a).astype(np.float64)
        y = (b[..., ch] if b.ndim == 3 else b).astype(np.float64)
        blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
        mx, my = blur(x), blur(y)
        vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
        maps.append((2 * mx * my + c1) * (2 * cxy + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))
    ssim_map = np.mean(maps, axis=0)
    return float(ssim_map[mask > 127].mean() if mask is not None else ssim_map.mean())


def synthetic_composite(image: np.ndarray, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recolour the object of a REAL photo so it no longer matches its scene.

    This is how the iHarmony4 benchmark (Cong et al., DoveNet, CVPR 2020) builds
    test cases: the untouched photo is the ground truth a harmonizer should recover.
    Random lightness contrast/brightness and colour-cast changes in Lab space.
    """
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    obj_mean = lab[alpha > 127].mean(0)
    gain = np.array([rng.uniform(0.75, 1.25), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)], dtype=np.float32)
    shift = np.array([rng.uniform(-15, 15), rng.uniform(-12, 12), rng.uniform(-12, 12)], dtype=np.float32)
    lab = (lab - obj_mean) * gain + obj_mean + shift
    lab[..., 0] = lab[..., 0].clip(0, 100)
    altered = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
    return blend(image, altered, alpha)
