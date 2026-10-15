"""
Module 2: scene geometry from a single photo, for physically plausible placement.

Depth Anything V2 (Yang et al., NeurIPS 2024), metric version, predicts the
distance of every pixel in metres. With a pinhole camera of focal length f
(pixels), an object of real height H standing at distance Z appears
h = f * H / Z pixels tall - so the object can be scaled automatically from its
real size. A plane fitted to the 3D floor points around the contact point gives
the floor's foreshortening, used to draw the contact shadow in perspective.
"""

import cv2
import numpy as np

DEPTH_MODELS = {"indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"}
FOV_DEG = 60.0  # assumed horizontal field of view, close to the cameras the metric models were trained on
_pipes: dict = {}


def depth_map(image: np.ndarray, scene: str = "indoor") -> np.ndarray:
    """Metric depth (metres) for every pixel of a BGR image."""
    import torch
    from PIL import Image
    from transformers import pipeline

    if scene not in _pipes:
        _pipes[scene] = pipeline("depth-estimation", model=DEPTH_MODELS[scene], device=0 if torch.cuda.is_available() else -1)
    depth = _pipes[scene](Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))["predicted_depth"].squeeze().cpu().numpy()
    return cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def focal_px(width: int) -> float:
    return width / 2 / np.tan(np.radians(FOV_DEG) / 2)


def object_scale(depth: np.ndarray, point: tuple[int, int], real_height_m: float) -> float:
    """Object height as a fraction of the image height, for an object of real_height_m standing at point."""
    x, y = point
    r = max(3, depth.shape[1] // 100)
    z = float(np.median(depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]))
    return focal_px(depth.shape[1]) * real_height_m / max(z, 0.1) / depth.shape[0]


def floor_foreshortening(depth: np.ndarray, point: tuple[int, int]) -> float:
    """How much the floor is compressed vertically at point (1 = seen from above, ~0 = grazing view):
    |n . r| for the normal n of a plane fitted to the 3D floor points around it and the viewing ray r."""
    h, w = depth.shape
    x, y = point
    f, cx, cy = focal_px(w), w / 2, h / 2
    ys, xs = np.mgrid[max(0, y - h // 20):min(h, y + h // 12), max(0, x - w // 10):min(w, x + w // 10)]
    z = depth[ys, xs]
    pts = np.stack([(xs - cx) * z / f, (ys - cy) * z / f, z], -1).reshape(-1, 3)
    for _ in range(2):  # least-squares plane, refitted once without the outliers
        centred = pts - pts.mean(0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        dist = np.abs(centred @ normal)
        pts = pts[dist <= 2 * dist.std() + 1e-6]
    ray = np.array([x - cx, y - cy, f])
    return float(np.clip(abs(normal @ ray) / np.linalg.norm(ray), 0.05, 1.0))


def depth_preview(depth: np.ndarray, point: tuple[int, int] | None = None) -> np.ndarray:
    """Colour-coded depth for figures (near = bright, far = dark), with the contact point marked."""
    d = np.log(np.clip(depth, 0.1, None))
    img = cv2.applyColorMap((255 * (d.max() - d) / max(np.ptp(d), 1e-6)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if point is not None:
        r = max(4, depth.shape[1] // 120)
        cv2.circle(img, point, r, (255, 255, 255), -1)
        cv2.circle(img, point, r, (0, 0, 0), max(1, r // 3))
    return img
