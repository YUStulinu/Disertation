"""
Module 2 shared harmonization code.

Every method takes the composite and the object's alpha mask IN THE COMPOSITE
(from common.composition.place) and recolours only the object. Outside the
mask the composite IS the background, so the reference comes from there.

Learned methods (pre-trained on the iHarmony4 benchmark):
  harmonizer - Harmonizer (Ke et al., ECCV 2022): EfficientNet-B0 predicting six global
               filters (temperature, brightness, contrast, saturation, highlight, shadow).
  pctnet     - PCT-Net (Guerreiro et al., CVPR 2023), ViT backbone: a colour transform per pixel.
  pctnet_cnn - PCT-Net with its CNN backbone.
  ensemble   - mean of the three outputs (default): errors of the individual models
               partly cancel, so it beats each of them on the benchmark (metrics.py).
Illumination-aware methods (heavier; common/intrinsic.py, common/relight.py):
  intrinsic  - Intrinsic Harmonization (Careaga et al., SIGGRAPH Asia 2023): albedo harmonization +
               re-shading of the object under the lighting estimated from the background.
  reshade    - the ensemble's colours, then only the re-shading step of Intrinsic Harmonization
               (its albedo step over-corrects strongly coloured objects, e.g. terracotta).
  iclight    - IC-Light (Zhang et al., ICLR 2025): diffusion relighting, only its lighting change is kept.
Baselines without training:
  reinhard   - Lab mean/std transfer (Reinhard et al., IEEE CG&A 2001), object vs background ring.
  histogram  - per-channel histogram matching, object vs background ring.
  vgg        - a 3x3 colour matrix + offset optimised so the object's VGG16 feature statistics
               (AdaIN, Huang & Belongie ICCV 2017) match the background's, up to --layer.
match_camera adds the low-level cues colour harmonization leaves out: it gives the object
the scene's sharpness (blur or unsharp mask) and adds the sensor noise it lacks.
"""

import sys

import cv2
import numpy as np

from common.composition import blend
from common.imaging import best_device
from common.paths import ROOT

THIRD_PARTY = ROOT / "third_party"
VGG_LAYERS = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}  # ReLU index in vgg16().features
METHOD_NAMES = {"none": "No harmonization", "reinhard": "Classical - Reinhard", "histogram": "Classical - Histogram",
                "vgg": "VGG16 statistics", "harmonizer": "DL - Harmonizer", "pctnet": "DL - PCT-Net (ViT)",
                "pctnet_cnn": "DL - PCT-Net (CNN)", "ensemble": "DL - Ensemble (3 models)",
                "intrinsic": "DL - Intrinsic (relit)", "reshade": "DL - Ensemble + reshading",
                "iclight": "DL - IC-Light (relit)"}
METHODS = list(METHOD_NAMES)
_models: dict = {}  # loaded networks, keyed by (name, device)


def reference_ring(alpha: np.ndarray) -> np.ndarray:
    """Boolean mask of the background around the object (falls back to all background)."""
    obj = alpha > 127
    ys, xs = np.nonzero(obj)
    r = max(15, int(0.25 * max(np.ptp(ys), np.ptp(xs)))) if len(ys) else 15
    ring = (cv2.dilate(obj.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0) & (alpha == 0)
    return ring if ring.sum() >= 100 else alpha == 0


# ── Classical baselines ───────────────────────────────────────────────────

def reinhard(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)  # true Lab ranges (L 0-100)
    obj, ref = lab[alpha > 127], lab[reference_ring(alpha)]
    out = cv2.cvtColor((lab - obj.mean(0)) * (ref.std(0) / (obj.std(0) + 1e-6)) + ref.mean(0), cv2.COLOR_Lab2BGR)
    return blend(image, np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), alpha)


def histogram(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    obj, ring = alpha > 127, reference_ring(alpha)
    out = image.copy()
    for c in range(3):
        src = np.cumsum(np.bincount(image[..., c][obj], minlength=256)) / obj.sum()
        ref = np.cumsum(np.bincount(image[..., c][ring], minlength=256)) / ring.sum()
        out[..., c] = np.searchsorted(ref, src).clip(0, 255).astype(np.uint8)[image[..., c]]
    return blend(image, out, alpha)


def vgg_features(image: np.ndarray, alpha: np.ndarray, layer: str = "relu3_3", iterations: int = 120,
                 max_side: int = 320, reg: float = 2.0) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    import torchvision.models as models

    device = best_device()
    if ("vgg", device) not in _models:
        _models["vgg", device] = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23].eval() \
            .requires_grad_(False).to(device)
    vgg, taps = _models["vgg", device], [i for i in VGG_LAYERS.values() if i <= VGG_LAYERS[layer]]

    ring = reference_ring(alpha)  # work on the object + its surroundings only, downscaled for speed
    ys, xs = np.nonzero((alpha > 0) | ring)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = min(1.0, max_side / max(y2 - y1, x2 - x1))
    size = (max(32, round((x2 - x1) * s)), max(32, round((y2 - y1) * s)))
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(cv2.resize(a, size, interpolation=cv2.INTER_AREA))).float().to(device)
    rgb = tensor(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None] / 255
    m_obj = tensor(alpha[y1:y2, x1:x2].astype(np.float32) / 255)[None, None]
    m_ref = tensor(ring[y1:y2, x1:x2].astype(np.float32))[None, None]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def stats(feat, m):  # masked per-channel mean / std
        m = F.interpolate(m, size=feat.shape[-2:], mode="area")
        mu = (feat * m).sum((2, 3)) / (m.sum() + 1e-6)
        return mu, (((feat - mu[..., None, None]) ** 2 * m).sum((2, 3)) / (m.sum() + 1e-6) + 1e-6).sqrt()

    def features(x):
        out, h = [], (x - mean) / std
        for i, module in enumerate(vgg[:max(taps) + 1]):
            h = module(h)
            if i in taps:
                out.append(h)
        return out

    with torch.no_grad():
        targets = [stats(f, m_ref) for f in features(rgb)]
    matrix = torch.eye(3, device=device, requires_grad=True)
    offset = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([matrix, offset], lr=0.02)
    for _ in range(iterations):
        recoloured = (torch.einsum("ij,bjhw->bihw", matrix, rgb) + offset.view(1, 3, 1, 1)).clamp(0, 1)
        loss = reg * (((matrix - torch.eye(3, device=device)) ** 2).sum() + (offset ** 2).sum())
        for f, (mu_t, sd_t) in zip(features(rgb * (1 - m_obj) + recoloured * m_obj), targets):
            mu, sd = stats(f, m_obj)  # dividing by the target scale makes the layers comparable
            loss = loss + (((mu - mu_t) ** 2).mean() + ((sd - sd_t) ** 2).mean()) / (sd_t.mean() ** 2 + 1e-6) / len(targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 @ matrix.detach().cpu().numpy().T
    out = np.clip(full + offset.detach().cpu().numpy(), 0, 1)
    return blend(image, cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), alpha)


# ── Learned harmonization (pre-trained on iHarmony4) ──────────────────────

def _load(name: str, device):
    """Load a third-party harmonization network once: (net, input mean, input std)."""
    import torch

    if (name, device) not in _models:
        if name == "harmonizer":
            sys.path.insert(0, str(THIRD_PARTY / "Harmonizer"))
            from src.model import Harmonizer  # third_party/Harmonizer/src

            net, weights, norm = Harmonizer(), THIRD_PARTY / "Harmonizer" / "pretrained" / "harmonizer.pth", None
        else:
            sys.path.insert(0, str(THIRD_PARTY / "PCTNet"))
            from iharm.mconfigs import BMCONFIGS  # third_party/PCTNet/iharm

            variant = "CNN" if name == "pctnet_cnn" else "ViT"
            cfg = BMCONFIGS[f"{variant}_pct"]
            net, norm = cfg["model"](**cfg["params"]), cfg["params"]["input_normalization"]
            weights = THIRD_PARTY / "PCTNet" / "pretrained_models" / f"PCTNet_{variant}.pth"
        net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        norm = norm or {"mean": [0, 0, 0], "std": [1, 1, 1]}
        _models[name, device] = (net.eval().to(device), *(torch.tensor(norm[k], device=device).view(1, 3, 1, 1)
                                                          for k in ("mean", "std")))
    return _models[name, device]


def learned(image: np.ndarray, alpha: np.ndarray, name: str = "harmonizer") -> np.ndarray:
    import torch
    import torch.nn.functional as F

    device = best_device()
    net, mean, std = _load(name, device)
    rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(device)
    mask = torch.from_numpy(alpha).float().div(255)[None, None].to(device)
    with torch.no_grad():
        if name == "harmonizer":
            out = net.restore_image(rgb, mask, net.predict_arguments(rgb, (mask > 0.5).float()))[-1]
        else:  # PCT-Net: parameters predicted at 256x256, colour transform applied at full resolution
            low = lambda t: F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
            x = (rgb - mean) / std
            out = net(low(x), x, low(mask), mask)["images_fullres"][None] * std + mean
    out = (out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def ensemble(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    outs = [learned(image, alpha, name).astype(np.float32) for name in ("harmonizer", "pctnet", "pctnet_cnn")]
    return (np.mean(outs, axis=0) + 0.5).astype(np.uint8)


def harmonize(image: np.ndarray, alpha: np.ndarray, method: str = "ensemble", **kw) -> np.ndarray:
    if method == "none":
        return image.copy()
    if method in ("harmonizer", "pctnet", "pctnet_cnn"):
        return learned(image, alpha, method)
    if method in ("intrinsic", "reshade", "iclight"):
        from common import intrinsic, relight

        (intrinsic if method == "iclight" else relight).unload()  # the two stacks do not fit 6 GB together
        if method == "iclight":
            return relight.relight(image, alpha, **kw)
        colours = ensemble(image, alpha) if method == "reshade" else image
        return intrinsic.intrinsic(colours, alpha, albedo=method == "intrinsic", **kw)
    return {"reinhard": reinhard, "histogram": histogram, "vgg": vgg_features, "ensemble": ensemble}[method](image, alpha, **kw)


# ── Camera consistency: blur and sensor noise ─────────────────────────────

def _blur_level(gray: np.ndarray, region: np.ndarray) -> float:
    """No-reference blur (Crete-Roffet et al., SPIE 2007): how little the region changes when
    re-blurred. 0 = sharp, 1 = completely blurred."""
    scores = []
    for axis, ksize in ((0, (1, 9)), (1, (9, 1))):
        sharp, reblur = np.abs(np.diff(gray, axis=axis)), np.abs(np.diff(cv2.blur(gray, ksize), axis=axis))
        r = region[1:] if axis == 0 else region[:, 1:]
        total = sharp[r].sum()
        scores.append(1 - np.maximum(sharp - reblur, 0)[r].sum() / max(total, 1e-6))
    return max(scores)


def _noise_level(gray: np.ndarray, region: np.ndarray) -> float:
    """Sensor noise sigma (Immerkaer, CVIU 1996), on the flatter half of the region."""
    response = np.abs(cv2.filter2D(gray, -1, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)))
    grad = np.hypot(cv2.Sobel(gray, -1, 1, 0), cv2.Sobel(gray, -1, 0, 1))
    flat = region & (grad <= np.median(grad[region]))
    return float(np.sqrt(np.pi / 2) / 6 * response[flat].mean()) if flat.any() else 0.0


def camera_noise(image: np.ndarray, size: tuple[int, int] = (640, 360)) -> float:
    """Sensor noise sigma of a camera frame at its own resolution (_noise_level on a central crop of
    at most `size`): the grain an object drawn into that frame should get. Measured where it is
    added - real cameras denoise and compress, so their noise does not follow the white-noise rule
    when the image is resized (a Full HD webcam frame measured 0.25 grey levels, its 480-row copy 0.42)."""
    h, w = image.shape[:2]
    cw, ch = min(w, size[0]), min(h, size[1])
    crop = image[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return _noise_level(gray, np.ones(gray.shape, bool))


def _sharpness(image: np.ndarray, t: float, px: float = 1.0) -> np.ndarray:
    """t > 0: Gaussian blur with sigma t; t < 0: unsharp mask with amount -t (sigma 1). px = the image's
    pixels per pixel of the image t is meant for: both radii are scaled by it."""
    if t >= 0:
        return cv2.GaussianBlur(image, (0, 0), t * px) if t > 0.05 else image
    return cv2.addWeighted(image, 1 - t, cv2.GaussianBlur(image, (0, 0), px), t, 0)


def camera_params(image: np.ndarray, alpha: np.ndarray, reference: np.ndarray | None = None) -> tuple[float, float]:
    """(t, noise): the sharpness change t (see _sharpness) that gives the object the blur level of
    the scene around it (or of the `reference` mask, e.g. the hand holding it), found by
    bisection, and the sensor noise sigma the object lacks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    obj = cv2.erode((alpha > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not obj.any():
        return 0.0, 0.0
    ring = reference if reference is not None and reference.sum() >= 100 else reference_ring(alpha)
    target, t = _blur_level(gray, ring), 0.0
    if abs(_blur_level(gray, obj) - target) > 0.02:
        lo, hi = -1.0, 2.5  # blur level grows with t: at most 1.0 sharpening amount, 2.5 px blur
        for _ in range(10):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _blur_level(_sharpness(gray, mid), obj) < target else (lo, mid)
        t = (lo + hi) / 2
    return t, float(np.sqrt(max(_noise_level(gray, ring) ** 2 - _noise_level(_sharpness(gray, t), obj) ** 2, 0)))


def apply_camera(image: np.ndarray, alpha: np.ndarray, t: float, noise: float, rng=None) -> np.ndarray:
    out = blend(image, _sharpness(image, t), alpha) if t else image
    grain = (rng or np.random.default_rng(0)).normal(0, noise, image.shape[:2])[..., None] * (alpha[..., None] / 255)
    return np.clip(out.astype(np.float32) + grain + 0.5, 0, 255).astype(np.uint8)


def match_camera(image: np.ndarray, alpha: np.ndarray, seed: int = 0) -> np.ndarray:
    """Give the object the scene's sharpness (blur it, or sharpen it with an unsharp mask), then
    add the sensor noise it lacks. The amount is found by bisection on the measured blur level."""
    return apply_camera(image, alpha, *camera_params(image, alpha), np.random.default_rng(seed))
