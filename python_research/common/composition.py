"""Module 2 shared composition code: placing a cut-out (with super-resolution when it must be
enlarged), inpainting the source, floor reflection and shadows."""

import cv2
import numpy as np

from common.paths import MODELS_DIR

SR_WEIGHTS = MODELS_DIR / "RealESRGAN_x4plus.pth"
SR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_sr: dict = {}


# ── Composition ───────────────────────────────────────────────────────────

def sr_weights_ready() -> bool:
    """Whether the Real-ESRGAN weights (~64 MB) are there - downloaded the first time they are needed;
    False (the cut-out is enlarged without super-resolution) if that fails, e.g. offline."""
    if not SR_WEIGHTS.exists():
        try:
            import torch

            SR_WEIGHTS.parent.mkdir(exist_ok=True)
            torch.hub.download_url_to_file(SR_URL, str(SR_WEIGHTS))  # a temporary file first: no partial weights
        except Exception as error:
            print(f"  Real-ESRGAN weights unavailable ({error}): no super-resolution.")
            return False
    return True


def super_resolve(bgra: np.ndarray) -> np.ndarray:
    """4x enlargement of a BGRA cut-out: Real-ESRGAN (Wang et al., ICCVW 2021) on the colour,
    bicubic on the alpha. Recovers sharp detail where plain interpolation would blur."""
    import torch
    from spandrel import ModelLoader

    from common.imaging import best_device

    device = best_device()
    if device not in _sr:
        _sr[device] = ModelLoader().load_from_file(str(SR_WEIGHTS)).eval().to(device)
    x = torch.from_numpy(cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(device)
    with torch.no_grad():
        y = _sr[device](x)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    colour = cv2.cvtColor((y * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.dstack([colour, cv2.resize(bgra[..., 3], colour.shape[1::-1], interpolation=cv2.INTER_CUBIC)])


def place(fg_bgra: np.ndarray, bg: np.ndarray, scale: float = 0.5, pos: tuple[float, float] = (0.5, 0.9),
          pos_px: tuple[int, int] | None = None, sr: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-blend a BGRA cut-out onto the background.

    scale: cut-out height as a fraction of the background height.
    pos:   bottom-centre anchor as (x, y) fractions of the background (pos_px overrides, in pixels).
    sr:    super-resolve the cut-out first when it has to be enlarged by more than 15%.
    Returns (composed BGR image, alpha mask 0-255 in BACKGROUND coordinates). The alpha
    mask is what every later step (harmonization, shadow, metrics) must use.
    """
    bg_h, bg_w = bg.shape[:2]
    th = max(1, round(bg_h * scale))
    if sr and th > 1.15 * fg_bgra.shape[0] and sr_weights_ready():
        fg_bgra = super_resolve(fg_bgra)
    fg = cv2.resize(fg_bgra, (max(1, round(fg_bgra.shape[1] * th / fg_bgra.shape[0])), th),
                    interpolation=cv2.INTER_AREA)
    cx, cy = pos_px or (round(pos[0] * bg_w), round(pos[1] * bg_h))
    x1, y1 = cx - fg.shape[1] // 2, cy - fg.shape[0]

    alpha = np.zeros((bg_h, bg_w), dtype=np.uint8)
    colour = np.zeros_like(bg)
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(bg_w, x1 + fg.shape[1]), min(bg_h, y1 + fg.shape[0])
    if xa >= xb or ya >= yb:
        raise ValueError("The foreground lands completely outside the background - adjust scale/pos.")
    crop = fg[ya - y1:yb - y1, xa - x1:xb - x1]
    alpha[ya:yb, xa:xb] = crop[:, :, 3]
    colour[ya:yb, xa:xb] = crop[:, :, :3]
    return blend(bg, colour, alpha), alpha


def blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """base * (1 - a) + top * a, with a = alpha / 255 (uint8 in, uint8 out)."""
    a = alpha.astype(np.float32)[..., None] / 255.0
    return np.clip(top.astype(np.float32) * a + base.astype(np.float32) * (1 - a) + 0.5, 0, 255).astype(np.uint8)


# ── Inpainting (fill the hole left in the source image) ───────────────────

def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow the mask by px pixels so the soft object border is filled too."""
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)) if px > 0 else mask


def inpaint_classical(image: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 5,
                      dilation: int = 3) -> np.ndarray:
    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}[method]
    return cv2.inpaint(image, dilate(mask, dilation), radius, flag)


_lama: dict = {}


def inpaint_lama(image: np.ndarray, mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    """LaMa (Fast Fourier Convolutions), the TorchScript model of simple-lama-inpainting
    (downloaded on first use). Loaded with map_location so it also runs on CPU-only PyTorch."""
    import torch
    from PIL import Image
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

    from common.imaging import best_device

    device = best_device()
    if device not in _lama:
        _lama[device] = torch.jit.load(download_model(LAMA_MODEL_URL), map_location=device).eval()
    img, msk = prepare_img_and_mask(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                                    Image.fromarray(dilate(mask, dilation)), device)
    with torch.inference_mode():
        out = (_lama[device](img, msk)[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[:image.shape[0], :image.shape[1]]  # LaMa pads to a multiple of 8


def inpaint(image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    return inpaint_lama(image, mask) if method == "lama" else inpaint_classical(image, mask, method)


# ── Floor reflection and shadows ───────────────────────────────────────────

def _base(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    ys, xs = np.nonzero(alpha > 127)
    return ys, xs, int(ys.max()), int(ys.max() - ys.min())


def add_reflection(image: np.ndarray, alpha: np.ndarray, strength: float = 0.3, fade: float = 0.35) -> np.ndarray:
    """Mirror image of the object in a glossy floor: the object flipped about its base line,
    fading with the distance from the base and slightly blurred, drawn under the object."""
    if strength <= 0 or not (alpha > 127).any():
        return image
    _, _, base, height = _base(alpha)
    n = min(height, image.shape[0] - base - 1)
    if n <= 0:
        return image
    rows = slice(base + 1, base + 1 + n)
    mirror = image[base - n + 1:base + 1][::-1]
    weight = strength * np.exp(-np.arange(n) / (fade * height))[:, None]
    a = alpha[base - n + 1:base + 1][::-1].astype(np.float32) / 255 * weight * (1 - alpha[rows] / 255)
    sigma = max(0.5, 0.01 * height)
    out = image.copy()
    out[rows] = blend(image[rows], cv2.GaussianBlur(mirror, (0, 0), sigma),
                      (cv2.GaussianBlur(a, (0, 0), sigma) * 255).astype(np.uint8))
    return out


def _darken_outside_object(image, shadow, object_alpha, opacity, blur):
    """Blur the shadow map, then darken by it - never on top of the object itself.

    blur=None picks a softness proportional to the object's width, so small and large
    objects get equally visible shadows.
    """
    if blur is None:
        blur = max(5, int(0.12 * np.ptp(np.nonzero(object_alpha > 127)[1])))
    shadow = cv2.GaussianBlur(shadow, (blur | 1,) * 2, 0)
    a = shadow.astype(np.float32) / 255.0 * opacity * (1 - object_alpha.astype(np.float32) / 255.0)
    return np.clip(image.astype(np.float32) * (1 - a[..., None]), 0, 255).astype(np.uint8)


def ellipse_shadow(image: np.ndarray, object_alpha: np.ndarray, opacity: float = 0.45, blur: int | None = None,
                   width_frac: float = 0.65, tilt: float | None = None) -> np.ndarray:
    """Soft contact shadow: a blurred ellipse under the object's base. `tilt` (from the depth map) is the
    floor's foreshortening, so the circle on the floor is drawn in perspective; without it 0.3 is assumed."""
    if not (object_alpha > 127).any():
        return image
    ys, xs, base, height = _base(object_alpha)
    base_xs = xs[ys > base - max(3, height // 20)]  # columns touching the floor (feet, vase foot)
    rx = max(4, int((np.ptp(base_xs) + 0.25 * np.ptp(xs)) * width_frac))
    shadow = np.zeros(object_alpha.shape, np.uint8)
    cv2.ellipse(shadow, (int(base_xs.mean()), base), (rx, max(3, int(rx * (tilt or 0.3)))), 0, 0, 360, 255, -1)
    return _darken_outside_object(image, shadow, object_alpha, opacity, blur)


def cast_shadow(image: np.ndarray, object_alpha: np.ndarray, angle_deg: float = 45.0, length: float = 0.35,
                opacity: float = 0.5, blur: int | None = None, tilt: float | None = None) -> np.ndarray:
    """Directional cast shadow: the silhouette laid flat on the floor plane behind the object.

    Every object point at height d above the base line maps to a floor point d*length away from
    the base, in direction angle_deg (0 = right, 90 = straight back); the backward component is
    foreshortened by the floor `tilt` when it is known.
    """
    if not (object_alpha > 127).any():
        return image
    _, _, base, _ = _base(object_alpha)
    a = np.radians(angle_deg)
    dx, dy = length * np.cos(a), length * np.sin(a) * (tilt / 0.3 if tilt else 1.0)
    # x' = x - dx*(y - base) ; y' = base + dy*(y - base)  (y - base <= 0 above the floor)
    m = np.float32([[1, -dx, dx * base], [0, dy, base * (1 - dy)]])
    shadow = cv2.warpAffine(object_alpha, m, object_alpha.shape[::-1])
    return _darken_outside_object(image, shadow, object_alpha, opacity, blur)


def add_shadow(image: np.ndarray, object_alpha: np.ndarray, mode: str = "ellipse", **kw) -> np.ndarray:
    if mode == "none":
        return image
    return ellipse_shadow(image, object_alpha, **kw) if mode == "ellipse" else cast_shadow(image, object_alpha, **kw)
