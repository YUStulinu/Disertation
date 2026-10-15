"""Module 2 shared segmentation code: YOLOv8-seg detection, SAM 2.1 mask refinement, matting, extraction."""

import cv2
import numpy as np

from common.paths import MODELS_DIR

YOLO_WEIGHTS = {"n": MODELS_DIR / "yolov8n-seg.pt", "m": MODELS_DIR / "yolov8m-seg.pt"}
SAM_WEIGHTS = MODELS_DIR / "sam2.1_b.pt"
_sam = None


def load_yolo(size: str = "n"):
    """YOLOv8 nano (~7 MB) or medium (~52 MB) segmentation model; downloaded on first use."""
    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    return YOLO(str(YOLO_WEIGHTS[size]))


def detect(model, image: np.ndarray):
    # retina_masks: masks at the original image resolution. Without it they come at the
    # letterboxed inference size (e.g. 448x640 for a 400x600 image) and a plain resize misaligns them.
    return model(image, verbose=False, retina_masks=True)[0]


def describe(result) -> list[tuple[str, float]]:
    """[(class name, confidence)] for every detection, in YOLO order."""
    if result.boxes is None:
        return []
    return [(result.names[int(b.cls[0])], float(b.conf[0])) for b in result.boxes]


def pick_detection(result, prefer_class: str | None = None, index: int = -1) -> int | None:
    """--index if valid, else the most confident detection of prefer_class, else the most confident overall."""
    dets = describe(result)
    if not dets or result.masks is None:
        return None
    if 0 <= index < len(dets):
        return index
    candidates = [i for i, (name, _) in enumerate(dets) if prefer_class in (None, name)]
    if not candidates:
        print(f"  Class '{prefer_class}' not found (detected: {[n for n, _ in dets]}); using the best detection.")
        candidates = range(len(dets))
    return max(candidates, key=lambda i: dets[i][1])


def detection_mask(result, idx: int, shape: tuple[int, int]) -> np.ndarray:
    """Binary uint8 mask (0/255) of detection `idx` at image size `shape` = (h, w)."""
    raw = result.masks.data[idx].cpu().numpy()
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.where(raw > 0.5, 255, 0).astype(np.uint8)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Close small holes inside the object, then remove small noise blobs outside it."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def sam_mask(image: np.ndarray, box) -> np.ndarray | None:
    """Sharper mask of the object inside a detection box: SAM 2.1 (Ravi et al., 2024) prompted
    with the box. Follows thin parts (handles, fingers) and leaves out what YOLO's coarse
    mask grabs around the object (e.g. a glass stand). Weights (~150 MB) download on first use."""
    global _sam
    from ultralytics import SAM

    if _sam is None:
        _sam = SAM(str(SAM_WEIGHTS))
    result = _sam(image, bboxes=[list(map(float, box))], verbose=False)[0]
    return None if result.masks is None else detection_mask(result, 0, image.shape[:2])


def segment(model, image: np.ndarray, prefer_class: str | None = None, index: int = -1, sam: bool = True):
    """Run YOLO (+ SAM refinement) -> (mask of the chosen object in source coordinates, raw YOLO result)."""
    result = detect(model, image)
    idx = pick_detection(result, prefer_class, index)
    if idx is None:
        raise RuntimeError("No objects detected in the source image.")
    name, conf = describe(result)[idx]
    print(f"  Selected detection [{idx}]: {name} (conf={conf:.2f})")
    mask = sam_mask(image, result.boxes[idx].xyxy[0].tolist()) if sam else None
    if mask is None or not mask.any():
        mask = detection_mask(result, idx, image.shape[:2])
    return refine_mask(mask), result


def soft_alpha(image: np.ndarray, mask: np.ndarray, radius: int | None = None) -> np.ndarray:
    """Binary mask -> soft alpha that follows the photo's own edges: guided filter (He et al.,
    ECCV 2010) with the image as guide, applied only in a thin band along the boundary (radius: the
    filter's, default 0.4% of the image's larger side; the band is twice as wide)."""
    r = radius or max(2, round(0.004 * max(image.shape[:2])))
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    p = (mask > 0).astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mean_i, mean_p = box(guide), box(p)
    a = (box(guide * p) - mean_i * mean_p) / (box(guide * guide) - mean_i ** 2 + 1e-3)
    alpha = box(a) * guide + box(mean_p - a * mean_i)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1,) * 2)
    band = (cv2.dilate(mask, k) > 0) & (cv2.erode(mask, k) == 0)
    return np.where(band, np.clip(alpha, 0, 1) * 255 + 0.5, mask).astype(np.uint8)


def foreground_colours(image: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Remove the old background that bleeds into semi-transparent edge pixels (the "halo"):
    solve I = a*F + (1-a)*B for the object colour F, with B = the inpainted source image."""
    a = alpha.astype(np.float32)[..., None] / 255
    fg = (image.astype(np.float32) - (1 - a) * background) / np.maximum(a, 0.05)
    edge = ((alpha > 0) & (alpha < 250))[..., None]
    return np.where(edge, np.clip(fg + 0.5, 0, 255), image).astype(np.uint8)


def extract_foreground(image: np.ndarray, mask: np.ndarray, background: np.ndarray | None = None,
                       matting: bool = True, pad: int = 10, radius: int | None = None) -> np.ndarray:
    """Tight BGRA cut-out of the object. With matting the alpha follows the photo's edges and,
    given the inpainted `background`, the edge colours are cleaned of the old background;
    without it the mask edge is just slightly blurred."""
    alpha = soft_alpha(image, mask, radius) if matting else cv2.GaussianBlur(mask, (7, 7), 0)
    colour = foreground_colours(image, alpha, background) if matting and background is not None else image
    ys, xs = np.nonzero(alpha)
    if len(ys) == 0:
        return np.dstack([colour, alpha])
    y1, x1 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    y2, x2 = min(image.shape[0], ys.max() + pad + 1), min(image.shape[1], xs.max() + pad + 1)
    return np.dstack([colour, alpha])[y1:y2, x1:x2]


def draw_detections(image: np.ndarray, result, only: list[str] | None = None) -> np.ndarray:
    """Coloured semi-transparent masks + boxes + labels for every (or only the listed) detection."""
    out = image.copy()
    colours = np.random.default_rng(42).integers(80, 240, size=(max(len(describe(result)), 1), 3)).tolist()
    for i, (name, conf) in enumerate(describe(result)):
        if only and name not in only:
            continue
        m = detection_mask(result, i, image.shape[:2]) > 0
        out[m] = (0.6 * out[m] + 0.4 * np.array(colours[i])).astype(np.uint8)
        x1, y1, x2, y2 = map(int, result.boxes[i].xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), colours[i], 2)
        cv2.putText(out, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colours[i], 2)
    return out


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a > 0, b > 0
    inter, union, total = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union else 1.0), (2 * inter / total if total else 1.0)
