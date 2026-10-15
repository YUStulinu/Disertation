"""Module 2 shared image helpers: I/O, labelled comparison figures, preview windows."""

import time
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def read_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Binary uint8 mask (0/255), optionally resized to `shape` = (h, w)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def read_composite(image_path: str | Path, mask_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A composed image + its placed alpha mask (default: <image>_mask.png saved by compose/pipeline)."""
    from common.paths import mask_path_for

    image = read_image(image_path)
    mask_path = Path(mask_path or mask_path_for(image_path))
    alpha = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if alpha is None:
        raise FileNotFoundError(f"Cannot read the placed mask {mask_path}. Create the composite with compose.py "
                                "or pipeline.py - they save it next to the image.")
    if alpha.shape != image.shape[:2]:
        raise ValueError(f"{mask_path.name} is {alpha.shape[::-1]} but the image is {image.shape[1::-1]}: it is not "
                         "the mask of this composite (a SOURCE-image mask cannot be reused here).")
    return image, alpha


def write_image(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def labeled_grid(panels: list[tuple[np.ndarray, str]], cols: int | None = None, height: int = 640,
                 label_h: int = 44) -> np.ndarray:
    """Side-by-side (or gridded) panels, all resized to one height (the first panel's, at most
    `height`), each with a caption bar. In a grid of several rows, panels are padded to equal width.

    cv2.putText only renders ASCII, so captions must avoid characters like em dashes.
    """
    h = min(panels[0][0].shape[0], height)
    tiles = []
    for img, text in panels:
        img = cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)
        tile = np.zeros((h + label_h, img.shape[1], 3), dtype=np.uint8)
        tile[label_h:] = img
        scale = min(0.8, 0.95 * img.shape[1] / max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0], 1))
        cv2.putText(tile, text, (8, label_h - 14), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        tiles.append(tile)
    cols = cols or len(tiles)
    if cols < len(tiles):
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def crop_object(image: np.ndarray, alpha: np.ndarray, margin: float = 0.25) -> np.ndarray:
    """Crop around the object (plus a margin) so edges, shadow and sharpness can be judged."""
    ys, xs = np.nonzero(alpha > 127)
    dy, dx = int(margin * np.ptp(ys)), int(margin * np.ptp(xs))
    return image[max(0, ys.min() - dy):ys.max() + dy, max(0, xs.min() - dx):xs.max() + dx]


_live_windows: set[str] = set()


def show_live(title: str, frame: np.ndarray, wait_ms: int = 1) -> int:
    """cv2.imshow + cv2.waitKey for the live camera windows. Returns the key pressed, lower-case
    (so keys work with Caps Lock on); Esc and the window's X button both return "q" - otherwise
    cv2.imshow would silently open a new window after the user closed it. The first frame also
    brings the window to the front, so the keys work at once."""
    cv2.imshow(title, frame)
    if title not in _live_windows:
        _live_windows.add(title)
        try:
            from ctypes import windll

            if hwnd := windll.user32.FindWindowW(None, title):
                windll.user32.SetForegroundWindow(hwnd)
        except ImportError:  # not Windows
            pass
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
        return ord("q")
    return key + 32 if ord("A") <= key <= ord("Z") else key


NOTE_S = 2.0  # how long a live window shows a short message such as "Saved ..."


class FfmpegWriter:
    """cv2.VideoWriter's write / release / isOpened, encoding H.264 with ffmpeg's x264 (the
    imageio-ffmpeg package): ~25-80 MB per minute of Full HD at crf 20 (more for a noisier camera
    image; the final demo recording: ~56 MB), where Media Foundation's fixed bitrate gives ~475 MB. The frames go to ffmpeg's stdin, so if the program is ended abruptly
    ffmpeg still gets the end of input and finishes a playable file."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], crf: int = 20):
        import subprocess

        import imageio_ffmpeg

        self._proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def isOpened(self) -> bool:
        return self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(memoryview(np.ascontiguousarray(frame)))  # no copy of the frame
        except (BrokenPipeError, OSError):  # ffmpeg stopped: keep the program running
            pass

    def release(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait()


def video_writer(path: Path, fps: float, size: tuple[int, int]):
    """An H.264 .mp4 writer (plays in every browser, so the video can go into results.html): x264
    through ffmpeg if imageio-ffmpeg is installed, else Windows Media Foundation (large files),
    else MPEG-4 Part 2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FfmpegWriter(path, fps, size)
        if writer.isOpened():
            return writer
    except (ImportError, OSError):
        pass
    writer = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer


def save_shot(frame: np.ndarray, name: str, folder: Path) -> Path:
    """The S key of the live demos: the frame shown, as <folder>/<name>_<n>.png (n = the first free number)."""
    n = 1
    while (folder / f"{name}_{n}.png").exists():
        n += 1
    return write_image(folder / f"{name}_{n}.png", frame)


def with_note(frame: np.ndarray, note: tuple[str, float] | None, top: bool = False) -> np.ndarray:
    """The frame with a short message in its bottom (or top) right corner, while note = (text,
    until perf_counter time) lasts - drawn on a copy, so a screenshot saved meanwhile does not show it."""
    if not note or time.perf_counter() > note[1]:
        return frame
    out = frame.copy()
    u = out.shape[0] / 480  # sizes are for a 480-row frame
    thick = max(1, round(u))
    (w, h), _ = cv2.getTextSize(note[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55 * u, thick)
    pad = round(8 * u)
    x, y = out.shape[1] - w - round(14 * u), (round(14 * u) + h) if top else (out.shape[0] - round(14 * u))
    cv2.rectangle(out, (x - pad, y - h - pad), (x + w + pad, y + pad), (20, 20, 20), -1)
    cv2.putText(out, note[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * u, (90, 220, 90), thick, cv2.LINE_AA)
    return out


class RateMeter:
    """Frames per second of a live loop, from the moving average of the intervals between frames
    (averaging the intervals, not their inverses, which would overstate an irregular rate)."""

    def __init__(self):
        self.last, self.interval = None, 0.0

    def tick(self) -> float:
        """Call once per frame; returns the current rate."""
        now = time.perf_counter()
        if self.last is not None and now > self.last:
            self.interval = 0.9 * self.interval + 0.1 * (now - self.last) if self.interval else now - self.last
        self.last = now
        return 1.0 / self.interval if self.interval else 0.0


def show(title: str, img: np.ndarray, max_w: int = 1400) -> None:
    """Preview window scaled to fit the screen; any key closes it."""
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, round(img.shape[0] * max_w / img.shape[1])))
    cv2.imshow(f"{title} (press any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def best_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
