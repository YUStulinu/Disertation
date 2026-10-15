"""
Module 1 shared code: MediaPipe pose extraction, skeleton drawing, sequence
preprocessing and the live classifier.

The classifier sees SEQUENCE_LENGTH skeleton samples taken SAMPLE_RATE times per
second (a ~1.9 s window), interpolated from the camera frames by their
timestamps. Sampling by time instead of by frame makes the model see the same
motion speed whatever the camera / pose-detection frame rate. These constants
are the single source of truth for training (dataset.py), Python live
inference and the C++ app (webcamwidget.h must use the same values).
"""

import time
from collections import deque

import cv2
import numpy as np

from common.paths import MODELS_DIR

POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
ONNX_PATH = MODELS_DIR / "action_classifier.onnx"
LABELS_PATH = MODELS_DIR / "action_classifier.labels.txt"

NUM_FEATURES = 33 * 2                 # x, y of the 33 MediaPipe landmarks
SEQUENCE_LENGTH = 30                  # samples fed to the classifier
SAMPLE_RATE = 15.0                    # samples per second
WINDOW_SECONDS = (SEQUENCE_LENGTH - 1) / SAMPLE_RATE
CLIP_SECONDS = 5.0                    # length of every recorded clip (record_session.py)
SMOOTHING = 0.8                       # live: weight of the previous probabilities (~0.2 s memory)
MIN_CONFIDENCE = 0.6                  # live: below this the action is reported as "uncertain"

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
CONNECTIONS = [  # upper body + legs (face and finger details left out for readability)
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


# ── Camera + MediaPipe ────────────────────────────────────────────────────

class PoseTracker:
    """MediaPipe PoseLandmarker in VIDEO mode (temporal tracking) with monotonic timestamps,
    or IMAGE mode for unrelated stills (video=False)."""

    def __init__(self, min_confidence: float = 0.3, num_poses: int = 1, video: bool = True):
        from mediapipe.tasks.python import BaseOptions, vision

        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_poses=num_poses,
            min_pose_detection_confidence=min_confidence, min_tracking_confidence=min_confidence))
        self.video, self.t0, self.last_ms = video, time.perf_counter(), -1

    def detect_all(self, frame_bgr: np.ndarray, t: float | None = None) -> list[np.ndarray]:
        """(33, 4) arrays [x, y, z, visibility] (normalized 0-1), one per person, the tallest first;
        t = frame time in seconds for video files (default: now)."""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            now = time.perf_counter() - self.t0 if t is None else t
            self.last_ms = max(int(now * 1000), self.last_ms + 1)
            result = self.landmarker.detect_for_video(image, self.last_ms)
        else:
            result = self.landmarker.detect(image)
        people = [np.array([[p.x, p.y, p.z, p.visibility] for p in pose], dtype=np.float32) for pose in result.pose_landmarks]
        return sorted(people, key=lambda p: -np.ptp(p[:, 1]))

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """The tallest (nearest) person, or None."""
        people = self.detect_all(frame_bgr)
        return people[0] if people else None

    def close(self) -> None:
        self.landmarker.close()


def open_webcam(index: int = 0, size: tuple[int, int] | None = None) -> cv2.VideoCapture:
    """The webcam, at its default resolution or the one closest to size = (width, height)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open the webcam (index {index}). Is another application using it?")
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray | None, min_visibility: float = 0.25) -> None:
    """Draw a (33, 4) landmark array onto a BGR frame, in place."""
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = {i: (int(x * w), int(y * h)) for i, (x, y, _z, v) in enumerate(landmarks) if v > min_visibility}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


# ── Sequence preprocessing ────────────────────────────────────────────────

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Center a (T, 33, 2) sequence on the hip midpoint and scale by the shoulder-hip distance,
    so an action looks the same wherever the person stands and however far from the camera."""
    hips = (sequence[:, LEFT_HIP] + sequence[:, RIGHT_HIP]) / 2.0
    scale = np.linalg.norm((sequence[:, LEFT_SHOULDER] + sequence[:, RIGHT_SHOULDER]) / 2.0 - hips, axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)  # avoid divide-by-zero on bad/occluded frames
    return ((sequence - hips[:, None]) / scale[:, None, None]).astype(np.float32)


def sample_window(times: np.ndarray, frames: np.ndarray, end: float, speed: float = 1.0) -> np.ndarray:
    """SEQUENCE_LENGTH frames linearly interpolated at 1/SAMPLE_RATE spacing, ending at `end`
    (speed > 1 covers a longer span = the action played faster), then normalized."""
    query = end - speed * np.arange(SEQUENCE_LENGTH)[::-1] / SAMPLE_RATE
    i = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    w = np.clip((query - times[i - 1]) / np.maximum(times[i] - times[i - 1], 1e-9), 0, 1)[:, None, None]
    return normalize_sequence(frames[i - 1] * (1 - w) + frames[i] * w)


# ── Live inference ────────────────────────────────────────────────────────

def load_labels(path=LABELS_PATH) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


class LiveActionClassifier:
    """Timestamped skeleton buffer -> time-sampled window -> ONNX classifier -> probabilities smoothed
    over time, "uncertain" below MIN_CONFIDENCE. Exactly what the C++ app does."""

    def __init__(self, onnx_path=ONNX_PATH, labels_path=LABELS_PATH):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path))
        self.labels = load_labels(labels_path)
        self.times: deque[float] = deque()
        self.frames: deque[np.ndarray] = deque()
        self.smoothed: np.ndarray | None = None

    @property
    def progress(self) -> float:
        """Share of the window already covered by buffered frames (1.0 = ready)."""
        return min(1.0, (self.times[-1] - self.times[0]) / WINDOW_SECONDS) if self.times else 0.0

    def push(self, landmarks: np.ndarray | None, t: float | None = None) -> tuple[str, float, np.ndarray] | None:
        """Add one frame (taken at time t, default now); returns (label, confidence, probabilities)
        when a person is visible and the window is covered, else None."""
        if landmarks is None:
            return None
        t = time.perf_counter() if t is None else t
        self.times.append(t)
        self.frames.append(landmarks[:, :2])
        while self.times[0] < t - WINDOW_SECONDS - 0.5:  # keep a little more than one window
            self.times.popleft()
            self.frames.popleft()
        if self.progress < 1.0:
            return None
        x = sample_window(np.array(self.times), np.stack(self.frames), t).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        logits = self.session.run(["action_logits"], {"keypoint_sequence": x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        self.smoothed = probs if self.smoothed is None else SMOOTHING * self.smoothed + (1 - SMOOTHING) * probs
        best = int(self.smoothed.argmax())
        label = self.labels[best] if self.smoothed[best] >= MIN_CONFIDENCE else "uncertain"
        return label, float(self.smoothed[best]), self.smoothed
