"""
Module 1 - Action Recognition: PyTorch dataset of skeleton windows.

Clips recorded by record_session.py (data/raw/<action>/*.npy, each
(num_frames, 33, 4): x, y, z, visibility over CLIP_SECONDS) are cut into
overlapping windows with common.pose.sample_window - the exact sampling the
live app uses. Splits are always made per CLIP, never per window, so windows
of one recording never land in both the training and the test set.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.paths import RAW_DIR
from common.pose import CLIP_SECONDS, SEQUENCE_LENGTH, WINDOW_SECONDS, sample_window

WINDOW_STEP_S = 1 / 6        # seconds between consecutive training windows
TRAIN_SPEEDS = (0.8, 1.0, 1.25)  # time-warp augmentation: the same action a bit slower / faster

# Left/right landmark pairs for the mirror augmentation (index 0, the nose, has no pair)
_MIRROR = np.arange(33)
for _l, _r in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _MIRROR[[_l, _r]] = _r, _l

Clip = tuple[Path, int]  # (path to .npy, label index)


def list_clips(data_dir: Path = RAW_DIR) -> tuple[list[Clip], list[str]]:
    """All clips + action labels (subfolder names, sorted for a stable encoding)."""
    labels = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and any(p.glob("*.npy")))
    if not labels:
        raise RuntimeError(f"No recorded clips under {data_dir}. Record some with record_session.py first.")
    return [(f, i) for i, label in enumerate(labels) for f in sorted((data_dir / label).glob("*.npy"))], labels


def split_clips(clips: list[Clip], per_class: int, seed: int = 42) -> tuple[list[Clip], list[Clip]]:
    """Stratified hold-out split: `per_class` clips of every action go to the test set."""
    rng, train, test = np.random.default_rng(seed), [], []
    for label in sorted({c[1] for c in clips}):
        group = [clips[i] for i in rng.permutation([i for i, c in enumerate(clips) if c[1] == label])]
        test, train = test + group[:per_class], train + group[per_class:]
    return train, test


def kfold_clips(clips: list[Clip], k: int = 5, seed: int = 0) -> list[tuple[list[Clip], list[Clip]]]:
    """Stratified k-fold split over clips -> [(train, test)] * k."""
    rng, folds = np.random.default_rng(seed), [[] for _ in range(k)]
    for label in sorted({c[1] for c in clips}):
        for j, i in enumerate(rng.permutation([i for i, c in enumerate(clips) if c[1] == label])):
            folds[j % k].append(clips[i])
    return [([c for f in folds if f is not fold for c in f], fold) for fold in folds]


def extract_windows(xy: np.ndarray, speeds=(1.0,)) -> np.ndarray:
    """(T, 33, 2) clip -> (N, SEQUENCE_LENGTH, 33, 2) normalized windows, for every speed."""
    times = np.arange(len(xy)) * (CLIP_SECONDS / len(xy))
    windows = []
    for speed in speeds:
        first_end = min(times[0] + speed * WINDOW_SECONDS, times[-1])
        windows += [sample_window(times, xy, end, speed) for end in np.arange(first_end, times[-1] + 1e-9, WINDOW_STEP_S)]
    return np.stack(windows)


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rotation (camera angle), scale (distance) and mirror (other side of the body)."""
    a = np.deg2rad(rng.uniform(-15.0, 15.0))
    out = sequence @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32).T * rng.uniform(0.9, 1.1)
    if rng.random() < 0.5:
        out = out[:, _MIRROR] * np.array([-1.0, 1.0], np.float32)
    return out.astype(np.float32)


class SkeletonWindowDataset(Dataset):
    """Every window of the given clips. Training sets add time-warped windows and random augmentation."""

    def __init__(self, clips: list[Clip], train: bool = False, seed: int = 0):
        parts = [extract_windows(np.load(p)[:, :, :2].astype(np.float32), TRAIN_SPEEDS if train else (1.0,))
                 for p, _ in clips]
        self.windows = np.concatenate(parts)
        self.targets = np.repeat([label for _, label in clips], [len(w) for w in parts])
        self.clip_ids = np.repeat(np.arange(len(clips)), [len(w) for w in parts])  # for clip-level voting
        self.train, self.rng = train, np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = augment_sequence(self.windows[idx], self.rng) if self.train else self.windows[idx]
        return torch.from_numpy(seq.reshape(SEQUENCE_LENGTH, -1)), int(self.targets[idx])
