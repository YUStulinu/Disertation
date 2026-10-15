"""
Module 1 - Week 1: inspect the recorded skeleton clips.

Without arguments: per action, every clip's frame count and average landmark
visibility, so bad recordings (too short, person barely visible) stand out.
With a clip: replays its skeleton frame by frame (only the skeleton is stored,
not the video). Press 'q' to stop early.

Usage:
    python playback_data.py
    python playback_data.py data/raw/walking/1.npy
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from common.imaging import show_live
from common.paths import RAW_DIR
from common.pose import CLIP_SECONDS, draw_skeleton


def summary() -> None:
    clips = sorted(RAW_DIR.glob("*/*.npy"))
    for action in sorted({c.parent.name for c in clips}):
        group = [c for c in clips if c.parent.name == action]
        print(f"\n{action} ({len(group)} clips)")
        for path in group:
            seq = np.load(path)  # (frames, 33, 4): x, y, z, visibility
            print(f"  {path.name:20s} frames={len(seq):4d}  avg_visibility={seq[:, :, 3].mean():.2f}")
    print(f"\nTotal: {len(clips)} clips in {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise or replay recorded skeleton clips")
    parser.add_argument("clip", nargs="?", type=Path, help="Clip to replay (omit for the dataset summary)")
    args = parser.parse_args()
    if args.clip is None:
        return summary()

    sequence = np.load(args.clip)
    delay = max(1, round(1000 * CLIP_SECONDS / len(sequence)))  # real-time playback
    print(f"Playing back {args.clip.name}: {len(sequence)} frames (press 'q' to quit)")
    for landmarks in sequence:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_skeleton(canvas, landmarks, min_visibility=0.3)
        if show_live("Skeleton playback (Q = quit)", canvas, delay) == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
