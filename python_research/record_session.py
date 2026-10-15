"""
Module 1 - Week 1 Day 4 / Week 5 Day 1: record the skeleton dataset.

The camera runs continuously; press keys IN THE CAMERA WINDOW:
  SPACE = record the next 5 s clip of the current action (after a 3 s countdown)
  S     = skip to the next action        Q = quit
Each clip is saved as data/raw/<action>/<timestamp>.npy, shape (frames, 33, 4):
x, y, z, visibility of the 33 MediaPipe landmarks; --save-video also keeps an
.mp4 next to it (for demos - training only needs the skeleton).

"idle" = standing still or small non-action movements: it gives the classifier
a "nothing is happening" class, so the moments between actions are not forced
into one of the real actions.

Usage:
    python record_session.py                     # every action until it has --target clips
    python record_session.py --action idle       # only this action
    python record_session.py --action walking --target 12 --save-video
"""

import argparse
import time

import cv2
import numpy as np

from common.imaging import show_live
from common.paths import RAW_DIR
from common.pose import CLIP_SECONDS, PoseTracker, draw_skeleton, open_webcam

ACTIONS = ["falling", "sitting", "walking", "lifting_object", "idle"]
COUNTDOWN_S = 3.0
MIN_DETECTED = 0.3  # reject clips where a pose was found in < 30% of the frames


def count_clips(action: str) -> int:
    return len(list((RAW_DIR / action).glob("*.npy")))


def overlay(frame, top: str, bottom: str, colour=(0, 220, 255)) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, top, (8, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.putText(frame, bottom, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def save_clip(action: str, keypoints: list[np.ndarray], video: list[np.ndarray] | None) -> None:
    path = RAW_DIR / action / f"{int(time.time())}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack(keypoints))
    if video:
        writer = cv2.VideoWriter(str(path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                 len(video) / CLIP_SECONDS, video[0].shape[1::-1])
        for frame in video:
            writer.write(frame)
        writer.release()
    print(f"Saved {path.name} ({action} {count_clips(action)}, {len(keypoints)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled skeleton clips")
    parser.add_argument("--action", help="Record only this action (default: all of " + ", ".join(ACTIONS) + ")")
    parser.add_argument("--target", type=int, default=10, help="Clips wanted per action")
    parser.add_argument("--save-video", action="store_true", help="Also save each clip as .mp4")
    args = parser.parse_args()

    pending = [a for a in ([args.action] if args.action else ACTIONS) if count_clips(a) < args.target]
    print("To record:", {a: f"{count_clips(a)}/{args.target}" for a in pending} or "nothing")
    tracker, cap = PoseTracker(), open_webcam()
    state, t_start, keypoints, video, total = "wait", 0.0, [], [], 0
    try:
        while pending:
            action = pending[0]
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            clean = frame.copy()
            landmarks = tracker.detect(frame)
            draw_skeleton(frame, landmarks)
            now = time.time()

            if state == "wait":
                overlay(frame, f"{action.upper()}  [{count_clips(action)}/{args.target}]",
                        "SPACE = record | S = skip | Q = quit")
            elif state == "countdown":
                if now - t_start >= COUNTDOWN_S:
                    state, t_start, keypoints, video, total = "record", now, [], [], 0
                overlay(frame, f"{action.upper()} - get ready!", f"Starting in {COUNTDOWN_S - (now - t_start):.1f}s")
            else:  # recording
                total += 1
                if landmarks is not None:
                    keypoints.append(landmarks)
                if args.save_video:
                    video.append(clean)
                overlay(frame, f"REC {action.upper()}  {CLIP_SECONDS - (now - t_start):.1f}s",
                        f"Pose found in {len(keypoints)}/{total} frames", (0, 60, 255))
                if now - t_start >= CLIP_SECONDS:
                    if len(keypoints) >= max(5, MIN_DETECTED * total):
                        save_clip(action, keypoints, video)
                        if count_clips(action) >= args.target:
                            pending.pop(0)
                    else:
                        print(f"Clip rejected: pose found in only {len(keypoints)}/{total} frames. Try again.")
                    state = "wait"

            key = show_live("Recording session (SPACE = record, S = skip, Q = quit)", frame)
            if key == ord("q"):
                break
            if key == ord(" ") and state == "wait":
                state, t_start = "countdown", now
            elif key == ord("s") and state != "record":
                print(f"Skipped: {pending.pop(0)}")
                state = "wait"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
    print("Done. Next: python train.py && python export_onnx.py")


if __name__ == "__main__":
    main()
