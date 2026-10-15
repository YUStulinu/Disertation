"""
Module 1 - live demo / end-to-end verification (Week 1 Day 3, Week 5 Day 5).

Runs exactly what the Qt/C++ app does - webcam -> MediaPipe skeleton ->
time-sampled window -> normalization -> ONNX Runtime - and shows the predicted
action with per-class probability bars. --pose-only shows just the skeleton.

Usage:
    python verify_pipeline.py [--pose-only] [--model models/action_classifier.onnx]
Keys: S = save the frame shown (images/python_app/<action>_<n>.png), Q or Esc = quit.
"""

import argparse
import time
from pathlib import Path

import cv2

from common.imaging import NOTE_S, RateMeter, save_shot, show_live, with_note
from common.paths import SCREENSHOTS_DIR
from common.pose import ONNX_PATH, LiveActionClassifier, PoseTracker, draw_skeleton, open_webcam


def draw_overlay(frame, classifier, prediction, person: bool, fps: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS {fps:.0f}  |  window {100 * classifier.progress:.0f}%", (8, h - 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    if not person or prediction is None:
        text = "No person detected" if not person else "Collecting frames..."
        cv2.putText(frame, text, (8, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
        return
    label, conf, probs = prediction
    cv2.putText(frame, f"{label}  {100 * conf:.0f}%", (8, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 220, 0) if conf > 0.7 else (0, 180, 255), 2)
    bar_w = (w - 16) // len(probs)
    for i, (name, p) in enumerate(zip(classifier.labels, probs)):
        x = 8 + i * bar_w
        cv2.rectangle(frame, (x, h - 30 - int(p * 20)), (x + bar_w - 6, h - 30),
                      (0, 200, 0) if name == label else (110, 110, 110), -1)
        cv2.putText(frame, name[:12], (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live action recognition (same pipeline as the C++ app)")
    parser.add_argument("--model", type=Path, default=ONNX_PATH)
    parser.add_argument("--pose-only", action="store_true", help="Only draw the skeleton")
    args = parser.parse_args()

    classifier = None if args.pose_only else LiveActionClassifier(args.model)
    tracker, cap = PoseTracker(), open_webcam()
    prediction, meter, note = None, RateMeter(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.perf_counter()
            fps = meter.tick()
            frame = cv2.flip(frame, 1)
            landmarks = tracker.detect(frame)
            draw_skeleton(frame, landmarks)
            if classifier:
                prediction = classifier.push(landmarks, now) or prediction
                draw_overlay(frame, classifier, prediction, landmarks is not None, fps)
            else:
                cv2.putText(frame, f"FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            key = show_live("Live action recognition (S = save, Q = quit)", with_note(frame, note, top=True))
            if key == ord("q"):
                break
            if key == ord("s"):  # the frame shown, without the note, named after the action shown
                name = prediction[0] if classifier and prediction and landmarks is not None else "frame"
                path = save_shot(frame, name, SCREENSHOTS_DIR)
                note = (f"Saved {path.name}", time.perf_counter() + NOTE_S)
                print("Saved:", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
