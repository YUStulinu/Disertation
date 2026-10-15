"""
Module 1 - Week 5 Day 3: live robustness test under different conditions.

Runs the exact live pipeline (MediaPipe -> time-sampled window -> ONNX) and logs
predictions only inside timed TRIALS, so the ground truth is always the action
actually being performed:
  1. press a number key (1..N) to choose the action you are about to perform
  2. a 3 s countdown lets you get into position (nothing is logged)
  3. perform the action during the 4 s trial - only these predictions are logged
     (start it as soon as REC appears and keep at it until the end)
  U removes the last trial (e.g. a wrong key or a botched attempt).
Scored per frame and per trial (majority vote of the trial's frames).

Usage:
    python robustness_test.py --condition normal     (also: fast, low_light, side_angle, far_distance)
    python robustness_test.py --report
Keys (in the camera window): 1..N = start a trial, U = undo the last trial, Q = quit.
"""

import argparse
import csv
import time
from collections import Counter

import cv2

from common.imaging import show_live
from common.paths import DATA_DIR
from common.plots import bar_chart
from common.pose import LiveActionClassifier, PoseTracker, draw_skeleton, open_webcam

LOG_DIR = DATA_DIR / "robustness_logs"
CONDITIONS = ["normal", "fast", "low_light", "side_angle", "far_distance"]
COUNTDOWN_S, TRIAL_S = 3.0, 4.0
LOG_HEADER = ["timestamp", "condition", "trial", "gt_label", "predicted", "confidence"]


def read_log(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_robustness_results() -> dict[str, dict]:
    """{condition: {frame_acc, trial_acc, frames, trials}} from the trial logs."""
    results = {}
    for path in sorted(LOG_DIR.glob("*.csv")):
        rows = read_log(path)
        trials: dict[str, list] = {}
        for r in rows:
            trials.setdefault(r["trial"], []).append(r)
        if rows:
            actions = sorted({r["gt_label"] for r in rows})
            results[path.stem] = {
                "frame_acc": sum(r["predicted"] == r["gt_label"] for r in rows) / len(rows),
                "trial_acc": sum(Counter(r["predicted"] for r in t).most_common(1)[0][0] == t[0]["gt_label"]
                                 for t in trials.values()) / len(trials),
                "frames": len(rows), "trials": len(trials),
                "per_action": {a: sum(r["predicted"] == a for r in rows if r["gt_label"] == a)
                               / sum(r["gt_label"] == a for r in rows) for a in actions}}
    return results


def print_report() -> None:
    """The results per condition and per action, plus a chart in data/final_metrics/."""
    results = load_robustness_results()
    conditions = sorted(results, key=lambda c: CONDITIONS.index(c) if c in CONDITIONS else len(CONDITIONS))
    actions = sorted({a for r in results.values() for a in r["per_action"]})
    short = [a.split("_")[0] for a in actions]
    print(f"\n{'Condition':<16}{'Frames':>8}{'Frame acc':>11}{'Trials':>8}{'Trial acc':>11}   frame acc per action: "
          + "  ".join(short))
    for cond in conditions:
        r = results[cond]
        per_action = "  ".join(f"{100 * r['per_action'][a]:>{len(n)}.0f}" if a in r["per_action"] else " " * len(n)
                               for a, n in zip(actions, short))
        print(f"{cond:<16}{r['frames']:>8}{100 * r['frame_acc']:>10.1f}%{r['trials']:>8}{100 * r['trial_acc']:>10.1f}%   "
              + per_action)
    if results:
        print("Saved:", bar_chart(conditions, {"Frame accuracy": [100 * results[c]["frame_acc"] for c in conditions],
                                               "Trial accuracy": [100 * results[c]["trial_acc"] for c in conditions]},
                                  "Live robustness test per condition (higher is better)", "Accuracy (%)",
                                  DATA_DIR / "final_metrics" / "module1_robustness.png", fmt="{:.1f}", xlim=(0, 112)))


def trials_per_action(path) -> tuple[int, Counter]:
    """(number of the last trial, trials done per action) of a log."""
    rows = read_log(path) if path.exists() else []
    actions = {r["trial"]: r["gt_label"] for r in rows}
    return max(map(int, actions), default=0), Counter(actions.values())


def run_test(condition: str) -> None:
    path = LOG_DIR / f"{condition}.csv"
    classifier = LiveActionClassifier()
    keys = {ord(str(i + 1)): label for i, label in enumerate(classifier.labels[:9])}
    print("Keys:", {chr(k): v for k, v in keys.items()}, f"| U = undo | Q = quit | appending trials to {path}")
    trial, done = trials_per_action(path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracker, cap = PoseTracker(), open_webcam()
    gt, phase, phase_end, prediction = None, "waiting", 0.0, ("-", 0.0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(LOG_HEADER)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.perf_counter()
                frame = cv2.flip(frame, 1)
                landmarks = tracker.detect(frame)
                draw_skeleton(frame, landmarks)
                out = classifier.push(landmarks, now)
                prediction = out[:2] if out else prediction
                if phase == "countdown" and now >= phase_end:
                    phase, phase_end = "trial", now + TRIAL_S
                elif phase == "trial" and now >= phase_end:
                    phase, done[gt] = "waiting", done[gt] + 1
                    print(f"  trial {trial} ({gt}) done")
                if phase == "trial" and out:
                    writer.writerow([f"{now:.3f}", condition, trial, gt, prediction[0], f"{prediction[1]:.4f}"])

                menu = "  ".join(f"{chr(k)} {label.split('_')[0]}:{done[label]}" for k, label in keys.items())
                status = {"waiting": (menu, (200, 200, 200)),  # e.g. "1 falling:2  2 idle:0 ..." = trials done
                          "countdown": (f"Get ready: {gt} in {phase_end - now:.1f}s", (0, 220, 255)),
                          "trial": (f"REC {gt}  {phase_end - now:.1f}s", (0, 60, 255))}[phase]
                lines = [(f"Pred: {prediction[0]} ({100 * prediction[1]:.0f}%)", (0, 220, 0)), status,
                         (f"Condition: {condition} | trials: {trial} | U = undo | Q = quit", (160, 160, 160))]
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, h - 100), (w, h), (20, 20, 20), -1)
                for i, (text, colour) in enumerate(lines):
                    cv2.putText(frame, text, (8, h - 72 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                key = show_live("Robustness test (Q = quit)", frame)
                if key == ord("q"):
                    break
                if key in keys and phase == "waiting":
                    gt, phase, phase_end, trial = keys[key], "countdown", now + COUNTDOWN_S, trial + 1
                elif key == ord("u") and phase == "waiting" and trial > 0:  # undo: rewrite the log without the last trial
                    f.flush()
                    rows = read_log(path)
                    keep = [r for r in rows if r["trial"] != str(trial)]
                    f.truncate(0)
                    writer.writerows([LOG_HEADER] + [[r[k] for k in LOG_HEADER] for r in keep])
                    f.flush()
                    print(f"  trial {trial} removed")
                    trial, done = trials_per_action(path)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            tracker.close()
    print_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live robustness test - Week 5 Day 3")
    parser.add_argument("--condition", default="normal", choices=CONDITIONS)
    parser.add_argument("--report", action="store_true", help="Summarise all saved logs")
    args = parser.parse_args()
    if args.report:
        print_report()
    else:
        run_test(args.condition)


if __name__ == "__main__":
    main()
