"""
Module 1 - Week 5: evaluation of the action classifier on UNSEEN data.

Default: k-fold cross-validation over clips - k fresh models are trained, each
scored only on the clips it never saw (with a small dataset every clip is tested
once). Reports per-class precision / recall / F1, macro and weighted averages,
accuracy per window (what the live app classifies) and per clip (majority vote)
and the Matthews Correlation Coefficient, and saves a CSV, an F1 chart and the
confusion matrix in data/final_metrics/ (*_cv or *_holdout).

  --holdout             score a model saved by `train.py --holdout N` on the clips it kept aside
  --compare OLD.pt      old vs that model on those held-out clips (data/model_comparison/)
  --checkpoint PATH     the model for --holdout / --compare (default: models/action_classifier.pt)
  --include-robustness  append the live robustness-test results (robustness_test.py)

Usage:
    python evaluate.py [--folds 5] [--include-robustness]
    python train.py --holdout 2 --out models/action_classifier_holdout.pt
    python evaluate.py --compare models/action_classifier_v1.pt --checkpoint models/action_classifier_holdout.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from common.paths import DATA_DIR, RAW_DIR
from common.plots import bar_chart, confusion_heatmap
from dataset import SkeletonWindowDataset, kfold_clips, list_clips
from model import load_checkpoint
from robustness_test import load_robustness_results
from train import CHECKPOINT_PATH, fit

OUTPUT_DIR = DATA_DIR / "final_metrics"


def predict(model, clips):
    """-> (window targets, window predictions, clip targets, clip majority-vote predictions)."""
    ds = SkeletonWindowDataset(clips)
    with torch.no_grad():
        preds = model(torch.from_numpy(ds.windows.reshape(len(ds), ds.windows.shape[1], -1))).argmax(1).numpy()
    votes = [np.bincount(preds[ds.clip_ids == i]).argmax() for i in range(len(clips))]
    return ds.targets, preds, np.array([label for _, label in clips]), np.array(votes)


def heldout_clips(ckpt: dict) -> list:
    if not ckpt.get("test_clips"):
        raise SystemExit("This model was trained on every clip. Use --checkpoint models/action_classifier_holdout.pt "
                         "(made by `python train.py --holdout 2 --out ...`) or the default cross-validation.")
    return [(RAW_DIR / p, ckpt["labels"].index(Path(p).parts[0])) for p in ckpt["test_clips"]]


def confusion(targets, preds, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)  # rows = actual, columns = predicted
    np.add.at(cm, (targets, preds), 1)
    return cm


def report(cm: np.ndarray, clip_acc: float, labels: list[str], protocol: str, tag: str, robust: dict) -> None:
    tp = cm.diagonal().astype(float)
    precision, recall = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    support, n = cm.sum(1), cm.sum()
    den = np.sqrt((n ** 2 - (cm.sum(0) ** 2).sum()) * (n ** 2 - (support ** 2).sum()))
    mcc = (cm.trace() * n - (cm.sum(0) * support).sum()) / den if den else 0.0
    rows = [[name, p, r, f, s] for name, p, r, f, s in zip(labels, precision, recall, f1, support)]
    rows += [["macro avg", precision.mean(), recall.mean(), f1.mean(), n],
             ["weighted avg", *(np.average(m, weights=support) for m in (precision, recall, f1)), n]]

    w = max(map(len, labels)) + 4
    print(f"\nMODULE 1 - {protocol} (per window)\n  {'class':<{w}}{'prec':>7}{'rec':>7}{'F1':>7}{'windows':>9}")
    for name, p, r, f, s in rows:
        print(f"  {name:<{w}}{p:>7.3f}{r:>7.3f}{f:>7.3f}{s:>9}")
    print(f"  window accuracy {100 * cm.trace() / n:.2f}%  |  clip accuracy {100 * clip_acc:.2f}%  |  MCC {mcc:.4f}")
    for cond, res in robust.items():
        print(f"  live/{cond:<13} frame accuracy {100 * res['frame_acc']:5.1f}%  trial accuracy "
              f"{100 * res['trial_acc']:5.1f}%  ({res['trials']} trials)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"module1_metrics_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerows([["protocol", protocol], ["class", "precision", "recall", "f1", "support_windows"],
                       *[[name, f"{p:.4f}", f"{r:.4f}", f"{q:.4f}", s] for name, p, r, q, s in rows],
                       ["window_accuracy", f"{cm.trace() / n:.4f}"], ["clip_accuracy", f"{clip_acc:.4f}"],
                       ["mcc", f"{mcc:.4f}"]])
        if robust:
            out.writerow(["condition", "frame_accuracy", "trial_accuracy", "trials"])
            out.writerows([c, f"{r['frame_acc']:.4f}", f"{r['trial_acc']:.4f}", r["trials"]] for c, r in robust.items())
    print("Saved:", OUTPUT_DIR / f"module1_metrics_{tag}.csv")
    print("Saved:", bar_chart(labels, {"F1": f1}, f"F1-score per class ({protocol})", "F1-score",
                              OUTPUT_DIR / f"module1_f1_{tag}.png", xlim=(0, 1.1)))
    print("Saved:", confusion_heatmap({f"Confusion matrix ({protocol})": cm}, labels,
                                      OUTPUT_DIR / f"module1_confusion_{tag}.png"))


def compare(old_path: Path, new_path: Path) -> None:
    (old, c_old), (new, c_new) = load_checkpoint(old_path), load_checkpoint(new_path)
    labels = c_new["labels"]
    if c_old["labels"] != labels:
        raise SystemExit(f"The two models know different actions: {c_old['labels']} vs {labels}.")
    clips = heldout_clips(c_new)
    cms = {name: confusion(*predict(m, clips)[:2], len(labels)) for name, m in (("Old model", old), ("New model", new))}
    for name, cm in cms.items():
        print(f"{name}: window accuracy {100 * cm.trace() / cm.sum():.1f}% on {len(clips)} held-out clips")
    out = DATA_DIR / "model_comparison"
    print("Saved:", confusion_heatmap(cms, labels, out / "confusion_comparison.png"))
    print("Saved:", bar_chart(labels, {k: cm.diagonal() / np.maximum(cm.sum(1), 1) for k, cm in cms.items()},
                              "Per-class accuracy on held-out clips: old vs new model", "Accuracy",
                              out / "accuracy_bars.png", fmt="{:.0%}", xlim=(0, 1.15)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the action classifier on unseen clips")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per fold")
    parser.add_argument("--holdout", action="store_true", help="Score the saved model on its held-out clips")
    parser.add_argument("--compare", type=Path, metavar="OLD.pt", help="Compare an older checkpoint to the saved one")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Model for --holdout / --compare")
    parser.add_argument("--include-robustness", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare(args.compare, args.checkpoint)

    if args.holdout:
        model, ckpt = load_checkpoint(args.checkpoint)
        (wt, wp, ct, cp), labels, protocol = predict(model, heldout_clips(ckpt)), ckpt["labels"], "held-out clips"
    else:
        clips, labels = list_clips()
        parts = []
        for i, (train, test) in enumerate(kfold_clips(clips, args.folds), 1):
            print(f"fold {i}/{args.folds}: training on {len(train)} clips, testing on {len(test)}")
            parts.append(predict(fit(train, len(labels), epochs=args.epochs, seed=i, verbose=False), test))
        (wt, wp, ct, cp), protocol = [np.concatenate(p) for p in zip(*parts)], f"{args.folds}-fold cross-validation"
    robust = load_robustness_results() if args.include_robustness else {}
    report(confusion(wt, wp, len(labels)), float((ct == cp).mean()), labels, protocol,
           "holdout" if args.holdout else "cv", robust)


if __name__ == "__main__":
    main()
