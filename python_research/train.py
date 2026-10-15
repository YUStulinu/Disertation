"""
Module 1 - Week 2: train the action classifier on skeleton windows.

By default every recorded clip is used - this is the model the live app runs;
its accuracy is measured separately, on unseen clips, by evaluate.py
(cross-validation). --holdout N keeps N clips per action aside (their paths are
stored in the checkpoint) for `evaluate.py --holdout` / `--compare`.

Usage:
    python train.py [--holdout 2] [--epochs 25]
    python export_onnx.py          # then: the C++ app / verify_pipeline.py use the new model
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.paths import MODELS_DIR, RAW_DIR
from dataset import SkeletonWindowDataset, list_clips, split_clips
from model import MODEL_CONFIG, SkeletonClassifier

CHECKPOINT_PATH = MODELS_DIR / "action_classifier.pt"


def fit(train_clips, num_classes: int, epochs: int = 25, lr: float = 2e-3, seed: int = 42,
        test_clips=None, verbose: bool = True) -> SkeletonClassifier:
    """Train a fresh model on the clips' windows (time-warped + augmented); optionally report test accuracy."""
    torch.manual_seed(seed)
    loader = DataLoader(SkeletonWindowDataset(train_clips, train=True, seed=seed), batch_size=32, shuffle=True)
    test = SkeletonWindowDataset(test_clips) if test_clips else None
    model = SkeletonClassifier(num_classes, **MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            out = model(x)
            loss = nn.functional.cross_entropy(out, y, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += len(y)
            correct += (out.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
        if verbose:
            msg = f"epoch {epoch:3d}/{epochs}  loss={loss_sum / total:.3f}  train_acc={correct / total:.2f}"
            if test:
                model.eval()
                with torch.no_grad():
                    pred = model(torch.from_numpy(test.windows.reshape(len(test), test.windows.shape[1], -1))).argmax(1)
                msg += f"  test_acc={(pred.numpy() == test.targets).mean():.2f}"
            print(msg)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the action classifier")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--holdout", type=int, default=0, metavar="N", help="Keep N clips per action aside for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    clips, labels = list_clips()
    train, test = split_clips(clips, args.holdout, args.seed) if args.holdout else (clips, [])
    print(f"{len(clips)} clips of {labels} -> {len(train)} for training, {len(test)} held out")
    model = fit(train, len(labels), args.epochs, seed=args.seed, test_clips=test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "labels": labels, "config": MODEL_CONFIG,
                "test_clips": [p.relative_to(RAW_DIR).as_posix() for p, _ in test]}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
