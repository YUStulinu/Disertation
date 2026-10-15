"""
Module 1 - Action Recognition: temporal classifier over skeleton windows.

Input (batch, 30, 66): x, y of the 33 landmarks for each time sample. The model
appends every landmark's frame-to-frame velocity (motion, not only pose: falling
and sitting down differ mostly in speed), runs a recurrent layer over the 30
steps, averages its outputs over time and classifies. The velocity is computed
inside the network, so the ONNX input stays (1, 30, 66) for the C++ app.
"""

from pathlib import Path

import torch
import torch.nn as nn

MODEL_CONFIG = {"cell": "lstm", "hidden_size": 64, "bidirectional": False, "velocity": True,
                "pooling": "mean", "dropout": 0.0}
LEGACY_CONFIG = {"velocity": False, "pooling": "last"}  # checkpoints saved before the config was stored


class SkeletonClassifier(nn.Module):
    def __init__(self, num_classes: int, cell: str = "lstm", hidden_size: int = 64, bidirectional: bool = False,
                 velocity: bool = True, pooling: str = "mean", dropout: float = 0.0):
        super().__init__()
        self.velocity, self.pooling = velocity, pooling
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](66 * (2 if velocity else 1), hidden_size,
                                                          batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.velocity:  # x[t] - x[t-1] (0 for the first step), scaled to the range of the positions
            x = torch.cat([x, 5 * (x - torch.cat([x[:, :1], x[:, :-1]], dim=1))], dim=-1)
        out, _ = self.rnn(x)
        return self.classifier(self.dropout(out.mean(1) if self.pooling == "mean" else out[:, -1]))


def load_checkpoint(path: Path) -> tuple[SkeletonClassifier, dict]:
    """Checkpoint saved by train.py -> (model in eval mode, checkpoint dict)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Run train.py first.")
    ckpt = torch.load(path, weights_only=False)
    model = SkeletonClassifier(len(ckpt["labels"]), **ckpt.get("config", LEGACY_CONFIG))
    model.load_state_dict({k.replace("lstm.", "rnn.", 1): v for k, v in ckpt["model_state"].items()})
    return model.eval(), ckpt
