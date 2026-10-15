"""
Module 1 - Action Recognition - Step 5: export the trained model to ONNX.

The C++/Qt app loads the .onnx file with ONNX Runtime's C++ API and never
needs PyTorch. The action names are saved next to it (.labels.txt), because
an ONNX graph only knows about tensors.

Usage:
    python export_onnx.py
"""

import numpy as np
import onnxruntime as ort
import torch

from common.pose import LABELS_PATH, NUM_FEATURES, ONNX_PATH, SEQUENCE_LENGTH
from dataset import SkeletonWindowDataset, list_clips
from model import load_checkpoint
from train import CHECKPOINT_PATH


def main() -> None:
    model, ckpt = load_checkpoint(CHECKPOINT_PATH)

    # Batch size stays fixed at 1: the live app classifies one window at a time,
    # and PyTorch's LSTM export misbehaves with a dynamic batch unless h0/c0 are exposed.
    torch.onnx.export(model, torch.zeros(1, SEQUENCE_LENGTH, NUM_FEATURES), str(ONNX_PATH),
                      input_names=["keypoint_sequence"], output_names=["action_logits"],
                      opset_version=17, dynamo=False)
    LABELS_PATH.write_text("\n".join(ckpt["labels"]), encoding="utf-8")
    print(f"Exported {ONNX_PATH} and {LABELS_PATH.name} ({ckpt['labels']})")

    # Verify on real windows: ONNX Runtime must reproduce the PyTorch logits.
    windows = SkeletonWindowDataset(list_clips()[0][:8]).windows
    session = ort.InferenceSession(str(ONNX_PATH))
    diff = 0.0
    for w in windows[::10]:
        x = w.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        diff = max(diff, float(np.abs(expected - session.run(None, {"keypoint_sequence": x})[0]).max()))
    if diff > 1e-4:
        raise RuntimeError(f"Verification FAILED: ONNX and PyTorch outputs differ by {diff}.")
    print(f"Verification passed: max |PyTorch - ONNX| = {diff:.2e}")


if __name__ == "__main__":
    main()
