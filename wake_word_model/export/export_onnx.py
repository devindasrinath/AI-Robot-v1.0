"""
Export WakeWordCNN to ONNX and verify with onnxruntime.
Run from project root: python wake_word_model/export/export_onnx.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import torch
import onnx
import onnxruntime as ort
import numpy as np
from wake_word_cnn import WakeWordCNN

ONNX_PATH = os.path.join(os.path.dirname(__file__), 'wake_word.onnx')
INPUT_SHAPE = (1, 1, 40, 98)

def export():
    model = WakeWordCNN()

    # Load trained weights if available
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'wake_word.pth')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print("No weights found — exporting with random weights (architecture check only)")

    model.eval()
    dummy = torch.randn(*INPUT_SHAPE)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        opset_version=13,
        input_names=["mel_input"],
        output_names=["logits"],
        dynamic_axes=None,   # static shapes required for tpu-mlir
        dynamo=False,        # force legacy exporter — new exporter can't produce opset 13
    )
    print(f"Exported: {ONNX_PATH}")

    # Verify ONNX model is valid
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX check passed")

    # Verify inference via onnxruntime
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    out = sess.run(None, {"mel_input": dummy.numpy()})
    print(f"ORT output shape: {out[0].shape}  values: {out[0]}")
    assert out[0].shape == (1, 2), f"Unexpected output shape: {out[0].shape}"
    print("Export and verification OK")

if __name__ == "__main__":
    export()
