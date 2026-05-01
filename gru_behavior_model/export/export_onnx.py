"""
Export NexuGRUPolicy to ONNX (opset 13, static shapes) for tpu-mlir.

Run: python gru_behavior_model/export/export_onnx.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import torch
import onnx
import onnxruntime as ort
import numpy as np
from gru_policy import NexuGRUPolicy, OBS_SIZE, HIDDEN_SIZE, NUM_ACTIONS

ONNX_PATH = os.path.join(os.path.dirname(__file__), 'gru_behavior.onnx')


def export():
    model = NexuGRUPolicy()

    weights_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gru_behavior.pth')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print("No weights found — exporting architecture only (random weights)")

    model.eval()
    obs_dummy    = torch.randn(1, 1, OBS_SIZE)
    hidden_dummy = torch.zeros(1, HIDDEN_SIZE)

    torch.onnx.export(
        model,
        (obs_dummy, hidden_dummy),
        ONNX_PATH,
        opset_version=13,
        input_names=["obs", "hidden_in"],
        output_names=["action_logits", "hidden_out"],
        dynamic_axes=None,   # static shapes required for tpu-mlir
        dynamo=False,        # force legacy exporter — new exporter can't produce opset 13
    )
    print(f"Exported: {ONNX_PATH}")

    # Verify
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    ops = set(n.op_type for n in onnx_model.graph.node)
    print(f"Opset: {onnx_model.opset_import[0].version}  Ops: {ops}")
    assert 'LSTM' not in ops and 'GRU' not in ops, "GRU/LSTM ops found — not supported by tpu-mlir!"
    print("No GRU/LSTM ops — safe for tpu-mlir")

    # ORT inference check
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    out = sess.run(None, {
        "obs":       obs_dummy.numpy(),
        "hidden_in": hidden_dummy.numpy(),
    })
    print(f"ORT action_logits shape: {out[0].shape}  hidden_out shape: {out[1].shape}")
    assert out[0].shape == (1, NUM_ACTIONS), f"Bad logits shape: {out[0].shape}"
    assert out[1].shape == (1, HIDDEN_SIZE), f"Bad hidden shape: {out[1].shape}"
    print("Export and verification OK")


if __name__ == "__main__":
    export()
