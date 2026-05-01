#!/usr/bin/env bash
# =============================================================================
# convert_bf16.sh — BF16 variant (no calibration needed, higher accuracy)
#
# Use when INT8 quantization degrades policy accuracy too much.
# BF16 is also supported on CV181x but uses more bandwidth.
# =============================================================================
set -euo pipefail

WORKSPACE="/workspace"
MODEL_NAME="gru_policy"
ONNX="${WORKSPACE}/export/${MODEL_NAME}.onnx"
MLIR="${MODEL_NAME}.mlir"
CVIMODEL="${MODEL_NAME}_bf16.cvimodel"
CHIP="cv181x"

echo "Step 1: ONNX -> MLIR"
model_transform.py \
  --model_name  "${MODEL_NAME}" \
  --model_def   "${ONNX}" \
  --input_shapes "[[1,10,8],[1,1,64]]" \
  --mlir        "${MLIR}"

echo "Step 2: MLIR -> cvimodel (BF16, no calibration required)"
model_deploy.py \
  --mlir     "${MLIR}" \
  --quantize BF16 \
  --chip     "${CHIP}" \
  --model    "${CVIMODEL}"

echo "Done! Output: ${CVIMODEL}"
