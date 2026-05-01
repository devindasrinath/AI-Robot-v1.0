#!/usr/bin/env bash
# =============================================================================
# convert.sh — Convert wake_word DS-CNN ONNX -> cvimodel for SG2002 NPU
#
# Run this INSIDE the Sophgo tpu-mlir Docker container:
#
#   docker run -it --rm \
#     -v $(pwd)/..:/workspace \
#     sophgo/tpuc_dev:latest \
#     bash /workspace/convert/convert.sh
#
# Output: wake_word.cvimodel
# =============================================================================
set -euo pipefail

echo "============================================================"
echo " Installing tpu-mlir"
echo "============================================================"
# Install from local wheel if available, otherwise fall back to PyPI
if [ -f /wheels/tpu_mlir-1.27-py3-none-any.whl ]; then
    pip install /wheels/tpu_mlir-1.27-py3-none-any.whl -q --ignore-requires-python
else
    pip install tpu_mlir -q --timeout 300 --retries 5
fi

WORKSPACE="/workspace"
MODEL_NAME="wake_word"
ONNX="${WORKSPACE}/export/${MODEL_NAME}.onnx"
MLIR="${MODEL_NAME}.mlir"
CALI_TABLE="${MODEL_NAME}_cali_table"
CVIMODEL="${WORKSPACE}/${MODEL_NAME}.cvimodel"
CALIB_DATA="${WORKSPACE}/calibration_data/samples"
CHIP="cv181x"
QUANT="INT8"

echo "============================================================"
echo " Step 1: ONNX -> MLIR"
echo " Input: [1,1,40,98]  (1ch, 40 mel bins, 98 frames ~1s)"
echo "============================================================"
model_transform.py \
  --model_name  "${MODEL_NAME}" \
  --model_def   "${ONNX}" \
  --input_shapes "[[1,1,40,98]]" \
  --mlir        "${MLIR}"

echo ""
echo "============================================================"
echo " Step 2: Calibration"
echo "============================================================"
run_calibration.py "${MLIR}" \
  --dataset    "${CALIB_DATA}" \
  --input_num  100 \
  -o           "${CALI_TABLE}"

echo ""
echo "============================================================"
echo " Step 3: MLIR -> cvimodel INT8"
echo "============================================================"
model_deploy.py \
  --mlir              "${MLIR}" \
  --quantize          "${QUANT}" \
  --calibration_table "${CALI_TABLE}" \
  --chip              "${CHIP}" \
  --model             "${CVIMODEL}"

echo ""
echo "============================================================"
echo " Step 4: GenericCpu check (quant/dequant only = OK)"
echo "============================================================"
grep "GenericCpu" "${MODEL_NAME}_cv181x_int8_sym_tpu.mlir" \
  && echo "WARNING: CPU ops found — check names above" \
  || echo "OK: no unexpected CPU ops"

echo ""
echo "Done! Output: ${CVIMODEL}"
