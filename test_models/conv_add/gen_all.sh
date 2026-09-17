#!/bin/bash
# gen_all.sh — conv_add 模型全链生成(阶段 1 一键)
# 产物: conv_add.onnx / net.json(dlc-to-json) / conv_add_repaired.dlc / 双 golden
# 依赖(只读): venv310、qairt SDK 2.48(/disk2/QCtools/qairt_2.48.40.260702)
set -euo pipefail
cd "$(dirname "$0")"
SDK=/disk2/QCtools/qairt_2.48.40.260702
PY=/disk2/Qwen35dev/revlibHtpPrepare/venv310/bin/python3
export PYTHONPATH=$SDK/lib/python

echo "[1/4] ONNX 模型 + ORT/fp16 参考 golden"
$PY gen_conv_add.py

echo "[2/4] converter → DLC zip(CRC 全零,官方 quirk)"
mkdir -p conv_add_qnn
$PY "$SDK/bin/x86_64-linux-clang/qairt-converter" \
  -i conv_add.onnx --float_bitwidth 16 -o conv_add_qnn/conv_add.cpp 2>&1 | tail -1

echo "[3/4] zip CRC 修复 → 可执行 DLC"
../../scripts/dlc_repair.sh conv_add_qnn/conv_add.cpp conv_add_qnn/conv_add_repaired.dlc

echo "[4/4] dlc-to-json → net.json(B线摄取格式)"
$PY "$SDK/bin/x86_64-linux-clang/qairt-dlc-to-json" \
  -i conv_add_qnn/conv_add_repaired.dlc -o conv_add_qnn/conv_add_net.json 2>&1 | tail -1

echo "DONE: conv_add.onnx / conv_add_qnn/conv_add_net.json / conv_add_qnn/conv_add_repaired.dlc"
echo "golden: Z_ort.f32.raw (ORT fp32) + Z_ref_f16.f32.raw (fp16 纪律 host 参考)"
echo "note: QNN-CPU 后端对 Transpose OpConfig 校验失败(quirk),QNN 工具链金标走 HTP off-target(阶段 9)"
