#!/bin/bash
# ===== Stage 2: PyTorch → ONNX 导出 (Example1) =====
# 机器: dev133 (编译机, 需 GPU)
# 产出: ONNX + encodings + tokenizer + test_vectors
# 耗时: ~30 min
#
# 用法: 在 dev133 上执行
#   bash stage2_export/run_example1.sh
#
# 前置:
#   - Stage 1 已完成 (HF 模型已下载)
#   - Python 环境: aimet-torch 2.16, transformers 4.53, torch 2.4+cu121, onnx 1.18
#   - GPU (CUDA) 用于量化校准
#   - QNN SDK 已安装

set -e

# ---- 配置 ----
QWEN3_PROJECT="${QWEN3_PROJECT:-/data01/rqilin/qwen3_llm_v2}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data01/rqilin/qnn-sdk}"
MODEL_ID="${MODEL_ID:-/data01/rqilin/models/Qwen3-4B-Base}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
ARN="${ARN:-1023}"  # Activation Retention Number (导出时用全 1023)

echo "===== Stage 2: PyTorch → ONNX 导出 ====="
echo "工程目录: $QWEN3_PROJECT"
echo "模型: $MODEL_ID"
echo "Context Length: $CONTEXT_LENGTH"
echo "ARN (导出): $ARN"

# ---- 设置环境 ----
export QNN_SDK_ROOT
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$QNN_SDK_ROOT/benchmarks/QNN:$PYTHONPATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

# ---- 修改配置文件 ----
CONFIG="$QWEN3_PROJECT/qwen3_local_config.json"
echo "修改配置: $CONFIG"

# 备份原配置
cp "$CONFIG" "${CONFIG}.bak"

# 用 Python 修改 JSON 配置
python3 << PYEOF
import json

config_path = "$CONFIG"
model_id = "$MODEL_ID"
cl = $CONTEXT_LENGTH
arn = $ARN
sdk = "$QNN_SDK_ROOT"

with open(config_path) as f:
    cfg = json.load(f)

# example1 配置
cfg["example1"]["MODEL_ID"] = model_id
cfg["example1"]["CONTEXT_LENGTH"] = cl
cfg["example1"]["ARN"] = arn
cfg["example1"]["QNN_SDK_ROOT"] = sdk
cfg["example1"]["MODEL_NAME"] = "qwen3"

# example2 配置 (保持一致)
cfg["example2"]["CL"] = cl
cfg["example2"]["EXPORT_AR"] = arn
cfg["example2"]["EXPORT_CONTEXT_LENGTH"] = cl
cfg["example2"]["onnx_name"] = "qwen3_base"

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"配置已更新: MODEL_ID={model_id}, CL={cl}, ARN={arn}")
PYEOF

# ---- 运行 Example1 (qwen3.py) ----
echo ""
echo "===== 运行 Example1: qwen3.py ====="
cd "$QWEN3_PROJECT/example1"

# Example1 完成: 加载 HF 模型 → AIMET 量化 → 导出 ONNX
python3 qwen3.py

# ---- 验证输出 ----
OUTPUT_DIR="$QWEN3_PROJECT/example1/output_dir_"
echo ""
echo "===== 验证输出 ====="
echo "ONNX 文件:"
ls -lh "$OUTPUT_DIR/onnx/"*.onnx 2>/dev/null || echo "  未找到 ONNX"
echo "Encodings:"
ls -lh "$OUTPUT_DIR/onnx/"*.encodings 2>/dev/null || echo "  未找到 encodings"
echo "Tokenizer:"
ls "$OUTPUT_DIR/tokenizer/" 2>/dev/null || echo "  未找到 tokenizer"

echo ""
echo "===== Stage 2 完成 ====="
echo "ONNX 已导出到: $OUTPUT_DIR"
echo ""
echo "下一步: bash stage3_compile/run_example2.sh"
