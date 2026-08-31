#!/bin/bash
# ===== Stage 1: 从 HuggingFace 下载模型 =====
# 机器: dev133 (编译机)
# 产出: safetensors 模型文件
# 耗时: ~10 min (取决于网络)
#
# 用法: 在 dev133 上执行
#   bash stage1_download/download_model.sh
#
# 前置: huggingface-cli 已安装 (pip install huggingface_hub)

set -e

# ---- 配置 ----
HF_MODEL_NAME="${HF_MODEL_NAME:-Qwen/Qwen3-4B-Base}"
MODEL_DIR="${MODEL_ID:-/data01/rqilin/models/Qwen3-4B-Base}"

echo "===== Stage 1: 下载模型 ====="
echo "模型: $HF_MODEL_NAME"
echo "目标: $MODEL_DIR"

# ---- 安装 huggingface_hub (如未安装) ----
if ! command -v huggingface-cli &>/dev/null; then
    echo "安装 huggingface_hub..."
    pip install huggingface_hub
fi

# ---- 下载 ----
mkdir -p "$MODEL_DIR"

# 方式1: huggingface-cli download (推荐)
huggingface-cli download "$HF_MODEL_NAME" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

# 方式2: 如果需要指定文件 (大模型只下载需要的)
# huggingface-cli download "$HF_MODEL_NAME" \
#     config.json tokenizer.json tokenizer_config.json \
#     model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
#     model.safetensors.index.json \
#     --local-dir "$MODEL_DIR"

# ---- 验证 ----
echo "===== 验证下载 ====="
ls -lh "$MODEL_DIR/"

echo ""
echo "===== Stage 1 完成 ====="
echo "模型已下载到: $MODEL_DIR"
echo ""
echo "下一步: bash stage2_export/run_example1.sh"
