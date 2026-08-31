#!/bin/bash
# ===== Stage 5: 板子上推理 =====
# 机器: Android 板子 (via adb)
# 产出: 文本生成
#
# 用法: 在本地 Windows 执行 (通过 adb shell)
#   bash stage5_infer/run_inference.sh
#
# 前置:
#   - Stage 4 已完成 (.bin + 配置已在板子上)
#   - QAIRT 已安装在板子上 (/data/qairt)
#   - env.sh 已配置 v73 库路径

set -e

ADB="${ADB_PATH:-adb}"
BOARD_MODEL_DIR="${BOARD_MODEL_DIR:-/data/qairt/qwen3}"
CONFIG="${BOARD_MODEL_DIR}/genie_qwen3_v73.json"
PROMPT="${1:-Hello, how are you?}"

echo "===== Stage 5: 推理 ====="
echo "配置: $CONFIG"
echo "Prompt: $PROMPT"
echo ""

# ---- 运行 genie-t2t-run ----
"$ADB" shell ". /data/qairt/env.sh && cd $BOARD_MODEL_DIR && genie-t2t-run -c genie_qwen3_v73.json -p '$PROMPT' 2>&1"

echo ""
echo "===== 推理完成 ====="
