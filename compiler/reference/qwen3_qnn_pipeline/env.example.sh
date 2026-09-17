#!/bin/bash
# ===== 环境变量配置 (复制此文件为 env.sh 并修改) =====

# dev133 连接
export DEV133_HOST="10.137.185.133"
export DEV133_USER="rqilin"
export DEV133_PASS="Wangba521."
export DEV133_FP="SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"

# dev133 路径
export QNN_SDK_ROOT="/data01/rqilin/qnn-sdk"
export QWEN3_PROJECT="/data01/rqilin/qwen3_llm_v2"
export MODEL_ID="/data01/rqilin/models/Qwen3-4B-Base"
export HF_MODEL_NAME="Qwen/Qwen3-4B-Base"

# 板子路径
export BOARD_MODEL_DIR="/data/qairt/qwen3"
export BOARD_BIN_DIR="/data/qairt/qwen3/serialized_v73"

# 本地工具路径 (Windows)
export ADB_PATH="C:/Users/RQILIN/AppData/Local/Programs/platform-tools/adb.exe"
export PLINK_PATH="C:/Users/RQILIN/AppData/Local/Temp/opencode/plink.exe"
export PSCP_PATH="C:/Users/RQILIN/AppData/Local/Temp/opencode/pscp.exe"

# 模型参数
export CONTEXT_LENGTH=2048
export ARN=128
export NUM_SPLITS=3
export SOC_ID=43        # v73
export DSP_ARCH="v73"
