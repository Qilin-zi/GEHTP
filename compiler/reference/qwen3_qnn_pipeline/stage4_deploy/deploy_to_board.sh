#!/bin/bash
# ===== Stage 4: 部署到板子 =====
# 机器: 本地 Windows (有 adb + plink)
# 产出: 板子上的 .bin + GenieX 配置 + tokenizer
# 耗时: ~5 min (adb push 34 MB/s)
#
# 用法: 在本地 Windows (PowerShell 或 Git Bash) 执行
#   bash stage4_deploy/deploy_to_board.sh
#
# 前置:
#   - Stage 3b 已完成 (v73 .bin 已生成在 dev133)
#   - adb 可连接板子
#   - plink/pscp 可连接 dev133

set -e

# ---- 配置 (从 env.sh 读取) ----
DEV133_HOST="${DEV133_HOST:-10.137.185.133}"
DEV133_USER="${DEV133_USER:-rqilin}"
DEV133_PASS="${DEV133_PASS:-Wangba521.}"
DEV133_FP="${DEV133_FP:-SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c}"

QWEN3_PROJECT="${QWEN3_PROJECT:-/data01/rqilin/qwen3_llm_v2}"
DEV133_BIN_DIR="$QWEN3_PROJECT/example2/host_linux/assets/artifacts/ar128-cl2048_v73"

BOARD_MODEL_DIR="${BOARD_MODEL_DIR:-/data/qairt/qwen3}"
BOARD_BIN_DIR="${BOARD_BIN_DIR:-/data/qairt/qwen3/serialized_v73}"

# 本地临时目录
LOCAL_TMP="${LOCAL_TMP:-/tmp/qwen3_transfer}"
mkdir -p "$LOCAL_TMP"

# 工具路径
ADB="${ADB_PATH:-adb}"
PLINK="${PLINK_PATH:-plink}"
PSCP="${PSCP_PATH:-pscp}"

NUM_SPLITS=3

echo "===== Stage 4: 部署到板子 ====="
echo "dev133: $DEV133_USER@$DEV133_HOST"
echo "板子目录: $BOARD_BIN_DIR"

# ---- Step 1: 从 dev133 下载 .bin 到本地 ----
echo ""
echo "===== Step 1: 从 dev133 下载 .bin ====="

# 先清理板子上的旧文件
$ADB shell "rm -f $BOARD_BIN_DIR/*.bin $BOARD_BIN_DIR/*.bin.bin 2>/dev/null" || true

for i in $(seq 1 $NUM_SPLITS); do
    BIN_FILE="ar128-cl2048_${i}_of_${NUM_SPLITS}_v73.serialized.bin"
    REMOTE_PATH="$DEV133_BIN_DIR/$BIN_FILE"
    LOCAL_PATH="$LOCAL_TMP/${i}_of_${NUM_SPLITS}_v73.bin"

    echo "  下载 ${i}/${NUM_SPLITS}: $BIN_FILE"

    # 用 plink cat 管道方式 (比 pscp 快 3 倍)
    "$PLINK" -ssh -batch -pw "$DEV133_PASS" -hostkey "$DEV133_FP" \
        "$DEV133_USER@$DEV133_HOST" "cat $REMOTE_PATH" > "$LOCAL_PATH"

    SIZE=$(stat -c%s "$LOCAL_PATH" 2>/dev/null || wc -c < "$LOCAL_PATH")
    echo "    本地: $SIZE bytes ($((SIZE / 1024 / 1024)) MB)"
done

# ---- Step 2: adb push 到板子 ----
echo ""
echo "===== Step 2: adb push 到板子 ====="

# 确保 adb root + 连接
"$ADB" root 2>/dev/null || true
sleep 2
"$ADB" wait-for-device

# 创建目录
"$ADB" shell "mkdir -p $BOARD_BIN_DIR"

for i in $(seq 1 $NUM_SPLITS); do
    LOCAL_PATH="$LOCAL_TMP/${i}_of_${NUM_SPLITS}_v73.bin"
    echo "  push ${i}/${NUM_SPLITS}..."
    "$ADB" push "$LOCAL_PATH" "$BOARD_BIN_DIR/"
done

# ---- Step 3: 推送 GenieX 配置 + tokenizer ----
echo ""
echo "===== Step 3: 推送配置文件 ====="

# GenieX 配置
"$ADB" push "stage4_deploy/genie_qwen3_v73.json" "$BOARD_MODEL_DIR/"

# Tokenizer (如果板子上没有)
"$ADB" shell "ls $BOARD_MODEL_DIR/tokenizer.json 2>/dev/null" || \
    "$ADB" push "stage4_deploy/tokenizer.json" "$BOARD_MODEL_DIR/" || \
    echo "  [警告] tokenizer.json 未找到,请手动推送"

# ---- Step 4: 验证 ----
echo ""
echo "===== Step 4: 验证 ====="
echo "板子上的 .bin 文件:"
"$ADB" shell "ls -lh $BOARD_BIN_DIR/"
echo ""
echo "板子上的配置文件:"
"$ADB" shell "ls -lh $BOARD_MODEL_DIR/genie_qwen3_v73.json $BOARD_MODEL_DIR/tokenizer.json 2>/dev/null"

# ---- 清理本地临时文件 ----
echo ""
echo "===== 清理本地临时文件 ====="
rm -f "$LOCAL_TMP"/*_v73.bin
rmdir "$LOCAL_TMP" 2>/dev/null || true

echo ""
echo "===== Stage 4 完成 ====="
echo "板子上已就绪: $BOARD_BIN_DIR/"
echo ""
echo "下一步: bash stage5_infer/run_inference.sh"
