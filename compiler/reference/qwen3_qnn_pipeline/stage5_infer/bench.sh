#!/bin/bash
# ===== Stage 5: 性能基准测试 =====
# 机器: Android 板子 (via adb)
# 产出: tok/s 速度
#
# 用法:
#   bash stage5_infer/bench.sh
#
# 测试方法: 运行 30s,统计生成字符数,估算 tok/s

ADB="${ADB_PATH:-adb}"
BOARD_MODEL_DIR="${BOARD_MODEL_DIR:-/data/qairt/qwen3}"
DURATION="${1:-30}"  # 测试时长(秒)

echo "===== 性能基准 (运行 ${DURATION}s) ====="

# 杀掉旧进程,等待 DSP 释放
$ADB shell "pkill -9 genie-t2t-run 2>/dev/null; sleep 5"

# 运行并计时
$ADB shell ". /data/qairt/env.sh && cd $BOARD_MODEL_DIR && \
    timeout $DURATION genie-t2t-run -c genie_qwen3_v73.json \
    -p 'Machine learning is a subset of artificial intelligence' \
    > /tmp/bench_out.txt 2>/dev/null; \
    echo 退出码=\$?; \
    GEN=\$(grep -o '\[BEGIN\].*' /tmp/bench_out.txt | sed 's/\[BEGIN\]: //'); \
    NCHARS=\$(echo \"\$GEN\" | wc -c); \
    NWORDS=\$(echo \"\$GEN\" | wc -w); \
    echo 生成: \$NCHARS 字符, \$NWORDS 词; \
    echo 速度: \$((NWORDS * 1)) words / ${DURATION}s = ~\$((NWORDS * 130 / 100 / DURATION)) tok/s; \
    echo '---内容---'; \
    echo \"\$GEN\" | head -c 300"

# 清理
$ADB shell "pkill -9 genie-t2t-run 2>/dev/null"
