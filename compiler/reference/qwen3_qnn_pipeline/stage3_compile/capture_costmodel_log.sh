#!/bin/bash
# ===== 抓 qnn-context-binary-generator 的 cost model 日志 =====
# 机器: dev133 (编译机)
# 产出: /tmp/qnn_costmodel_split1.log + 过滤后的关键行
# 耗时: ~3 min (只跑 1 个分片)
#
# 用法: 在 dev133 上执行
#   bash stage3_compile/capture_costmodel_log.sh
#
# 原理:
#   1. 开启 QNN verbose 日志环境变量
#   2. 重跑 split 1 的 context binary 生成 (输出到 /tmp 避免污染正式产物)
#   3. 全量日志写 /tmp/qnn_costmodel_split1.log
#   4. grep 出 cost model / partition / kernel / vtcm 相关行打印
#
# 注意: 此脚本只读 DLC, 不修改正式 .bin 产物 (输出目录用单独的 _debug 目录)

set -e

# ---- 配置 ----
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data01/rqilin/qnn-sdk}"
QWEN3_PROJECT="${QWEN3_PROJECT:-/data01/rqilin/qwen3_llm_v2}"
ARN="${ARN:-128}"
CL="${CONTEXT_LENGTH:-2048}"
NUM_SPLITS="${NUM_SPLITS:-3}"
SPLIT="${SPLIT:-1}"          # 默认抓第 1 片
SOC_ID="${SOC_ID:-43}"       # v73
DSP_ARCH="${DSP_ARCH:-v73}"

ARTIFACTS="$QWEN3_PROJECT/example2/host_linux/assets/artifacts"
DLC_DIR="$ARTIFACTS/ar${ARN}-cl${CL}"
DLC="$DLC_DIR/${SPLIT}_of_${NUM_SPLITS}/compiled_model/ar${ARN}-cl${CL}_${SPLIT}_of_${NUM_SPLITS}_quantized.dlc"

# 输出 (独立目录, 不覆盖正式产物)
OUT_DIR="/tmp/qnn_costmodel_debug"
CONF_DIR="/tmp/qnn_costmodel_conf"
LOG_FILE="/tmp/qnn_costmodel_split${SPLIT}.log"
FILTER_FILE="/tmp/qnn_costmodel_split${SPLIT}_filtered.log"

# ---- 设置 QNN 环境 ----
export QNN_SDK_ROOT
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$QNN_SDK_ROOT/benchmarks/QNN:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export HEXAGON_TOOLS_DIR="$QNN_SDK_ROOT/bin/x86_64-linux-clang"

# ---- 开启 verbose 日志 (QNN SDK 通用 + HTP backend 专用) ----
# QNN_LOG_LEVEL: error|warn|info|debug|verbose
export QNN_LOG_LEVEL="debug"
export QNN_HTP_BACKEND_LOG_LEVEL="5"        # 0=off, 5=verbose
export QNN_HTP_NET_RUN_LOG_LEVEL="5"
export QNN_MODEL_DLC_LOG_LEVEL="5"
export QNN_CONTEXT_BINARY_GENERATOR_LOG_LEVEL="5"
# 部分 SDK 版本用这些
export QNN_VERBOSE="1"
export QNN_SDK_VERBOSE="1"
# 让 HTP op infra 打印 cost/partition/kernel 选择
export QNN_HTP_INFRASTRUCTURE_LOG_LEVEL="5"
export QNN_HTP_PERF_INFRASTRUCTURE_LOG_LEVEL="5"

echo "===== 配置 ====="
echo "DLC:      $DLC"
echo "输出目录: $OUT_DIR"
echo "日志:     $LOG_FILE"
echo "QNN_LOG_LEVEL=$QNN_LOG_LEVEL"
echo ""

# ---- 检查 DLC ----
if [ ! -f "$DLC" ]; then
    echo "[错误] DLC 不存在: $DLC"
    echo "请先运行 stage3_compile/run_example2.sh"
    exit 1
fi
if [ ! -r "$DLC" ]; then
    echo "[错误] DLC 不可读 (root 拥有), 用 dzdo su 运行:"
    echo "  dzdo su -c \"bash $0\""
    exit 1
fi

# ---- 准备配置 (最简 PerfSetting, 与 gen_v73_context_binary.py 一致) ----
mkdir -p "$OUT_DIR" "$CONF_DIR"
GRAPH="ar${ARN}-cl${CL}_${SPLIT}_of_${NUM_SPLITS}"
BIN_NAME="${GRAPH}_v73_debug.serialized"

cat > "$CONF_DIR/htp_${SPLIT}.json" << EOF
{
    "backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so",
        "config_file_path": "$CONF_DIR/perf_${SPLIT}.conf"
    }
}
EOF

cat > "$CONF_DIR/perf_${SPLIT}.conf" << EOF
{
    "devices": [{
        "soc_id": ${SOC_ID},
        "dsp_arch": "${DSP_ARCH}",
        "pd_session": "unsigned"
    }]
}
EOF

# ---- 运行 (stdout+stderr 合并 tee 到日志) ----
# 关键参数:
#   --log_level=verbose              最大日志级别
#   --profiling_level=detailed       per-Op timing
#   --save_backend_op_mapping        生成 op→kernel 映射 chrometrace.json
echo "===== 运行 qnn-context-binary-generator (verbose + detailed profiling) ====="
echo "日志实时写入: $LOG_FILE"
echo ""

qnn-context-binary-generator \
    --backend libQnnHtp.so \
    --model libQnnModelDlc.so \
    --input_output_tensor_mem_type memhandle \
    --output_dir "$OUT_DIR" \
    --config_file "$CONF_DIR/htp_${SPLIT}.json" \
    --binary_file "$BIN_NAME" \
    --dlc_path "$DLC" \
    --log_level verbose \
    --profiling_level detailed \
    --save_backend_op_mapping \
    2>&1 | tee "$LOG_FILE" || true

LOG_LINES=$(wc -l < "$LOG_FILE")
LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
echo ""
echo "===== 日志统计 ====="
echo "总行数: $LOG_LINES"
echo "大小:   $LOG_SIZE"
echo "文件:   $LOG_FILE"

# ---- 过滤 cost model 相关行 ----
echo ""
echo "===== 过滤 cost model 相关行 → $FILTER_FILE ====="

# 关键词 (大小写不敏感)
PATTERN='cost|partition|kernel|vtcm|tiling|tile|fusion|fuse|layout|reorder|data move|dm |op selection|select|perf|plan|schedule|op id|graph opt|optimize'

grep -iE "$PATTERN" "$LOG_FILE" > "$FILTER_FILE" 2>/dev/null || true

FILTER_LINES=$(wc -l < "$FILTER_FILE" 2>/dev/null || echo 0)
echo "过滤后行数: $FILTER_LINES"
echo "文件:       $FILTER_FILE"
echo ""

# ---- 打印关键行预览 (前 80 行) ----
echo "===== 关键行预览 (前 80 行) ====="
if [ "$FILTER_LINES" -gt 0 ]; then
    head -n 80 "$FILTER_FILE"
else
    echo "[未匹配到 cost model 关键词]"
    echo "可能 verbose 没生效。尝试以下排错:"
    echo "  1. 确认 QNN SDK 版本支持这些 LOG_LEVEL 变量"
    echo "  2. 查看完整日志: less $LOG_FILE"
    echo "  3. 试加 --debug_flag 或 --verbose 参数 (见 qnn-context-binary-generator --help)"
    echo ""
    echo "--- 日志前 40 行 (看是否真有 debug 输出) ---"
    head -n 40 "$LOG_FILE"
fi

echo ""
echo "===== 完成 ====="
echo "完整日志:   $LOG_FILE"
echo "过滤后日志: $FILTER_FILE"
echo "下一步在 Windows 拉回查看:"
echo "  pscp -pw <pass> rqilin@<dev133>:$LOG_FILE ."
echo "  pscp -pw <pass> rqilin@<dev133>:$FILTER_FILE ."
