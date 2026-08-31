#!/bin/bash
# ===== Stage 3a: ONNX → DLC (Example2: qnn_model_prepare.py) =====
# 机器: dev133 (编译机)
# 产出: 量化后的 DLC 文件 (ar128-cl2048, 3 个分片)
# 耗时: ~60 min
#
# 用法: 在 dev133 上执行
#   bash stage3_compile/run_example2.sh
#
# 注意: 此脚本只跑到 DLC 生成步骤。context binary 生成步骤
#       会因 PerfSetting 配置问题失败(已知问题),DLC 已成功生成。
#       v73 context binary 用 gen_v73_context_binary.py 独立生成。
#
# 前置:
#   - Stage 2 已完成 (ONNX 已导出)
#   - QNN SDK 工具链: qairt-converter, qairt-quantizer, qnn-context-binary-generator
#   - dzdo su root 权限 (DLC 文件会被 root 拥有)

set -e

# ---- 配置 ----
QWEN3_PROJECT="${QWEN3_PROJECT:-/data01/rqilin/qwen3_llm_v2}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data01/rqilin/qnn-sdk}"
CL="${CONTEXT_LENGTH:-2048}"
ARN="${ARN:-128}"
NUM_SPLITS="${NUM_SPLITS:-3}"

echo "===== Stage 3a: ONNX → DLC ====="
echo "工程: $QWEN3_PROJECT"
echo "CL=$CL, ARN=$ARN, splits=$NUM_SPLITS"

# ---- 设置环境 ----
export QNN_SDK_ROOT
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$QNN_SDK_ROOT/benchmarks/QNN:$PYTHONPATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export HEXAGON_TOOLS_DIR="$QNN_SDK_ROOT/bin/x86_64-linux-clang"

# ---- 修改 example2 配置 (v73 参数) ----
CONFIG="$QWEN3_PROJECT/qwen3_local_config.json"
python3 << PYEOF
import json

config_path = "$CONFIG"
with open(config_path) as f:
    cfg = json.load(f)

# example2: 转换/编译参数
cfg["example2"]["CL"] = $CL
cfg["example2"]["ARNs"] = [$ARN]       # 单 AR 模型
cfg["example2"]["EXPORT_AR"] = 1023
cfg["example2"]["EXPORT_CONTEXT_LENGTH"] = $CL
cfg["example2"]["onnx_name"] = "qwen3_base"
cfg["example2"]["num_splits"] = $NUM_SPLITS
cfg["example2"]["soc_id"] = 43          # v73
cfg["example2"]["dsp_arch"] = "v73"
cfg["example2"]["num_cores"] = 1
cfg["example2"]["core_id"] = [0]
cfg["example2"]["vtcm_mb"] = 16

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("example2 配置已更新: v73, ARN=$ARN, splits=$NUM_SPLITS")
PYEOF

# ---- 运行 qnn_model_prepare.py ----
# 注意: 此脚本会执行:
#   1. ONNX 分割 (num_splits=3)
#   2. ONNX → DLC (qairt-converter)
#   3. DLC 量化 (qairt-quantizer)
#   4. context binary 生成 (qnn-context-binary-generator) ← 这步可能失败
#
# 即使步骤 4 失败,DLC (步骤 1-3 的产出) 已生成,可用于 Stage 3b
echo ""
echo "===== 运行 qnn_model_prepare.py ====="
echo "注意: context binary 步骤可能失败(已知 PerfSetting 问题),DLC 会成功生成"
cd "$QWEN3_PROJECT/example2/host_linux"

# 用 dzdo su 以 root 运行 (DLC 文件会被 root 拥有)
# 写临时脚本避免 dzdo su 引号嵌套问题,路径直接写入不依赖环境变量继承
cd "$QWEN3_PROJECT/example2/host_linux"
ROOT_SCRIPT=$(mktemp /tmp/qwen3_stage3a.XXXX.sh)
cat > "$ROOT_SCRIPT" << EOF
#!/bin/bash
cd "$QWEN3_PROJECT/example2/host_linux"
export QNN_SDK_ROOT="$QNN_SDK_ROOT"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$QNN_SDK_ROOT/benchmarks/QNN"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:\$LD_LIBRARY_PATH"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:\$PATH"
python3 qnn_model_prepare.py
EOF
chmod +x "$ROOT_SCRIPT"
dzdo su -c "bash $ROOT_SCRIPT" || true
rm -f "$ROOT_SCRIPT"

# ---- 修复 DLC 文件权限 ----
echo ""
echo "===== 修复 DLC 权限 ====="
ARTIFACTS="$QWEN3_PROJECT/example2/host_linux/assets/artifacts"
chmod -R a+r "$ARTIFACTS"/*/compiled_model/*.dlc 2>/dev/null || \
    dzdo su -c "chmod -R a+r $ARTIFACTS" 2>/dev/null || true

# ---- 验证 DLC ----
echo ""
echo "===== 验证 DLC 文件 ====="
for i in $(seq 1 $NUM_SPLITS); do
    DLC="$ARTIFACTS/ar${ARN}-cl${CL}/${i}_of_${NUM_SPLITS}/compiled_model/ar${ARN}-cl${CL}_${i}_of_${NUM_SPLITS}_quantized.dlc"
    if [ -f "$DLC" ]; then
        SIZE=$(du -h "$DLC" | cut -f1)
        echo "  [OK] ${i}_of_${NUM_SPLITS}: $SIZE"
    else
        echo "  [MISSING] ${i}_of_${NUM_SPLITS}: $DLC"
    fi
done

echo ""
echo "===== Stage 3a 完成 ====="
echo "DLC 已生成。下一步用独立脚本生成 v73 context binary:"
echo "  python stage3_compile/gen_v73_context_binary.py"
