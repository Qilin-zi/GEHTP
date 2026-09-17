#!/bin/bash
# ===== 一键脚本: 在 dev133 上跑 Stage 1-3 =====
# 机器: dev133 (编译机)
# 产出: v73 context binary (.bin)
# 总耗时: ~2 小时
#
# 用法:
#   scp 此文件到 dev133, 然后执行:
#   bash run_on_dev133.sh

set -e

# ===== 配置 =====
QNN_SDK_ROOT="/data01/rqilin/qnn-sdk"
QWEN3_PROJECT="/data01/rqilin/qwen3_llm_v2"
MODEL_ID="/data01/rqilin/models/Qwen3-4B-Base"
HF_MODEL_NAME="Qwen/Qwen3-4B-Base"

# 模型参数
CONTEXT_LENGTH=2048
ARN=128
NUM_SPLITS=3
SOC_ID=43        # v73
DSP_ARCH="v73"

export QNN_SDK_ROOT
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$QNN_SDK_ROOT/benchmarks/QNN"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"

echo "############################################################"
echo "# Qwen3 → QNN HTP v73 端到端流程 (dev133)                   #"
echo "############################################################"
echo ""

# ===== Stage 1: 下载模型 =====
echo "===== [1/5] Stage 1: 下载模型 ====="
if [ -d "$MODEL_ID" ] && ls "$MODEL_ID"/*.safetensors >/dev/null 2>&1; then
    echo "模型已存在,跳过下载: $MODEL_ID"
else
    mkdir -p "$MODEL_ID"
    pip install -q huggingface_hub 2>/dev/null || true
    huggingface-cli download "$HF_MODEL_NAME" --local-dir "$MODEL_ID"
fi
echo ""

# ===== Stage 2: 导出 ONNX (Example1) =====
echo "===== [2/5] Stage 2: 导出 ONNX ====="
OUTPUT_DIR="$QWEN3_PROJECT/example1/output_dir_"

if [ -f "$OUTPUT_DIR/onnx/qwen3_base.onnx" ]; then
    echo "ONNX 已存在,跳过: $OUTPUT_DIR/onnx/qwen3_base.onnx"
else
    # 更新配置 (用 env 文件传参给 Python,避免引号嵌套)
    CFG_FILE=$(mktemp /tmp/qwen3_cfg.XXXX.py)
    cat > "$CFG_FILE" << PYEOF
import json
p = "$QWEN3_PROJECT/qwen3_local_config.json"
with open(p) as f: c = json.load(f)
c['example1']['MODEL_ID'] = "$MODEL_ID"
c['example1']['CONTEXT_LENGTH'] = $CONTEXT_LENGTH
c['example1']['ARN'] = 1023
c['example1']['QNN_SDK_ROOT'] = "$QNN_SDK_ROOT"
c['example1']['MODEL_NAME'] = 'qwen3'
c['example2']['CL'] = $CONTEXT_LENGTH
c['example2']['EXPORT_AR'] = 1023
c['example2']['EXPORT_CONTEXT_LENGTH'] = $CONTEXT_LENGTH
c['example2']['onnx_name'] = 'qwen3_base'
c['example2']['ARNs'] = [$ARN]
c['example2']['num_splits'] = $NUM_SPLITS
c['example2']['soc_id'] = $SOC_ID
c['example2']['dsp_arch'] = "$DSP_ARCH"
with open(p,'w') as f: json.dump(c, f, indent=2)
print('配置已更新')
PYEOF
    python3 "$CFG_FILE"
    rm -f "$CFG_FILE"

    cd "$QWEN3_PROJECT/example1"
    python3 qwen3.py
fi
echo ""

# ===== Stage 3a: ONNX → DLC (Example2) =====
echo "===== [3/5] Stage 3a: ONNX → DLC ====="
ARTIFACTS="$QWEN3_PROJECT/example2/host_linux/assets/artifacts"
DLC_CHECK="$ARTIFACTS/ar${ARN}-cl${CONTEXT_LENGTH}/1_of_${NUM_SPLITS}/compiled_model/ar${ARN}-cl${CONTEXT_LENGTH}_1_of_${NUM_SPLITS}_quantized.dlc"

if [ -f "$DLC_CHECK" ]; then
    echo "DLC 已存在,跳过"
else
    # 把 root 命令写到临时脚本,避免 dzdo su 引号嵌套问题
    cd "$QWEN3_PROJECT/example2/host_linux"
    ROOT_SCRIPT=$(mktemp /tmp/qwen3_root.XXXX.sh)
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
fi

# 修复 DLC 权限
echo "修复 DLC 权限..."
chmod -R a+r "$ARTIFACTS"/*/compiled_model/*.dlc 2>/dev/null || \
    dzdo su -c "chmod -R a+r $ARTIFACTS" 2>/dev/null || true
echo ""

# ===== Stage 3b: 生成 v73 Context Binary =====
echo "===== [4/5] Stage 3b: 生成 v73 Context Binary ====="
V73_DIR="$ARTIFACTS/ar${ARN}-cl${CONTEXT_LENGTH}_v73"
V73_ALL_OK=true
for i in $(seq 1 $NUM_SPLITS); do
    [ -f "$V73_DIR/ar${ARN}-cl${CONTEXT_LENGTH}_${i}_of_${NUM_SPLITS}_v73.serialized.bin" ] || V73_ALL_OK=false
done

if $V73_ALL_OK; then
    echo "v73 .bin 已存在,跳过"
else
    # 用环境变量传参给 Python,避免硬编码
    export GEN_V73_DLC_DIR="$ARTIFACTS/ar${ARN}-cl${CONTEXT_LENGTH}"
    export GEN_V73_OUT_DIR="$V73_DIR"
    export GEN_V73_CONF_DIR="$ARTIFACTS/ar${ARN}-cl${CONTEXT_LENGTH}_v73_conf"
    export GEN_V73_NUM_SPLITS=$NUM_SPLITS
    export GEN_V73_ARN=$ARN
    export GEN_V73_CL=$CONTEXT_LENGTH
    export GEN_V73_SOC_ID=$SOC_ID
    export GEN_V73_DSP_ARCH=$DSP_ARCH

    python3 << 'PYEOF'
import json, os, subprocess, sys

QNN_SDK_ROOT = os.environ["QNN_SDK_ROOT"]
DLC_DIR = os.environ["GEN_V73_DLC_DIR"]
OUT_DIR = os.environ["GEN_V73_OUT_DIR"]
CONF_DIR = os.environ["GEN_V73_CONF_DIR"]
NUM_SPLITS = int(os.environ["GEN_V73_NUM_SPLITS"])
ARN = int(os.environ["GEN_V73_ARN"])
CL = int(os.environ["GEN_V73_CL"])
SOC_ID = int(os.environ["GEN_V73_SOC_ID"])
DSP_ARCH = os.environ["GEN_V73_DSP_ARCH"]

os.environ["PATH"] = f"/usr/bin:/usr/local/bin:{QNN_SDK_ROOT}/bin/x86_64-linux-clang:" + os.environ.get("PATH","")
os.environ["PYTHONPATH"] = f"{QNN_SDK_ROOT}/lib/python:{QNN_SDK_ROOT}/benchmarks/QNN"
os.environ["LD_LIBRARY_PATH"] = f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang:" + os.environ.get("LD_LIBRARY_PATH","")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)

for i in range(1, NUM_SPLITS+1):
    graph = f"ar{ARN}-cl{CL}_{i}_of_{NUM_SPLITS}"
    htp = {"backend_extensions": {"shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{CONF_DIR}/perf_{i}.conf"}}
    with open(f"{CONF_DIR}/htp_{i}.json","w") as f: json.dump(htp, f, indent=4)
    perf = {"devices": [{"soc_id": SOC_ID, "dsp_arch": DSP_ARCH, "pd_session": "unsigned"}]}
    with open(f"{CONF_DIR}/perf_{i}.conf","w") as f: json.dump(perf, f, indent=4)

    dlc = f"{DLC_DIR}/{i}_of_{NUM_SPLITS}/compiled_model/{graph}_quantized.dlc"
    bin_name = f"{graph}_v73.serialized"
    print(f"\n--- {i}/{NUM_SPLITS} ---")
    print(f"DLC: {dlc}")
    r = subprocess.run(["qnn-context-binary-generator",
        "--backend","libQnnHtp.so","--model","libQnnModelDlc.so",
        "--input_output_tensor_mem_type","memhandle","--output_dir",OUT_DIR,
        "--config_file",f"{CONF_DIR}/htp_{i}.json","--binary_file",bin_name,
        "--dlc_path",dlc], capture_output=True, text=True, timeout=600)
    out_file = f"{OUT_DIR}/{bin_name}.bin"
    if os.path.exists(out_file):
        print(f"  [OK] {os.path.getsize(out_file)//1024//1024} MB")
    else:
        print(f"  [FAIL]")
        if r.stdout: print(r.stdout[-500:])
        if r.stderr: print(r.stderr[-500:])
        sys.exit(1)
print("\n[成功] v73 context binary 全部生成")
PYEOF
fi
echo ""

# ===== 汇总 =====
echo "===== [5/5] 汇总 ====="
echo "v73 Context Binary 文件:"
ls -lh "$V73_DIR"/*.bin 2>/dev/null || echo "  [警告] 未找到 .bin 文件"

echo ""
echo "############################################################"
echo "# dev133 阶段完成!                                          #"
echo "#                                                           #"
echo "# v73 .bin 路径: $V73_DIR/                                  #"
echo "#                                                           #"
echo "# 下一步: 在 Windows 上运行 run_on_windows.ps1              #"
echo "############################################################"
