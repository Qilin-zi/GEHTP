#!/bin/bash
# conv_add_pipeline.sh — GEHTP conv2d+add 端到端一键 (阶段 10, M2)
# =====================================================================
# 流程: 模型/金标(gen_all + gen_io_rounds) → hnnx_compile tagged
#       (主 + 1KB 溢出变体) → wtop_emit (主 + spill) → kernels 编译/签名
#       → 推板 52f67807 → 运行例 37 → 拉结果 → 判据汇总
# 用法: conv_add_pipeline.sh [device]      # 不带参数 = 仅 host 全链
# 依赖(只读): venv310、qairt SDK、Hexagon SDK、SWIV
# 设备纪律: 仅 52f67807; CMA 耗尽停下通知用户; wt_exec_shutdown 成败必调(例内)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TM="$ROOT/test_models/conv_add"
BL="$ROOT/blobs_conv_add"
CB="$ROOT/compiler/build_linux"
KD="$ROOT/kernels"
SDK=/disk2/QCtools/qairt_2.48.40.260702
PY=/disk2/Qwen35dev/revlibHtpPrepare/venv310/bin/python3
DEVICE="${ANDROID_SERIAL:-52f67807}"
DEVDIR="/data/local/tmp/hvxhmx23/g37"

echo "=== [1/6] 模型 + golden (host) ==="
cd "$TM"
bash gen_all.sh >/dev/null 2>&1 || true
[ -f conv_add.onnx ] || { echo "gen_all.sh failed"; exit 1; }
$PY gen_io_rounds.py

mkdir -p "$BL"
echo "=== [2/6] hnnx_compile tagged (主 + 溢出变体) ==="
unzip -p conv_add_qnn/conv_add.cpp model.params.bin > conv_add_qnn/model.params.bin 2>/dev/null || true
"$CB/hnnx_compile" --net-json conv_add_qnn/conv_add_net.json \
    --weights-bin conv_add_qnn/model.params.bin --format tagged \
    --output conv_add_qnn/conv_add.bin | tail -1
"$CB/hnnx_compile" --net-json conv_add_qnn/conv_add_net.json \
    --weights-bin conv_add_qnn/model.params.bin --format tagged \
    --vtcm-budget 1024 \
    --output conv_add_qnn/conv_add_spill.bin | tail -1

echo "=== [3/6] wtop_emit (主 + spill) ==="
"$CB/wtop_emit" --bin conv_add_qnn/conv_add.bin --input-f16 in0.f16.raw \
    --out "$BL/blob.wtop" --manifest "$BL/manifest.json" | tail -1
"$CB/wtop_emit" --bin conv_add_qnn/conv_add_spill.bin --input-f16 in0.f16.raw \
    --out "$BL/blob_spill.wtop" --manifest "$BL/manifest_spill.json" | tail -1
cp in0.f16.raw in1.f16.raw in2.f16.raw gold0.f16.raw gold1.f16.raw gold2.f16.raw "$BL/"

if [ "${1:-}" != "device" ]; then
    echo "=== host 闭环完成 (device 步: conv_add_pipeline.sh device) ==="
    echo "blobs: $BL/{blob.wtop, blob_spill.wtop, in0..2, gold0..2}"
    exit 0
fi

echo "=== [4/6] kernels 编译/签名 + 推板 ==="
cd "$KD"
SDK_HEX=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0
HT="$SDK_HEX/tools/HEXAGON_Tools/19.0.07"
CC="$HT/Tools/bin/hexagon-clang"
./build_libs.sh >/dev/null 2>&1
"$HT/Tools/bin/hexagon-llvm-objdump" -d lib/libhvxhmx_v23.so | grep -q vgather && { echo "vgather 铁律违反"; exit 1; }
INC_HEX="-I$SDK_HEX/incs -I$SDK_HEX/incs/stddef \
         -I$SDK_HEX/rtos/qurt/computev81/include \
         -I$SDK_HEX/rtos/qurt/computev81/include/qurt"
$CC -mv81 -O2 -mhvx -mhmx -shared -fPIC -std=gnu11 -Wall -mhvx-length=128B \
    $INC_HEX -I"$KD/include" -I"$KD/examples/common" \
    -o /tmp/test_37.so "$KD/examples/37_conv2d_add/main.c" \
    "$KD/examples/common/example_util.c" -L"$KD/lib" -lhvxhmx_v23 -lc -ldl -lgcc
python3 /disk2/QCtools/swiv_build_utility.py -i /tmp/test_37.so -o /tmp/test_37.so.signed 2>&1 | tail -1

adb -s "$DEVICE" shell "mkdir -p $DEVDIR" >/dev/null
adb -s "$DEVICE" push "$KD/lib/libhvxhmx_v23.signed.so" /data/local/tmp/hvxhmx23/libhvxhmx_v23.so >/dev/null
adb -s "$DEVICE" push /tmp/test_37.so.signed "/data/local/tmp/hvxhmx23/test_37_conv2d_add.so" >/dev/null
adb -s "$DEVICE" shell "echo 'FARF=0xFFFFFFFF' > /data/local/tmp/hvxhmx23/test_37_conv2d_add.so.farf" 2>/dev/null || true

echo "=== [5/6] 推 blob/输入/金标 + 运行例 37 ==="
adb -s "$DEVICE" push "$BL/blob.wtop" "$BL/blob_spill.wtop" "$DEVDIR/" >/dev/null
adb -s "$DEVICE" push "$BL"/in0.f16.raw "$BL"/in1.f16.raw "$BL"/in2.f16.raw "$DEVDIR/" >/dev/null
adb -s "$DEVICE" push "$BL"/gold0.f16.raw "$BL"/gold1.f16.raw "$BL"/gold2.f16.raw "$DEVDIR/" >/dev/null
adb -s "$DEVICE" shell "cd /data/local/tmp/hvxhmx23 && ADSP_LIBRARY_PATH=/data/local/tmp/hvxhmx23 CDSP_LIBRARY_PATH=/data/local/tmp/hvxhmx23 ./run_main_on_hexagon 3 test_37_conv2d_add.so" > "$BL/device_out.txt" 2>&1 || true
adb -s "$DEVICE" shell "cat /data/local/tmp/hvxhmx23/37_conv2d_add.txt" > "$BL/result_37.txt" 2>/dev/null || true

echo "=== [6/6] 判据汇总 ==="
if [ ! -s "$BL/result_37.txt" ]; then
    echo "=== M2 FAILED: no result file (device run failed) ==="
    tail -5 "$BL/device_out.txt" 2>/dev/null || true
    exit 1
fi
cat "$BL/result_37.txt"
PASS_N=$(grep -c "\[PASS\]" "$BL/result_37.txt" || true)
FAIL_N=$(grep -c "\[FAIL\]" "$BL/result_37.txt" || true)
echo "--- PASS=$PASS_N FAIL=$FAIL_N ---"
if [ "$FAIL_N" -gt 0 ] || [ "$PASS_N" -lt 10 ]; then
    echo "=== M2 FAILED ==="; exit 1
fi
echo "=== M2 ALL GREEN ==="
