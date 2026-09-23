#!/bin/bash
# run_qwen3_06b_device.sh — Qwen3-0.6B prefill 设备门一键跑 (GEHTP 最小 LLM 线)
# 前置: 设备 52f67807 在线且空闲 (PORTAL §4 纪律); skel 必须是 08-10 版 (md5 380f3cbf)。
# 产物: out_p{0..3}.f16.raw + judge 打印 (cos>=0.9999 + top1 32/32/prompt)。
set -euo pipefail
DEV=52f67807
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="$ROOT/test_models/qwen3_06b/cpu_golden"
BLOB="$ROOT/test_assets/qwen3_06b/qwen3_06b.wtop"
SKEL_MD5=380f3cbfa3084d7d1a6cf87ae7dbe964

echo "== [0/4] 前置检查 =="
state=$(adb -s $DEV get-state 2>/dev/null || echo off)
[ "$state" = "device" ] || { echo "FAIL: $DEV offline ($state)"; exit 1; }
occ=$(adb -s $DEV shell "ps -A 2>/dev/null | grep run_main_on_hexagon" || true)
[ -z "$occ" ] || { echo "FAIL: 设备被占用: $occ"; exit 1; }
md5=$(adb -s $DEV shell "md5sum /data/local/tmp/hvxhmx23/librun_main_on_hexagon_skel.so" | awk '{print $1}' | tr -d '\r')
[ "$md5" = "$SKEL_MD5" ] || { echo "FAIL: skel 陈旧 ($md5 != $SKEL_MD5) — 按 RUNBOOK §4 换 hvxhmx_libs 版"; exit 1; }
adb -s $DEV shell "su 0 getenforce | grep -qi permissive || su 0 setenforce 0" >/dev/null
adb -s $DEV shell "su 0 getenforce"
# 通道探针: capability API 死活 (0x39=通道死, 勿继续, 先整板恢复)
probe=$(adb -s $DEV shell "cd /data/local/tmp/hvxhmx23 && ADSP_LIBRARY_PATH=. CDSP_LIBRARY_PATH=. ./run_main_on_hexagon 3 benchone_20260910_194104.so 2>&1 | head -2" || true)
if echo "$probe" | grep -q "Capability API failed"; then
    echo "FAIL: fastrpc 通道死 (0x39) — 需整板重启/断电恢复, 勿再试探 (失败 open 会加重楔死)"; exit 1
fi
echo "通道活; skel 正确; 开始 4 prompts 设备 run"

mkdir -p "$ROOT/test_assets/qwen3_06b/device_out"
for j in 0 1 2 3; do
    echo "== [$((j+1))/4] prompt$j =="
    "$ROOT/scripts/gehtp" run "$BLOB" \
        --input "$GOLD/embed_p${j}.f16.raw" \
        --output "$ROOT/test_assets/qwen3_06b/device_out/out_p${j}.f16.raw" --device $DEV
done

echo "== judge =="
/home/speech/miniforge3/bin/python3 "$ROOT/scripts/judge_qwen3_06b.py" \
    --golden "$GOLD/hf_logits_fp32.npy" \
    --device-dir "$ROOT/test_assets/qwen3_06b/device_out"
