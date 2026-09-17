#!/bin/bash
# gehtp_quickstart.sh — GEHTP 正门快速上手 sample
# =====================================================================
# 演示两命令用户旅程: ONNX → compile → 上板 run → 输出对拍。
# 模型: test_models/conv_add (Transpose→Conv2d→Add→Transpose, f16)
# 用法: scripts/gehtp_quickstart.sh [工作目录]     默认 /tmp/gehtp_quickstart
# 设备: 52f67807 (ANDROID_SERIAL 可覆盖); 互斥约定见脚本内 ps 检查
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TM="$ROOT/test_models/conv_add"
WORK="${1:-/tmp/gehtp_quickstart}"
DEVICE="${ANDROID_SERIAL:-52f67807}"
GEHTP="$ROOT/scripts/gehtp"
mkdir -p "$WORK"

echo "########## [0/4] 设备互斥检查 (多窗口约定: 有主就等) ##########"
if adb -s "$DEVICE" shell "ps 2>/dev/null | grep -q run_main_on_hexagon" 2>/dev/null; then
    echo "设备 $DEVICE 上有 run_main_on_hexagon 在跑(别的窗口), 等 60s..."
    sleep 60
    adb -s "$DEVICE" shell "ps 2>/dev/null | grep -q run_main_on_hexagon" 2>/dev/null \
        && { echo "仍占用, 退出。稍后再试。"; exit 1; }
fi
echo "设备空闲 ✓"

echo "########## [1/4] gehtp compile (host: ONNX → wtop blob) ##########"
"$GEHTP" compile "$TM/conv_add.onnx" -o "$WORK/conv_add.wtop"
# 产物: $WORK/conv_add.wtop (+ .manifest.json + .work/ 中间产物)

echo "########## [2/4] gehtp setup (一次性: 库+通用 runner 上板) ##########"
"$GEHTP" setup --device "$DEVICE"

echo "########## [3/4] gehtp run ×3 (三组输入轮换注入) ##########"
for i in 0 1 2; do
    "$GEHTP" run "$WORK/conv_add.wtop" \
        --input "$TM/in$i.f16.raw" --output "$WORK/out$i.f16.raw" --device "$DEVICE"
done

echo "########## [4/4] 输出 vs golden (f16, 期望逐字节全同) ##########"
python3 - "$WORK" "$TM" <<'EOF'
import struct, sys
work, tm = sys.argv[1], sys.argv[2]
bad_total = 0
for i in range(3):
    a = open(f"{work}/out{i}.f16.raw", "rb").read()
    g = open(f"{tm}/gold{i}.f16.raw", "rb").read()
    assert len(a) == len(g) == 65536, f"size {len(a)} != 65536"
    ua = struct.unpack("<32768H", a); ug = struct.unpack("<32768H", g)
    bad = sum(1 for x, y in zip(ua, ug)
              if x != y and ((x & 0x7c00) == 0x7c00 or (y & 0x7c00) == 0x7c00
                             or abs(x - y) > 1))
    exact = sum(1 for x, y in zip(ua, ug) if x == y)
    print(f"round{i}: byte-exact={exact}/32768 bad(>1ULP)={bad}")
    bad_total += bad
print("=== SAMPLE ALL GREEN ===" if bad_total == 0 else "=== SAMPLE FAILED ===")
sys.exit(0 if bad_total == 0 else 1)
EOF

echo
echo "下一步: spill 变体 (编译期内存规划, runlist 插 SPILL/FILL 算子):"
echo "  $GEHTP compile $TM/conv_add.onnx -o $WORK/conv_add_spill.wtop --ddr-budget 4096"
echo "  $GEHTP run $WORK/conv_add_spill.wtop --input $TM/in0.f16.raw --output $WORK/out_spill.f16.raw"
