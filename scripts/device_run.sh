#!/bin/bash
# GEHTP: 设备上板脚本模板(未经修改的副本,阶段 9/10 将改写为 GEHTP 原生版)
# 来源: /disk2/V81Dev/vtcm_engine_probe/run.sh (2026-08-31 拷贝)
# 用途: 构建/签名/上板/拉取 一键模板 —— 关键模式:su 0 + /sdcard 中转 + shell 属主目录
# run.sh — vtcm_engine_probe 构建/签名/上板/拉取 (一键)
# ==================================================
# 用法:
#   ./run.sh          # 编译+签名 (本地, 无设备)
#   ./run.sh device   # 编译+签名+push 52f67807+运行+拉结果到本目录
#
# 依赖 V2.3 树只读引用 (include/lib/common), 产物全部落本目录.
# 设备目录 /data/local/tmp/ep 为 shell 属主 (hvxhmx23 是 root+SELinux 只读),
# 首次运行时从 hvxhmx23 su 拷贝 run_main_on_hexagon 装载器.
set -euo pipefail

DIR="$(cd "$(dirname "$BASH_SOURCE[0]")" && pwd)"
LIB="/disk2/V81Dev/hvxhmx_libsV2.3"
SDK="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
HT="$SDK/tools/HEXAGON_Tools/19.0.07"
CC="$HT/Tools/bin/hexagon-clang"
SWIV="${SWIV_TOOL:-/disk2/QCtools/swiv_build_utility.py}"
DEVICE="${DEVICE:-52f67807}"
DEVDIR="/data/local/tmp/ep"

INC_HEX="-I$SDK/incs -I$SDK/incs/stddef \
         -I$SDK/rtos/qurt/computev81/include \
         -I$SDK/rtos/qurt/computev81/include/qurt"
CFLAGS="-mv81 -O2 -mhvx -mhmx -shared -fPIC -std=gnu11 -Wall -mhvx-length=128B"
LDFLAGS="-lc -ldl -lgcc"

SO="$DIR/test_vtcm_engine_probe.so"

echo "=== build ==="
$CC $CFLAGS $INC_HEX -I"$LIB/include" -I"$LIB/examples/common" \
    -o "$SO" "$DIR/main.c" "$LIB/examples/common/example_util.c" \
    -L"$LIB/lib" -lhvxhmx_v23 $LDFLAGS
echo "  so: $(wc -c < "$SO") bytes"

# 铁律: 产物 vgather 必须为 0 (V81 errata 崩 CDSP)
NG=$(hexagon-llvm-objdump -d "$SO" 2>/dev/null | grep -c vgather || true)
echo "  vgather count: $NG"
[ "$NG" = "0" ] || { echo "FATAL: vgather != 0"; exit 1; }

echo "=== SWIV sign ==="
python3 "$SWIV" -i "$SO" -o "${SO}.signed" 2>&1 | tail -1

if [ "${1:-}" != "device" ]; then
    echo "=== local-only done (run '$0 device' when 52f67807 is free) ==="
    exit 0
fi

echo "=== device prep ($DEVICE:$DEVDIR, root 通道) ==="
adb() { command adb -s "$DEVICE" "$@"; }
# /data/local/tmp 标 system_data_file + Enforcing → shell 域禁写, 全程 su 0;
# 推送经 /sdcard (shell 可写) 中转. 首次从 hvxhmx23(root 目录) 拷装载器 + v23 库.
adb shell "su 0 sh -c 'mkdir -p $DEVDIR && \
    cp /data/local/tmp/hvxhmx23/run_main_on_hexagon \
       /data/local/tmp/hvxhmx23/librun_main_on_hexagon_skel.so \
       /data/local/tmp/hvxhmx23/libhvxhmx_v23.so $DEVDIR/ && chmod 755 $DEVDIR/*'" >/dev/null
adb shell "su 0 ls $DEVDIR/run_main_on_hexagon" >/dev/null

echo "=== device run ==="
adb push "${SO}.signed" "/sdcard/test_vtcm_engine_probe.so" >/dev/null
adb shell "su 0 cp /sdcard/test_vtcm_engine_probe.so $DEVDIR/ && su 0 chmod 644 $DEVDIR/test_vtcm_engine_probe.so"
adb shell "su 0 sh -c 'echo FARF=0xFFFFFFFF > $DEVDIR/test_vtcm_engine_probe.so.farf'" 2>/dev/null || true
adb shell "su 0 sh -c 'cd $DEVDIR && ADSP_LIBRARY_PATH=$DEVDIR CDSP_LIBRARY_PATH=$DEVDIR \
    ./run_main_on_hexagon 3 test_vtcm_engine_probe.so'" 2>&1 | grep -E 'return|Successfully|ERROR' | head -2 || true
adb shell "su 0 cat /data/local/tmp/hvxhmx23/vtcm_engine_probe.txt" > "$DIR/vtcm_engine_probe.txt"
echo "=== result: $DIR/vtcm_engine_probe.txt ($(wc -l < "$DIR/vtcm_engine_probe.txt") lines) ==="
grep -c "EP cfg=" "$DIR/vtcm_engine_probe.txt" || true
