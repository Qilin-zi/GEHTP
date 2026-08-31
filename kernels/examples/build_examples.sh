#!/bin/bash
# build_examples.sh — V2.2 示例编译/签名/部署/实机运行 (一键, 01-22)
# =====================================================================
# 用法:
#   ./build_examples.sh            # 跑全部 (01-15 V2.1 回归 + 16-22 新单元)
#   ./build_examples.sh 20         # 只跑 20_dualdomain (含双域编排)
#   ./build_examples.sh all
#
# 与 V2.1 差异:
#   - 链接 -lhvxhmx_v23; 设备目录 /data/local/tmp/hvxhmx23
#   - 部署 assets (s256/smallm) + host 打包 oplist blob + K2560 金标
#   - 例 20: ser → dom3/dom4 并发 a/b, host analyze_dd.py 对拍切分等价
#   - 例 21: 设备 W3 报告行与 host wt_inspect 输出逐行 diff
#   - 全部结果落 results/, 末尾汇总 PASS/FAIL
set -euo pipefail

DEVICE="${DEVICE:-52f67807}"
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
HT="$SDK/tools/HEXAGON_Tools/19.0.07/Tools"
CC="$HT/bin/hexagon-clang"
SWIV="${SWIV_TOOL:-/disk2/QCtools/swiv_build_utility.py}"

INC_HEX="-I$SDK/incs -I$SDK/incs/stddef \
         -I$SDK/rtos/qurt/computev81/include \
         -I$SDK/rtos/qurt/computev81/include/qurt"
CFLAGS="-mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC -std=gnu11 -Wall"
# libqurt.a 非 PIC 不可静态链 (R_HEX_32_6_X); qurt/HAP 符号运行时由 DSP 进程解析
# (全部已闭合模块的测试 .so 同款链法)
LDFLAGS="-lc -ldl -lgcc"

DEVDIR="/data/local/tmp/hvxhmx23"
BUILD="$LIB/build"
RES="$LIB/results"
BLOBS="$BUILD/blobs"
EXAMPLES=(01_runtime_init 02_convf16_gemm 03_convbbb_int8 04_convhbh_u16 \
          05_i16_weight_convs 06_dwconv 07_add 08_divide 09_activation \
          10_reduction 11_lookup_unpack 12_multitile_gemm 13_compat_dlsym \
          14_hmx_peak_gemm 15_v2_llm_ops 16_wtcache_pin 17_w4a16_gemm \
          18_smallm_gemv 19_gdn_sm 20_dualdomain 21_oplist_exec \
          22_dualcore_threads 23_fence 24_arena 25_harness 26_wpool \
          27_pxbridge 28_gdn_tree 29_kvcache 30_graph_step 31_gemm_dispatch 32_rbr \
          33_bledger 34_dmaring 35_btrack 36_absoak 37_conv2d_add)

adb() { command adb -s "$DEVICE" "$@"; }

# ---------- 0. 库 + host 工具 + blob ----------
mkdir -p "$BUILD" "$RES" "$BLOBS"
if [ ! -f "$LIB/lib/libhvxhmx_v23.signed.so" ]; then
    echo "=== lib missing → build_libs.sh ==="
    "$LIB/build_libs.sh"
fi
if [ ! -x "$BUILD/host/pack_oplist" ] || [ ! -x "$BUILD/host/wt_inspect" ]; then
    echo "=== host tools ==="
    mkdir -p "$BUILD/host"
    g++ -O2 -std=c++14 -I"$LIB/include" -I"$LIB/host/vendor" \
        -o "$BUILD/host/pack_oplist" "$LIB/host/pack_oplist.cc" \
        "$LIB/host/vendor/weight_pack.cc" \
        "$LIB/src/runtime/oplist_parse.c" "$LIB/src/runtime/wt_sha256.c"
    gcc -O2 -I"$LIB/include" \
        -o "$BUILD/host/wt_inspect" "$LIB/host/wt_inspect.c" \
        "$LIB/src/runtime/oplist_parse.c" "$LIB/src/runtime/wt_sha256.c" \
        "$LIB/src/runtime/wt_w3.c"
fi
# GEHTP 例37 资产(conv_add_pipeline.sh 产出; 缺失则跳过推送)
G37_DIR="${GEHTP_37_DIR:-$LIB/../../blobs_conv_add}"
if [ -f "$G37_DIR/blob.wtop" ]; then
    adb shell "mkdir -p $DEVDIR/g37" >/dev/null 2>&1
    adb push "$G37_DIR"/*.wtop "$G37_DIR"/in*.f16.raw "$G37_DIR"/gold*.f16.raw "$DEVDIR/g37/" >/dev/null 2>&1 || true
fi

for tag in w4 w5; do
    if [ ! -f "$BLOBS/blob_$tag.wtop" ]; then
        "$BUILD/host/pack_oplist" --t10 "$LIB/assets/s2560" --out "$BLOBS" --tag "$tag"
    fi
done
"$BUILD/host/wt_inspect" "$BLOBS/blob_w5.wtop" > "$BLOBS/w3_w5.host.txt"

# ---------- 1. 部署 ----------
echo "=== deploy (device=$DEVICE) ==="
adb shell "mkdir -p $DEVDIR/assets" >/dev/null 2>&1
adb push "$LIB/lib/libhvxhmx_v23.signed.so" "$DEVDIR/libhvxhmx_v23.so" >/dev/null
adb shell "rm -rf $DEVDIR/assets/s256 $DEVDIR/assets/smallm" >/dev/null 2>&1
adb push "$LIB/assets/s256" "$DEVDIR/assets/" >/dev/null
adb push "$LIB/assets/smallm" "$DEVDIR/assets/" >/dev/null
adb push "$BLOBS/blob_w4.wtop" "$DEVDIR/" >/dev/null
adb push "$BLOBS/blob_w5.wtop" "$DEVDIR/" >/dev/null
adb push "$BLOBS/rms_w.f16.raw" "$DEVDIR/" >/dev/null
adb push "$LIB/assets/s2560/Y_gold.raw" "$DEVDIR/" >/dev/null
adb push "$LIB/assets/s256/Y_gold_2563.raw" "$DEVDIR/assets/s256/" >/dev/null
# 例 13 dlopen V2.1 库 — 设备 hvxhmx_libs 目录已有现成 signed 件
adb shell "cp /data/local/tmp/hvxhmx_libs/libhvxhmx_v2.so $DEVDIR/libhvxhmx_v2.so 2>/dev/null || true"
# 例 20 dump 目录: DSP 上 system() 无 shell, 必须 host 预建
adb shell "rm -rf $DEVDIR/dd_out; mkdir -p $DEVDIR/dd_out" >/dev/null 2>&1
# skel 必须用 hvxhmx_libs 的 2026-08-10 版 (支持 dom3/4 unsigned PD;
# /data/local/tmp 顶层 2022 旧版报 0x80000406)
adb shell "cp /data/local/tmp/hvxhmx_libs/librun_main_on_hexagon_skel.so $DEVDIR/ 2>/dev/null || cp /data/local/tmp/librun_main_on_hexagon_skel.so $DEVDIR/ 2>/dev/null || true"
if adb shell "test -x /data/local/tmp/hvxhmx_libs/run_main_on_hexagon"; then
    adb shell "cp /data/local/tmp/hvxhmx_libs/run_main_on_hexagon $DEVDIR/"
elif adb shell "test -x /data/local/tmp/run_main_on_hexagon"; then
    adb shell "cp /data/local/tmp/run_main_on_hexagon $DEVDIR/"
else
    echo "ERROR: run_main_on_hexagon not found on device" >&2
    exit 1
fi
adb shell "chmod 755 $DEVDIR/*" >/dev/null 2>&1

run_shell() { adb shell "cd $DEVDIR && ADSP_LIBRARY_PATH=$DEVDIR CDSP_LIBRARY_PATH=$DEVDIR $1"; }

pull_result() {   # pull_result <name> → results/<name>.txt (仅设备文件非空才覆盖, 防清空旧结果)
    if adb shell "test -s $DEVDIR/$1.txt"; then
        adb shell "cat $DEVDIR/$1.txt" > "$RES/$1.txt"
    else
        echo "WARNING: $1.txt missing or empty on device, keeping local old file" >&2
    fi
}

# ---------- 2. 单例构建+运行 ----------
run_one() {
    local EX="$1"
    local SRC="$LIB/examples/$EX/main.c"
    local UTIL="$LIB/examples/common/example_util.c"
    local SO="$BUILD/test_${EX}.so"
    if [ ! -f "$SRC" ]; then echo "  skip (no source): $EX"; return 0; fi

    echo "=== build $EX ==="
    # 编译错误必须中断 (曾经静默吞错 → 跑陈旧 .so)
    if ! $CC $CFLAGS $INC_HEX -I"$LIB/include" -I"$LIB/examples/common" \
        -o "$SO" "$SRC" "$UTIL" -L"$LIB/lib" -lhvxhmx_v23 $LDFLAGS; then
        echo "  !! COMPILE FAILED: $EX"
        return 1
    fi
    echo "  so: $(wc -c < "$SO") bytes"
    python3 "$SWIV" -i "$SO" -o "${SO}.signed" 2>&1 | tail -1
    adb push "${SO}.signed" "$DEVDIR/test_${EX}.so" >/dev/null 2>&1
    adb shell "echo 'FARF=0xFFFFFFFF' > $DEVDIR/test_${EX}.so.farf" 2>/dev/null || true

    case "$EX" in
    20_dualdomain)
        echo "  run: ser (dom3, 16 步) → a/b (dom3+dom4 并发, 8+8)"
        run_shell "./run_main_on_hexagon 3 test_20_dualdomain.so dd ser 0 16" \
            2>&1 | grep -E 'return|ERROR' | head -1 || true
        run_shell "./run_main_on_hexagon 3 test_20_dualdomain.so dd a 0 8" \
            >/dev/null 2>&1 &
        local PA=$!
        run_shell "./run_main_on_hexagon 4 test_20_dualdomain.so dd b 8 8" \
            >/dev/null 2>&1 &
        local PB=$!
        wait $PA || true
        wait $PB || true
        for t in ser a b; do pull_result "20_dualdomain_$t"; done
        python3 "$LIB/host/analyze_dd.py" "$RES"
        # 门统一归并到 20_dualdomain.txt (per-tag 文件留作 sha 证据, 无门)
        mv "$RES/20_dualdomain_host.txt" "$RES/20_dualdomain.txt"
        ;;
    *)
        echo "  run (CDSP PD 3):"
        run_shell "./run_main_on_hexagon 3 test_${EX}.so" 2>&1 \
            | grep -E 'return|Successfully|ERROR' | head -2 || true
        pull_result "$EX"
        ;;
    esac

    if [ -s "$RES/$EX.txt" ]; then
        if [ "$EX" = "21_oplist_exec" ]; then
            # W3 行 host/设备逐行对拍 (host wt_inspect vs 设备 result 内 W3 行)
            sed 's/^ *//' "$RES/$EX.txt" | grep '"t":"W3"' > "$RES/w3_device.txt"
            if diff -q "$BLOBS/w3_w5.host.txt" "$RES/w3_device.txt" >/dev/null; then
                echo "[PASS] w3_host_device_lines_identical err=0 tol=0" >> "$RES/$EX.txt"
            else
                echo "[FAIL] w3_host_device_lines_identical (host=$BLOBS/w3_w5.host.txt dev=$RES/w3_device.txt)" >> "$RES/$EX.txt"
            fi
        fi
        sed 's/^/    /' "$RES/$EX.txt"
    else
        echo "    !! no result file for $EX"
    fi
    echo ""
}

SEL="${1:-all}"
if [ "$SEL" = "all" ]; then
    for ex in "${EXAMPLES[@]}"; do run_one "$ex"; done
else
    # 支持 "02" 或 "02_convf16_gemm" 两种写法
    for ex in "${EXAMPLES[@]}"; do
        case "$ex" in
            "$SEL"|"$SEL"_*) run_one "$ex" ;;
        esac
    done
fi

# ---------- 3. 汇总 ----------
echo "=== summary (results/) ==="
total_p=0; total_f=0; bad_list=""
for f in "$RES"/*.txt; do
    name=$(basename "$f" .txt)
    p=$(grep -c '^\s*\[PASS\]' "$f" || true)
    fl=$(grep -c '^\s*\[FAIL\]' "$f" || true)
    total_p=$((total_p+p)); total_f=$((total_f+fl))
    if [ "$fl" -gt 0 ]; then
        bad_list="$bad_list $name($fl)"
        printf "  %-30s %3d pass %3d FAIL\n" "$name" "$p" "$fl"
    fi
done
printf "  %-30s %3d pass %3d FAIL\n" "TOTAL" "$total_p" "$total_f"
if [ "$total_f" -eq 0 ]; then echo "=== ALL GREEN ==="; else echo "=== FAILURES:${bad_list} ==="; fi
echo "=== done ==="
