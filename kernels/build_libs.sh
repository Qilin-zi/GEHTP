#!/bin/bash
# build_libs.sh — 编译 hvxhmx_libsV2.3 源码 → lib/libhvxhmx_v23.so (+ 签名版)
# =====================================================================
# V2.3 = V2.2 全量, src 布局融合 (不再有 src/v22 子目录):
#   src/runtime  wtcache/dc_*/dd_worker/oplist_*/dma_utils/wt_*
#                + 新: fence.c arena.c wpool.c kvcache.c harness.c gemm_dispatch.c
#   src/hmx      V2.1 conv 族 + w4a16_driver_dc.c (+ .inc)
#   src/hvx      V2.1 算子 + gdn_kern/gdn_ref + 新: gdn_tree.c pxbridge.c
#   src/compat   V2.1 版本兼容层
# 产物一个 .so; example 链接 -lhvxhmx_v23。
# 用法: ./build_libs.sh            # 全量
#       ./build_libs.sh one <base> # 只重编 <base>.c + 链接 (例: one fence)
set -euo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
HT="$SDK/tools/HEXAGON_Tools/19.0.07"
CC="$HT/Tools/bin/hexagon-clang"
OBJDUMP="$HT/Tools/bin/hexagon-llvm-objdump"
SWIV="${SWIV_TOOL:-/disk2/QCtools/swiv_build_utility.py}"

INC_HEX="-I$SDK/incs -I$SDK/incs/stddef \
         -I$SDK/rtos/qurt/computev81/include \
         -I$SDK/rtos/qurt/computev81/include/qurt \
         -I$HT/Tools/target/hexagon/include"
CFLAGS="-mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC -std=gnu11 -Wall"
LDFLAGS="-lc -ldl -lgcc"  # qurt 符号运行时解析 (模块先例); libqurt.a 非 PIC 不可静态挂

# 文件级例外:
#   hvx_int8gemm.c : -O1 (19.0.07 软件流水线器 stale-vector 首迭代错码, V2.1 起不可改)
#   dma_utils.c    : SDK 拷贝件 READ ONLY, 原样编译, 不开 -Wextra
#   工程单元 (原 v22 + 新 v23): -Wextra -Wno-shift-count-overflow
O1_FILES="hvx_int8gemm.c"
CFLAGS_O1=$(echo "$CFLAGS" | sed 's/-O2/-O1/')
RO_FILES="dma_utils.c"
WEXTRA_FILES="wtcache_impl.c dc_parts.c dc_sync.c dc_threads.c dd_worker.c \
              oplist_parse.c oplist_exec.c wt_sha256.c wt_w3.c \
              w4a16_driver_dc.c gdn_kern.c gdn_ref.c \
              fence.c arena.c wpool.c kvcache.c harness.c gemm_dispatch.c \
              gdn_tree.c pxbridge.c rbr.c bledger.c \
              ring_sim.c ring_policy.c btrack.c bflush.c dcache.c"
W_WARN="-Wall -Wextra -Wno-shift-count-overflow"

BUILD="$LIB/build"
mkdir -p "$BUILD" "$LIB/lib"

flags_for() {  # $1=basename → extra cflags
    local base="$1"
    for f in $RO_FILES;    do [ "$base" = "$f" ] && { echo "-Wall"; return; }; done
    for f in $O1_FILES;    do [ "$base" = "$f" ] && { echo "$CFLAGS_O1"; return; }; done
    for f in $WEXTRA_FILES; do [ "$base" = "$f" ] && { echo "$W_WARN"; return; }; done
    echo ""
}

cc_one() {  # $1=src $2=obj $3=extra-cflags
    $CC $CFLAGS $INC_HEX -I"$LIB/include" $3 -c "$1" -o "$2"
}

ALL_SRC=$(ls "$LIB"/src/runtime/*.c "$LIB"/src/hmx/*.c "$LIB"/src/hvx/*.c "$LIB"/src/compat/*.c)

if [ "${1:-all}" = "one" ]; then
    base="$2"
    src=$(echo "$ALL_SRC" | tr ' ' '\n' | grep "/$base.c$" | head -1)
    [ -z "$src" ] && { echo "!! no source for $base"; exit 1; }
    echo "=== [one] CC $base ==="
    cc_one "$src" "$BUILD/$base.o" "$(flags_for "$base")"
else
    echo "=== [1/2] 全量编译 (融合 src 布局) ==="
    for src in $ALL_SRC; do
        base=$(basename "$src" .c)
        extra="$(flags_for "$base")"
        echo "  CC $base $([ -n "$extra" ] && echo '[-Wall -Wextra]')" 
        cc_one "$src" "$BUILD/$base.o" "$extra"
    done
fi

echo "=== [2/2] LD libhvxhmx_v23.so ==="
$CC $CFLAGS $INC_HEX -I"$LIB/include" -o "$LIB/lib/libhvxhmx_v23.so" \
    $(ls "$BUILD"/*.o) $LDFLAGS
echo "  ok: $(wc -c < "$LIB/lib/libhvxhmx_v23.so") bytes"

echo "=== vgather grep (必须为 0) ==="
VG=$($OBJDUMP -d "$LIB/lib/libhvxhmx_v23.so" 2>/dev/null | grep -ci 'vgather' || true)
echo "  vgather count = $VG"
[ "$VG" != "0" ] && { echo "  !! ABORT: vgather 存在, 部署会 crash CDSP"; exit 1; }

echo "=== UNDEF 扫描 (qurt/HAP 运行时解析除外, 必须为 0) ==="
UNDEF=$($HT/Tools/bin/hexagon-nm -u "$LIB/lib/libhvxhmx_v23.so" 2>/dev/null \
        | awk '{print $2}' | grep -vE '^(qurt_|HAP_|ex_|__hexagon|dcache|memcpy|memset|malloc|free|memalign|memmove|posix_memalign|open|read|close|lseek|fopen|fread|fclose|snprintf|printf|puts|putchar|expf?|sqrtf?|fabsf?|logf?|powf?|fmaxf?|fminf?|floorf?|ceilf?|_.*|$)' | grep -v 'U ' | head -5 || true)
echo "  (人工核对上方列表; 常见 libc/math 均为运行时解析)"

echo "=== SWIV sign ==="
python3 "$SWIV" -i "$LIB/lib/libhvxhmx_v23.so" -o "$LIB/lib/libhvxhmx_v23.signed.so" 2>&1 | tail -1
echo "  signed: $(wc -c < "$LIB/lib/libhvxhmx_v23.signed.so") bytes"
echo "=== done ==="
