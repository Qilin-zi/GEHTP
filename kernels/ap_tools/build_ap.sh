#!/bin/bash
# build_ap.sh — 编 AP 侧 Android ARM 工具 (gehtp_proxy) → build/ap_tools/
# 用法: ./build_ap.sh            # 编 + 签免 (AP 侧无签名要求)
#       DEVICE=<serial> ./build_ap.sh push   # 编 + 推到 DEVROOT
set -euo pipefail
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${HEXAGON_SDK_ROOT:-/disk1/toolchains/Hexagon_SDK/6.6.0.0}"
NDK_CC="$SDK/tools/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang"
DEVROOT="${GEHTP_DEV_TMP:-/data/local/tmp/hvxhmx23}"
BUILD="$LIB/build/ap_tools"
mkdir -p "$BUILD"

echo "=== build gehtp_proxy (aarch64 android, static) ==="
"$NDK_CC" -O2 -Wall -static -o "$BUILD/gehtp_proxy" "$LIB/ap_tools/gehtp_proxy.c"
ls -la "$BUILD/gehtp_proxy"

if [ "${1:-}" = "push" ]; then
    adb -s "${DEVICE:-d0f1784}" push "$BUILD/gehtp_proxy" "$DEVROOT/gehtp_proxy"
    adb -s "${DEVICE:-d0f1784}" shell "chmod 755 $DEVROOT/gehtp_proxy"
fi
