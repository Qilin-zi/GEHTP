#!/usr/bin/env bash
# run_host_ci.sh — host 侧 CI 门 (无设备): 构建 + ctest + 数值门冒烟
# 供 .github/workflows/ci.yml 与本地同口径使用。
# 设备侧验收门 (golden cos / op_ts 交叉校验) 见 check_device_golden.py /
# check_op_ts.py — 设备产物存在时才跑, 不在本脚本范围。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/compiler/build_linux"

echo "== [1/3] 构建编译器 =="
cmake -S "$ROOT/compiler" -B "$BUILD" >/dev/null
cmake --build "$BUILD" -j"$(nproc)" >/dev/null

echo "== [2/3] ctest =="
ctest --test-dir "$BUILD" --output-on-failure

echo "== [3/3] golden checker 冒烟 (正例 + 负例) =="
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 - "$TMP" <<'EOF'
import sys, numpy as np
tmp = sys.argv[1]
rng = np.random.default_rng(7)
a = rng.standard_normal((4, 64)).astype(np.float16)
a.tofile(f"{tmp}/gold.f16.raw")
(a + rng.standard_normal((4, 64)).astype(np.float16) * 0.01).tofile(f"{tmp}/dev_ok.f16.raw")
(a + 3.0).tofile(f"{tmp}/dev_bad.f16.raw")  # 平移: cos 明显 < 1
EOF
# 正例: 近似副本须过门
python3 "$ROOT/scripts/check_device_golden.py" \
    --device "$TMP/dev_ok.f16.raw" --gold "$TMP/gold.f16.raw" \
    --gold-dtype f16 --cos-min 0.99 --rows 4
# 负例: 平移样本须 fail (exit 1), 反转 && 保证脚本在负例失败时报错退出
if python3 "$ROOT/scripts/check_device_golden.py" \
        --device "$TMP/dev_bad.f16.raw" --gold "$TMP/gold.f16.raw" \
        --gold-dtype f16 --cos-min 0.99 --rows 4 >/dev/null 2>&1; then
    echo "FATAL: 负例未拦截 (checker 失效)" >&2
    exit 1
fi

echo "HOST CI PASS"
