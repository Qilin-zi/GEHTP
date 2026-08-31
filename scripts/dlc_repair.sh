#!/bin/bash
# dlc_repair.sh — 修复 qairt-converter 2.48 产出的 zip(DLC 容器)零 CRC 问题
# =====================================================================
# 实测事实: 2.48 converter 写的 zip 所有成员 CRC 字段为 0,
# 官方 libQnnModelDlc / qairt-net-run 因此 "Failed to construct"。
# 修复 = 逐成员解出(unzip -p 容忍坏 CRC)重打包为 ZIP_STORED(正确 CRC)。
# 用法: dlc_repair.sh <input.zip/dlc> <output.dlc>
set -euo pipefail
IN="${1:?usage: dlc_repair.sh <input> <output>}"
OUT="${2:?}"
command -v unzip >/dev/null || { echo "need unzip"; exit 1; }

NAMES=("dlc.metadata.history.2.3.0" "dlc.metadata2.3.0" "model" "model.params" "model.params.bin")
python3 - "$IN" "$OUT" <<'EOF'
import subprocess, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
names = ["dlc.metadata.history.2.3.0", "dlc.metadata2.3.0",
         "model", "model.params", "model.params.bin"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for n in names:
        # unzip 遇坏 CRC 返回码 1 或 2(警告)但内容完整;stdout 非空即接受
        r = subprocess.run(["unzip", "-p", src, n], capture_output=True)
        assert len(r.stdout) > 0, f"unzip {n} rc={r.returncode}: {r.stderr[:200]}"
        z.writestr(n, r.stdout)
print("repaired ->", out)
EOF
