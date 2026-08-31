#!/usr/bin/env python3
"""gen_smallm_assets.py — V2.2 example 18 (small-M GEMV pad-256) 资产生成

从 V2.2 assets/s256 (t10 256³ 资产) 解码 act v0/v1 → 取真实行 → pad 到 M=256
(raw=32768) → 重打包 crouton surface:
  assets/smallm/act_p1_v0.raw   (真实 M=1, 行0 = v0 行0)
  assets/smallm/act_p1_v1.raw   (真实 M=1, 行0 = v1 行0; pad 行不变性对照)
  assets/smallm/act_p16_v0.raw  (真实 M=16, 行0..15 = v0 行0..15)

设备侧用例无需数值金标: 判据 = p1 行0 与 full-256 行0 byte-exact +
两次 p1 (v0/v1) 的 pad 行 byte-equal + M=1 与 M=256 invoke 同价。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv_crouton16 import inv_crouton16_bytes, pack_a16_crouton16_row4_surface

LIB = Path(__file__).resolve().parents[1]
SRC = LIB / "assets" / "s256"
OUT = LIB / "assets" / "smallm"
M = K = N = 256
PAD = 32768


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = inv_crouton16_bytes((SRC / "act_variants" / "v0.raw").read_bytes(), M, K)
    v1 = inv_crouton16_bytes((SRC / "act_variants" / "v1.raw").read_bytes(), M, K)
    assert v0.shape == (M, K) and v1.shape == (M, K)

    def emit(name, rows):
        act = np.full((M, K), PAD, dtype=np.uint16)
        act[:rows.shape[0]] = rows
        surf = pack_a16_crouton16_row4_surface(act)
        assert np.array_equal(inv_crouton16_bytes(surf.tobytes(), M, K), act), "round-trip"
        (OUT / name).write_bytes(surf.tobytes())
        print(f"{name}: rows={rows.shape[0]} pad_rows={M - rows.shape[0]} bytes={surf.nbytes}")

    emit("act_p1_v0.raw", v0[:1])
    emit("act_p1_v1.raw", v1[:1])
    emit("act_p16_v0.raw", v0[:16])
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
