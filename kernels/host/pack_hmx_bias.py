#!/usr/bin/env python3
"""pack_hmx_bias.py — 005 US3b (T019) 合成 blob: 32 对齐 + bias 的 matmul

OP_MATMUL_F16_BIAS (opcode 36) 单 op, M=32/K=1024/N=2048 (全 32 对齐 → HMX 候选形)。
编译器发射门 (US3a) 对该组合硬报错, 故走本低阶组包器 (ver 1 blob, 16B 槽记录,
同 pack_oplist.cc 布局)。数据 LCG 确定性生成。
用法: pack_hmx_bias.py <out_dir>
产物: blob_hmx_bias.wtop + manifest_hmx_bias.json
"""
import json
import struct
import sys
import os

import numpy as np

M, K, N = 32, 1024, 2048
OP_MATMUL_F16_BIAS = 36


def lcg_u32(state):
    while True:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        yield state[0]


def gen(shape, dtype_scale, seed):
    st = [seed]
    g = lcg_u32(st)
    x = np.empty(int(np.prod(shape)), dtype=np.float32)
    for i in range(x.size):
        x[i] = (next(g) / 0xFFFFFFFF) * 2.0 - 1.0
    x = x.reshape(shape) * dtype_scale
    return x.astype(np.float16)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    act = gen((M, K), 1.0, 101)       # [-1, 1]
    wgt = gen((K, N), 0.05, 202)      # [-0.05, 0.05] (acc 量级 ~0.9)
    bias = gen((N,), 2.0, 303)        # [-2, 2]
    slots = [act.tobytes(), wgt.tobytes(), bias.tobytes()]

    blob = bytearray(b"WTOP")
    blob += struct.pack("<HH", 1, 0x1234)          # ver 1, endian chk
    blob += struct.pack("<II", len(slots), 1)      # n_slots, n_ops
    offs, cur = [], 0
    for s in slots:
        offs.append(cur)
        cur += len(s)
        cur = (cur + 127) & ~127
    for i, s in enumerate(slots):
        blob += struct.pack("<IIII", len(s), 1, offs[i], 0)
    args = [0x8000 | 0, 0x8000 | 1, 0, M, K, N, 0, 0x8000 | 2]
    blob += struct.pack("<HH", OP_MATMUL_F16_BIAS, len(args))
    for a in args:
        blob += struct.pack("<I", a)
    while len(blob) % 128:
        blob += b"\0"
    weight_off = len(blob)
    for i, s in enumerate(slots):
        while len(blob) < weight_off + offs[i]:
            blob += b"\0"
        blob += s

    path = os.path.join(out, "blob_hmx_bias.wtop")
    open(path, "wb").write(blob)
    man = {
        "tag": "hmx_bias", "ver": 1, "endian_chk": 0x1234,
        "n_slots": len(slots), "n_ops": 1, "blob_bytes": len(blob),
        "weight_off": weight_off,
        "slots": [{"i": i, "len": len(s), "offset": offs[i]} for i, s in enumerate(slots)],
        "ops": [{"i": 0, "opcode": OP_MATMUL_F16_BIAS, "args": args}],
        "MKN": [M, K, N],
    }
    open(os.path.join(out, "manifest_hmx_bias.json"), "w").write(json.dumps(man, indent=1))
    print(f"[pack] {path}: {len(blob)} bytes, M={M} K={K} N={N}")


if __name__ == "__main__":
    main()
