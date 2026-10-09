#!/usr/bin/env python3
"""cmp_l0.py — hook 锚点对拍 + last-writer 溯源 (004 US2 取证链)

cmp_hook 的进阶版: 锚点 temp 若非 op idx 自己的输出, 自动找 last-writer op j
(其输出 temp == 该 temp, j < idx), 比 host_id_<oids[j]> —— 免手工挑
「终态检查点」。writer 表由 blob op args + opcode→out_arg 契约构建
(gehtp_prof.parse_wtop, 无外部工具依赖)。

用法: cmp_l0.py <spec|"idx:temp,..."> --wtop blob.wtop --manifest m.json \
        --host-dir D [--dev-dir D]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gehtp_prof import parse_wtop  # noqa: E402

# opcode → 输出 temp 的 args 下标 (oplist_parse.h 注册契约; None = 不产 temp)
# 注意: 与旧 v3-era 编号不同, 以当前 oplist_parse.h 枚举为准。
OUT_ARG = {
    1: 2,   # OP_MATMUL_W4A16
    2: 2,   # OP_RMSNORM_F16
    4: 1,   # OP_SILU_F16
    5: 1,   # OP_IM2COL
    6: 3,   # OP_CONV2D_F16
    7: 2,   # OP_ADD_F16
    10: 1,  # OP_TRANSPOSE_F16
    11: 1,  # OP_UNARY_F16
    12: 2,  # OP_BINARY_F16
    13: 1,  # OP_SOFTMAX_F16
    14: 8,  # OP_CONCAT_F16
    15: 1,  # OP_STRIDED_SLICE_F16
    16: 1,  # OP_SPLIT_F16
    17: 1,  # OP_REDUCE_F16
    18: 1,  # OP_CUMSUM_F32
    19: 2,  # OP_CONV1D_SSM_F16
    20: 2,  # OP_GATHER_F16
    21: 1,  # OP_ARGMAX_F16
    24: 2,  # OP_MATMUL_F16
    25: 3,  # OP_RMSNORM2_F16
    26: 1,  # OP_BROADCAST_F16
    27: 1,  # OP_TRANSPOSE_GEN_F16
    28: 3,  # OP_SCATTER_ND_F16
    34: 1,  # OP_PAD_F16
    35: 2,  # OP_MATMUL_F16_SPLIT
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--wtop", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--host-dir", required=True)
    ap.add_argument("--dev-dir", default="/tmp/prof")
    args = ap.parse_args()

    _, _, ops = parse_wtop(args.wtop)
    m = json.load(open(args.manifest))
    oids, names = m["op_ids"], m["op_names"]

    def out_temp(i):
        code, a = ops[i][0], ops[i][1]
        oa = OUT_ARG.get(code)
        if oa is None or oa >= len(a):
            return None
        t = a[oa]
        return t if t < 0x8000 else None  # slot 引用不进 temp 表

    lastw = {}
    for i in range(len(ops)):
        t = out_temp(i)
        if t is not None:
            lastw.setdefault(t, []).append(i)

    def find_writer(temp, before):
        ws = [i for i in lastw.get(temp, []) if i < before]
        return ws[-1] if ws else None

    first_bad = None
    for tok in args.spec.strip().split(","):
        idx, temp = map(int, tok.split(":"))
        dp = os.path.join(args.dev_dir, f"hook_{idx}_{temp}.f16.raw")
        if not os.path.exists(dp):
            print(f"op{idx:<5} t{temp:<5} DEV_MISSING")
            continue
        d = np.fromfile(dp, dtype=np.float16).astype(np.float32)
        if out_temp(idx) == temp:
            ref_idx, role = idx, "OUT"
        else:
            ref_idx, role = find_writer(temp, idx), "IN(writer)"
        if ref_idx is None:
            print(f"op{idx:<5} t{temp:<5} 无法溯源 writer")
            continue
        hp = os.path.join(args.host_dir, f"host_id_{oids[ref_idx]}.f32.raw")
        if not os.path.exists(hp):
            print(f"op{idx:<5} t{temp:<5} {role} op{ref_idx} HOST_MISSING")
            continue
        h = np.fromfile(hp, dtype=np.float32)
        n = min(d.size, h.size)
        if n == 0:
            print(f"op{idx:<5} t{temp:<5} {role} op{ref_idx} EMPTY")
            continue
        dd, hh = d[:n], h[:n]
        c = float(dd @ hh / (np.linalg.norm(dd) * np.linalg.norm(hh) + 1e-30))
        md = float(np.max(np.abs(dd - hh)))
        tag = "OK " if c >= 0.999 else ("warn" if c >= 0.99 else "BAD")
        print(f"op{idx:<5} t{temp:<5} {role:<9} writer=op{ref_idx:<5} {names[ref_idx]:<16} "
              f"{tag} cos={c:.6f} max|d|={md:.4f}")
        if tag == "BAD" and first_bad is None:
            first_bad = (idx, ref_idx)
    print("---")
    print("首个 BAD 锚点:", first_bad if first_bad is not None else "(全部 OK/无 BAD)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
