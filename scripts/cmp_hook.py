#!/usr/bin/env python3
"""cmp_hook.py — 设备 hook dump vs host 参照 dump 逐锚点对拍 (004 US2 取证链)

设备: <dev-dir>/hook_<idx>_<temp>.f16.raw (f16, op 执行完后 temp 原文;
      scripts/gehtp run --hook "idx:temp,..." 拉回到 prof 目录)
host: <host-dir>/host_id_<manifest.op_ids[idx]>.f32.raw (f32, host 正确链 dump)
判读: OK cos>=0.999 / warn cos>=0.99 / BAD 其余; 首个 BAD 锚点 = 发散定位。

对拍纪律: 锚点须取「该 gid 在 runlist 中最后一个写终态 temp 的 op」——
code-26 Eltwise_Binary 等分解中间步与终态同 gid, 中间步 temp 比较必 BAD
(伪影, 非数值 bug; 详见 004 T008 取证)。

用法: cmp_hook.py <spec|"idx:temp,..."|--spec-file f> --manifest m.json \
        --host-dir D [--dev-dir D (默认 /tmp/prof)]
"""
import argparse
import json
import os
import sys

import numpy as np


def load_spec(args):
    # 显式 --spec 必须定义: 否则会被 argparse 前缀匹配吸进 --spec-file
    if args.spec_opt:
        return args.spec_opt.strip()
    if args.spec_file:
        return open(args.spec_file).read().strip()
    return args.spec.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default=None)
    ap.add_argument("--spec", dest="spec_opt", default=None,
                    help="hook spec 字符串 (亦可作位置参数)")
    ap.add_argument("--spec-file")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--host-dir", required=True)
    ap.add_argument("--dev-dir", default="/tmp/prof")
    args = ap.parse_args()
    if not args.spec and not args.spec_opt and not args.spec_file:
        ap.error("需 spec 位置参数、--spec 或 --spec-file")

    m = json.load(open(args.manifest))
    oids, names = m["op_ids"], m["op_names"]
    spec = load_spec(args)

    first_bad = None
    for tok in spec.split(","):
        idx_s, temp_s = tok.split(":")
        idx, temp = int(idx_s), int(temp_s)
        name = names[idx] if idx < len(names) else "?"
        oid = oids[idx] if idx < len(oids) else -1
        dp = os.path.join(args.dev_dir, f"hook_{idx}_{temp}.f16.raw")
        hp = os.path.join(args.host_dir, f"host_id_{oid}.f32.raw")
        if not os.path.exists(dp):
            print(f"op{idx:<6} {name:<18} oid={oid:<6} DEV_MISSING")
            if first_bad is None:
                first_bad = idx
            continue
        if not os.path.exists(hp):
            print(f"op{idx:<6} {name:<18} oid={oid:<6} HOST_MISSING")
            continue
        d = np.fromfile(dp, dtype=np.float16).astype(np.float32)
        h = np.fromfile(hp, dtype=np.float32)
        n = min(d.size, h.size)
        if n == 0:
            print(f"op{idx:<6} {name:<18} oid={oid:<6} EMPTY dev={d.size} host={h.size}")
            continue
        dd, hh = d[:n], h[:n]
        c = float(dd @ hh / (np.linalg.norm(dd) * np.linalg.norm(hh) + 1e-30))
        md = float(np.max(np.abs(dd - hh)))
        if d.size != h.size:
            # 元素数不匹配 = op 少写/多写(Bn/batch/形状 bug 签名, 004 T008 的
            # Bn=1 即 dev 1 元素 vs host 8 元素), 直接判 SIZE 异常
            tag = "SIZE"
        else:
            tag = "OK " if c >= 0.999 else ("warn" if c >= 0.99 else "BAD")
        print(f"op{idx:<6} {name:<18} oid={oid:<6} {tag} cos={c:.6f} max|d|={md:.4f} "
              f"nelem dev={d.size} host={h.size}")
        if tag in ("BAD", "SIZE") and first_bad is None:
            first_bad = idx
    print("---")
    print("首个 BAD 锚点:", first_bad if first_bad is not None else "(全部 OK/无 BAD)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
