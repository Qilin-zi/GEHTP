#!/usr/bin/env python3
"""check_device_golden.py — 设备输出 vs golden 数值门 (验收闭环 T0)

判据: 全局余弦 cos / 最大绝对差 max|d| / 逐行 argmax 一致率 top1。
阈值显式给 (--cos-min/--top1-min), 不达标 exit 1 — 供 CI/脚本串联。

共享基线口径 (gehtp-cross-session-coordination): 0.8B 线 cos 门 = 0.998646 不回归;
judge_logits.py 的 0.9999 是 M4.4 期望值不是验收门, 勿混用。

单文件对拍:
  check_device_golden.py --device out.f16.raw --gold logits_0.f32.raw \
      --gold-dtype f32 --cos-min 0.998646 --top1-min 32/32
批量对拍 (HF golden 目录, 同 judge_logits 口径):
  check_device_golden.py --device out.f16.raw \
      --gold-pattern "test_models/qwen35_08b/cpu_golden/hf/logits_{i}.f32.raw" \
      --gold-dtype f32 --prompts 4 --cos-min 0.998646
"""
import argparse
import glob
import sys

import numpy as np


def load_raw(path, dtype):
    dt = {"f16": np.float16, "f32": np.float32}[dtype]
    return np.fromfile(path, dtype=dt).astype(np.float32)


def compare(dev, gold):
    if dev.size != gold.size:
        sys.exit(f"size mismatch: device {dev.size} vs gold {gold.size} elems")
    d, g = dev.ravel(), gold.ravel()
    n_bad = int(np.count_nonzero(~np.isfinite(d)))
    if n_bad:
        sys.exit(f"device 输出含 {n_bad} 个 NaN/Inf — 数值已坏, 无需比 golden")
    cos = float(d @ g / (np.linalg.norm(d) * np.linalg.norm(g) + 1e-30))
    maxd = float(np.max(np.abs(d - g)))
    rows = dev.shape[0] if dev.ndim == 2 else 1
    if rows > 1:
        dd = dev if dev.ndim == 2 else dev.reshape(1, -1)
        gg = gold if gold.ndim == 2 else gold.reshape(1, -1)
        agree = int((dd.argmax(axis=1) == gg.argmax(axis=1)).sum())
    else:
        agree = 1 if d.argmax() == g.argmax() else 0
    return cos, maxd, agree, rows


def parse_top1(spec):
    hit, total = spec.split("/")
    return int(hit), int(total)


def main():
    ap = argparse.ArgumentParser(description="设备输出 vs golden 数值门")
    ap.add_argument("--device", required=True, help="设备输出 raw (gehtp run 拉回)")
    ap.add_argument("--device-dtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--gold", help="单个 golden raw 文件")
    ap.add_argument("--gold-dtype", default="f32", choices=["f16", "f32"],
                    help="HF golden 惯例 f32; 设备自对拍用 f16")
    ap.add_argument("--gold-pattern", help="批量: 含 {i} 的通配, i=0..prompts-1")
    ap.add_argument("--prompts", type=int, default=0)
    ap.add_argument("--cos-min", type=float, required=True,
                    help="cos 阈值; 0.8B 基线口径 0.998646")
    ap.add_argument("--top1-min", default=None,
                    help="形如 32/32; 省略则不设 top1 门")
    ap.add_argument("--rows", type=int, default=0,
                    help="device 多行 logits 时行数 (0=单行/平铺)")
    args = ap.parse_args()

    if not args.gold and not args.gold_pattern:
        ap.error("--gold 或 --gold-pattern 必给其一")
    if args.gold_pattern and "{i}" not in args.gold_pattern:
        ap.error("--gold-pattern 需含 {i}")

    dev = load_raw(args.device, args.device_dtype)
    if args.rows > 1:
        if dev.size % args.rows:
            sys.exit(f"device {dev.size} elems 不能按 rows={args.rows} 整除")
        dev = dev.reshape(args.rows, dev.size // args.rows)

    top1_hit = top1_total = None
    if args.top1_min:
        top1_hit, top1_total = parse_top1(args.top1_min)

    pairs = []
    if args.gold:
        pairs.append(("", args.gold))
    else:
        for i in range(args.prompts):
            p = args.gold_pattern.format(i=i)
            if not glob.glob(p):
                sys.exit(f"golden 缺失: {p}")
            pairs.append((f"prompt{i} ", p))

    failed = False
    for tag, gpath in pairs:
        gold = load_raw(gpath, args.gold_dtype)
        dev_c = dev
        if dev.ndim == 2 and gold.size == dev.size:
            gold = gold.reshape(dev.shape)
        elif dev.ndim == 2 and gold.size == dev.shape[1]:
            dev_c = dev[-1:]  # golden 仅末位置 → 只比末行
        cos, maxd, agree, rows = compare(dev_c, gold)
        line = (f"{tag}cos={cos:.6f} max|d|={maxd:.4f} top1={agree}/{rows}")
        ok = cos >= args.cos_min
        if top1_total is not None:
            need = top1_total if len(pairs) == 1 else rows
            line += f" (门 {top1_hit}/{top1_total})" if len(pairs) == 1 else ""
            ok = ok and agree >= (top1_hit if len(pairs) == 1 else need)
        print(("PASS " if ok else "FAIL ") + line)
        failed = failed or not ok

    if failed:
        print("GATE FAIL — 数值回归, 勿部署", file=sys.stderr)
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
