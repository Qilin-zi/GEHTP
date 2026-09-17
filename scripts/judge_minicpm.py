#!/usr/bin/env python3
"""judge_minicpm.py — C4 判据: minicpm5_2b 设备输出 vs gold.f16.raw

判据(PORTAL §3 纪律: 长链 f16 用值差不位判):
  1. logits 值差 max|d| 与均值 (期望量级参考 L3 基准 0.000488, 全模型 2492 op 链更长, 实测记录)
  2. cos 相似度 (期望 ≥0.9999)
  3. top1 argmax 逐位置命中率 (期望 16/16)
用法: judge_minicpm.py --out <设备输出.f16.raw> --gold <gold.f16.raw> [--shape 1,16,130560]
"""
import argparse
import numpy as np


def load_raw(path, n):
    """按文件大小嗅探 dtype: n*2=f16, n*4=f32/int32(位模式经 f16 不可能)。返回 f32 数组。"""
    import os
    sz = os.path.getsize(path)
    if sz == n * 2:
        return np.fromfile(path, dtype=np.float16).astype(np.float32)
    if sz == n * 4:
        return np.fromfile(path, dtype=np.float32)
    raise SystemExit(f"[FAIL] {path} 字节数 {sz} 既非 {n}×2 也非 ×4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--shape", default="1,16,130560")
    args = ap.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))
    n = int(np.prod(shape))

    out = load_raw(args.out, n)
    gold = load_raw(args.gold, n)
    out = out.reshape(shape)
    gold = gold.reshape(shape)

    d = np.abs(out - gold)
    denom = np.abs(gold).max()
    cos = float(np.dot(out.ravel(), gold.ravel()) /
                (np.linalg.norm(out) * np.linalg.norm(gold) + 1e-30))
    print(f"max|d|   = {d.max():.6f}  (gold |max|={denom:.3f}, 相对={d.max()/denom:.6f})")
    print(f"mean|d|  = {d.mean():.6f}")
    print(f"cos      = {cos:.7f}")

    o_top = out.argmax(-1).ravel()   # [16]
    g_top = gold.argmax(-1).ravel()
    hit = int((o_top == g_top).sum())
    print(f"top1     = {hit}/{o_top.size}  out_ids={o_top.tolist()}")
    print(f"           gold_ids={g_top.tolist()}")

    ok = (cos >= 0.9999) and (hit == o_top.size)
    print("[PASS] minicpm5_2b 判据全达" if ok else "[FAIL] 判据未达")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
