#!/usr/bin/env python3
"""judge_qwen3_06b.py — Qwen3-0.6B 设备 f16 logits 判定 (GEHTP 最小 LLM 线)

判据 (对齐 0.8B 计划 G3 门): 每 prompt cos>=0.9999 且 top1 32/32 全中。
设备输出 = output_temp 274, f16 raw [1,32,151936] (9723904 B)。
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)      # hf_logits_fp32.npy [4,32,151936]
    ap.add_argument("--device-dir", required=True)  # out_p{j}.f16.raw
    ap.add_argument("--cos-th", type=float, default=0.9999)
    args = ap.parse_args()

    g = np.load(args.golden).astype(np.float64)     # [4,32,V]
    n_pass = 0
    for j in range(4):
        try:
            h = np.fromfile(f"{args.device_dir}/out_p{j}.f16.raw", dtype=np.float16)
        except OSError as e:
            print(f"[p{j}] MISSING: {e}")
            continue
        want = g.shape[1] * g.shape[2]
        if h.size != want:
            print(f"[p{j}] SIZE MISMATCH: got {h.size} elems, want {want} — 输出截断/错误")
            continue
        h = h.reshape(1, *g.shape[1:]).astype(np.float64)
        a, b = h.ravel(), g[j].ravel()
        cos = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        top1 = int((h.argmax(-1) == g[j:j + 1].argmax(-1)).sum())
        vd = float(np.abs(a - b).max())
        ok = cos >= args.cos_th and top1 == g.shape[1]
        n_pass += ok
        print(f"[p{j}] cos={cos:.6f} top1={top1}/{g.shape[1]} max_valdiff={vd:.5f} -> {'PASS' if ok else 'FAIL'}")
    print(f"=== DEVICE GATE: {n_pass}/4 {'ALL GREEN' if n_pass == 4 else 'NOT GREEN'} ===")


if __name__ == "__main__":
    main()
