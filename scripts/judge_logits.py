#!/usr/bin/env python3
"""judge_logits.py — 0.8B prefill/decode 精度门 (004 US2 双门, FR-010)

prefill 门:
  judge_logits.py <dev_out.f16.raw> [prompt_idx]
  设备输出 [32,V] vs HF fp32 golden, 基线门 cos>=0.99864 / top1>=29/32。
  (FR-010 字面 0.998643 为单次记录值; 跨构建 f16 噪声带 ±4e-6 —— 同基线族
  实测 0.998642/0.998645/0.998646, 门取族下沿。)

decode 双门:
  judge_logits.py <dev_out.f16.raw> [prompt_idx] --row R --device-prefill <prefill_out.f16.raw>
  设备输出 [1,V] (单 token 前向):
    主门(设备自洽): vs 设备自身 prefill 第 R 行, cos>=0.9999 且 top1 精确
                    (同权重同来源, 精确捕状态 bug, 不依赖 HF)
    基线门(跨设备): vs HF logits_{i} 第 R 行, cos>=0.99864 且 top1 精确
  行锚: P1 无 KV 用 --row 0 (SC-001: seq=1 blob @ pos0 == prefill 首位);
        P2 KV 接线后 --row 31 (SC-002: 真 decode 步 == prefill 末位 == 第 32 token)
"""
import argparse
import sys

import numpy as np

VOCAB = 248320
SEQ = 32
GOLD = "/disk1/gehtp/test_models/qwen35_08b/cpu_golden"
COS_BASELINE = 0.99864   # 基线门下沿 (FR-010; 见模块注释)
TOP1_BASELINE = 29       # /32
COS_SELF = 0.9999        # 设备自洽主门


def load_dev(path):
    a = np.fromfile(path, dtype=np.float16).astype(np.float32)
    if a.size == VOCAB:
        a = a.reshape(1, VOCAB)
    elif a.size == SEQ * VOCAB:
        a = a.reshape(SEQ, VOCAB)
    else:
        sys.exit(f"device output {a.size} elems, 既非 {VOCAB} 也非 {SEQ*VOCAB}")
    return a


def cos(a, b):
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev_out")
    ap.add_argument("prompt_idx", nargs="?", type=int, default=0)
    ap.add_argument("--row", type=int, default=None,
                    help="decode 模式: 对拍 prefill 第 R 行 (P1: 0; P2: 31/末位)")
    ap.add_argument("--device-prefill", default=None,
                    help="decode 主门参照: 设备自身 prefill 输出 [32,V] f16.raw")
    args = ap.parse_args()

    dev = load_dev(args.dev_out)
    gold_full = np.fromfile(f"{GOLD}/hf/logits_{args.prompt_idx}.f32.raw",
                            dtype=np.float32).reshape(SEQ, VOCAB)

    if args.row is None:
        # prefill 门 (整 [32,V])
        c = cos(dev, gold_full)
        agree = int((dev.argmax(axis=1) == gold_full.argmax(axis=1)).sum())
        maxd = float(np.max(np.abs(dev - gold_full)))
        ok = c >= COS_BASELINE and agree >= TOP1_BASELINE
        print(f"prefill{args.prompt_idx}: cos={c:.6f} top1={agree}/{SEQ} max|d|={maxd:.4f} "
              f"(门 cos>={COS_BASELINE}/top1>={TOP1_BASELINE})")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    # decode 双门 ([1,V] vs 第 row 行)
    r = args.row if args.row >= 0 else SEQ + args.row
    g = gold_full[r:r + 1]
    gates = []
    if args.device_prefill:
        pref = load_dev(args.device_prefill)
        if pref.shape[0] != SEQ:
            sys.exit(f"--device-prefill 需 [32,{VOCAB}], got {pref.shape}")
        p = pref[r:r + 1]
        c = cos(dev, p)
        hit = int(dev.argmax() == p.argmax())
        gates.append(("主门(设备自洽, dev vs dev-prefill 行%d)" % r,
                      c, hit, 1, c >= COS_SELF and hit == 1))
    c = cos(dev, g)
    hit = int(dev.argmax() == g.argmax())
    maxd = float(np.max(np.abs(dev - g)))
    gates.append((f"基线门(dev vs HF logits_{args.prompt_idx} 行{r})",
                  c, hit, 1, c >= COS_BASELINE and hit == 1))
    allok = True
    for name, c, hit, n, ok in gates:
        print(f"{name}: cos={c:.6f} top1={hit}/{n} max|d|={maxd:.4f} {'PASS' if ok else 'FAIL'}")
        allok = allok and ok
    print("PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
