#!/usr/bin/env python3
"""judge_logits.py — M4.4 判据: 设备 prefill 输出 vs HF fp32 golden

设备输出 = output_temp 的 f16 logits [seq=32, vocab=248320]
HF golden = cpu_golden/hf/logits_N.f32.raw [32, 248320] f32
判据 (M4.2 值差纪律, 长链禁位判据):
  cos    全局余弦相似度          期望 ≥ 0.9999
  max|d| 最大绝对值差            参考值(供判读, 不设硬门)
  top1   逐位置 argmax 一致率    期望 32/32 (next-token 可用性)
用法: judge_logits.py <device_out.f16.raw> [prompt_idx 0..3, 默认0] [--all]
"""
import sys
import numpy as np

VOCAB = 248320
SEQ = 32
GOLD = "/disk2/GEHTP/test_models/qwen35_08b/cpu_golden"


def load_dev(path):
    a = np.fromfile(path, dtype=np.float16).astype(np.float32)
    if a.size == VOCAB:  # 仅末位置 logits 的情形
        a = a.reshape(1, VOCAB)
    elif a.size == SEQ * VOCAB:
        a = a.reshape(SEQ, VOCAB)
    else:
        sys.exit(f"device output {a.size} elems, 既非 {VOCAB} 也非 {SEQ*VOCAB}")
    return a


def main():
    dev_path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "--all" else 0
    dev = load_dev(dev_path)
    gold = np.fromfile(f"{GOLD}/hf/logits_{idx}.f32.raw", dtype=np.float32).reshape(SEQ, VOCAB)
    if dev.shape[0] == 1:  # 末位置对齐
        gold = gold[-1:, :]
    flat_d, flat_g = dev.ravel(), gold.ravel()
    cos = float(flat_d @ flat_g / (np.linalg.norm(flat_d) * np.linalg.norm(flat_g) + 1e-30))
    maxd = float(np.max(np.abs(flat_d - flat_g)))
    top1_dev = dev.argmax(axis=1)
    top1_gold = gold.argmax(axis=1)
    agree = int((top1_dev == top1_gold).sum())
    print(f"prompt{idx}: shape={dev.shape} cos={cos:.6f} max|d|={maxd:.4f} "
          f"top1={agree}/{dev.shape[0]}")
    ok = cos >= 0.9999 and agree == dev.shape[0]
    print("PASS" if ok else "CHECK (供判读: 看 top1 错位位置与 cos 量级)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
