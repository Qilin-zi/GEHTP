#!/usr/bin/env python3
"""golden_f16_weights.py — f16 权重域 golden (G1 判据校准用)

host_run 的权重 = qairt-converter --float_bitwidth 16 的 f16 转换,
而 cpu_golden/hf 是 fp32 权重前向 → 两域系统差 (实测 cos 0.9987)。
本脚本: 同 QPM 模型代码 + 同种子 prompts, 但全部权重先 round-trip
f32→f16→f32 (等价 f16 转换权重), 前向仍 f32 —— 产 host 链的同域真值。

用法: golden_f16_weights.py --model-dir /4090disk2/Qwen3.5-0.8B --out /tmp/golden_f16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_qwen35 import build_model  # noqa: E402  (复用 hf 装载路径)
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--n-prompts", type=int, default=4)
    args = ap.parse_args()

    model, cfg = build_model(args.model_dir, "hf")  # hf fp32 装载
    # 全权重 f16 round-trip (等价 converter f16 转换)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.data.half().float())
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(42)  # 与 golden_qwen35.run_prompts 同种子
    for i in range(args.n_prompts):
        ids = rng.integers(0, 2000, (1, args.seq), dtype=np.int64)
        with torch.no_grad():
            logits = model(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.ones(1, args.seq, dtype=torch.long),
                position_ids=torch.arange(args.seq, dtype=torch.long).unsqueeze(0),
                use_cache=False,
            ).logits.float().numpy()
        logits.astype(np.float32).tofile(os.path.join(args.out, f"logits_{i}.f32.raw"))
        ids.tofile(os.path.join(args.out, f"input_{i}.i64.raw"))
        print(f"[golden-f16w] prompt {i}: logits {logits.shape} mean={logits.mean():.6f}")


if __name__ == "__main__":
    main()
