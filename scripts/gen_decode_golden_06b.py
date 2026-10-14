#!/usr/bin/env python3
"""gen_decode_golden_06b.py — Qwen3-0.6B 自回归贪心 golden (HF fp32 参考).

与设备 decode 对拍口径: 设备逐 token 喂 (position 0..), 每步 argmax 贪心;
本脚本用 HF fp32 逐步前向 (input_ids = 当前全前缀) 取 logits[-1] argmax,
产出同样的贪心序列 + 每步全 logits (供逐位 cos/top1 取证)。

用法:
  gen_decode_golden_06b.py --model-dir /disk1/qwen3_06b \
      --prompt "The capital of France is" --max-new 32 \
      --out test_models/qwen3_06b/decode/golden_tokens.json
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/disk1/qwen3_06b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out", default="test_models/qwen3_06b/decode/golden_tokens.json")
    ap.add_argument("--save-logits", action="store_true",
                    help="另存每步 logits 行 f32 npy (逐位 cos 取证)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model_dir)
    prompt_ids = tk(args.prompt, return_tensors=None)["input_ids"]
    print(f"[golden] prompt={args.prompt!r} ids={prompt_ids} n={len(prompt_ids)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(dev).eval()
    print(f"[golden] HF fp32 on {dev}")

    all_ids = list(prompt_ids)
    generated = []
    step_logits = []
    with torch.no_grad():
        for step in range(args.max_new):
            inp = torch.tensor([all_ids], dtype=torch.long, device=dev)
            logits = model(input_ids=inp).logits            # [1, T, V]
            row = logits[0, -1].float().cpu().numpy()       # [V] 当前步预测
            nxt = int(row.argmax())
            generated.append(nxt)
            step_logits.append(row)
            all_ids.append(nxt)

    text = tk.decode(all_ids)
    print(f"[golden] generated_ids={generated}")
    print(f"[golden] full={all_ids}")
    print(f"[golden] text={text!r}")

    out = {
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "generated_ids": generated,
        "all_ids": all_ids,
        "max_new": args.max_new,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[golden] wrote {args.out}")

    if args.save_logits:
        np.save(args.out.replace(".json", "_logits.npy"),
                np.stack(step_logits).astype(np.float32))
        print(f"[golden] wrote logits {os.path.basename(args.out).replace('.json','_logits.npy')}")


if __name__ == "__main__":
    main()
