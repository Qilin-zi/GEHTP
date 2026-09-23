#!/usr/bin/env python3
"""golden_qwen3_06b_q4.py — Qwen3-0.6B Q4 域 golden (GEHTP Q4 线)

GGUF Q4_0 反量化权重 → 与 export_qwen3_06b_onnx.py 同源的前向模块 → 4 prompts
logits 金标。镜像 blob 供给:
  - 194 个 Q4_0 matmul (含 token_embd=lm_head) → 反量化 f32;
  - blk.0/1/2.ffn_down 是 Q4_1 (打包器不认, blob 回落 f16 池) → 这里用 HF f16 语义;
  - RMSNorm 用 GGUF F32 (与 f16 池同源)。

输出: q4_logits_fp32.npy [4,32,151936] + 与 HF fp32 金标的 cos/top1 对照
(量化域差距预期 cos≈0.99, top1 高重合 — 这是 Q4 线的正确性口径, 非退化)。

用法: golden_qwen3_06b_q4.py --gguf /disk2/models/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf \
      --model-dir /4090disk2/Qwen3-0.6B --out-dir test_models/qwen3_06b/cpu_golden_q4
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_qwen3_06b_onnx import Qwen3Prefill, tokenize_4  # noqa: E402
from golden_qwen35 import gguf_read_full  # noqa: E402
from safetensors.torch import load_file

GGUF_Q4_0 = 2
GGUF_Q4_1 = 3
GGUF_F32 = 0

ROLE_TO_ATTR = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def f16_to_f32(h):
    sign = (h & 0x8000) << 16
    ex = (h >> 10) & 0x1F
    mn = h & 0x3FF
    if ex == 0:
        u = sign
    elif ex == 31:
        u = sign | 0x7F800000 | (mn << 13)
    else:
        u = sign | ((ex - 15 + 127) << 23) | (mn << 13)
    return struct.unpack("<f", struct.pack("<I", u))[0]


def dequant_q4_0(data, ne0, ne1):
    """GGUF Q4_0 → f32 [ne1, ne0] (numpy 序 = ne 反转)。"""
    n_blocks = ne0 // 32
    out = np.zeros((ne1, ne0), dtype=np.float32)
    pos = 0
    for i in range(ne1):
        for b in range(n_blocks):
            d = f16_to_f32(struct.unpack("<H", data[pos:pos + 2])[0])
            nib = data[pos + 2:pos + 18]
            for j in range(16):
                out[i, b * 32 + j] = d * ((nib[j] & 0xF) - 8)
                out[i, b * 32 + 16 + j] = d * ((nib[j] >> 4) - 8)
            pos += 18
    assert pos == len(data), f"Q4_0 尺寸不符 {pos} vs {len(data)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", default="/4090disk2/Qwen3-0.6B")
    ap.add_argument("--out-dir", default="test_models/qwen3_06b/cpu_golden_q4")
    args = ap.parse_args()

    gguf = open(args.gguf, "rb").read()
    arch, tensors, data_start = gguf_read_full(args.gguf)

    # 反量化 Q4_0 权重。GGUF ne 序: ne0=最快维=in(K) (attn_output 亦如此:
    # dims=(2048,1024) → numpy [1024,2048] = HF o_proj [out=1024,in=2048]);
    # numpy 序 (ne 反转) 即 HF [out,in], 直接可用。
    sd = {}
    for name, (gtype, dims, offset, nbytes) in tensors.items():
        if gtype != GGUF_Q4_0:
            continue
        ne0, ne1 = dims[0], dims[1]
        w = dequant_q4_0(gguf[data_start + offset:data_start + offset + nbytes], ne0, ne1)
        if name == "token_embd.weight":
            sd["model.embed_tokens.weight"] = torch.from_numpy(w)  # lm_head (tied)
        else:
            parts = name.split(".")
            layer = int(parts[1])
            role = ROLE_TO_ATTR[parts[2]]
            sd[f"model.layers.{layer}.{role}.weight"] = torch.from_numpy(w)
    # F32 norms
    for name, (gtype, dims, offset, nbytes) in tensors.items():
        if gtype != GGUF_F32:
            continue
        w = np.frombuffer(gguf[data_start + offset:data_start + offset + nbytes],
                          dtype=np.float32)
        if name == "output_norm.weight":
            sd["model.norm.weight"] = torch.from_numpy(w)
        else:
            parts = name.split(".")
            layer = int(parts[1])
            if parts[2] == "attn_norm":
                sd[f"model.layers.{layer}.input_layernorm.weight"] = torch.from_numpy(w)
            elif parts[2] == "ffn_norm":
                sd[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.from_numpy(w)
            elif parts[2] == "attn_q_norm":
                sd[f"model.layers.{layer}.self_attn.q_norm.weight"] = torch.from_numpy(w)
            elif parts[2] == "attn_k_norm":
                sd[f"model.layers.{layer}.self_attn.k_norm.weight"] = torch.from_numpy(w)

    # Q4_1 ×3 → HF f16 语义 (镜像 blob 的 f16 池回落)
    hf = load_file(os.path.join(args.model_dir, "model.safetensors"))
    for l in (0, 1, 2):
        k = f"model.layers.{l}.mlp.down_proj.weight"
        sd[k] = hf[k].to(torch.float32).contiguous()
    hf_embed = hf["model.embed_tokens.weight"].to(torch.float32).contiguous()
    # 注意: sd 里的 embed 保持 Q4 token_embd (lm_head 用它);
    # 输入侧查表用 HF embed (与设备输入 embed_p*.f16.raw 同源)

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    m = Qwen3Prefill(sd, cfg)
    m.bake(32)
    m.eval()

    ids = tokenize_4(args.model_dir, 32)
    with torch.no_grad():
        outs = []
        for j in range(ids.shape[0]):                   # 模块为 batch-1 编写
            emb = hf_embed[ids[j:j + 1]]                # [1,32,1024] fp32
            outs.append(m(emb))                         # [1,32,V]
        out = torch.cat(outs, dim=0)                    # [4,32,V]
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "q4_logits_fp32.npy"), out.numpy())

    # 与 HF fp32 金标对照 (量化域口径)
    hf_gold = np.load(os.path.join(os.path.dirname(args.out_dir),
                                   "cpu_golden/hf_logits_fp32.npy"))
    a = out.numpy().astype(np.float64).reshape(4, -1)
    b = hf_gold.astype(np.float64).reshape(4, -1)
    cos = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    top1 = (out.argmax(-1) == torch.from_numpy(hf_gold).argmax(-1)).float().mean().item()
    print(f"[q4 golden] vs HF fp32: cos={cos:.6f} top1={top1*100:.2f}% "
          f"(量化域口径, 预期 cos≈0.99)")
    print(f"[q4 golden] wrote {args.out_dir}/q4_logits_fp32.npy {tuple(out.shape)}")


if __name__ == "__main__":
    main()
