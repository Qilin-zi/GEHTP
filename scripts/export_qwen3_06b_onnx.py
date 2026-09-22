#!/usr/bin/env python3
"""export_qwen3_06b_onnx.py — Qwen3-0.6B 香草 prefill ONNX 导出 (GEHTP 最小 LLM 线)

与 export_qwen35_onnx.py (Qwen3.5 混合 GDN) 的差异:
  - 纯 torch 手写前向 (~标准 LLaMA 类: RMSNorm+RoPE+GQA+SwiGLU), 不依赖 transformers
    modeling 版本 (tf 5.7 masking_utils API 漂移风险归零);
  - **embedding 折叠到 host 侧**: 图输入 = inputs_embeds fp32 [1,seq,1024],
    绕开引擎 f16-only 边界与 int Gather/Cast 暗雷 (A3);
  - position_ids / causal mask / RoPE cos/sin 全部烘焙为常量缓冲 (seq 固定);
  - 输出 = 全 logits fp32 [1,seq,151936] (converter --float_bitwidth 16 落到 f16)。

自检门: ① torch 模块 vs HF Qwen3ForCausalLM (fp32) logits cos≥0.999999 + top1 全同
        ② ORT (fp32) vs torch 模块 cos≥0.999999
golden 模式另产: 4 prompts 的 inputs_embeds f16 raw (设备输入) + HF fp32 logits 金标。

用法:
  export_qwen3_06b_onnx.py export --model-dir /4090disk2/Qwen3-0.6B --seq 32 \
      --out test_models/qwen3_06b/model.onnx
  export_qwen3_06b_onnx.py golden --model-dir /4090disk2/Qwen3-0.6B --seq 32 \
      --out-dir test_models/qwen3_06b/cpu_golden
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL_DIR_DEF = "/4090disk2/Qwen3-0.6B"


# --------------------------------------------------------------------------
# 纯 torch Qwen3 prefill (手写, 无 transformers 依赖)
# --------------------------------------------------------------------------
class Qwen3Prefill(nn.Module):
    """inputs_embeds [1,S,1024] fp32 → logits [1,S,151936] fp32。embedding host 折叠。"""

    def __init__(self, sd: dict, cfg: dict):
        super().__init__()
        L = cfg["num_hidden_layers"]
        H = cfg["hidden_size"]
        nh = cfg["num_attention_heads"]
        nkv = cfg["num_key_value_heads"]
        dh = cfg["head_dim"]
        eps = cfg["rms_norm_eps"]
        theta = cfg["rope_theta"]
        self.L, self.H, self.nh, self.nkv, self.dh, self.eps = L, H, nh, nkv, dh, eps
        self.scale = 1.0 / (dh ** 0.5)
        self.seq = None  # set by bake()

        def w(name):
            return torch.nn.Parameter(sd[name].to(torch.float32).contiguous(), requires_grad=False)

        self.embed = w("model.embed_tokens.weight")          # [V,H] 仅 host 侧用 + lm_head 共享
        self.final_norm = w("model.norm.weight")
        for i in range(L):
            p = f"model.layers.{i}."
            for k in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                      "self_attn.q_norm", "self_attn.k_norm",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                      "input_layernorm", "post_attention_layernorm"):
                setattr(self, f"l{i}_{k.replace('.', '_')}", w(p + k + ".weight"))

    def bake(self, seq: int):
        """烘焙 RoPE cos/sin 与因果掩码为常量缓冲。"""
        self.seq = seq
        half = self.dh // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                       # [S,half]
        emb = torch.cat([freqs, freqs], dim=-1)                # [S,dh]
        self.register_buffer("rope_cos", emb.cos()[None, None], persistent=False)   # [1,1,S,dh]
        self.register_buffer("rope_sin", emb.sin()[None, None], persistent=False)
        row = torch.arange(seq).unsqueeze(1)
        col = torch.arange(seq).unsqueeze(0)
        keep = row >= col
        mask = torch.where(keep, torch.zeros(()), torch.full((), -30000.0))  # f16 安全下限内
        self.register_buffer("cmask", mask[None, None], persistent=False)          # [1,1,S,S]

    @staticmethod
    def _rms(x, w, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def _rope(self, x):  # x [1,nh,S,dh]
        half = self.dh // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * self.rope_cos + rot * self.rope_sin

    def forward(self, inputs_embeds):  # [1,S,H] fp32
        S = inputs_embeds.shape[1]
        x = inputs_embeds
        for i in range(self.L):
            g = lambda k: getattr(self, f"l{i}_{k}")
            # --- attention ---
            h = self._rms(x, g("input_layernorm"), self.eps)
            q = h @ g("self_attn_q_proj").t()            # [1,S,nh*dh]
            k = h @ g("self_attn_k_proj").t()
            v = h @ g("self_attn_v_proj").t()
            q = q.view(1, S, self.nh, self.dh).transpose(1, 2)  # [1,nh,S,dh]
            k = k.view(1, S, self.nkv, self.dh).transpose(1, 2)
            v = v.view(1, S, self.nkv, self.dh).transpose(1, 2)
            # QK norm (Qwen3 特征): 每头 dh 维 RMSNorm
            q = self._rms(q, g("self_attn_q_norm"), self.eps)
            k = self._rms(k, g("self_attn_k_norm"), self.eps)
            q = self._rope(q)
            k = self._rope(k)
            rep = self.nh // self.nkv
            k = k.unsqueeze(2).expand(1, self.nkv, rep, S, self.dh).reshape(1, self.nh, S, self.dh)
            v = v.unsqueeze(2).expand(1, self.nkv, rep, S, self.dh).reshape(1, self.nh, S, self.dh)
            att = torch.matmul(q, k.transpose(-1, -2)) * self.scale + self.cmask
            att = torch.softmax(att, dim=-1)
            att = torch.matmul(att, v)                          # [1,nh,S,dh]
            att = att.transpose(1, 2).reshape(1, S, self.nh * self.dh)
            x = x + att @ g("self_attn_o_proj").t()
            # --- MLP (SwiGLU) ---
            h = self._rms(x, g("post_attention_layernorm"), self.eps)
            gate = torch.nn.functional.silu(h @ g("mlp_gate_proj").t())
            up = h @ g("mlp_up_proj").t()
            x = x + (gate * up) @ g("mlp_down_proj").t()
        x = self._rms(x, self.final_norm, self.eps)
        return x @ self.embed.t()                               # tied lm_head [1,S,V]


def load_sd(model_dir):
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return sd, cfg


def build(model_dir, seq):
    sd, cfg = load_sd(model_dir)
    m = Qwen3Prefill(sd, cfg)
    m.bake(seq)
    m.eval()
    return m, cfg


def hf_golden_logits(model_dir, input_ids):
    """HF 参考前向 (fp32), 返回 [1,S,V] logits。"""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        return hf(input_ids=input_ids).logits


def make_embeds(model_dir, input_ids):
    """host 侧 embedding 查表: input_ids [1,S] int64 → f16 raw [1,S,1024]。"""
    sd, _ = load_sd(model_dir)
    emb = sd["model.embed_tokens.weight"].to(torch.float32)
    return emb[input_ids]                                      # [1,S,H] fp32


PROMPTS = [
    "The capital of France is",
    "1+1=",
    "List three colors:",
    "Q: What is the largest planet in the Solar System? A:",
]


def tokenize_4(model_dir, seq):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_dir)
    out = []
    for p in PROMPTS:
        ids = tk(p, return_tensors=None)["input_ids"][:seq]
        ids = ids + [tk.eos_token_id] * (seq - len(ids))       # 右 pad eos
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)                 # [4,S]


def cmd_export(args):
    m, cfg = build(args.model_dir, args.seq)
    x = torch.zeros(1, args.seq, cfg["hidden_size"], dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        m, (x,), args.out,
        input_names=["inputs_embeds"], output_names=["logits"],
        opset_version=args.opset, dynamo=False,
        do_constant_folding=True,
    )
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # --- 自检 1: torch 模块 vs HF ---
    ids = tokenize_4(args.model_dir, args.seq)[:1]             # prompt0
    emb = make_embeds(args.model_dir, ids)
    with torch.no_grad():
        mine = m(emb)
    ref = hf_golden_logits(args.model_dir, ids)
    cos = torch.nn.functional.cosine_similarity(mine.flatten(1).double(), ref.flatten(1).double()).item()
    top1_same = (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"[selfcheck:torch-vs-HF] cos={cos:.8f} top1_match={top1_same*100:.1f}%")
    assert cos >= 0.999999 and top1_same == 1.0, "手写模块与 HF 不一致, 拒绝交付"

    # --- 自检 2: ORT vs torch ---
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"inputs_embeds": emb.numpy()})[0]
    cos2 = np.dot(out.ravel().astype(np.float64), mine.numpy().ravel().astype(np.float64)) / (
        np.linalg.norm(out) * np.linalg.norm(mine.numpy()))
    print(f"[selfcheck:ORT-vs-torch] cos={cos2:.8f}")
    assert cos2 >= 0.999999, "ORT 与 torch 不一致, 拒绝交付"
    print("[export] ALL SELFCHECKS GREEN")


def cmd_golden(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ids = tokenize_4(args.model_dir, args.seq)                 # [4,S]
    ref = hf_golden_logits(args.model_dir, ids)                # [4,S,V] fp32
    np.save(os.path.join(args.out_dir, "hf_logits_fp32.npy"), ref.numpy())
    for j in range(ids.shape[0]):
        emb = make_embeds(args.model_dir, ids[j:j+1])          # [1,S,H] fp32
        emb.to(torch.float16).numpy().tofile(os.path.join(args.out_dir, f"embed_p{j}.f16.raw"))
    meta = {"prompts": PROMPTS, "seq": args.seq, "input_ids": ids.tolist(),
            "judge": "cos>=0.9999 + top1 32/32 per prompt (f16 长链判据)"}
    json.dump(meta, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[golden] wrote {args.out_dir} (hf_logits_fp32.npy {ref.shape} + 4 embed raw)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "golden"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", default=MODEL_DIR_DEF)
        p.add_argument("--seq", type=int, default=32)
        p.add_argument("--opset", type=int, default=17)
        p.add_argument("--out", default="test_models/qwen3_06b/model.onnx")
        p.add_argument("--out-dir", default="test_models/qwen3_06b/cpu_golden")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.cmd == "export":
        cmd_export(args)
    else:
        cmd_golden(args)


if __name__ == "__main__":
    main()
