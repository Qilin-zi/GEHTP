#!/usr/bin/env python3
"""replay_attn_layer3.py — 全注意力层 L3 HF 侧手动复刻, 输出全锚点 npy (GEHTP M4.1d)

与 host_run 的 /tmp/host_id_<oid>.f32.raw 逐锚点对拍。
公式 = modeling_qwen3_5.py Qwen3_5Attention(874-938 行)+ DecoderLayer.

用法:
  replay_attn_layer3.py --model-dir /disk2/Qwen3.5-0.8B \
      --layer-dir /disk2/GEHTP/test_models/qwen35_08b/layers/layer_3 \
      --out /tmp/anchor3
"""
import argparse
import os
import sys

QPM_ROOT = "/disk2/Qwen35dev/qwen3.5_4b_base/example1"
sys.path.insert(0, QPM_ROOT)

import transformers.utils.import_utils as _tf_import
_tf_import.is_causal_conv1d_available = lambda *a, **k: False
_tf_import.is_flash_linear_attention_available = lambda *a, **k: False

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import transformers.masking_utils as _mu
import transformers.modeling_attn_mask_utils as _mau
from huggingface.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from huggingface.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def load_remapped(model, model_dir):
    sd_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    raw = load_file(sd_path)
    remapped = {}
    for k, v in raw.items():
        if k.startswith("model.visual.") or k.startswith("mtp."):
            continue
        if k.startswith("model.language_model."):
            remapped["model." + k[len("model.language_model."):]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    tied = {k for k in missing if "lm_head" in k or "embed_tokens" in k}
    if tied:
        model.tie_weights()
        missing = [k for k in missing if k not in tied]
    assert not missing, f"weights missing: {missing}"
    return model


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # transformers GLM 风格(无 interleave): cos/sin [seq, dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(attn, query, key, value, mask, dropout, scaling, **kw):
    q_len = query.shape[2]
    attn_w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        causal_mask = mask[:, :, :, : key.shape[-2]]
        attn_w = attn_w + causal_mask
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn_w, value)
    return out, attn_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Qwen3_5Config.from_pretrained(args.model_dir)
    model = Qwen3_5ForCausalLM(cfg.text_config)
    model = model.to(torch.float32).eval()
    model = load_remapped(model, args.model_dir)

    layer = model.model.layers[3]
    attn = layer.self_attn

    def rd(name):
        return np.fromfile(os.path.join(args.layer_dir, name), dtype=np.float32)

    hidden = torch.from_numpy(rd("layer_3_in.f32.raw")).reshape(1, 32, 1024)
    mask = torch.from_numpy(rd("layer_3_mask.f32.raw")).reshape(1, 1, 32, 32)
    cos = torch.from_numpy(rd("layer_3_cos.f32.raw")).reshape(1, 32, 64)
    sin = torch.from_numpy(rd("layer_3_sin.f32.raw")).reshape(1, 32, 64)

    A = {}
    with torch.no_grad():
        # ---- 注意力(q_norm/k_norm 2 kv 头共享; head_dim=256) ----
        h_norm = layer.input_layernorm(hidden)
        A["rms_in"] = h_norm

        # HF: q_proj → [1,32,4096] = 8 头 × head_dim(256) × 2 (q+gate)
        #      k_proj/v_proj → [1,32,512] = 2 KV 头 × 256
        q_proj = attn.q_proj(h_norm)                                   # [1,32,4096]
        A["q_proj"] = q_proj
        qp_v = q_proj.view(1, 32, 8, 2, 256)
        q = qp_v[..., 0, :]                                            # [1,32,8,256]
        gate_chunk = qp_v[..., 1, :]                                   # [1,32,8,256]
        k_proj = attn.k_proj(h_norm)                                   # [1,32,512]
        A["k_proj"] = k_proj
        k = k_proj.reshape(1, 32, 2, 256)
        v_proj = attn.v_proj(h_norm)                                   # [1,32,512]
        A["v_proj"] = v_proj
        v = v_proj.reshape(1, 32, 2, 256)
        A["q_raw"], A["k_raw"], A["v_raw"] = q, k, v

        q_n = attn.q_norm(q).transpose(1, 2)                           # [1,8,32,256]
        k_n = attn.k_norm(k).transpose(1, 2)                           # [1,2,32,256]
        A["q_norm"], A["k_norm"] = q_n, k_n

        # partial RoPE: 只旋转前 64 维(rotary_dim), 后 192 直通
        rotary_dim = cos.shape[-1]                                     # 64
        c1, s1 = cos.unsqueeze(1), sin.unsqueeze(1)                    # [1,1,32,64]
        q_rot = (q_n[..., :rotary_dim] * c1) + (rotate_half(q_n[..., :rotary_dim]) * s1)
        q_rot = torch.cat([q_rot, q_n[..., rotary_dim:]], dim=-1)
        k_rot = (k_n[..., :rotary_dim] * c1) + (rotate_half(k_n[..., :rotary_dim]) * s1)
        k_rot = torch.cat([k_rot, k_n[..., rotary_dim:]], dim=-1)
        A["q_rot"], A["k_rot"] = q_rot, k_rot

        # GQA: k/v repeat_interleave 8/2=4
        k_rep = k_rot.repeat_interleave(4, dim=1)                      # [1,8,32,256]
        v_rep = v.transpose(1, 2).repeat_interleave(4, dim=1)          # [1,8,32,256]
        A["k_rep"], A["v_rep"] = k_rep, v_rep

        scaling = attn.scaling  # 1/sqrt(256)
        attn_w = torch.matmul(q_rot, k_rep.transpose(2, 3)) * scaling  # [1,8,32,32]
        A["qk_scaled"] = attn_w
        attn_m = attn_w + mask                                        # causal add
        A["qk_masked"] = attn_m
        probs = F.softmax(attn_m, dim=-1, dtype=torch.float32)
        A["softmax"] = probs
        attn_out = torch.matmul(probs, v_rep)                          # [1,8,32,256]
        A["attn_out"] = attn_out
        attn_flat = attn_out.transpose(1, 2).contiguous().reshape(1, 32, 2048)
        A["attn_flat"] = attn_flat

        # 门控(HF: attn_output * sigmoid(gate))
        A["gate_chunk"] = gate_chunk
        gated = attn_flat * torch.sigmoid(gate_chunk.reshape(1, 32, 2048))
        A["gated_attn"] = gated

        o_out = attn.o_proj(gated)                                     # [1,32,1024]
        A["o_proj"] = o_out

        # ---- 残差 + MLP ----
        resid1 = hidden + o_out
        A["resid1"] = resid1
        post_norm = layer.post_attention_layernorm(resid1)
        A["post_norm"] = post_norm
        mlp = layer.mlp(post_norm)
        A["mlp_out"] = mlp
        layer_out = resid1 + mlp
        A["layer_out"] = layer_out

    for name, t in A.items():
        np.save(os.path.join(args.out, f"{name}.npy"), t.detach().cpu().numpy().astype(np.float32))
        print(f"[anchor] {name} {tuple(t.shape)}")

    gold = rd("layer_3_out.f32.raw").reshape(1, 32, 1024)
    a64 = A["layer_out"].numpy().astype(np.float64).ravel()
    g64 = gold.astype(np.float64).ravel()
    print("[selfcheck] layer_out vs golden cos=%.9f" %
          (np.dot(a64, g64) / (np.linalg.norm(a64) * np.linalg.norm(g64))))


if __name__ == "__main__":
    main()
