#!/usr/bin/env python3
"""replay_gdn_layer0.py — GDN L0 HF 侧手动复刻, 输出全锚点 npy (GEHTP M4.1c)

与 host_run 的 /tmp/host_id_<oid>.f32.raw 逐锚点对拍。锚点按 HF 前向公式
(modeling_qwen3_5.py torch_chunk_gated_delta_rule + GatedDeltaNet.forward)
严格顺序计算; 布局 = HF 自然布局 ([1,32,16,128] 等), 对拍脚本负责
与 net.json 4D 布局的转置对齐。

用法:
  replay_gdn_layer0.py --model-dir /disk2/Qwen3.5-0.8B \
      --layer-in /disk2/GEHTP/test_models/qwen35_08b/layers/layer_0/layer_0_in.f32.raw \
      --out /tmp/anchor
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


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer-in", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Qwen3_5Config.from_pretrained(args.model_dir)
    model = Qwen3_5ForCausalLM(cfg.text_config)
    model = model.to(torch.float32).eval()
    model = load_remapped(model, args.model_dir)

    layer = model.model.layers[0]
    ldn = layer.linear_attn

    hidden = torch.from_numpy(np.fromfile(args.layer_in, dtype=np.float32))
    hidden = hidden.reshape(1, 32, 1024)

    A = {}  # anchor name -> tensor
    with torch.no_grad():
        # ---- 输入归一化 + 投影 ----
        h_norm = layer.input_layernorm(hidden)                       # [1,32,1024]
        A["rms_in"] = h_norm

        mixed_qkv = ldn.in_proj_qkv(h_norm)                          # [1,32,6144]
        A["qkv_proj"] = mixed_qkv
        mixed_qkv_t = mixed_qkv.transpose(1, 2)                      # [1,6144,32]
        A["qkv_t"] = mixed_qkv_t

        # causal depthwise conv (padding=k-1 左填, 取前 seq) + silu
        conv_out = F.silu(ldn.conv1d(mixed_qkv_t))[:, :, :32]        # [1,6144,32]
        A["conv_silu"] = conv_out
        conv_bt = conv_out.transpose(1, 2)                           # [1,32,6144]
        A["conv_bt"] = conv_bt

        q, k, v = torch.split(conv_bt, [2048, 2048, 2048], dim=-1)
        q = q.reshape(1, 32, 16, 128)
        k = k.reshape(1, 32, 16, 128)
        v = v.reshape(1, 32, 16, 128)
        A["q_raw"], A["k_raw"], A["v_raw"] = q, k, v

        z = ldn.in_proj_z(h_norm).reshape(1, 32, 16, 128)            # [1,32,16,128]
        b = ldn.in_proj_b(h_norm)                                    # [1,32,16]
        a = ldn.in_proj_a(h_norm)                                    # [1,32,16]
        A["z"], A["b"], A["a"] = z, b, a

        beta = b.sigmoid()                                           # [1,32,16]
        A["beta"] = beta
        g = -ldn.A_log.float().exp() * F.softplus(a.float() + ldn.dt_bias)  # [1,32,16]
        A["g"] = g

        q_ln = l2norm(q, dim=-1, eps=1e-6)
        k_ln = l2norm(k, dim=-1, eps=1e-6)
        A["q_ln"], A["k_ln"] = q_ln, k_ln

        # ---- chunk_gated_delta_rule (torch fallback 公式) ----
        seq = 32
        chunk_size = 64
        pad_size = chunk_size - seq % chunk_size
        scale = 1 / (128 ** 0.5)

        # transpose(1,2) → [1,16,32,128] f32, pad → 64
        qp = F.pad(q_ln.transpose(1, 2), (0, 0, 0, pad_size))
        kp = F.pad(k_ln.transpose(1, 2), (0, 0, 0, pad_size))
        vp = F.pad(v.transpose(1, 2), (0, 0, 0, pad_size))
        betap = F.pad(beta.transpose(1, 2), (0, pad_size))
        gp = F.pad(g.transpose(1, 2), (0, pad_size))
        qp = qp * scale
        A["q_scaled_pad"] = qp
        A["k_pad"] = kp
        A["v_pad"] = vp

        v_beta = vp * betap.unsqueeze(-1)                            # [1,16,64,128]
        k_beta = kp * betap.unsqueeze(-1)
        A["vb"] = v_beta
        A["kb"] = k_beta

        gcum = gp.cumsum(dim=-1)                                     # [1,16,64]
        A["gcum"] = gcum
        decay_mask = ((gcum.unsqueeze(-1) - gcum.unsqueeze(-2)).tril().exp()).tril()
        A["decay_mask"] = decay_mask                                 # [1,16,64,64]

        mask0 = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=0)
        mask1 = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1)
        attn = -((k_beta @ kp.transpose(-1, -2)) * decay_mask).masked_fill(mask0, 0)
        A["attn_pre"] = attn                                        # [1,16,64,64]

        attn_iter = attn.clone()
        for i in range(1, chunk_size):
            row = attn_iter[..., i, :i].clone()
            sub = attn_iter[..., :i, :i].clone()
            attn_iter[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        A["attn_iter"] = attn_iter

        attnI = attn_iter + torch.eye(chunk_size)                    # [1,16,64,64]
        A["attnI"] = attnI

        # HF 函数内: value 变量被重赋值 = attnI @ v_beta (state 递推的中介)
        value = attnI @ v_beta                                       # [1,16,64,128]
        A["attnI_at_vb"] = value

        k_cumdecay = attnI @ (k_beta * gp.exp().unsqueeze(-1))       # [1,16,64,128]
        A["k_cumdecay"] = k_cumdecay

        # chunk 0 (initial_state=None): core = attn_qk @ value,
        # attn_qk = (q@k^T * dm).masked_fill(严格上三角, 0) — 无负号无迭代无 eye
        attn_qk = ((qp @ kp.transpose(-1, -2)) * decay_mask).masked_fill(mask1, 0)
        A["attn_qk"] = attn_qk                                      # [1,16,64,64]
        core = attn_qk @ value                                       # [1,16,64,128]
        A["core_pad"] = core
        core = core[:, :, :seq]                                      # [1,16,32,128]
        core_t = core.transpose(1, 2).contiguous()                   # [1,32,16,128]
        A["core_t"] = core_t
        core_flat = core_t.reshape(1, 32, 2048)
        A["core_flat"] = core_flat

        # ---- 门控 norm + out_proj ----
        # HF: core→[-1,head_v_dim](128 维上 norm), z 同, norm 后 flatten
        core_hd = core_t.reshape(-1, 128)                            # [512,128]
        z_hd = z.reshape(-1, 128)
        var = core_hd.pow(2).mean(-1, keepdim=True)
        nrm = core_hd * torch.rsqrt(var + 1e-6)                      # f32 norm
        A["core_norm_unnorm"] = nrm
        gate = ldn.norm.weight * nrm.to(torch.float32)
        A["gate_weighted"] = gate
        silu_z = F.silu(z_hd.to(torch.float32))
        A["silu_z"] = silu_z
        gated = gate * silu_z                                        # [512,128]
        A["gated_norm"] = gated
        gated_flat = gated.reshape(1, 32, 2048)
        out_proj = ldn.out_proj(gated_flat)                          # [1,32,1024]
        A["out_proj"] = out_proj

        # ---- 残差 + MLP ----
        resid1 = hidden + out_proj
        A["resid1"] = resid1
        post_norm = layer.post_attention_layernorm(resid1)
        A["post_norm"] = post_norm
        mlp = layer.mlp(post_norm)
        A["mlp_out"] = mlp
        layer_out = resid1 + mlp
        A["layer_out"] = layer_out

    for name, t in A.items():
        path = os.path.join(args.out, f"{name}.npy")
        np.save(path, t.detach().cpu().numpy().astype(np.float32))
        print(f"[anchor] {name} {tuple(t.shape)} -> {path}")

    gfile = np.fromfile(args.layer_in, dtype=np.float32).reshape(1, 32, 1024)
    gold = np.fromfile(
        os.path.join(os.path.dirname(args.layer_in), "layer_0_out.f32.raw"),
        dtype=np.float32).reshape(1, 32, 1024)
    a64 = A["layer_out"].detach().cpu().numpy().astype(np.float64).ravel()
    g64 = gold.astype(np.float64).ravel()
    cos = float(np.dot(a64, g64) / (np.linalg.norm(a64) * np.linalg.norm(g64)))
    print(f"[selfcheck] layer_out vs golden cos={cos:.9f}")


if __name__ == "__main__":
    main()
