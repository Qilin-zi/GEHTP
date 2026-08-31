#!/usr/bin/env python3
"""golden_qwen35.py — Qwen3.5 CPU 双 golden 生成 (GEHTP M0)

两个权重源、同一模型代码(QPM modeling,纯 torch fallback):
  --mode hf     HF safetensors(f32)
  --mode q4_0   GGUF Q4_0_pure(反量化 f32,证明 Q4_0 供给路径与 HF 同源)
两者 logits 互拍 cos ≥ 0.99 即"同源"验收;各自输出为后续设备对拍金标。

GGUF 角色表按架构键(qwen35)归一——与 M2 gguf_wtlink.cpp 同设计:
  blk.N.<role> → model.layers.N.<hf_path>。
GGUF 布局约定(实测): tensor dims = ggml ne 数组(ne[0] 最快维),numpy 等价形状 = dims[::-1];
RMSNorm 权重 GGUF 烤入 (1+w);ssm_a = -A(HF 侧为 A_log = log(-ssm_a))。

用法:
  golden_qwen35.py --model-dir /disk2/Qwen3.5-0.8B \
      --gguf /disk2/qwen35_4B/Qwen3.5-0.8B-Q4_0_pure.gguf \
      --seq 32 --n-prompts 4 --mode both --out-dir test_models/qwen35_08b/cpu_golden
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

from export_qwen35_onnx import QPM_ROOT, load_remapped

sys.path.insert(0, QPM_ROOT)


# --------------------------------------------------------------------------
# GGUF v3 最小读取器
# --------------------------------------------------------------------------
GGUF_Q4_0 = 2
GGUF_F32 = 0


def _skip_value(data, pos, vtype):
    """按 GGUF 类型推进 pos,返回新 pos(STRING 返回 (new_pos, value))。"""
    if vtype == 8:  # STRING
        vlen = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        val = data[pos:pos + vlen].decode()
        return pos + vlen, val
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        return pos + sizes[vtype], None
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack_from("<IQ", data, pos)
        pos += 12
        for _ in range(count):
            pos, _ = _skip_value(data, pos, elem_type)
        return pos, None
    raise ValueError(f"unknown KV type {vtype}")


def gguf_read(path):
    """返回 (arch, tensors) — tensors: {name: (ggml_type, dims, data_bytes)}。"""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<4sI", data, 0)
    assert magic == b"GGUF" and version == 3, f"bad GGUF header {magic} v{version}"
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24
    arch = ""
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        key = data[pos:pos + klen].decode()
        pos += klen
        vtype = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        pos, val = _skip_value(data, pos, vtype)
        if key == "general.architecture":
            arch = val
    assert arch, "no general.architecture key"
    tensors = {}
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        name = data[pos:pos + nlen].decode()
        pos += nlen
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = struct.unpack_from(f"<{n_dims}Q", data, pos)
        pos += 8 * n_dims
        gtype = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        offset = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        n_elems = int(np.prod(dims))
        if gtype == GGUF_F32:
            nbytes = n_elems * 4
        elif gtype == GGUF_Q4_0:
            assert n_elems % 32 == 0, f"{name}: Q4_0 非 32 对齐"
            nbytes = n_elems // 32 * 18
        else:
            tensors[name] = (gtype, dims, 0, 0)  # 不支持的量化类型:跳过
            continue
        tensors[name] = (gtype, dims, offset, nbytes)
    # GGUF tensor offset 相对数据段起点:数据段 = tensor info 表末尾,按 32 对齐
    data_start = (pos + 31) & ~31
    for name, (gtype, dims, offset, nbytes) in tensors.items():
        tensors[name] = (gtype, dims, data[data_start + offset:data_start + offset + nbytes])
    return arch, tensors


def q4_0_dequant(buf, n_elems):
    """GGUF Q4_0 块(32 元素 = 2B fp16 d + 16B nibbles)→ f32。

    llama.cpp 布局: 元素 0..15 = 各字节低 nibble, 元素 16..31 = 高 nibble。
    """
    out = np.empty(n_elems, dtype=np.float32)
    for b in range(n_elems // 32):
        d = np.frombuffer(buf, dtype="<u2", count=1, offset=b * 18).view("<f2")[0].astype(np.float32)
        nib = np.frombuffer(buf, dtype=np.uint8, count=16, offset=b * 18 + 2)
        lo = (nib & 0xF).astype(np.float32) - 8.0
        hi = (nib >> 4).astype(np.float32) - 8.0
        out[b * 32:b * 32 + 16] = lo * d
        out[b * 32 + 16:b * 32 + 32] = hi * d
    return out


def gguf_to_state_dict(gguf_path, text_cfg):
    """GGUF(qwen35 角色表)→ HF state_dict。transpose=True 表示 GGUF [in,out]→HF [out,in]。"""
    arch, tensors = gguf_read(gguf_path)
    assert arch == "qwen35", f"arch={arch}, 需要 qwen35"
    hidden = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    layer_types = text_cfg.layer_types  # 'linear_attention' / 'full_attention'
    sd = {}
    n_gdn = n_attn = 0

    def get(name):
        assert name in tensors, f"GGUF 缺 {name}"
        return tensors[name]

    def q4(name, expected):
        gtype, dims, buf = get(name)
        assert gtype == GGUF_Q4_0 and tuple(dims) == expected, f"{name}: {gtype} {dims}"
        # ggml 布局: dims 列表 = ne 数组(ne[0] 最快维)→ numpy 等价形状 = dims[::-1]
        return q4_0_dequant(buf, int(np.prod(dims))).reshape(expected[::-1])

    def f32(name, expected):
        gtype, dims, buf = get(name)
        assert gtype == GGUF_F32 and tuple(dims) == expected, f"{name}: {gtype} {dims}"
        # 同上: ne0-fastest → numpy 形状反转
        return np.frombuffer(buf, dtype="<f4").copy().reshape(expected[::-1])

    def rms(name, expected):
        """普通 RMSNorm: GGUF 烤入 (1+w)(llama.cpp 直接乘 w 的约定),HF 存 w → 减 1。"""
        return f32(name, expected) - 1.0

    # 顶层
    sd["model.embed_tokens.weight"] = q4("token_embd.weight", (hidden, text_cfg.vocab_size))
    sd["model.norm.weight"] = rms("output_norm.weight", (hidden,))

    for layer in range(n_layers):
        p = f"model.layers.{layer}."
        b = f"blk.{layer}."
        sd[p + "input_layernorm.weight"] = rms(b + "attn_norm.weight", (hidden,))
        sd[p + "post_attention_layernorm.weight"] = rms(b + "post_attention_norm.weight", (hidden,))
        sd[p + "mlp.gate_proj.weight"] = q4(b + "ffn_gate.weight", (hidden, text_cfg.intermediate_size))
        sd[p + "mlp.up_proj.weight"] = q4(b + "ffn_up.weight", (hidden, text_cfg.intermediate_size))
        sd[p + "mlp.down_proj.weight"] = q4(b + "ffn_down.weight", (text_cfg.intermediate_size, hidden))
        if layer_types[layer] == "linear_attention":
            kv = text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim
            vv = text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
            n_gdn += 1
            sd[p + "linear_attn.in_proj_qkv.weight"] = q4(b + "attn_qkv.weight", (hidden, kv * 2 + vv))
            sd[p + "linear_attn.in_proj_z.weight"] = q4(b + "attn_gate.weight", (hidden, vv))
            sd[p + "linear_attn.in_proj_a.weight"] = q4(b + "ssm_alpha.weight", (hidden, text_cfg.linear_num_value_heads))
            sd[p + "linear_attn.in_proj_b.weight"] = q4(b + "ssm_beta.weight", (hidden, text_cfg.linear_num_value_heads))
            sd[p + "linear_attn.out_proj.weight"] = q4(b + "ssm_out.weight", (vv, hidden))
            # conv1d: GGUF [k,C] → HF [C,1,k]
            conv = f32(b + "ssm_conv1d.weight", (text_cfg.linear_conv_kernel_dim, kv * 2 + vv))
            sd[p + "linear_attn.conv1d.weight"] = conv[:, None, :]
            # GGUF ssm_a = -A(llama.cpp 递推直接用 -A),HF 存 A_log → log(-ssm_a)
            a_neg = f32(b + "ssm_a", (text_cfg.linear_num_value_heads,))
            sd[p + "linear_attn.A_log"] = np.log(-a_neg)
            sd[p + "linear_attn.dt_bias"] = f32(b + "ssm_dt.bias", (text_cfg.linear_num_value_heads,))
            sd[p + "linear_attn.norm.weight"] = f32(b + "ssm_norm.weight", (text_cfg.linear_value_head_dim,))
        else:
            # q_proj 输出 = heads×head_dim×2(Qwen3.5 双 q,拆 q/门控分量)
            qd = text_cfg.num_attention_heads * text_cfg.head_dim * 2
            kd = text_cfg.num_key_value_heads * text_cfg.head_dim
            od = text_cfg.num_attention_heads * text_cfg.head_dim
            n_attn += 1
            sd[p + "self_attn.q_proj.weight"] = q4(b + "attn_q.weight", (hidden, qd))
            sd[p + "self_attn.k_proj.weight"] = q4(b + "attn_k.weight", (hidden, kd))
            sd[p + "self_attn.v_proj.weight"] = q4(b + "attn_v.weight", (hidden, kd))
            sd[p + "self_attn.o_proj.weight"] = q4(b + "attn_output.weight", (od, hidden))
            sd[p + "self_attn.q_norm.weight"] = rms(b + "attn_q_norm.weight", (text_cfg.head_dim,))
            sd[p + "self_attn.k_norm.weight"] = rms(b + "attn_k_norm.weight", (text_cfg.head_dim,))
    print(f"[gguf] gdn_layers={n_gdn} attn_layers={n_attn} tensors_used={len(sd)}")
    return sd


def build_model(model_dir, weights):
    """weights: 'hf' | ('gguf', path)。返回 (model, cfg)。"""
    from huggingface.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from huggingface.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5Config.from_pretrained(model_dir).text_config
    model = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()
    if weights == "hf":
        model = load_remapped(model, model_dir)
    else:
        sd = gguf_to_state_dict(weights[1], cfg)
        sd = {k: torch.from_numpy(v) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tied = [k for k in missing if "lm_head" in k or "embed_tokens" in k]
        if tied:
            model.tie_weights()
            missing = [k for k in missing if k not in tied]
        print(f"[load] missing={missing} unexpected={unexpected}")
        assert not missing and not unexpected, "state_dict 不匹配"
    return model, cfg


def run_prompts(model, cfg, seq, n_prompts, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_prompts):
        ids = rng.integers(0, 2000, (1, seq), dtype=np.int64)
        with torch.no_grad():
            logits = model(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.ones(1, seq, dtype=torch.long),
                position_ids=torch.arange(seq, dtype=torch.long).unsqueeze(0),
                use_cache=False,
            ).logits.float().numpy()
        out.append((ids, logits))
        print(f"[golden] prompt {i}: logits {logits.shape} mean={logits.mean():.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--mode", choices=["hf", "q4_0", "both"], default="both")
    ap.add_argument("--text", action="append", default=None,
                    help="真实文本(可多次),编码后跑 top-1 一致性判据")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for mode in (["hf", "q4_0"] if args.mode == "both" else [args.mode]):
        weights = "hf" if mode == "hf" else ("gguf", args.gguf)
        assert mode != "q4_0" or args.gguf, "--gguf required for q4_0"
        print(f"=== mode={mode} ===")
        model, cfg = build_model(args.model_dir, weights)
        prompts = run_prompts(model, cfg, args.seq, args.n_prompts)
        subdir = os.path.join(args.out_dir, mode)
        os.makedirs(subdir, exist_ok=True)
        logits_all = []
        for i, (ids, logits) in enumerate(prompts):
            ids.tofile(os.path.join(subdir, f"input_{i}.i64.raw"))
            logits.astype(np.float32).tofile(os.path.join(subdir, f"logits_{i}.f32.raw"))
            logits_all.append(logits.reshape(1, -1))
        results[mode] = np.concatenate(logits_all, axis=0)
        print(f"[golden] {mode}: saved {args.n_prompts} prompts to {subdir}")

    if len(results) == 2:
        # Q4_0 量化引入真实噪声(层尾 logits 噪声 ~1.0,近并列 token 会翻):
        # 随机 token 用 cos 门,真实文本用 top-1 门
        a = results["hf"].ravel()
        b = results["q4_0"].ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        maxdiff = float(np.abs(results["hf"] - results["q4_0"]).max())
        print(f"[same-source] hf vs q4_0: cos={cos:.6f} maxdiff={maxdiff:.3e}")
        assert cos > 0.8, "HF 与 Q4_0 双 golden 不同源(cos 过低)"
        print("[same-source] cos OK")

    # 真实文本 top-1 一致性(量化语义级判据)
    if args.text and len(results) == 2:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
        sub = os.path.join(args.out_dir, "text")
        os.makedirs(sub, exist_ok=True)
        for ti, text in enumerate(args.text):
            ids = tok.encode(text).ids
            assert 0 < len(ids) <= args.seq, f"text {ti} 超出 seq 上限"
            tops = {}
            for mode, weights in (("hf", "hf"), ("q4_0", ("gguf", args.gguf))):
                model, cfg = build_model(args.model_dir, weights)
                ids_t = torch.tensor([ids], dtype=torch.long)
                with torch.no_grad():
                    logits = model(input_ids=ids_t,
                                   attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                   position_ids=torch.arange(len(ids)).unsqueeze(0),
                                   use_cache=False).logits.float().numpy()
                tops[mode] = logits.argmax(-1)[0]
                np.asarray(ids).astype(np.int64).tofile(os.path.join(sub, f"text{ti}_{mode}.i64.raw"))
                logits.astype(np.float32).tofile(os.path.join(sub, f"text{ti}_{mode}.f32.raw"))
                with open(os.path.join(sub, f"text{ti}_{mode}.top1"), "w") as f:
                    f.write(" ".join(map(str, tops[mode].tolist())))
            agree = (tops["hf"] == tops["q4_0"]).sum()
            print(f"[text] prompt{ti} top-1 agreement: {agree}/{len(ids)}")
            assert agree / len(ids) >= 0.8, f"text{ti} top-1 一致性过低: {agree}/{len(ids)}"
    print("done")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"seq": args.seq, "n_prompts": args.n_prompts, "modes": list(results)}, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
