#!/usr/bin/env python3
"""export_qwen3vl_4b_onnx.py — Qwen3-VL-4B 纯文本 prefill ONNX 导出 (GEHTP)

与 export_qwen35_4b_onnx.py 同型, 两处部署形态对齐生产(qwen3-vl-4cores eaglet):
  1. 主输入 = inputs_embeds [1,seq,2560] f16(embedding 走 CPU LUT, 同 QNN 生产;
     也满足 gehtp portal 单运行时输入 f16-only 约束)
  2. position_ids/attention_mask = 图内常量(定长 prefill, 位置恒 0..seq-1;
     常折后 rope cos/sin 全部预烘焙, 消掉 rotary Transpose 类 emit 缺口)

模型: transformers 5.14 原生 Qwen3VLTextModel(text_config, eager attention)
  + tied lm_head(= embed_tokens.weight 共享存储, 单份 initializer)。
权重: HF safetensors 键 model.language_model.* → 去前缀; 弃 model.visual.*。

用法:
  export_qwen3vl_4b_onnx.py --model-dir /disk2/Qwen3-VL-4B-Instruct --seq 32 \
      --out /disk2/GEHTP/test_models/qwen3vl_4b/model.onnx
产物: model.onnx(+外部权重) / in_embeds.f16.raw / golden_logits.f32.npy / meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

# transformers 5.x create_causal_mask 走 vmap(sdpa_mask_recent_torch),
# jit.trace 下 functorch vmap 崩。替换为等价纯 torch 实现(同 qwen35 导出器):
# 加性 4D 掩码 [bsz,1,q,kv]: 0=参与, -inf=屏蔽, 上三角屏蔽(因果)。
import transformers.masking_utils as _mu


def _jit_causal_mask(config, inputs_embeds, attention_mask=None, cache_position=None,
                     past_key_values=None, position_ids=None, **kwargs):
    bsz, q_len, _ = inputs_embeds.shape
    row = torch.arange(q_len).unsqueeze(1)  # [q,1]
    col = torch.arange(q_len).unsqueeze(0)  # [1,q]
    keep = row >= col
    m = torch.where(keep,
                    torch.zeros((), dtype=inputs_embeds.dtype),
                    torch.full((), float("-inf"), dtype=inputs_embeds.dtype))
    m = m.unsqueeze(0).unsqueeze(0)  # [1,1,q,q]
    if attention_mask is not None:
        pad = torch.where(attention_mask == 0,
                          torch.full((), float("-inf"), dtype=m.dtype),
                          torch.zeros((), dtype=m.dtype))
        m = m + pad.unsqueeze(1).unsqueeze(-1)
    return m


_mu.create_causal_mask = _jit_causal_mask

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLTextConfig)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


class TextPrefill(torch.nn.Module):
    """inputs_embeds-only prefill: 位置/掩码全常量, 单运行时输入。

    cos/sin 预烘为 buffer 常量 + rotary_emb.forward 覆写直接返回:
    rotary 预处理链(Gather/MatMul/Transpose/Slice/ScatterND/Unsqueeze)整链不进图——
    1) 图更小; 2) 绕开 loader 低 id param/weight const 互撞缺陷(实证见
    memory gehtp-qwen3vl4b-portal-line)。数值与原模块逐位等价(同一模块跑出)。"""

    def __init__(self, text_model, lm_head, seq):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        pos = torch.arange(seq, dtype=torch.long).view(1, 1, seq).expand(4, 1, seq).contiguous()
        self.register_buffer("pos_ids", pos, persistent=False)
        # 原模块预烘(text_model.forward 内部: pos[1:] 送 rotary_emb)
        rotary = text_model.rotary_emb
        dummy = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
        with torch.no_grad():
            cos, sin = rotary(dummy, pos[1:].contiguous())  # [3,1,seq,head_dim] fp32
        self.register_buffer("cos_b", cos.contiguous(), persistent=True)
        self.register_buffer("sin_b", sin.contiguous(), persistent=True)
        # 实例属性覆写 class 方法: 返回预烘常量, 不再算
        rotary.forward = lambda x, position_ids: (self.cos_b, self.sin_b)

    def forward(self, inputs_embeds):
        out = self.text_model(inputs_embeds=inputs_embeds, attention_mask=None,
                              position_ids=self.pos_ids, use_cache=False)
        return self.lm_head(out.last_hidden_state)


def load_remapped(text_model, model_dir):
    """HF 多模态 safetensors → Qwen3VLTextModel state_dict (弃 visual)。"""
    import glob
    shards = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))) or \
             sorted(glob.glob(os.path.join(model_dir, "model.safetensors-*.safetensors"))) or \
             [os.path.join(model_dir, "model.safetensors")]
    raw = {}
    for sh in shards:
        raw.update(load_file(sh))
    print(f"[load] {len(shards)} shard(s) -> {len(raw)} keys")
    remapped, n_skipped = {}, 0
    for k, v in raw.items():
        if k.startswith("model.visual."):
            n_skipped += 1
            continue
        if k.startswith("model.language_model."):
            remapped[k[len("model.language_model."):]] = v
        else:
            remapped[k] = v  # lm_head.weight (非 tied 时)
    missing, unexpected = text_model.load_state_dict(remapped, strict=False)
    print(f"[load] total={len(raw)} remapped={len(remapped)} skipped={n_skipped}")
    print(f"[load] missing={missing}")
    print(f"[load] unexpected={unexpected}")
    assert not missing, f"weights missing: {missing}"
    assert not unexpected, f"weights unexpected: {unexpected}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = Qwen3VLConfig.from_pretrained(args.model_dir)
    text_cfg: Qwen3VLTextConfig = cfg.text_config
    # eager attention: 展开为 matmul/softmax 原语链(编译器无融合注意力 op,
    # sdpa 在 trace 下的分解路径不如 eager 可控)
    text_cfg._attn_implementation = "eager"
    print(f"[cfg] vocab={text_cfg.vocab_size} hidden={text_cfg.hidden_size} "
          f"layers={text_cfg.num_hidden_layers} heads={text_cfg.num_attention_heads} "
          f"kv_heads={text_cfg.num_key_value_heads} head_dim={text_cfg.head_dim} "
          f"mrope_section={text_cfg.rope_scaling.get('mrope_section')}")

    text_model = Qwen3VLTextModel(text_cfg).to(torch.float32).eval()
    load_remapped(text_model, args.model_dir)
    # tie_word_embeddings=true: lm_head 与 embed_tokens 共享
    lm_head = torch.nn.Linear(text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
    lm_head.weight = text_model.embed_tokens.weight
    n_params = sum(p.numel() for p in text_model.parameters())
    print(f"[model] text params={n_params} (lm_head tied)")

    seq = args.seq
    wrapper = TextPrefill(text_model, lm_head, seq).eval()

    # 真实 token 经 embed_tokens 构造输入(幅度真实, 利于 f16 长链判读);
    # 设备注入用 f16 舍入版, ORT golden 用同一份 f16→fp32 回读(与设备逐位同源)
    torch.manual_seed(42)
    token_ids = torch.randint(0, text_cfg.vocab_size, (1, seq), dtype=torch.long)
    with torch.no_grad():
        embeds_fp32 = text_model.embed_tokens(token_ids)  # [1,seq,2560] fp32
    embeds_f16 = embeds_fp32.to(torch.float16)
    ort_input = embeds_f16.to(torch.float32)  # f16 舍入后无损回 fp32

    with torch.no_grad():
        logits = wrapper(ort_input)
    print(f"[fwd] logits shape={tuple(logits.shape)} mean={logits.float().mean().item():.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (ort_input,),
        args.out,
        input_names=["inputs_embeds"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[export] {args.out} ({size_mb:.1f} MB, opset={args.opset})")

    # 后处理(仅小 proto,不加载外部数据): 递归剥离悬空图输入
    # (jit trace 把未被消费的 traced 值提升为图输入, 如 text_position_ids 切片)
    import onnx as _onnx
    _m = _onnx.load(args.out, load_external_data=False)
    _declared = {"inputs_embeds"}

    def _consumed(g, acc):
        for n in g.node:
            for i in n.input:
                acc.add(i)
            for a in n.attribute:
                if a.type == a.GRAPH:
                    _consumed(a.g, acc)
                elif a.type == a.GRAPHS:
                    for gg in a.graphs:
                        _consumed(gg, acc)

    def _prune(g):
        acc = set()
        _consumed(g, acc)
        kept = [i for i in g.input if i.name in acc or i.name in _declared]
        dangling = [i.name for i in g.input if i.name not in acc and i.name not in _declared]
        if dangling:
            del g.input[:]
            g.input.extend(kept)
            print(f"[export] strip dangling inputs: {dangling}")
        for n in g.node:
            for a in n.attribute:
                if a.type == a.GRAPH:
                    _prune(a.g)

    _prune(_m.graph)
    _onnx.save(_m, args.out)

    # ORT 一致性自检(与设备同源输入: f16 舍入版)
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"inputs_embeds": ort_input.numpy()})[0]
    a = logits.float().numpy().ravel()
    b = np.asarray(out).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    maxdiff = float(np.abs(np.asarray(out) - logits.float().numpy()).max())
    print(f"[ort] cos={cos:.9f} maxdiff={maxdiff:.3e}")
    assert cos > 0.99999 and maxdiff < 1e-3, "ORT 与模型前向不一致"
    print("[ort] OK")

    # 设备/golden 资产
    out_dir = os.path.dirname(os.path.abspath(args.out))
    in_path = os.path.join(out_dir, "in_embeds.f16.raw")
    embeds_f16.numpy().tofile(in_path)
    gold_path = os.path.join(out_dir, "golden_logits.f32.npy")
    np.save(gold_path, np.asarray(out, dtype=np.float32))
    meta = {
        "model_dir": os.path.abspath(args.model_dir),
        "seq": seq,
        "input": {"file": os.path.basename(in_path), "shape": [1, seq, text_cfg.hidden_size],
                  "dtype": "float16"},
        "golden": {"file": os.path.basename(gold_path), "shape": list(out.shape),
                   "dtype": "float32", "source": "onnxruntime CPU (fp32 graph)"},
        "token_ids": token_ids[0].tolist(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[assets] {in_path} + {gold_path} + meta.json")


if __name__ == "__main__":
    main()
