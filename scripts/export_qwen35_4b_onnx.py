#!/usr/bin/env python3
"""export_qwen35_onnx.py — Qwen3.5 纯文本 prefill ONNX 导出 (GEHTP M0)

QPM modeling 代码(纯 torch fallback,GDN=cumsum 链)+ HF safetensors:
  1. Qwen3_5Config.from_pretrained(model_dir) → text_config
  2. Qwen3_5ForCausalLM(text_config) → fp32
  3. safetensors 键重映射: model.language_model.* → model.*(弃 visual/mtp)
  4. torch.onnx.export 固定 seq prefill(input_ids/attention_mask/position_ids)
  5. onnxruntime 与模型前向一致性自检(max diff / cos)

依赖: /home/speech/miniforge3/bin/python3 (torch/transformers/onnxruntime/safetensors)
用法:
  export_qwen35_onnx.py --model-dir /disk2/Qwen3.5-0.8B --seq 32 \
      --out /disk2/GEHTP/test_models/qwen35_08b/model.onnx
"""
import argparse
import os
import sys

QPM_ROOT = "/disk2/Qwen35dev/qwen3.5_4b_base/example1"
sys.path.insert(0, QPM_ROOT)

# 强制 torch fallback: 环境若装了 CUDA 版 causal_conv1d/FLA,QPM 会走 GPU kernel 崩
import transformers.utils.import_utils as _tf_import
_tf_import.is_causal_conv1d_available = lambda *a, **k: False
_tf_import.is_flash_linear_attention_available = lambda *a, **k: False

import numpy as np
import torch
from safetensors.torch import load_file

# transformers 4.57 的 create_causal_mask 走 vmap 建掩码(sdpa_mask_recent_torch),
# jit.trace 下 functorch vmap 崩(unordered_map::at)。替换为等价纯 torch 实现:
# 加性 4D 掩码 [bsz,1,q,kv]: 0=参与, -inf=屏蔽, 上三角屏蔽(因果); 附带 attention_mask 填充语义。
import transformers.masking_utils as _mu
import transformers.modeling_attn_mask_utils as _mau


def _jit_causal_mask(config, input_embeds, attention_mask=None, cache_position=None,
                     past_key_values=None, position_ids=None, **kwargs):
    # 因果掩码: keep = (row >= col) → 0, 否则 -inf。
    # 不用 torch.triu(diagonal=1): torch 2.10 导出 Trilu 会丢 k 属性(k=1 变 k=0,
    # 实测小实验)导致对角线被误掩 → softmax 全 -inf 行 → NaN。
    bsz, q_len, _ = input_embeds.shape
    row = torch.arange(q_len).unsqueeze(1)  # [q,1]
    col = torch.arange(q_len).unsqueeze(0)  # [1,q]
    keep = row >= col
    m = torch.where(keep,
                    torch.zeros((), dtype=input_embeds.dtype),
                    torch.full((), float("-inf"), dtype=input_embeds.dtype))
    m = m.unsqueeze(0).unsqueeze(0)  # [1,1,q,q]
    if attention_mask is not None:
        pad = torch.where(attention_mask == 0,
                          torch.full((), float("-inf"), dtype=m.dtype),
                          torch.zeros((), dtype=m.dtype))
        m = m + pad.unsqueeze(1).unsqueeze(-1)
    return m


_mu.create_causal_mask = _jit_causal_mask
_mau.create_causal_mask = _jit_causal_mask

from huggingface.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from huggingface.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def load_remapped(model, model_dir):
    """HF 多模态 safetensors → ForCausalLM 纯文本 state_dict."""
    import glob
    shards = sorted(glob.glob(os.path.join(model_dir, "model.safetensors-*.safetensors"))) or              sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    assert shards, f"no safetensors shards in {model_dir}"
    raw = {}
    for sh in shards:
        raw.update(load_file(sh))
    print(f"[load] {len(shards)} shard(s) -> {len(raw)} keys")
    remapped = {}
    n_skipped = 0
    for k, v in raw.items():
        if k.startswith("model.visual.") or k.startswith("mtp."):
            n_skipped += 1
            continue
        if k.startswith("model.language_model."):
            remapped["model." + k[len("model.language_model."):]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[load] total={len(raw)} remapped={len(remapped)} skipped={n_skipped}")
    print(f"[load] missing={missing}")
    print(f"[load] unexpected={unexpected}")
    # tie_word_embeddings=True: lm_head.weight 与 embed_tokens 共享,补绑定即可
    tied = {k for k in missing if "lm_head" in k or "embed_tokens" in k}
    if tied:
        model.tie_weights()
        missing = [k for k in missing if k not in tied]
    assert not missing, f"weights missing: {missing}"
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = Qwen3_5Config.from_pretrained(args.model_dir)
    text_cfg = cfg.text_config
    print(f"[cfg] vocab={text_cfg.vocab_size} hidden={text_cfg.hidden_size} "
          f"layers={text_cfg.num_hidden_layers} heads={text_cfg.num_attention_heads}")

    model = Qwen3_5ForCausalLM(text_cfg)
    model = model.to(torch.float32).eval()
    model = load_remapped(model, args.model_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params}")

    torch.manual_seed(42)
    seq = args.seq
    input_ids = torch.randint(0, 2000, (1, seq), dtype=torch.long)
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids, use_cache=False).logits
    print(f"[fwd] logits shape={tuple(logits.shape)} mean={logits.float().mean().item():.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask, position_ids),
        args.out,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        # use_cache 必须显式 False: merge_with_config_defaults 会把 None 合并成
        # config.use_cache=True → 输出带 DynamicCache 对象 → JIT 展平崩
        kwargs={"use_cache": False},
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[export] {args.out} ({size_mb:.1f} MB, opset={args.opset})")

    # 后处理(仅小 proto,不加载外部数据):
    # 1) logits_to_keep=0 的整段切片: trace 产 Gather(data, 空 indices 图输入),
    #    ONNX 空 indices=全量 → 换 Identity
    # 2) 递归剥离悬空输入(jit trace 把未被消费的 traced 值提升为图输入,
    #    如 text_position_ids 切片)→ 否则 ORT/转换器报缺 feed
    import onnx as _onnx
    _m = _onnx.load(args.out, load_external_data=False)
    # 空索引 Gather 的 indices = trace 自造图输入(如 onnx::Gather_4);
    # 声明输入(input_ids 等)也是图输入但索引非空 → 排除
    _declared = {"input_ids", "attention_mask", "position_ids"}
    idx_names = {i.name for i in _m.graph.input if i.name not in _declared}
    for n in _m.graph.node:
        if n.op_type == "Gather" and len(n.input) >= 2 and n.input[1] in idx_names:
            n.op_type = "Identity"
            while len(n.input) > 1:
                n.input.pop()
            del n.attribute[:]
            print(f"[export] patched empty-indices Gather {n.name} -> Identity")

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
        kept = [i for i in g.input if i.name in acc]
        dangling = [i.name for i in g.input if i.name not in acc]
        if dangling:
            del g.input[:]
            g.input.extend(kept)
            print(f"[export] strip dangling inputs: {dangling}")
        for n in g.node:
            for a in n.attribute:
                if a.type == a.GRAPH:
                    _prune(a.g)

    _prune(_m.graph)

    # 3) Trilu 降级: qairt-converter 2.48 缺 onnx_trilu 翻译(实测唯一缺口)。
    #    trace 中间张量形状多为动态 → 运行时形状掩码链(免 rank,负索引 Slice):
    #      s=Shape(X); r2=Slice(s,[-2]); c2=Slice(s,[-1])
    #      c1=Unsqueeze(Range(0,c2,1),[0]); r1=Unsqueeze(Range(0,r2,1),[1])
    #      keep = (c1 >= r1+k)   [upper]   /  Not(c1 > r1+k)   [tril]
    #      Out = Where(keep, X, zero标量)   ← Where 广播免显式前导维
    import onnx.shape_inference as _si
    _m2 = _si.infer_shapes(_m)
    _dtype_of = {}
    for _vi in _m2.graph.value_info:
        _t = _vi.type.tensor_type
        if _t.HasField("elem_type"):
            _dtype_of[_vi.name] = _t.elem_type
    _n_trilu = 0
    _new_nodes = []
    _new_inits = []

    def _scalar(name, value, dtype_np):
        _a = np.array(value, dtype=dtype_np)
        _init = _onnx.numpy_helper.from_array(_a, name)
        _new_inits.append(_init)
        return _init.name

    def _vec1(name, value):
        """1 元素向量(Unsqueeze/Squeeze 的 axes 输入要求 1-D tensor)。"""
        _a = np.array([value], dtype=np.int64)
        _init = _onnx.numpy_helper.from_array(_a, name)
        _new_inits.append(_init)
        return _init.name

    for _i, _n in enumerate(list(_m.graph.node)):
        if _n.op_type != "Trilu":
            continue
        _dtype = _dtype_of.get(_n.input[0])
        if _dtype is None:
            raise ValueError(f"Trilu {_n.name}: 无法确定输入 dtype")
        _np_dtype = _onnx.helper.tensor_dtype_to_np_dtype(_dtype)
        _upper = 1
        _k = 0
        for _a in _n.attribute:
            if _a.name == "upper":
                _upper = _a.i
            elif _a.name == "k":
                _k = _a.i
        _p = f"{_n.name}_tri"
        _s = _p + "_s"
        _r2 = _p + "_r2"
        _c2 = _p + "_c2"
        _r2s = _p + "_r2s"
        _c2s = _p + "_c2s"
        _r1 = _p + "_r1"
        _c1 = _p + "_c1"
        _keep = _p + "_keep"
        _rng_r = _p + "_rng_r"
        _rng_c = _p + "_rng_c"
        _nodes = [
            _onnx.helper.make_node("Shape", [_n.input[0]], [_s], name=_p + "_shape"),
            _onnx.helper.make_node("Slice", [_s, _scalar(_p + "_s_start", [-2], np.int64),
                                           _scalar(_p + "_s_end", [-1], np.int64)],
                                   [_r2], name=_p + "_sl2"),
            _onnx.helper.make_node("Slice", [_s, _scalar(_p + "_s_start1", [-1], np.int64),
                                           _scalar(_p + "_s_end1", [2**31 - 1], np.int64)],
                                   [_c2], name=_p + "_sl1"),
            _onnx.helper.make_node("Squeeze", [_r2, _vec1(_p + "_axsq", 0)],
                                   [_r2s], name=_p + "_sq2"),
            _onnx.helper.make_node("Squeeze", [_c2, _vec1(_p + "_axsq1", 0)],
                                   [_c2s], name=_p + "_sq1"),
            _onnx.helper.make_node("Range", [_scalar(_p + "_rg0", 0, np.int64), _r2s,
                                             _scalar(_p + "_rg1", 1, np.int64)],
                                   [_rng_r], name=_p + "_rngr"),
            _onnx.helper.make_node("Range", [_scalar(_p + "_rg0c", 0, np.int64), _c2s,
                                             _scalar(_p + "_rg1c", 1, np.int64)],
                                   [_rng_c], name=_p + "_rngc"),
            _onnx.helper.make_node("Unsqueeze", [_rng_r, _vec1(_p + "_ax1", 1)],
                                   [_r1], name=_p + "_usr"),
            _onnx.helper.make_node("Unsqueeze", [_rng_c, _vec1(_p + "_ax0", 0)],
                                   [_c1], name=_p + "_usc"),
        ]
        if _k != 0:
            _r1k = _p + "_r1k"
            _nodes.append(_onnx.helper.make_node(
                "Add", [_r1, _scalar(_p + "_k", _k, np.int64)], [_r1k], name=_p + "_addk"))
        else:
            _r1k = _r1
        if _upper:
            _nodes.append(_onnx.helper.make_node("GreaterOrEqual", [_c1, _r1k], [_keep],
                                                 name=_p + "_ge"))
        else:
            _nodes.append(_onnx.helper.make_node("Greater", [_c1, _r1k], [_p + "_gt"],
                                                 name=_p + "_gt"))
            _nodes.append(_onnx.helper.make_node("Not", [_p + "_gt"], [_keep],
                                                 name=_p + "_not"))
        _zero = _scalar(_p + "_zero", (0 if _np_dtype != np.bool_ else False), _np_dtype)
        _nodes.append(_onnx.helper.make_node("Where", [_keep, _n.input[0], _zero],
                                             _n.output[:1], name=_p + "_where"))
        _new_nodes.append((_i, _nodes))
        _n_trilu += 1

    _m.graph.initializer.extend(_new_inits)
    _off = 0
    for _i, _nodes in _new_nodes:
        _m.graph.node.remove(_m.graph.node[_i + _off])  # 去掉原 Trilu
        for _j, _n in enumerate(_nodes):
            _m.graph.node.insert(_i + _off + _j, _n)
        _off += len(_nodes) - 1
    print(f"[export] lowered {_n_trilu} Trilu nodes -> runtime-shape Where chains")
    _onnx.save(_m, args.out)

    # ORT 一致性自检
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    })[0]
    a = logits.float().numpy().ravel()
    b = np.asarray(out).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    maxdiff = float(np.abs(np.asarray(out) - logits.float().numpy()).max())
    print(f"[ort] cos={cos:.9f} maxdiff={maxdiff:.3e}")
    assert cos > 0.99999 and maxdiff < 1e-3, "ORT 与模型前向不一致"
    print("[ort] OK")


if __name__ == "__main__":
    main()
