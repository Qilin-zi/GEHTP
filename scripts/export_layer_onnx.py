#!/usr/bin/env python3
"""export_layer_onnx.py — Qwen3.5 单层 ONNX 导出 + HF 层输入/输出 golden (GEHTP M4.1)

分层数值门的前置:从 HF 全模型前向中 dump 某层的输入(与位置嵌入),导出
"单层前向" ONNX(hidden_states → 层输出),ORT 自检,并落盘:
  layer_N_in.f32.raw   [1,seq,hidden] f32 层输入(= 设备输入)
  layer_N_out.f32.raw  [1,seq,hidden] f32 HF 层输出(golden)
  layer_N_cos.f32.raw / layer_N_sin.f32.raw  (仅全注意力层)
  layer_N.onnx

GDN 层(layer_types[i]=="linear_attention"):输入 = hidden + attention_mask。
全注意力层(full_attention):另加 position_embeddings(cos/sin)+ position_ids。

依赖: /home/speech/miniforge3/bin/python3 (torch/transformers/onnxruntime/safetensors)
用法:
  export_layer_onnx.py --model-dir /disk2/Qwen3.5-0.8B --layer 0 --seq 32 \
      --out /disk2/GEHTP/test_models/qwen35_08b/layers/layer_0
"""
import argparse
import os
import sys

QPM_ROOT = "/disk2/Qwen35dev/qwen3.5_4b_base/example1"
sys.path.insert(0, QPM_ROOT)

# 强制 torch fallback(同 export_qwen35_onnx.py)
import transformers.utils.import_utils as _tf_import
_tf_import.is_causal_conv1d_available = lambda *a, **k: False
_tf_import.is_flash_linear_attention_available = lambda *a, **k: False

import numpy as np
import torch
from safetensors.torch import load_file

import transformers.masking_utils as _mu
import transformers.modeling_attn_mask_utils as _mau


def _jit_causal_mask(config, input_embeds, attention_mask=None, cache_position=None,
                     past_key_values=None, position_ids=None, **kwargs):
    bsz, q_len, _ = input_embeds.shape
    row = torch.arange(q_len).unsqueeze(1)
    col = torch.arange(q_len).unsqueeze(0)
    keep = row >= col
    m = torch.where(keep,
                    torch.zeros((), dtype=input_embeds.dtype),
                    torch.full((), float("-inf"), dtype=input_embeds.dtype))
    m = m.unsqueeze(0).unsqueeze(0)
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


class GdnLayerWrapper(torch.nn.Module):
    """linear_attention 层: 全 1 mask prefill 时层收到的 attention_mask=None
    (Qwen3_5TextModel._update_linear_attn_mask 实测), 只需 hidden。"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        return self.layer(hidden_states, None, None, None, None, None)


class AttnLayerWrapper(torch.nn.Module):
    """full_attention 层: 需 causal mask(4D)+ cos/sin + position_ids。"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, causal_mask, cos, sin, position_ids):
        return self.layer(hidden_states, (cos, sin), causal_mask,
                          position_ids, None, None)


def write_raw(path, t):
    t = t.detach().cpu().to(torch.float32).contiguous()
    t.numpy().tofile(path)
    print(f"[dump] {path} {tuple(t.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    cfg = Qwen3_5Config.from_pretrained(args.model_dir)
    text_cfg = cfg.text_config
    print(f"[cfg] hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers}")
    print(f"[cfg] layer_types={text_cfg.layer_types}")

    model = Qwen3_5ForCausalLM(text_cfg)
    model = model.to(torch.float32).eval()
    model = load_remapped(model, args.model_dir)

    layer = model.model.layers[args.layer]
    ltype = text_cfg.layer_types[args.layer]
    print(f"[layer] idx={args.layer} type={ltype}")
    assert ltype in ("linear_attention", "full_attention")

    # ---- 手动复刻 TextModel 主循环, 截取第 args.layer 层的输入/输出 ----
    # (Qwen3_5TextModel.forward 实测: GDN 层 mask = linear_attn_mask, 全 1 时为
    #  None; 全注意力层 mask = 4D causal_mask; pe 循环外算一次)
    seq = args.seq
    input_ids = torch.randint(0, 2000, (1, seq), dtype=torch.long)
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    tm = model.model  # text_cfg 构造 → model.model 直接是 Qwen3_5TextModel
    with torch.no_grad():
        inputs_embeds = tm.embed_tokens(input_ids)
        pe = tm.rotary_emb(inputs_embeds, position_ids)
        causal_mask = _jit_causal_mask(tm.config, inputs_embeds, attention_mask)
        causal_mask = causal_mask.to(inputs_embeds.dtype)
        hidden = inputs_embeds
        for i in range(args.layer):
            lm = None if text_cfg.layer_types[i] == "linear_attention" else causal_mask
            hidden = tm.layers[i](hidden, position_embeddings=pe,
                                  attention_mask=lm, position_ids=position_ids,
                                  past_key_values=None, use_cache=False,
                                  cache_position=None)
        layer_in = hidden.detach().clone()
        lm = None if ltype == "linear_attention" else causal_mask
        ref_out = tm.layers[args.layer](
            hidden, position_embeddings=pe, attention_mask=lm,
            position_ids=position_ids, past_key_values=None,
            use_cache=False, cache_position=None).detach().clone()
    print(f"[loop] layer {args.layer} in {tuple(layer_in.shape)} out {tuple(ref_out.shape)}")
    cos, sin = pe
    hidden = layer_in

    os.makedirs(args.out, exist_ok=True)
    write_raw(os.path.join(args.out, f"layer_{args.layer}_in.f32.raw"), hidden)
    write_raw(os.path.join(args.out, f"layer_{args.layer}_out.f32.raw"), ref_out)
    if ltype == "full_attention":
        write_raw(os.path.join(args.out, f"layer_{args.layer}_mask.f32.raw"), causal_mask)
        write_raw(os.path.join(args.out, f"layer_{args.layer}_cos.f32.raw"), cos)
        write_raw(os.path.join(args.out, f"layer_{args.layer}_sin.f32.raw"), sin)
        write_raw(os.path.join(args.out, f"layer_{args.layer}_pos_ids.i64.raw"), position_ids)

    # ---- 导出 ONNX ----
    wrapper = GdnLayerWrapper(layer) if ltype == "linear_attention" else AttnLayerWrapper(layer)
    onnx_path = os.path.join(args.out, f"layer_{args.layer}.onnx")
    if ltype == "linear_attention":
        torch.onnx.export(
            wrapper, (hidden,), onnx_path,
            input_names=["hidden_states"],
            output_names=["out"], opset_version=17, dynamo=False,
            do_constant_folding=True)
    else:
        torch.onnx.export(
            wrapper, (hidden, causal_mask, cos, sin, position_ids),
            onnx_path,
            input_names=["hidden_states", "causal_mask", "cos", "sin", "position_ids"],
            output_names=["out"], opset_version=17, dynamo=False,
            do_constant_folding=True)
    print(f"[export] {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # 悬空输入剥离(jit trace 把未消费值提升为图输入; 全注意力层 position_ids
    # 在 use_cache=False prefill 路径下可能不被消费)
    import onnx as _onnx
    _m = _onnx.load(onnx_path, load_external_data=False)

    def _consumed(g, acc):
        for n in g.node:
            for i in n.input:
                acc.add(i)
            for a in n.attribute:
                if a.type == a.GRAPH:
                    _consumed(a.g, acc)

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

    # Trilu 降级: qairt-converter 2.48 缺 onnx_trilu 翻译(实测唯一缺口)。
    # trace 中间张量形状多为动态 → 运行时形状掩码链(免 rank,负索引 Slice):
    #   s=Shape(X); r2=Slice(s,[-2]); c2=Slice(s,[-1])
    #   c1=Unsqueeze(Range(0,c2,1),[0]); r1=Unsqueeze(Range(0,r2,1),[1])
    #   keep = (c1 >= r1+k)   [upper]   /  Not(c1 > r1+k)   [tril]
    #   Out = Where(keep, X, zero标量)   ← Where 广播免显式前导维
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
        _m.graph.node.remove(_m.graph.node[_i + _off])
        for _j, _n in enumerate(_nodes):
            _m.graph.node.insert(_i + _off + _j, _n)
        _off += len(_nodes) - 1
    print(f"[export] lowered {_n_trilu} Trilu nodes -> runtime-shape Where chains")
    _onnx.save(_m, onnx_path)

    # ---- ORT 自检 ----
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    feeds = {"hidden_states": hidden.numpy()}
    if ltype == "full_attention":
        feeds["causal_mask"] = causal_mask.numpy()
        feeds["cos"] = cos.numpy()
        feeds["sin"] = sin.numpy()
        feeds["position_ids"] = position_ids.numpy()
    graph_in = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in graph_in}
    ort_out = sess.run(None, feeds)[0]
    a = ref_out.numpy().reshape(-1).astype(np.float64)
    b = ort_out.reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    maxdiff = float(np.abs(a - b).max())
    print(f"[ort] cos={cos:.9f} maxdiff={maxdiff:.3e}")
    assert cos >= 0.9999, "ORT self-check failed"


if __name__ == "__main__":
    main()
