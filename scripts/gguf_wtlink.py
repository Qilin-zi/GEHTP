#!/usr/bin/env python3
"""gguf_wtlink.py — net.json 权重节点 ↔ GGUF 张量匹配器 (GEHTP M2)

按架构键(qwen35)的角色表, 把 net.json 的权重消费节点名
(/model/layers.N.<block>.<role>/<OpType>) 映射到 GGUF 张量名
(blk.N.<gguf_role>) 并做形状交叉验证。双向完备性检查 fail-fast。

输出 match_map.json:
  { "<node_name>": {"gguf_name": ..., "ggml_type": 2|0, "dims": [gguf 序],
                    "nbytes": ...}, ... }

GGUF 布局实测契约(见 golden_qwen35.py docstring):
  - tensor offset 相对数据段起点; dims = ggml ne 数组(ne[0] 最快维)
  - Q4_0 块 18B(低 nibble 前 16 元素); RMSNorm 权重烤入 (1+w); ssm_a = -A

用法:
  gguf_wtlink.py --net-json model_net.json \
      --gguf Qwen3.5-0.8B-Q4_0_pure.gguf --out match_map.json
"""
import argparse
import json
import math
import struct
import sys

# 007 US2 (T012): golden_qwen35 的 import 链死依赖已删盘目录
# (/disk2/GEHTP/scripts, /disk2/Qwen35dev); gguf_read_full 本体自包含,
# 就地内联(与 golden_qwen35.py 逐行同语义, numpy.prod → math.prod)。
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


def gguf_read_full(path):
    """返回 (arch, tensors, data_start) — tensors: {name: (ggml_type, dims, data_bytes)}。"""
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
        n_elems = math.prod(dims)
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
        tensors[name] = (gtype, dims, offset, nbytes)
    return arch, tensors, data_start

# GGUF 角色表(arch=qwen35; 由 golden_qwen35.py 的实测映射固化)
ROLE_MAP = {
    "in_proj_qkv": "attn_qkv",
    "in_proj_z": "attn_gate",
    "in_proj_a": "ssm_alpha",
    "in_proj_b": "ssm_beta",
    "out_proj": "ssm_out",
    "conv1d": "ssm_conv1d",
    "A_log": "ssm_a",
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
    "q_norm": "attn_q_norm",
    "k_norm": "attn_k_norm",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
}
# linear_attn 内的 norm = ssm_norm(特殊 block 前缀)
BLOCK_PREFIX = {"linear_attn", "self_attn", "mlp"}


def parse_node_layer_role(node_name):
    """节点名路径 → (layer, block, role)。

    '/model/layers.0/linear_attn/in_proj_qkv/MatMul' → (0, 'linear_attn', 'in_proj_qkv')
    'rms_norm_/model/layers.0/input_layernorm/' → (0, 'norm', 'input_layernorm')
    """
    parts = node_name.split("/")
    # 顶层: '/model/norm/' 与 '/model/embed_tokens/' 与 '/lm_head/MatMul'
    # lm_head = embed 转置 tie(007 US2 实测: onnx__MatMul_208884 == E.T, 抽样
    # 3000 点 diff=0; GGUF 无 output.weight = llama.cpp tied 约定同证)
    if "lm_head" in parts:
        return None, "lm_head", "lm_head"
    if "model" in parts:
        mi = parts.index("model")
        if mi + 1 < len(parts) and parts[mi + 1] == "norm":
            return None, "model", "norm"
        if mi + 1 < len(parts) and parts[mi + 1] == "embed_tokens":
            return None, None, "embed_tokens"
    # 找 layers.N 段(2.48 实测: 可能带 'rms_norm_' 前缀)
    for i, p in enumerate(parts):
        if p.startswith("layers.") and p[7:].isdigit():
            layer = int(p[7:])
            block = parts[i + 1] if i + 1 < len(parts) else ""
            role = parts[i + 2] if i + 2 < len(parts) else ""
            if role == "" and block in ROLE_MAP:
                # 'layers.0/input_layernorm/' 形态: role = block
                role, block = block, "norm"
            if role == "" and block == "linear_attn":
                continue  # 无角色信息, 由权重张量名通道解析
            if block in BLOCK_PREFIX or block == "norm" or role in ROLE_MAP or role in ("A_log", "dt_bias"):
                return layer, block, role
            return None
    return None


def parse_weight_tensor_role(tname):
    """权重张量名 → (layer, block, role)。

    'model.layers.0.linear_attn.A_log' → (0, 'linear_attn', 'A_log')
    'model.embed_tokens.weight' → (None, None, 'embed_tokens')
    'model.norm.weight' → (None, 'model', 'norm')
    """
    parts = tname.split(".")
    if parts[0] != "model":
        return None
    if parts[1] == "layers" and len(parts) >= 5 and parts[2].isdigit():
        return int(parts[2]), parts[3], parts[4]
    if len(parts) == 3 and parts[1] == "embed_tokens":
        return None, None, "embed_tokens"
    if len(parts) == 3 and parts[1] == "norm":
        return None, "model", "norm"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-json", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arch, tensors, data_start = gguf_read_full(args.gguf)
    assert arch == "qwen35", f"arch={arch}, 需要 qwen35"

    with open(args.net_json) as f:
        net = json.load(f)
    nodes = net["graph"]["nodes"]
    g_tensors = net["graph"]["tensors"]

    match = {}
    unmatched_net = []
    for name, node in nodes.items():
        loc = parse_node_layer_role(name)
        wt_inputs = [t for t in node["input_names"]
                     if t.startswith("model.") or t.startswith("onnx::")]
        if not wt_inputs:
            continue
        layer, block, role = loc if loc else (None, None, "")
        valid_roles = set(ROLE_MAP) | {"A_log", "dt_bias", "embed_tokens", "norm",
                                       "lm_head"}
        if role not in valid_roles:
            # 节点名路径无有效角色(如 .../linear_attn/Add)→ 权重张量名通道
            for t in wt_inputs:
                wl = parse_weight_tensor_role(t)
                if wl:
                    layer, block, role = wl
                    break
        if role not in valid_roles:
            continue
        # 顶层(layer=None)特殊: embed_tokens / norm
        if role == "embed_tokens":
            gguf_name = "token_embd.weight"
        elif role == "lm_head":
            # tied lm_head: 源 = token_embd(实测 onnx__MatMul_208884 == E.T)
            gguf_name = "token_embd.weight"
        elif role == "norm" and block == "linear_attn":
            gguf_name = f"blk.{layer}.ssm_norm.weight"
        elif role == "norm":
            gguf_name = "output_norm.weight"
        elif role in ROLE_MAP:
            gguf_name = f"blk.{layer}.{ROLE_MAP[role]}.weight"
        else:
            # A_log/dt_bias 等无 .weight 后缀
            if role == "A_log":
                gguf_name = f"blk.{layer}.ssm_a"
            elif role == "dt_bias":
                gguf_name = f"blk.{layer}.ssm_dt.bias"
            else:
                continue
        if gguf_name not in tensors:
            unmatched_net.append((name, gguf_name))
            continue
        gtype, dims, offset, nbytes = tensors[gguf_name]
        # 形状交叉验证: net.json 权重张量 dims(HF 序)与 GGUF dims(ggml 序)互转
        wt_shape = None
        for t in wt_inputs:
            if t in g_tensors:
                wt_shape = g_tensors[t].get("dims")
                break
        if wt_shape and len(dims) >= 2:
            # 压掉 net 侧的 1 维(conv1d 为 [1,k,1,C] 4D 布局), 比较有效形状
            eff_shape = [d for d in wt_shape if d != 1] or wt_shape
            # GGUF 布局: 一般张量 ne0 最快(numpy 序 = dims 反转);
            # ssm_conv1d 是 llama.cpp 转置存储的例外(numpy 序 = dims 原序)
            hf_from_gguf = list(dims) if role == "conv1d" else list(reversed(dims))
            if tuple(hf_from_gguf) != tuple(eff_shape):
                unmatched_net.append((name, f"{gguf_name} 形状不符 gguf{dims} vs net{wt_shape}"))
                continue
        match[name] = {"gguf_name": gguf_name, "ggml_type": gtype,
                       "dims": list(dims), "nbytes": nbytes,
                       "file_offset": data_start + offset}

    # 双向完备性
    # 白名单: A_log 被 ONNX 常量折叠(实测 net.json 无 A_log 张量)→ 由 .bin 池供应
    folded_exempt = {t for t in tensors if t.endswith(".ssm_a")}
    matched_gguf = {m["gguf_name"] for m in match.values()}
    unmatched_gguf = sorted(set(tensors) - matched_gguf - folded_exempt)
    if unmatched_net or unmatched_gguf:
        print(f"[wtlink] FAIL: net 侧未匹配 {len(unmatched_net)}, gguf 侧未匹配 {len(unmatched_gguf)}")
        for u in unmatched_net[:10]:
            print(f"  net: {u}")
        for u in unmatched_gguf[:10]:
            print(f"  gguf: {u}")
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(match, f, indent=2)
    tsv = args.out.replace(".json", ".tsv")
    with open(tsv, "w") as f:
        f.write("node\tgguf_name\tggml_type\tfile_offset\tnbytes\tdims\n")
        for k, v in match.items():
            f.write(f"{k}\t{v['gguf_name']}\t{v['ggml_type']}\t{v['file_offset']}\t"
                    f"{v['nbytes']}\t{','.join(map(str, v['dims']))}\n")
    print(f"[wtlink] OK: {len(match)} 权重节点 ↔ GGUF 双向完备 → {args.out} + {tsv}")


if __name__ == "__main__":
    main()
