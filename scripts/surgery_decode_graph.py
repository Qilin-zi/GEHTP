#!/usr/bin/env python3
"""surgery_decode_graph.py — T018: seq=1 prefill net.json → 单步 decode 图手术.

把 0.8B (Qwen3.5) seq=1 全前向图变换为持久状态单 token decode 图:
  - 双输入: input_ids i32[1] (原样, 喂 embedding Gather — 泛型 Gather 的
    indices 坐标展开只在单元素 idx 下=行查表, 97 元 idx 会逐位查错行,
    设备 op20 同理) + 融合输入面 decode_in i32[96]: [cos 64×f16 | sin 64×f16
    | mask 64×f16] (u16 偏移: cos 0, sin 64, mask 128; MAX_POS=64)
  - 6 full-attn 层: RoPE 烘焙链 → 输入面; GQA 展开/KV Where/NaN 守卫 → 删除;
    插入 KvAppendF16 + KvGatherKT/V (状态表 0x6000|idx); mask 变加性 (-inf)
  - 18 GDN 层: conv+silu → Conv1dSsmStepF16(含 SiLU); chunk 机器(~600 节点)
    → DeltaChunkF16 (q_raw=1: kernel 内 1/√D 关闭, 图内 q̂ 直喂)
  - 删除三输入/不可达子图 (从 logits 反向可达性剪枝); params.bin 不动 (零追加)

新算子类型注册在 graph_prepare/ops.cpp/wtop_ops 三侧 (T018c/d)。
用法: surgery_decode_graph.py --net NET.json --out-dir DIR [--max-pos 64]
"""
import argparse
import copy
import json
import os
import sys

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
GDN_LAYERS = [l for l in range(24) if l not in FULL_ATTN_LAYERS]
HIDDEN, VOCAB = 1024, 248320
N_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 256          # full-attn
GDN_H, GDN_D = 16, 128                            # linear-attn heads/dim
GDN_C, GDN_DC = 6144, 4                           # conv 内通道/窗
ROT_DIM = 64                                      # partial_rotary 0.25 × head_dim

DT_I32, DT_F16 = 50, 534
U32 = "306"   # scalar_params 数值键码


def fatal(msg):
    print(f"surgery: FATAL {msg}", file=sys.stderr)
    sys.exit(1)


def u16_face_layout(max_pos):
    cos, sin, mask = 0, ROT_DIM, 2 * ROT_DIM
    total = mask + max_pos
    if total % 2:
        total += 1
    return dict(cos=cos, sin=sin, mask=mask, n32=total // 2)


class Surgery:
    def __init__(self, net, max_pos):
        self.g = net["graph"]
        self.nodes = self.g["nodes"]
        self.tensors = self.g["tensors"]
        self.max_pos = max_pos
        self.face = u16_face_layout(max_pos)
        self.max_id = max(t["id"] for t in self.tensors.values())
        self.audit = []

    def say(self, msg):
        self.audit.append(msg)
        print(msg)

    def tdims(self, name):
        return self.tensors[name]["dims"]

    def new_tensor(self, name, dims, dtype=DT_F16, ttype=3):
        if name in self.tensors:
            fatal(f"tensor {name} 已存在")
        self.max_id += 1
        rank = len(dims)
        self.tensors[name] = {
            "id": self.max_id, "type": ttype, "dataFormat": 0,
            "data_type": dtype, "unquantized_data_type": dtype,
            "permute_order_to_src": list(range(rank)),
            "quant_params": {"definition": 2147483647, "encoding": 2147483647,
                             "is_overridden": False,
                             "scale_offset": {"bitwidth": 0, "minimum": 0.0, "maximum": 0.0,
                                              "scale": 0.0, "offset": 0, "is_symmetric": False,
                                              "is_fixed_point": True}},
            "dims": dims, "is_dynamic_dims": [], "is_quantizable": True,
            "is_updateable": False,
        }
        return name

    def new_node(self, name, ntype, inputs, outputs, sp=None):
        if name in self.nodes:
            fatal(f"node {name} 已存在")
        scalar = {k: {U32: int(v)} for k, v in (sp or {}).items()}
        scalar["packageName"] = {"1544": "qti.aisw"}
        self.nodes[name] = {
            "package": "qti.aisw", "type": ntype, "scalar_params": scalar,
            "param_map": {}, "input_names": inputs, "output_names": outputs,
            "macs_per_inference": "0",
        }
        return name

    def node(self, name):
        nd = self.nodes.get(name)
        if nd is None:
            fatal(f"节点缺失: {name}")
        return nd

    def rebind(self, node_name, slot, new_tensor):
        nd = self.node(node_name)
        old = nd["input_names"][slot]
        nd["input_names"][slot] = new_tensor
        self.say(f"  rebind {node_name}[{slot}] {old.split('/')[-1]} -> {new_tensor.split('/')[-1]}")

    def set_dims(self, tensor, dims):
        self.tensors[tensor]["dims"] = dims

    # ---------------- 图头: 融合输入面 ----------------
    def build_input_face(self):
        f = self.face
        self.new_tensor("/decode/decode_in", [f["n32"]], DT_I32, 0)
        self.new_tensor("/decode/face_cos", [1, 1, 1, ROT_DIM])
        self.new_tensor("/decode/face_sin", [1, 1, 1, ROT_DIM])
        self.new_tensor("/decode/face_mask", [1, 1, 1, self.max_pos])
        for nm, beg, out in (("FaceCos", f["cos"], "/decode/face_cos"),
                             ("FaceSin", f["sin"], "/decode/face_sin"),
                             ("FaceMask", f["mask"], "/decode/face_mask")):
            self.new_node(f"/decode/{nm}", "DecodeFaceF16", ["/decode/decode_in"], [out],
                          {"u16_begin": beg, "u16_len": out and self.tensors[out]["dims"][-1]})
        self.say(f"input face: decode_in i32[{f['n32']}] u16 布局 {f}")

    # ---------------- full-attn 层手术 ----------------
    def surgery_full_attn(self, layer, kv_idx):
        p = f"/model/layers.{layer}/self_attn"
        cos, sin, mask = "/decode/face_cos", "/decode/face_sin", "/decode/face_mask"
        self.say(f"L{layer} full-attn (kv state {kv_idx}):")

        # RoPE 烘焙链 → 输入面 (Unsqueeze_12=cos / Unsqueeze_13=sin 的消费点)
        self.rebind(f"{p}/Mul_2", 1, cos)
        self.rebind(f"{p}/Mul_4", 1, cos)
        self.rebind(f"{p}/Mul_3", 1, sin)
        self.rebind(f"{p}/Mul_5", 1, sin)

        # KV append: k=Concat_9 (post-RoPE), v=Transpose_2; 输出= k 面直通
        k_in = f"{p}/Concat_9_output_0"
        v_in = f"{p}/Transpose_2_output_0"
        k_face = f"{p}/kv_k_face"
        self.new_tensor(k_face, copy.deepcopy(self.tdims(k_in)))
        self.new_node(f"{p}/kv_append", "KvAppendF16", [k_in, v_in], [k_face],
                      {"state_idx": kv_idx, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})

        # KV gather 双面 (n_kv 维 = max_pos; 尾部由加性 mask -inf 湮灭)
        kt = f"{p}/kv_kt"
        vt = f"{p}/kv_v"
        self.new_tensor(kt, [1, N_HEADS, HEAD_DIM, self.max_pos])
        self.new_tensor(vt, [1, N_HEADS, self.max_pos, HEAD_DIM])
        self.new_node(f"{p}/kv_gather_kt", "KvGatherKT", [k_face], [kt],
                      {"state_idx": kv_idx, "n_heads": N_HEADS,
                       "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})
        self.new_node(f"{p}/kv_gather_v", "KvGatherV", [k_face], [vt],
                      {"state_idx": kv_idx, "n_heads": N_HEADS,
                       "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})

        # scores 链: Mul_11 W ← gather kT; 维度 1 → max_pos
        self.rebind(f"{p}/Mul_11", 0, kt)
        self.set_dims(f"{p}/Mul_11_output_0", [1, N_HEADS, HEAD_DIM, self.max_pos])
        self.set_dims(f"{p}/MatMul_output_0", [1, N_HEADS, 1, self.max_pos])
        # mask: Where_2 链 → 加性 mask 面 (-inf 无效位)
        self.rebind(f"{p}/Add_3", 1, mask)
        self.set_dims(f"{p}/Add_3_output_0", [1, N_HEADS, 1, self.max_pos])
        self.set_dims(f"{p}/Softmax_output_0", [1, N_HEADS, 1, self.max_pos])
        # probs·V: NaN 守卫/GQA v 展开 → gather v
        self.rebind(f"{p}/MatMul_1", 0, f"{p}/Softmax_output_0")
        self.rebind(f"{p}/MatMul_1", 1, vt)

    # ---------------- GDN 层手术 ----------------
    def surgery_gdn(self, layer, conv_idx, delta_idx):
        p = f"/model/layers.{layer}/linear_attn"
        self.say(f"L{layer} GDN (conv state {conv_idx}, delta state {delta_idx}):")

        # conv 链 (conv→slice→silu→…) → 单步 op (kernel 含 SiLU)
        conv = self.node(f"{p}/Conv_2d")
        x_in = conv["input_names"][0]              # [1,1,1,6144] (in_proj qkv fc 视图)
        w_in = conv["input_names"][1]              # onnx::Conv_XXXX [1,4,1,6144]
        split_in = self.node(f"{p}/Split")["input_names"][0]   # silu 出 (Transpose_1_output_0)
        y = f"{p}/conv_step_out"
        self.new_tensor(y, copy.deepcopy(self.tdims(split_in)))
        self.new_node(f"{p}/conv_step", "Conv1dSsmStepF16", [x_in, w_in], [y],
                      {"state_idx": conv_idx, "C": GDN_C, "d_conv": GDN_DC})
        self.rebind(f"{p}/Split", 0, y)

        # chunk 机器 → 单步 delta (q_raw=1: 不做 kernel 内 1/√D, q̂ 直喂)
        qh = f"{p}/Mul_6_output_0"     # q̂ (l2norm×pow)
        kh = f"{p}/Mul_5_output_0"     # k̂ (l2norm×rms)
        vh = f"{p}/Cast_5_output_0"    # v (Transpose_4 出, [1,16,1,128])
        beta = f"{p}/Sigmoid_1_output_0"
        g = f"{p}/Mul_1_output_0"      # ĝ (全负)
        t13_in = self.node(f"{p}/Transpose_13")["input_names"][0]  # Slice_385_output_0
        dy = f"{p}/delta_step_out"
        self.new_tensor(dy, copy.deepcopy(self.tdims(t13_in)))
        self.new_node(f"{p}/delta_step", "DeltaChunkF16",
                      [kh, vh, qh, beta, g], [dy],
                      {"state_idx": delta_idx, "H": GDN_H, "D": GDN_D, "q_raw": 1})
        self.rebind(f"{p}/Transpose_13", 0, dy)

    # ---------------- embedding ----------------
    def surgery_embed(self):
        # 不动 embedding Gather: indices 维持原 input_ids 输入 (单元素行查表)。
        # 泛型 Gather 的 indices 坐标展开语义下, 多元素 idx 会逐输出位查
        # idx[i%n] 错行 (host ops.cpp / 设备 op20 同款), id 不并入 decode_in。
        g = self.node("/model/embed_tokens/Gather")
        idx = g["input_names"][1]
        if self.tensors[idx]["type"] != 0:
            fatal(f"embedding indices {idx} 非 type0 输入")
        self.embed_idx = idx
        self.say(f"embedding Gather indices 维持原输入 {idx}")

    # ---------------- 可达性剪枝 ----------------
    def prune_unreachable(self):
        producer = {}
        for name, nd in self.nodes.items():
            for out in nd["output_names"]:
                producer[out] = name
        keep_nodes, queue = set(), ["logits"]
        seen_t = set()
        while queue:
            t = queue.pop()
            if t in seen_t:
                continue
            seen_t.add(t)
            n = producer.get(t)
            if n is None or n in keep_nodes:
                continue
            keep_nodes.add(n)
            queue.extend(self.nodes[n]["input_names"])
        dropped_nodes = [n for n in self.nodes if n not in keep_nodes]
        for n in dropped_nodes:
            del self.nodes[n]
        # 无引用的 type0/3 张量丢弃 (type4/2 保留: params.bin 声明序对齐不动)
        refd = set(seen_t)
        for nd in self.nodes.values():
            refd.update(nd["input_names"]); refd.update(nd["output_names"])
        dropped_t = [t for t, ti in self.tensors.items()
                     if t not in refd and ti["type"] in (0, 3)]
        for t in dropped_t:
            del self.tensors[t]
        self.say(f"prune: 节点 {len(self.nodes) + len(dropped_nodes)} -> {len(self.nodes)} "
                 f"(删 {len(dropped_nodes)}); 张量 -> {len(self.tensors)} (删 {len(dropped_t)})")

    def run(self):
        self.build_input_face()
        kv_i = 0
        conv_i, delta_i = 0, 0
        for l in range(24):
            if l in FULL_ATTN_LAYERS:
                self.surgery_full_attn(l, kv_i); kv_i += 1
            else:
                self.surgery_gdn(l, conv_i, delta_i); conv_i += 1; delta_i += 1
        self.surgery_embed()
        self.prune_unreachable()
        # 自检: 恰两输入 = embedding ids + decode 面
        ins = {t for t, ti in self.tensors.items() if ti["type"] == 0}
        if ins != {self.embed_idx, "/decode/decode_in"}:
            fatal(f"输入面异常: {ins}")
        for n, nd in self.nodes.items():     # 悬空引用自检
            for i in nd["input_names"]:
                if i not in self.tensors:
                    fatal(f"{n} 引用缺失张量 {i}")
        return kv_i, conv_i, delta_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--params", default=None, help="随行复制(字节不动)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pos", type=int, default=64)
    args = ap.parse_args()

    net = json.load(open(args.net))
    s = Surgery(net, args.max_pos)
    kv_n, conv_n, delta_n = s.run()
    n_states = kv_n + conv_n + delta_n
    kv_slot = KV_HEADS * HEAD_DIM * 2 * 2 * args.max_pos
    conv_slot = (GDN_DC - 1) * GDN_C * 4
    delta_slot = GDN_H * GDN_D * GDN_D * 4
    print(f"\n状态表: {n_states} 项 = {kv_n}×KV {kv_slot}B + {conv_n}×conv {conv_slot}B "
          f"+ {delta_n}×delta {delta_slot}B, 共 "
          f"{(kv_n * kv_slot + conv_n * conv_slot + delta_n * delta_slot) / 1e6:.1f}MB DDR")

    os.makedirs(args.out_dir, exist_ok=True)
    out_net = os.path.join(args.out_dir, "decode_net.json")
    with open(out_net, "w") as f:
        json.dump(net, f, ensure_ascii=False)
    if args.params:
        import shutil
        shutil.copyfile(args.params, os.path.join(args.out_dir, "decode_params.bin"))
    with open(os.path.join(args.out_dir, "surgery_audit.txt"), "w") as f:
        f.write("\n".join(s.audit) + "\n")
    print(f"写出 {out_net} (nodes={len(net['graph']['nodes'])}, tensors={len(net['graph']['tensors'])})")


if __name__ == "__main__":
    main()
