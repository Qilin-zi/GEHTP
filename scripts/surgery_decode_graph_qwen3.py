#!/usr/bin/env python3
"""surgery_decode_graph_qwen3.py — Qwen3-0.6B (28 层 dense) seq=1 prefill → decode 图手术.

与 0.8B 版 (surgery_decode_graph.py) 的差异:
  - 28 层全 full-attn (无 GDN), flat 节点名 (无 /model/layers.N 前缀; RmsNorm 带
    rms_norm_/ 前缀), 故用 **结构性发现** (type+shape+生产者/消费者拓扑), 不硬编码名;
  - full rotary (ROT_DIM=128=head_dim) + QK-norm + GQA rep=2;
  - scores = 干净 MatMul(q@k^T), 非 0.8B 的 Mul+reduce;
  - embedding Gather 保留 (input_ids [1,1] i64 进图查表, tied 权重 004 op20 已验证)。

新算子: KvAppendF16 / KvGatherKT / KvGatherV / DecodeFaceF16 (已注册三侧)。
用法: surgery_decode_graph_qwen3.py --net decode_net.json --params decode_params.bin \
        --out-dir DECODE_OUT [--max-pos 64]
"""
import argparse
import copy
import json
import os
import sys

HIDDEN, VOCAB = 1024, 151936
N_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
N_LAYERS = 28
ROT_DIM = 128                            # full rotary

DT_I32, DT_F16 = 50, 534
U32 = "306"


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
        self.producer = {}
        self.consumer = {}
        for n, nd in self.nodes.items():
            for o in nd["output_names"]:
                self.producer[o] = n
            for i in nd["input_names"]:
                self.consumer.setdefault(i, []).append(n)

    # ---- 基础 helpers ----
    def say(self, msg):
        self.audit.append(msg)
        print(msg)

    def tdims(self, name):
        return self.tensors[name]["dims"]

    def cons(self, t, ntype=None):
        cs = self.consumer.get(t, [])
        return [c for c in cs if ntype is None or self.nodes[c]["type"] == ntype]

    def one(self, t, ntype=None, what=""):
        cs = self.cons(t, ntype)
        if len(cs) != 1:
            fatal(f"{what} 张量 {t} 消费节点应恰 1 个 (type={ntype}), 得 {cs}")
        return cs[0]

    def out(self, n):
        return self.nodes[n]["output_names"][0]

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

    def rebind(self, node_name, slot, new_tensor):
        nd = self.nodes[node_name]
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
                          {"u16_begin": beg, "u16_len": self.tensors[out]["dims"][-1]})
        self.say(f"input face: decode_in i32[{f['n32']}] u16 布局 {f}")

    # ---------------- 层发现 ----------------
    def find_layers(self):
        """沿残差链返回 28 个 input_layernorm RmsNorm 节点名 (有序)。"""
        # embedding Gather 输出 = 残差流起点
        ins = [t for t, ti in self.tensors.items() if ti["type"] == 0 and t != "/decode/decode_in"]
        assert len(ins) == 1, f"应恰 1 个原始输入 (input_ids), 得 {ins}"
        input_ids = ins[0]
        gather = self.one(input_ids, "Gather", "embedding")
        residual = self.out(gather)  # [1,1,1024]
        layers = []
        for _ in range(N_LAYERS):
            inn = self.one(residual, "RmsNorm", "input_layernorm")
            layers.append(inn)
            # attn 残差 Add = 消费 residual 的 Eltwise_Binary
            attn_add = self.one(residual, "Eltwise_Binary", "attn residual")
            # post_norm = RmsNorm 消费 attn_add 输出; mlp_add = 另一 Eltwise_Binary 消费者
            aout = self.out(attn_add)
            post = self.one(aout, "RmsNorm", "post_norm")
            mlp_add = [c for c in self.cons(aout, "Eltwise_Binary") if c != post]
            if len(mlp_add) != 1:
                fatal(f"层 {len(layers)} mlp 残差 Add 识别异常: {mlp_add}")
            residual = self.out(mlp_add[0])
        return layers, input_ids

    def discover_attn(self, inn):
        """从 input_layernorm 结构性发现注意力手术点, 返回 dict。"""
        h = self.out(inn)                      # [1,1,1024]
        # q/k/v proj FC: seq=1 时 pre-reshape 被 converter 折叠, FC 直吃 h;
        # 更一般情形经 pre-reshape 中转
        fcs = self.cons(h, "FullyConnected")
        if not fcs:
            pre = self.cons(h, "Reshape")
            assert len(pre) == 3, f"{inn} 下游应 3 pre-reshape, 得 {pre}"
            fcs = [self.one(self.out(pr), "FullyConnected", "qkv proj") for pr in pre]
        assert len(fcs) == 3, f"{inn} 下游应 3 FC, 得 {fcs}"
        projs = {}
        for fc in fcs:
            post = self.one(self.out(fc), "Reshape", "post reshape")
            tr = self.one(self.out(post), "Transpose", "qkv transpose")
            d = self.tensors[self.out(tr)]["dims"]
            if d[1] == N_HEADS:
                projs["q"] = tr                     # [1,16,1,128]
            else:
                nxt = self.cons(self.out(tr))
                if any(self.nodes[c]["type"] == "RmsNorm" for c in nxt):
                    projs["k"] = tr                 # [1,8,1,128] → k_norm
                else:
                    projs["v"] = tr                 # [1,8,1,128] → GQA v
        assert set(projs) == {"q", "k", "v"}, f"qkv 识别异常 {projs.keys()}"

        def rope(branch_tr, nh):
            """给定 q 或 k 的 Transpose 节点, 返回 (mul_cos, mul_rsin, add_rope)。"""
            norm = self.one(self.out(branch_tr), "RmsNorm", "qk_norm")
            no = self.out(norm)
            slices = self.cons(no, "StridedSlice")
            assert len(slices) == 2, f"qk RoPE 应 2 Slice, 得 {slices}"
            mul_cos = self.one(no, "Eltwise_Binary", "qk*cos")
            # 哪个 slice 直接进 Concat, 哪个进 Neg
            neg = None
            for s in slices:
                for c in self.cons(self.out(s)):
                    if self.nodes[c]["type"] == "Eltwise_Unary":
                        neg = c
            assert neg is not None, "RoPE Neg 缺失"
            concat = self.one(self.out(neg), "Concat", "rot concat")
            mul_rsin = self.one(self.out(concat), "Eltwise_Binary", "rot*sin")
            # add_rope = 同时消费 mul_cos 与 mul_rsin 的 Add
            a1 = set(self.cons(self.out(mul_cos), "Eltwise_Binary"))
            a2 = set(self.cons(self.out(mul_rsin), "Eltwise_Binary"))
            adds = a1 & a2
            assert len(adds) == 1, f"RoPE Add 识别异常 {adds}"
            return mul_cos, mul_rsin, list(adds)[0]

        q_cos, q_sin, q_rope = rope(projs["q"], N_HEADS)
        k_cos, k_sin, k_rope = rope(projs["k"], KV_HEADS)
        v_trans = projs["v"]

        # GQA 展开 (k: k_rope → Unsqueeze → Expand → Reshape → Transpose(k^T))
        k_unsq = self.one(self.out(k_rope), "Reshape", "GQA k unsq")
        k_exp = self.one(self.out(k_unsq), "Eltwise_Binary", "GQA k expand")
        k_resh = self.one(self.out(k_exp), "Reshape", "GQA k reshape")
        k_tr = self.one(self.out(k_resh), "Transpose", "k^T")
        # v 展开 (v_trans → Unsqueeze_1 → Expand_1 → Reshape_4)
        v_unsq = self.one(self.out(v_trans), "Reshape", "GQA v unsq")
        v_exp = self.one(self.out(v_unsq), "Eltwise_Binary", "GQA v expand")
        v_resh = self.one(self.out(v_exp), "Reshape", "GQA v reshape")

        # scores 链
        matmul_s = self.one(self.out(q_rope), "MatMul", "q@k^T")
        mul_scale = self.one(self.out(matmul_s), "Eltwise_Binary", "scale")
        add_mask = self.one(self.out(mul_scale), "Eltwise_Binary", "mask")
        softmax = self.one(self.out(add_mask), "Softmax", "softmax")
        matmul_v = self.one(self.out(softmax), "MatMul", "probs·V")

        return dict(q_cos=q_cos, q_sin=q_sin, k_cos=k_cos, k_sin=k_sin,
                    k_rope=k_rope, v_trans=v_trans, matmul_s=matmul_s,
                    mul_scale=mul_scale, add_mask=add_mask, softmax=softmax,
                    matmul_v=matmul_v)

    # ---------------- full-attn 层手术 ----------------
    def surgery_layer(self, layer, kv_idx, a):
        cos, sin, mask = "/decode/face_cos", "/decode/face_sin", "/decode/face_mask"
        self.say(f"L{layer} full-attn (kv state {kv_idx}):")

        # RoPE 烘焙链 → 输入面 (cos/sin 常量均在 input[1])
        self.rebind(a["q_cos"], 1, cos)
        self.rebind(a["q_sin"], 1, sin)
        self.rebind(a["k_cos"], 1, cos)
        self.rebind(a["k_sin"], 1, sin)

        # KV append: k=post-RoPE, v=v 转置输出
        k_in = self.out(a["k_rope"])
        v_in = self.out(a["v_trans"])
        k_face = f"/decode/L{layer}/kv_k_face"
        self.new_tensor(k_face, copy.deepcopy(self.tdims(k_in)))
        self.new_node(f"/decode/L{layer}/kv_append", "KvAppendF16", [k_in, v_in], [k_face],
                      {"state_idx": kv_idx, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})

        # KV gather 双面
        kt = f"/decode/L{layer}/kv_kt"
        vt = f"/decode/L{layer}/kv_v"
        self.new_tensor(kt, [1, N_HEADS, HEAD_DIM, self.max_pos])
        self.new_tensor(vt, [1, N_HEADS, self.max_pos, HEAD_DIM])
        self.new_node(f"/decode/L{layer}/kv_gather_kt", "KvGatherKT", [k_face], [kt],
                      {"state_idx": kv_idx, "n_heads": N_HEADS,
                       "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})
        self.new_node(f"/decode/L{layer}/kv_gather_v", "KvGatherV", [k_face], [vt],
                      {"state_idx": kv_idx, "n_heads": N_HEADS,
                       "kv_heads": KV_HEADS, "head_dim": HEAD_DIM})

        # scores: q@k^T 的 k^T 输入 ← gather kT (原 Transpose_3/ GQA k 展开变死码被剪)
        self.rebind(a["matmul_s"], 1, kt)
        self.set_dims(self.out(a["matmul_s"]), [1, N_HEADS, 1, self.max_pos])
        self.set_dims(self.out(a["mul_scale"]), [1, N_HEADS, 1, self.max_pos])
        # mask: 加性 -inf 面
        self.rebind(a["add_mask"], 1, mask)
        self.set_dims(self.out(a["add_mask"]), [1, N_HEADS, 1, self.max_pos])
        self.set_dims(self.out(a["softmax"]), [1, N_HEADS, 1, self.max_pos])
        # probs·V: V 输入 ← gather v (原 GQA v 展开变死码被剪)
        self.rebind(a["matmul_v"], 1, vt)

    # ---------------- 可达性剪枝 ----------------
    def prune_unreachable(self):
        # producer 须在新增节点后重建 (__init__ 的映射不含 KV/face 新节点)
        producer = {}
        for n, nd in self.nodes.items():
            for o in nd["output_names"]:
                producer[o] = n
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
        layers, input_ids = self.find_layers()
        assert len(layers) == N_LAYERS, f"应 {N_LAYERS} 层, 得 {len(layers)}"
        for i, inn in enumerate(layers):
            a = self.discover_attn(inn)
            self.surgery_layer(i, i, a)
        self.prune_unreachable()
        ins = {t for t, ti in self.tensors.items() if ti["type"] == 0}
        if ins != {input_ids, "/decode/decode_in"}:
            fatal(f"输入面异常: {ins}")
        for n, nd in self.nodes.items():
            for i in nd["input_names"]:
                if i not in self.tensors:
                    fatal(f"{n} 引用缺失张量 {i}")
        return len(layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--params", default=None, help="随行复制(字节不动)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pos", type=int, default=64)
    args = ap.parse_args()

    net = json.load(open(args.net))
    s = Surgery(net, args.max_pos)
    kv_n = s.run()
    kv_slot = KV_HEADS * HEAD_DIM * 2 * 2 * args.max_pos
    print(f"\n状态表: {kv_n} 项 = {kv_n}×KV {kv_slot}B = {kv_n * kv_slot / 1e6:.1f}MB DDR")

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
