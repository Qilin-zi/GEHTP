#!/usr/bin/env python3
"""wtlink_qwen3_06b.py — Qwen3-0.6B net.json FC 节点 ↔ GGUF 张量匹配器

与 gguf_wtlink.py (qwen35) 同构, 适配香草 qwen3 (llama 式 GGUF 角色)。
角色分配依据 (2026-09-22 实测对齐):
  - net.json FC 节点序 = ONNX/torch 前向序: 每层 q,k,v,o,gate,up,down
    (attention BMM 节点夹在 v/o 之间, 不在权重消费集内);
  - GGUF blk.L.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight
    与之一一对应; 形状交叉验证 (GGUF ne 序 dims 反转 = HF [K,N]);
  - lm_head (/MatMul_252) → token_embd.weight (tie_word_embeddings; output.weight
    是 type-14 别名, 无数据);
  - blk.0/1/2.ffn_down = Q4_1 (本链打包器只认 Q4_0) → **排除**, 回落 f16 池
    (数值正确, 且 f16 比 Q4_1 更准)。

用法:
  wtlink_qwen3_06b.py --net-json <work>/qwen3_06b_net.json \
      --gguf /disk2/models/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf \
      --out test_models/qwen3_06b/match_map.json
"""
import argparse
import json
import sys

sys.path.insert(0, "/disk1/GEHTP/scripts")
from golden_qwen35 import gguf_read_full  # noqa: E402

ROLES = ["attn_q", "attn_k", "attn_v", "attn_output",
         "ffn_gate", "ffn_up", "ffn_down"]
Q4_1_EXCLUDE = {f"blk.{l}.ffn_down.weight" for l in (0, 1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-json", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arch, tensors, data_start = gguf_read_full(args.gguf)
    assert arch == "qwen3", f"arch={arch}, 需要 qwen3"
    print(f"[wtlink] gguf: {arch}, {len(tensors)} 张量")

    net = json.load(open(args.net_json))
    nodes = net["graph"]["nodes"]
    g_tensors = net["graph"]["tensors"]

    fc = []
    for name, n in nodes.items():
        if n.get("type") != "FullyConnected":
            continue
        wnames = [t for t in n["input_names"] if t.startswith("onnx::")]
        assert len(wnames) == 1, f"{name}: 权重输入 {wnames}"
        tt = g_tensors[wnames[0]]
        dims = tt["dims"]
        if tt.get("permute_order_to_src"):
            dims = list(reversed(dims))  # 逻辑 [K,N]
        fc.append((name, dims))
    print(f"[wtlink] net.json FC 节点: {len(fc)} (应 197 = 28×7 + lm_head)")

    match = {}
    problems = []
    for idx, (name, dims) in enumerate(fc):
        K, N = dims[0], dims[1]
        if idx == len(fc) - 1:
            assert K == 1024 and N == 151936, f"lm_head 形状异常 {dims}"
            gguf_name = "token_embd.weight"
        else:
            layer, role = idx // 7, ROLES[idx % 7]
            gguf_name = f"blk.{layer}.{role}.weight"
        if gguf_name in Q4_1_EXCLUDE:
            print(f"[wtlink] 排除 Q4_1: {name} → {gguf_name} (回落 f16 池)")
            continue
        if gguf_name not in tensors:
            problems.append((name, gguf_name, "gguf 无此张量"))
            continue
        gtype, gdims, offset, nbytes = tensors[gguf_name]
        # GGUF ne 序 (ne[0] 最快) → numpy 序 = 反转; 期望 [N, K], 逻辑比对 [K, N]
        if gtype == 2:  # Q4_0: ne 反转后为 [N, K]
            expect = (N, K)
            if tuple(reversed(gdims)) != expect and tuple(gdims) != expect:
                problems.append((name, gguf_name,
                                 f"形状 gguf{gdims} vs 期望[{N},{K}]"))
                continue
        match[name] = {"gguf_name": gguf_name, "ggml_type": gtype,
                       "dims": list(gdims), "nbytes": nbytes,
                       "file_offset": data_start + offset}

    if problems:
        print("[wtlink] FAIL:")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(match, f, indent=2)
    tsv = args.out.replace(".json", ".tsv")
    with open(tsv, "w") as f:
        f.write("node\tgguf_name\tggml_type\tfile_offset\tnbytes\tdims\n")
        for k, v in match.items():
            f.write(f"{k}\t{v['gguf_name']}\t{v['ggml_type']}\t{v['file_offset']}\t"
                    f"{v['nbytes']}\t{','.join(map(str, v['dims']))}\n")
    print(f"[wtlink] OK: {len(match)} 节点 (排除 Q4_1 ×{len(Q4_1_EXCLUDE)}) "
          f"→ {args.out} + {tsv}")


if __name__ == "__main__":
    main()
