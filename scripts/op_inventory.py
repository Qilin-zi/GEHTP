#!/usr/bin/env python3
"""op_inventory.py — QNN net.json 算子清单统计 (GEHTP M0)

对 net.json 的 graph.nodes 做: op type 计数、输入/输出张量形状、静态参数计数、
scalar_params 键采样。产出 op_inventory.json 供编译器测试断言。

用法:
  op_inventory.py --net-json model_net.json --out op_inventory.json
  op_inventory.py --net-json model_net.json --expect op_types.txt   # 校验 op 型 ⊆ 期望集
"""
import argparse
import json
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-json", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect", default=None,
                    help="期望 op 型列表文件(每行一个),超出则报错")
    args = ap.parse_args()

    with open(args.net_json) as f:
        net = json.load(f)

    graph = net.get("graph", {})
    nodes = graph.get("nodes", [])
    # 兼容两种 schema: list(旧) 与 dict(按节点名键控, 2.48 qairt-dlc-to-json)
    if isinstance(nodes, dict):
        nodes = [{"name": k, **v} for k, v in nodes.items()]
    type_counts = Counter(n["type"] for n in nodes)
    total_params = sum(int(n.get("params_count", 0) or 0) for n in nodes)
    scalar_keys = Counter()
    io_shapes = {}
    n_with_scalar = 0
    for n in nodes:
        for k in (n.get("scalar_params") or {}):
            scalar_keys[k] += 1
        if n.get("scalar_params"):
            n_with_scalar += 1
    # 输入/输出张量: tensors 区(dict 按名或 list)取 type=INPUT/OUTPUT
    tensors = graph.get("tensors", {})
    if isinstance(tensors, dict):
        tensors = [{"name": k, **v} for k, v in tensors.items()]
    for t in tensors:
        ttype = t.get("type")
        if ttype == "INPUT" or (isinstance(ttype, int) and ttype == 0):
            io_shapes[t.get("name", t.get("id"))] = {
                "type": "INPUT",
                "dims": t.get("dims"),
                "data_type": t.get("data_type"),
            }
        elif ttype == "OUTPUT" or (isinstance(ttype, int) and ttype == 1):
            io_shapes[t.get("name", t.get("id"))] = {
                "type": "OUTPUT",
                "dims": t.get("dims"),
                "data_type": t.get("data_type"),
            }

    inv = {
        "net_json": args.net_json,
        "n_nodes": len(nodes),
        "n_params": total_params,
        "op_types": dict(sorted(type_counts.items())),
        "n_nodes_with_scalar_params": n_with_scalar,
        "scalar_param_keys": dict(sorted(scalar_keys.items())),
        "io_tensors": io_shapes,
    }
    print(f"[inventory] nodes={len(nodes)} params={total_params} op_types={len(type_counts)}")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:32s} {c}")
    print(f"[inventory] scalar_params keys: {dict(sorted(scalar_keys.items()))}")
    print(f"[inventory] io tensors:")
    for name, s in io_shapes.items():
        print(f"  {name}: {s}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(inv, f, indent=2)
        print(f"[inventory] written {args.out}")

    if args.expect:
        with open(args.expect) as f:
            allowed = {line.strip() for line in f if line.strip()}
        extra = set(type_counts) - allowed
        if extra:
            print(f"[inventory] FAIL: 出现期望集外 op 型: {extra}")
            sys.exit(1)
        print(f"[inventory] OK: 全部 {len(type_counts)} 种 op 型 ⊆ 期望集 ({len(allowed)} 种)")


if __name__ == "__main__":
    main()
