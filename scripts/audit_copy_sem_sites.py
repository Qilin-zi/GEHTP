#!/usr/bin/env python3
"""audit_copy_sem_sites.py — A3-① ScatterNd/Pad/Cast/Reshape copy-sem 站点静态审计

背景: wtop_emit 的 op_copy_sem 对 Reshape/Pad/ScatterNd/Cast 四类统一发射
OP_UNARY_F16 恒等拷贝 (第一输入直通)。恒等对各型的真实含义:
  Reshape : 恒等 = 语义正确 (连续同字节)               → 恒安全
  ScatterNd: 恒等 = 丢弃 scatter 更新                   → 仅当输出不到 logits (死端) 安全
  Pad     : 恒等 = 丢弃 padding                        → 仅当 pad 区域下游不被读 安全
  Cast    : 恒等 = 位模式直通; host Cast 语义 = int32 位模式→float 数值 → 直接冲突

审计口径 (全部静态, 不改 emit):
  1) 可达性: 反向 BFS 自 Output 边界节点 — 站点输出是否到 logits
  2) ScatterNd 形态: reduction 键 / indices·updates 来源 (const?)
  3) Pad 形态: pads 参数 + 下游是否 StridedSlice 裁回
  4) Cast 形态: 输入数据类型线索

产出:
  - docs/A3_COPY_SEM_AUDIT.md (分类表 + 处置 + L3 问题回答)
  - stdout 汇总; 末尾静态断言: 存在「可达且未判安全」站点则 rc=1 (入仓作门禁)

用法: audit_copy_sem_sites.py [model_net.json] [--report docs/A3_COPY_SEM_AUDIT.md]
"""
import json
import sys
import argparse
from collections import deque, Counter

COPY_SEM_TYPES = ("ScatterNd", "Pad", "Cast", "Reshape")


def load_net(path):
    nj = json.load(open(path))
    return nj["graph"]["nodes"], nj["graph"].get("tensors", {})


def build_maps(nodes):
    """tensor -> producer node name; node -> consumers list."""
    prod = {}
    for name, nd in nodes.items():
        for out in nd.get("output_names", []):
            prod[out] = name
    consumers = {name: [] for name in nodes}
    for name, nd in nodes.items():
        for inp in nd.get("input_names", []):
            p = prod.get(inp)
            if p:
                consumers[p].append(name)
    return prod, consumers


def reachable_to_output(nodes, consumers):
    """反向 BFS: 从 Output 边界节点沿 input_names 回溯所有祖先。"""
    roots = [n for n, nd in nodes.items()
             if nd.get("type") == "Output" or n == "Output" or n.endswith("/Output")]
    seen = set()
    dq = deque(roots)
    while dq:
        cur = dq.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for inp in nodes.get(cur, {}).get("input_names", []):
            # input_names 是张量名; 找生产者
            pass
    return seen, roots


def ancestors(nodes, prod, tensors):
    """反向 BFS 自图输出 (tensors 里 type==1 者, 如 logits), 经 prod 映射张量→节点。"""
    out_tensors = [t for t, td in tensors.items() if td.get("type") == 1]
    roots = sorted({prod[t] for t in out_tensors if t in prod})
    seen = set()
    dq = deque(roots)
    while dq:
        cur = dq.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for inp in nodes.get(cur, {}).get("input_names", []):
            p = prod.get(inp)
            if p and p not in seen:
                dq.append(p)
    return seen, roots, out_tensors


def scalar_keys(nd):
    return {k: list(v.keys()) for k, v in nd.get("scalar_params", {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net_json", nargs="?",
                    default="test_models/qwen35_08b/conv/model_net.json")
    ap.add_argument("--report", default="docs/A3_COPY_SEM_AUDIT.md")
    args = ap.parse_args()

    nodes, tensors = load_net(args.net_json)
    prod, consumers = build_maps(nodes)
    anc, roots, out_tensors = ancestors(nodes, prod, tensors)

    # 收集 copy-sem 站点
    sites = {t: [] for t in COPY_SEM_TYPES}
    for name, nd in nodes.items():
        t = nd.get("type")
        if t in sites:
            sites[t].append(name)

    rows = []          # (type, name, reachable, form_note, verdict)
    verdict_count = Counter()

    for t, names in sites.items():
        for name in sorted(names):
            nd = nodes[name]
            outs = nd.get("output_names", [])
            reach = any(c in anc for c in consumers.get(name, [])) or name in anc
            ins = nd.get("input_names", [])
            sp = nd.get("scalar_params", {})
            if t == "Reshape":
                verdict, note = "SAFE", "恒等=语义正确(连续同字节)"
            elif t == "ScatterNd":
                red = [k for k in sp if "reduction" in k.lower()]
                idx_src = ins[1] if len(ins) > 1 else "?"
                upd_src = ins[2] if len(ins) > 2 else "?"
                idx_const = prod.get(idx_src, "")
                idx_is_const = "Const" in idx_const or nodes.get(idx_src, {}).get("type") in (None,)
                note = f"reduction={red or '无'}; indices←{idx_src}; updates←{upd_src}"
                verdict = "SAFE_DEADEND" if not reach else "REVIEW"
            elif t == "Pad":
                pads = sp.get("pads", sp.get("pad_amounts", "?"))
                # 下游是否 StridedSlice (pad 后被裁回的迹象)
                cons_types = Counter(nodes[c].get("type") for c in consumers.get(name, []))
                note = f"pads={pads}; 下游={dict(cons_types)}"
                verdict = "SAFE_DEADEND" if not reach else "REVIEW"
            else:  # Cast
                note = f"输入←{ins[0] if ins else '?'}"
                verdict = "SAFE_DEADEND" if not reach else "REVIEW"
            rows.append((t, name, reach, note, verdict))
            verdict_count[(t, verdict)] += 1

    # 汇总
    total = len(rows)
    review = [r for r in rows if r[4] == "REVIEW"]
    print(f"== A3-① copy-sem 审计: {args.net_json}")
    print(f"图输出张量: {out_tensors}; 产出节点: {roots}; 反向可达节点 {len(anc)}/{len(nodes)}")
    for (t, v), c in sorted(verdict_count.items()):
        print(f"  {t:10s} {v:14s} × {c}")
    print(f"  合计 {total}; 待人工 REVIEW (可达且非恒安全) = {len(review)}")

    # REVIEW 明细 (全列, 这是审计的核心产出)
    if review:
        print("\n== REVIEW 明细 (可达非恒安全):")
        for t, name, reach, note, verdict in review[:200]:
            print(f"  [{t}] {name}\n      {note}")

    # 报告落盘
    with open(args.report, "w") as f:
        f.write("# A3-① copy-sem 站点静态审计报告\n\n")
        f.write(f"- 输入: `{args.net_json}`\n")
        f.write(f"- 图输出张量: {out_tensors}; 产出节点: {roots}; 反向可达 {len(anc)}/{len(nodes)} 节点\n")
        f.write(f"- 站点合计 {total}: " +
                ", ".join(f"{t}/{v}×{c}" for (t, v), c in sorted(verdict_count.items())) + "\n\n")
        f.write("## 判据\n\n"
                "- Reshape: 恒等=语义正确 → 恒安全\n"
                "- ScatterNd/Pad/Cast: 输出不到 logits (死端) → SAFE_DEADEND; 可达 → REVIEW\n\n")
        f.write("## REVIEW 明细\n\n")
        for t, name, reach, note, verdict in review:
            f.write(f"- [{t}] `{name}` — {note}\n")
        f.write("\n## 全站点 CSV\n\n```\ntype,name,reachable,verdict,note\n")
        for t, name, reach, note, verdict in rows:
            f.write(f"{t},{name},{int(reach)},{verdict},{note}\n")
        f.write("```\n")
    print(f"\n报告 → {args.report}")

    # 静态断言门: 存在 REVIEW 即 rc=1 (需人工处置或判死端证据)
    sys.exit(1 if review else 0)


if __name__ == "__main__":
    main()
