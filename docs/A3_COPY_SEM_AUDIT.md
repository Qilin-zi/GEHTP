# A3 copy-sem 审计 + 数值判决报告 (2026-09-17, 105 会话)

> 输入: `test_models/qwen35_08b/conv/model_net.json` (15129 节点)
> 工具: `scripts/audit_copy_sem_sites.py` (静态) + `GEHTP_HOST_*_IDENTITY` 设备恒等仿真 (数值)
> 结论先行: **ScatterNd 恒等 = 数值死刑 (L0 cos 0.102), A3② 真语义上设备进关键路径;
> Pad 恒等在 L0 无害 (有挂账); Cast 待全模型判决; Reshape 恒安全。**

---

## 1. 静态审计 (audit_copy_sem_sites.py)

图输出张量 `logits` (tensors type==1), 产出节点 `/lm_head/MatMul_post_reshape`,
反向可达 15129/15129 — **全部 copy-sem 站点均可达 logits, 无死端**:

| 型 | 站点 | 可达 | 静态判 |
|---|---|---|---|
| Reshape | 4180 | 全 | SAFE (恒等=语义正确, 连续同字节) |
| ScatterNd | 1154 | 全 | REVIEW → 数值判决 §2 |
| Pad | 90 | 全 | REVIEW → §3 |
| Cast | 1 | 是 (`/model/rotary_emb/Cast_2`) | REVIEW → §4 |

ScatterNd 形态 (全部同构): `ScatterNd(data←计算链, indices←Concat const, updates←Reshape←计算链, reduction=0)`;
host 语义注释 (ops.cpp:894): "GDN chunk gated delta rule: 每头 (h,0,0) 坐标写块"。
18 个 GDN 层 × 64 站/层 = 1152 (+2) — 全在 linear_attn 内。

## 2. ScatterNd 数值判决 (L0 GDN 层, host 设备恒等仿真)

方法: host execute 真语义 vs 仿真设备 emit (UNARY 恒等拷贝 = data 直通丢弃 scatter 更新),
同输入同权重对比输出 (gold_ncf = HF 锚点, 本报告 §6 已重证其正确性)。

| 实验 | vs | cos | 判 |
|---|---|---|---|
| 真语义 (修复后链) | gold | **0.999999** (max\|d\|=0.00023) | 基线正确 |
| 恒等仿真 (scatter+pad+cast) | 真语义 | **0.102233** (max\|d\|=0.9327) | **死刑** |
| 仅 ScatterNd 恒等 | 真语义 | **0.102233** | **全罪** |
| 仅 Pad 恒等 | 真语义 | **1.0** | L0 无害 (§3 挂账) |

**判决: ScatterNd 必须真语义上设备 (A3②), 进 G2 前置关键路径。**
L0 的 64 站 ScatterNd 写的是 GDN delta rule 的每头状态/矩阵块 — 丢弃更新 = GDN 数学塌掉。

## 3. Pad 判决 (L0)

L0 共 5 站 Pad (scheme=0 CONSTANT, val=0): `[1,32,128,16]→[1,64,128,16]` ×3, `[1,32,16]→[1,64,16]` ×2
(chunk 对齐补零)。恒等仿真 (输入前置 + 尾部补零模拟) vs 真语义 cos=1.0 — 前段内容逐位一致,
下游对补零区不敏感 (GDN chunk 掩码覆盖)。
**挂账**: 设备上恒等的尾部 = temp 池陈旧垃圾而非零 — 若下游某形态读尾区即破;
全模型 90 站的终审走 P1 恒等仿真 + 设备 run 门 (本报告随 P1 更新)。

## 4. Cast 判决 (待全模型)

全模型唯一 Cast = `/model/rotary_emb/Cast_2` (RoPE position 链, int32 位模式→float 数值)。
设备恒等 = int32 位模式当 f16 读 = 位置值变 denormal ≈ 0 — **若未被 pass_const_fold 折叠吃掉,
RoPE cos/sin 全错**。L3 注意力层不含此 Cast (cos/sin 烘焙输入), 故 L3 设备绿不构成证据。
判决实验: P1 全模型 host 恒等仿真 (GEHTP_HOST_CAST_IDENTITY=1) + blob 静态检查该站是否折叠。

## 5. L3 问题回答 (任务书 A3 注意项)

**"L3 的 ScatterNd 为何恒等安全" → 伪命题**: L3 net.json 含 **0 个 ScatterNd、0 个 Pad、0 个 Cast**
(仅 12 型, copy-sem 仅 Reshape — 恒安全)。L3 设备绿从未考验恒等拷贝路径,
"L3 在恒等拷贝下通过" 不构成任何 ScatterNd 证据。

## 6. 附带重大发现: VL 在途 Gather 重写回归 (已修)

审计中发现现行 host 链 (8936c01+VL) L0 输出 vs HF 锚点 cos 仅 0.53。
bisect 定罪: **VL 在途的 Gather 通用化重写** — 前导 size-1 维 compact 后 axis 未同步左移:
L0 Gather tbl `[1,16,1,64,64]` axis=3 (converter 按原始 rank 写), compact `[16,1,64,64]` 后真轴=2,
VL 用 axis=3 取末维 64 → 全层数据错位 (embedding 修复保留, GDN 标量查表被打烂)。
修复: `axis -= (原始rank - 压缩后rank)` (ops.cpp, 105 会话)。
验证: 修复链 vs HF 锚点 cos=1.0, vs gold_ncf 0.999999; pre-VL ops.cpp 对照组同为 1.0;
4090 VL 二进制对照组同为 0.53 (回归属 VL 在途, 非本树新引入)。
**注意: 该 bug 在 VL 在途内, 104 基线树的 host_run 当前对 L0/全模型 GDN 层输出均不可信
(全模型 66×18 GDN gather 同型) — 修复须尽快回灌 104。**

## 7. 静态断言门 (入仓)

`scripts/audit_copy_sem_sites.py` 末尾断言: 存在「可达且非恒安全」站点即 rc=1。
当前 rc=1 (ScatterNd×1154 + Pad×90 + Cast×1 全可达) — 门保持红色直到:
A3② 落地 (ScatterNd 出 copy-sem) + Pad/Cast 判决闭合。
