# 013 附加实测 — w4 线 0.8B 当前树全链重编 + 板跑（2026-10-14，设备 d0f1784）

## 动机

013 原始 spec 承诺「w4 线对 007 q4 锚 25.7s 实质下降」；T011 已把**原生 W4A16** E2E 收益移出范围（M%256 NO-GO）。本次以**当前树全链重编**（编译+发射）兑现该口径：007 的 q4 锚 blob 复用 003 时代 tagged.bin（无 gdn 融合、无编译期折叠），从未与 f16 线同等受益。

## 链路

1. **编译**（直通模式，~2 min）：`hnnx_compile --net-json qwen35_08b_v2_net.json --weights-bin model.params.bin --format tagged --no-pass fc_bias_fold --opt-stats --output qwen35_08b_v13.tagged.bin`（.scratch_013/compile_v13.log）
   - `gdn_block: recognized=18 lowered=18 coverage=18/18`（融合烤入）
   - `before=26399 after=1617`（shape_fold 308 + eltwise_const_fold 54）
   - `[memplan] vtcm auto off: scratch_max=16760832 + head ≥ VTCM 16MB` — lm_head 分块 16.76MB 顶满 16MB，VTCM auto 撤退（与 010/011 全部现存 0.8B 正典一致：TEMPOFF/VTCL 从未在 0.8B 打开，非本次回归）
2. **发射**：`wtop_emit --bin v13.tagged.bin --input-f16 ids,mask,pos --dma-runlist --gguf Qwen35-0.8B.gguf --match 007 match_map.tsv`
   - 输入绑定逐槽核验（GEHTP_DEBUG_INPUTS）：k=0 op1←ids / k=1 op2←mask / k=2 op3←pos（与 010b 设备实证序一致，本次板跑精度反证）
   - `WTOP OK: slots=439 ops=920 bytes=471917952 (gguf hits=285 miss=147) fold=92`（旧图 11504 ops → 920）
   - `w4 469565440 B (447.8MB) / f16 1291328 B，w4 占比 99.7%`；w4 字节与 007d 逐字节同量（469565440）
   - manifest：out_temp=**114**（旧图 2796）、input_slot=0、input_elems=32
   - blob md5 `5be4bd7f136ab762a9e401b031c0e4cc`
3. **板跑**：lib_018（013 树，签名版落地普通名）+ runner_018b + cl020/ 私有目录；单 blob + 4 运行时输入文件（job `input` 覆盖输入槽）

## 双门结果

### 精度门（冻结门 pooled cos≥0.98 + top1≥20/32，vs cpu_golden/q4_0）

| prompt | pooled cos | top1 | 门 |
|---|---|---|---|
| 0 | 0.999802 | 31/32 | PASS |
| 1 | 0.999763 | 31/32 | PASS |
| 2 | 0.999963 | 32/32 | PASS |
| 3 | 0.999954 | 31/32 | PASS |

**4/4 PASS**，全部远超门。注意 `.scratch_007/judge_q4d.py` 内嵌的 0.99864/29 是**占位门（按构造不可达，q4_gate.md 明示）**，判分以 0.98/20 冻结门为准（本文件用独立脚本）。

### 性能门（vs 007 q4 锚 exec 25.7s / wall 27.4s）

| 口径 | 013 重编 w4 线 | 007 锚 | 变化 |
|---|---|---|---|
| exec | **15.09s** | 25.7s | **−41.3%** |
| wall | 16.3s | 27.4s | −40.5% |

op_timing 分解（920 ops，sum_op_us=15069372）：

| opcode | n | total | 007 锚对照 |
|---|---|---|---|
| 24 MATMUL_F16 | 199 | 9.39s | 10.6s（−11%） |
| 35 lm_head | 32 | 4.59s | 4.61s（持平，路径未动） |
| 37 gdn_block 融合 | 18 | 0.65s | 旧图未融合、散在 ~10.5s 杂项（**主刀**） |
| 25 | 79 | 0.25s | — |
| 其余 | ~590 | ~0.19s | — |

## 诚实归属

- 收益来源 = **当前树全链重编**：gdn 融合（009/c77665a 默认开，18/18）+ 编译期折叠 + 发射折叠，**不是 013 A/C 段机制**——0.8B 图 VTCM auto 规划撤退（无 TAG_MEM_PLAN→无 TEMPOFF/VTCL），q4 线无 W4A16 op（CRVL 不发），A/C 机制在 0.8B 上本就不触（与 A 段收口注记一致）。
- 007 锚的 25.7s 是「旧图+当代 lib」口径；本实测补齐「当代图」欠账，w4 线与 f16 线（2.76s）同源同代。

## 资产

- `.scratch_013/`：compile_v13.log、qwen35_08b_v13.tagged.bin（4.07GB）、qwen35_08b_q4_v13.wtop（471917952B）+ manifest + w4report、out_q4v13_{0..3}.f16.raw、q4v13_{0..3}.device.txt
- 板上：cl020/{qwen35_08b_q4_v13.wtop, input_q4v13_{0..3}.i32.raw}（后续 D 段/merge 验证可直接复用）

## 坑

- 复用 003 时代 tagged.bin 发 q4（本日首试）= 无 gdn 融合的裸图（ops 11504，无 VTCL/TEMPOFF 亦无门可期）；**q4 线必须重编译**而非只重发射。
- 判分脚本占位门陷阱（见上）。
- out_temp 每图必读 manifest（本图 114，007 图 1620，010 图 2796）。
