# 020: decode 静态 DMA 调度 (编译期固定搬运流程)

2026-11-04 立项。前置 018 (W4TS, 57ecb84 已合入主线)。分支待定 off `57ecb84`。

> **north-star**：本线是**批处理 (M>1)** 的必要前件 —— 批处理要求权重搬运与
> 计算解耦成一条可复用的连续流，本线把这条流在编译期固定下来。单序列收益
> 仅 ~4%（诚实标注），真正的 8–16× 杠杆是下游批处理 + 2 NSP 双序列吞吐
> （见 [[gehtp-decode-batching-feasibility]] [[gehtp-2nsp-perf-target]]）。

## 问题

decode 稳态 37.9ms/步 (0.6B Q4, W4TS 探针 57 归因)：

| 成分 | 时间 | 占比 | 现状 |
|---|---|---|---|
| lm_head (97MB 大块) | 6.1ms | 16% | 16GB/s 已到峰值 |
| 252 层 GEMV (~2MB/个) | 25ms | 66% | ~15GB/s 接近峰值 |
| 其余 (delta/conv/GDN/norm + 间隙 + setup) | ~7ms | 18% | 剩余大头 |

matmul 已跑在 ~15-16GB/s（峰值附近），`exec_matmul_w4_stream` 的 8 槽 ring 已在
**op 内**流水（`dma_wait_for_idle` 循环结束时实为 no-op）。剩余开销在 **op 之间**：

- 每 op 前奏：激活量化 (f16→f32→q8_0, 串行) + 提交 8 desc + 等首组；
- 分发循环 `wt_exec_run_range` 严格串行（op i 全完才 op i+1），前奏不重叠。

**核心洞察**：decode 图是静态的 —— 每 token 跑同一套 253 个 op、同样的权重地址、
同样的顺序，变的只有激活数据。因此权重搬运可以**编译期排成一条连续 DMA 流，
运行期逐 token 重放**。fence_handoff 已利用「权重只读」这一静态性（flush 一次），
本线把同一洞察再推一步：权重的**顺序**也是静态的。

## 既有资产盘点 (本方案零新格式，复用 018)

| 件 | 状态 | 位置 |
|---|---|---|
| W4TS 内核 + 8 槽 ring | ✅ 018 已上线 | `exec_matmul_w4_stream` oplist_exec.c:2369 |
| fence_handoff 有界集合 (权重只读 flush-once) | ✅ 018 已上线 | 同上 :2422 |
| 异步 DMA 菊花链 submit+晚 poll | ✅ 主线 | `dma_desc_submit/dma_desc_is_done` dma_utils.c |
| 编译期遍历 ops 判 bit5 (arena 预算) | ✅ 018 已上线 | `hmx_arena_max_of` wtop_emit.cpp:829 |
| blob v3 footer 先例 (注解/carve/gdn) | ✅ 主线 | oplist_parse.h :62-69 |

## 方案 (编译期全固定，执行器零运行时决策)

1. **契约**：新增 **DMA schedule footer**（编译器产出，v3 扩展，非新 opcode）。
   记录 decode 图内每个 W4TS 权重组：`{ op_idx u32, src_offset u64, len u32, vtcm_dst u32 }`。
   编译器在 `hmx_arena_max_of` 遍历 `em.ops` 判 bit5 处，顺带把每个 bit5 matmul 的
   权重槽偏移/长度按 op 顺序排成一张全局表（每 token 完全一样，emit 一次）。
2. **执行器两层拆分**（`exec_matmul_w4_stream` 内部，不拆 TU —— 013 拆分 NO-GO 前科）：
   - **DMA 重放层**：读 footer，把整条 decode 图权重搬运当一条连续流跑，全局双缓冲
     （大小 = max 单 op 组，数百 KB ≪ 16MB VTCM），提前 N 组流水；
   - **计算消费层**：每个 matmul 从双缓冲取它那组 tile 算，算完**不 drain**，让位给
     下一个 op 的预取。
   - 删掉 `dma_wait_for_idle()` 的跨 op 排空语义（改成仅图末排空一次）；
     `dc_dma_reset()` 的 per-op 计数改按 footer 段记。
3. **2 NSP 口径**：footer 按域生成（每域独立 schedule），双域各跑一路序列，
   各自那条连续 DMA 流独立喂，互不抢带宽。单序列分片到双域不加速（op 更碎，
   延迟更主导 —— [[gehtp-014-t020-f16-shard-speedup-fail]] 前科）。

## 数值声明 (诚实)

DMA 顺序变化不改变搬运的字节内容（同一权重、同一长度、同一目的地），故本线对
W4TS 输出应为**位级同**（与 018 的 q8_0 量化误差无关，那已在上游消化）。若双缓冲
导致 tile 落点 offset 变化，仅影响累加顺序不改变数学结果，精度门用 cos 兜底。

## 门

- G1 零回归：无 bit5 的 f16/Q4 blob 输出位级同（footer 只加在 bit5 op 上）；ctest 全绿。
- G2 精度：0.6B decode cos ≥ 0.9999 / argmax == golden（应位级同；非位级同须呈报用户裁决）。
- G3 性能：0.6B Q4 decode 37.9ms → 目标 ~36.5ms（**~4%**，诚实锚：matmul 已 15-16GB/s）。
- G4 2 NSP：双域各跑一路序列，合计吞吐 ~2×，单域位级同 018 锚。

## 非目标

- 批处理 (M>1) 本身（本线是它的前件，批处理另立 021）。
- 非 matmul op (delta/conv/GDN/norm) 的 memory-bound 优化（013 US-A/US-C 线，非 DMA 调度）。
- prefill (M≥32 HMX 路径不动)。
- 4B 线的 40-bit 流（019 已立，本线不碰 40-bit 寻址）。

## 假设 (合理默认，可在 /speckit-plan 调整)

- **双缓冲粒度 = max 单 op 组**：编译期闭式求得（同 `hmx_arena_max_of`），数百 KB ≪ 16MB
  VTCM；vtcm_dst 用组粒度偏移编码。若某 op 组异常大（如 lm_head）则回退 per-op ring 不参与 footer。
- **footer 覆盖面 = 仅 W4TS matmul (bit5)**：本线先吃 ~4% 的 op 间 matmul 启动延迟；
  非 matmul op（RMSNorm/conv/delta/GDN）权重搬运是 013 US-A/US-C 的 VTCM 驻留线，不并入 footer。
- **2 NSP footer = 编译期生成两份**（每域一份独立 schedule），符合「编译期全固定、执行器零决策」；
  运行期不做 offset 平移。
