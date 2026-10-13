# Research: MatMul HMX 流水线

## R1 — HMX acquire/release 的成本与「外提」可行性

**Decision**: 把 `hmx_enable_execution` / `hmx_unit_acquire` / `hmx_unit_release` / `hmx_disable_execution` 从 `(r,c)` tile 循环内提到循环外，整段走查 acquire 一次、release 一次。

**Rationale**:
- 两层锁，成本不同（[hmx_common.c:223-273](kernels/src/runtime/hmx_common.c#L223)）：
  - `hmx_enable_execution` = `HAP_compute_res_hmx_lock2(ctx, HMX_SHARED)` —— HAP 资源/电源域握手，贵。
  - `hmx_unit_acquire` = `memw_locked` 软件自旋锁 —— 便宜。
  - 当前每 tile 都付两层成本；M=32 时 `n_col=192` → 192 次。
- 累加器边界已由 `mxclracc.hf` 提供，acquire/release 不是 accumulate/readout 的同步点（见 [hmx_convf16.c:109-141](kernels/src/hmx/hmx_convf16.c#L109) 每 tile 的 asm 序列：`mxclracc` → `bias=mxmem2` → 32×`mxmem:deep` → `cvt.hf=acc` → `mxmem=cvt`）。
- 参考实现 `hmx_convf16.c` 本身也是逐 `(mt,nt)` acquire/release——即「逐 tile 握手」是从参考里**照搬**的，不是「已知最优」，无证据表明外提不安全。

**Alternatives considered**: 逐 tile 保持现状（保留正确性、无收益）；只外提 spinlock 而保留 HAP lock 逐 tile（省小头留大头，收益有限）。

**Risk**: 跨 tile 背靠背 `mxmem:deep` 装载是否需显式 drain 未在文档确认，必须设备侧位级门验证，不作纸上假设。单线程 runtime 无 HMX 竞争，长持 HAP lock 反而避免重复电源门控，可能是正收益。

## R2 — tile 间装载与 MAC 重叠（软件流水）

**Decision**: 在「外提握手」之后，进一步软件流水 tile 走查：tile c+1 的 `mxmem:deep` 装载与 tile c 的累加/读出重叠。

**Rationale**: MX "deep" 内存的 `:deep` 后缀表明硬件已对 K 循环内做双缓冲；tile 级流水是在 `cvt.hf=acc` 读出边界之外再加一层。是否可行取决于累加器读出与下一 tile `mxmem:deep` 之间能否乱序/重叠。

**Alternatives considered**: 仅外提握手（步骤化、风险低、先拿 192→1 的大头）；完全流水（收益更大但需设备逐位验证）。

**Risk**: 累加器读出与下一装载重叠若破坏部分和顺序 → 数值漂移；必须由 FR-003/FR-004 的位级门兜底。

## R3 — op 级权重预取需要异步 DMA

**Decision**: 新增 `dc_dma_submit`（`dmstart` 触发即返）+ `dc_dma_wait`（poll `dma_desc_is_done`）拆分当前阻塞的 `dc_dma_once`。

**Rationale**: 当前 [dc_dma_once_ex](kernels/src/runtime/dc_parts.c#L156) 完全阻塞（mutex → poll-idle → `dmstart` → poll-idle → poll-done），每调 ~700µs 固定开销（mutex + 两次 poll-idle）。已有原语 `dmstart`/`dmwait`/`dmlink`/`dma_desc_is_done`（[dma_cmds.h](kernels/include/dma_cmds.h)、[dma_desc.h](kernels/include/dma_desc.h)）支持异步；`dmlink` 链式（T013）当前被禁（R-D1 只走 `dmstart` 分支）。

**Alternatives considered**: 继续用阻塞 `dc_dma_once`（w_pp 时 ~0.3ms+0.7ms 固定，可接受但不重叠）；启用 `dmlink` 链式（省固定开销但仍是阻塞提交）。

**Risk**: 异步 DMA 的 fence 顺序（FC_CPU→FC_DMA clean、DMA→HMX invalidate）必须在 submit/wait 间保持铁律；双缓冲暂存区占用需预算约束（FR-007）。

## 结论与推荐顺序

1. **先做 R1（外提握手）**：最小改动、对准 ~100ms 主体、正确性风险最低（`mxclracc` 已提供边界），是 192→1 的大头收益。
2. **再做 R3（异步 DMA 预取）**：次目标，需新增 submit/wait 接口，收益受 w_pp 限制（~0.3ms/op）。
3. **最后 R2（tile 软件流水）**：收益最大但风险最高，依赖 R1 落地后的设备位级验证结论。

---

# 实证后记 (2026-10-13 设备数据, 推翻/修正上列三研究的假设)

## R1 外提握手 — 假设证伪: 零收益

lift1 板测: 位级同 md5, wall 14120→14116ms (0.03% 噪声)。hmx_enable_execution/unit_acquire 的逐 tile 成本可忽略 (软锁 ~ns 级)。外提保留 (192→1/op, 卫生改善)。

## R2 tile 流水 — 双重死亡

1. **数据**: 逐 (r,c) tile 固定 ~531µs 且与 n_dot(K) 无关 (K=64 与 K=1024 同价), 逐 op 零截距线性 → 装载/发射/MAC 都不是成本, 无流水化对象。
2. **ISA**: 汇编器硬规则 `packet has HMX weight without an activation` — act/wgt 装载必须同包配对; act K 链驻留跨列复用在 v81 不存在。链式形态 (ggml range=2048·n_dot−1) 板测性能中性 (44.0 vs 44.4ms); n_dot>32 时 range 溢出 16 位 dW 字段 → 4 op 回落标量 = lift2 的 +84% 全部来源。

## R3 异步预取 — 收益对象消失

真瓶颈 = **标量 crouton 权重 pack** (~16ns/元素, 531µs/tile), 且它是**供给侧问题**: blob 漏 --dma-runlist → w_pp 预打包路径未启用。重 emit 后 pack 消失 (13.58s→2.76s, −79.7%, 位级同)。pp 后串行 DMA Σ=22.6ms = 0.82% wall = 异步化收益上限; 大 FC 乒乓双缓冲 25MB > VTCM 16MB 物理不可行; 700µs 固定开销假设不复存在 (22.6ms ≈ 纯传输 66.5GB/s)。**US3 整体 NO-GO。**

## 教训 (方法论)

- "固定成本/tile 且与 K 无关"的签名 ⇒ 先查 pack/staging, 不查计算引擎。
- Σdma_us=0 + 槽计数 (无 [pp]) 是供给侧断链的最快指纹; 改内核前先对 emit 侧旗标。
- 结论排序修正: 供给侧(emit 旗标) > 引擎执行细节; R1→R2→R3 的推荐顺序在本 blob 上全部失效。
