# 020 plan: decode 静态 DMA 调度 (编译期固定搬运流程)

> 依赖链: **020 (本线, 连续 DMA 流) → 021 (批处理 M>1, 一次流服务 N 序列)**。
> 单做 020 只值 ~4% (matmul 已 15-16GB/s); 020 的真正价值是给 021 提供
> 「权重搬运与计算解耦的连续流」载体。见文末 021 依赖链。

## 改动面

| 文件 | 改什么 |
|---|---|
| `kernels/include/oplist_parse.h` | v3 新增 DMA schedule footer 契约 + `wt_blob` 增 `dma_base/n_dma` 字段 |
| `compiler/tools/wtop_emit.cpp` | 在 `hmx_arena_max_of` 遍历处顺带生成 footer (扫描 bit5 matmul) |
| `kernels/src/runtime/oplist_exec.c` | DMA 重放层 + `exec_matmul_w4_stream` 改消费全局流 (不拆 TU, 013 前科) |
| `kernels/host/pack_dma_sched_ab.cc` | 合成两臂组包器 (基线 vs footer) |

## footer 契约 (v3 扩展, 非新 opcode)

记录结构 (每 W4TS 权重组一条, 16B 对齐):

```
{ op_idx u32, src_offset u64, len u32, vtcm_dst u32 }
```

编译器扫描 decode 图内所有 bit5 matmul (`em.ops` 遍历, 同 `hmx_arena_max_of`),
把每个 op 的权重组按 op 顺序摊平成一张全局表。**每 token 完全一样, emit 一次**。
权重 `src_offset` 取 slot 内偏移 (u64, size-agnostic — [[gehtp-streaming-weight-roadmap]]);
`vtcm_dst` 用「组粒度偏移」编码 (双缓冲组索引, 非绝对地址)。

## 执行器设计 (oplist_exec.c, 不拆 TU)

```
DMA 重放层 (每 token 一次, 全局):
  读 footer → 摊平所有权重组的 (src, len, vtcm_dst)
  全局 VTCM 双缓冲 [bufA | bufB], 大小 = max 单组 × 2 (数百 KB ≪ 16MB)
  提前 N 组 desc 菊花链提交, 持续重放 (晚 poll, 同 018 ring 纪律)

计算消费层 (per matmul, 仍走 wt_exec_run_range 循环):
  exec_matmul_w4_stream 改为: 从全局双缓冲取本 op 组 tile 算
  (激活量化仍 per-op 串行; 权重已在缓冲, 免冷启动)
  算完不 drain (dma_wait_for_idle 语义从「per-op 排空」改「仅图末一次」)

图末: dma_wait_for_idle 一次
```

关键: `dc_dma_reset()` 的 per-op 计数改按 footer 段记 (op_ts 归因仍可回溯到 op)。

## 为什么 >4% (诚实边界)

全局流让 DMA 引擎在**非 matmul op (delta/conv/GDN) 计算期间**持续预取下一个
matmul 的权重, 把这部分计算与搬运重叠。吃掉的不仅是 ~4% op 间启动, 还有
部分 18% 非 matmul 时间 —— **但上限受 DDR 总带宽约束** (非 matmul op 自身也
DDR-bound, 与预取争带宽)。真实收益在 4% 与 4%+18% 之间, 实测定 (探针)。

## 数值声明

DMA 顺序变化不改变搬运字节内容 (同权重/同长度/同目的地) → 对 W4TS 输出应**位级同**
(q8_0 量化误差已在上游 018 消化)。若双缓冲使 tile 落点 offset 变, 仅累加序变,
精度门用 cos 兜底 (≥0.9999), 非位级同须呈报用户裁决。

## emit 判别 (wtop_emit.cpp)

不动 `op_matmul.cpp` 的 bit5 emit 判别 (018 已定)。footer 生成只在
`hmx_arena_max_of` 的 bit5 分支处追加: 命中 `(a[6]&32u)` 的 op, 追加其权重组到
footer 列表, 不改变任何既有 emit/arena 逻辑。

## 对拍资产

`pack_dma_sched_ab.cc`: 同确定性 Q4_0 权重字节, 两臂 blob 仅 footer 有无:
- A: 无 footer → per-op ring (018 基线)
- B: 有 footer → 全局流
例 54: 两臂跑板, 输出 cos/max_rel_err + 各臂 µs → GB/s (G2 精度 + 性能差)。

## 风险

1. **DDR 争用**: 全局预取与非 matmul op 的 DDR 读争带宽, 可能净负 → 探针实测,
   若净负则 footer 只覆盖 matmul 段、非 matmul 段不预取 (回退 per-op)。
2. **VTCM 布局冲突**: 双缓冲与 013 US-A 的 VTCM 布局/carve 区冲突 → 复用
   `hmx_arena_max_of` 闭式预算, 双缓冲并入 arena 上界, 不另起 runtime 决策。
3. **SPLIT chunk 边界**: lm_head 等 SPLIT matmul 的 n0 分块, footer 的 op_idx
   须对齐到 chunk 粒度; lm_head 组异常大 (>双缓冲) → 回退 per-op ring 不参与 footer。
4. **013 拆分前科**: 全部新代码留在 oplist_exec.c static, 不 de-static / 不拆 TU。

## 021 依赖链 (north-star)

020 交付「编译期固定的连续 DMA 流」后, 021 批处理在此之上:

1. **N 序列 KV cache**: 28 层 × 2(K+V) × 1024 × 64 × 2B ≈ 7.3MB/序列, 32 序列 = 235MB DDR。
2. **连续批调度器**: 聚合 N 条请求 → 一次 M=N forward (现 `gehtp_server.py` 每 token 单次 M=1)。
3. **批-M decode 图**: M 从 1→N, 同 matmul 权重读不变、MACs × N → 算术强度 32×,
   memory-bound 翻 compute-bound。020 的 footer 流正是喂批的载体 (一次流服务 N 序列)。

021 目标: 0.6B decode 8.46 → 65-85 tok/s (对标/超 llama.cpp 45.2), 2 NSP 双序列再 ×2。
单序列延迟不降 (换吞吐不换快), 单序列提速须投机解码 (另线)。
