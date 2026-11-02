# 018: W4 decode tile 流式搬运 (编译期固定)

2026-10-28 立项。分支 `018-w4-decode-stream` off `9014d83`。

## 问题

decode M<32 Q4 GEMV (`exec_matmul_q4_0_gemv`, oplist_exec.c:2139) = HVX load 单元
散读 DDR + 逐元素 nibble→f32 反量化。实测: 0.6B q4 decode 186.4ms/步 (5.36 t/s),
慢于 f16 118.2ms; 单 op 带宽 ~2-6GB/s。探针 51/52 (2026-10-27) 钉死根因:

- HVX vmemu→DDR 是 ~10-11GB/s 硬墙 (MLP 1-8 全平, 寄存器溢出前无提升);
- 散访问形态 (行跨步) 下 VTCM 驻留不救 (探针 52: 0.99×);
- 唯三出路 (llama.cpp ggml-hexagon 45 t/s 实证, 同 v81 硬件):
  **① 权重 32 行 2D tile 布局 (散→连续) ② DMA 引擎流式搬 (66.7GB/s, 非 HVX load)
  ③ 行并行 rmpy int8 matvec (32 输出列/指令链, 无水平归约)**。

## 既有资产盘点 (本方案零新格式)

| 件 | 状态 | 位置 |
|---|---|---|
| 640B tile repack (ct 主序, A 序 nibble) | ✅ 已上线 (007 US2 Route A) | `repack_q4_0_tiles` wtop_ops.hpp:455 |
| rmpy 行并行 matvec 内核 | ✅ vendored 未接线 | `tiled_vec_dot_q4_0_32x1` hvx-mm-kernels-tiled.h:381 |
| f32→q8_0_tiled 激活量化器 | ✅ vendored | `quantize_row_f32_q8_0_tiled` :192 |
| f16→f32 HVX 转换 | ✅ | `hvx_vec_f16_to_f32` hvx-base.h:146 |
| 异步 DMA 菊花链 submit+晚poll | ✅ 主线 | `dma_desc_submit/dma_desc_is_done` dma_utils.c:133/200 |
| wtcache 链纪律 (unlink 防野指针) | ✅ | wtcache_impl.c:32-36 |
| op 内 VTCM arena 自划先例 | ✅ | exec_matmul_f16_hmx off_a/off_w/off_o :2444 |
| Q4 decode 基线内核 (对拍 golden) | ✅ | exec_matmul_q4_0_gemv :2139 |

## 方案 (编译期全固定, 执行器零运行时决策)

1. **契约**: `OP_MATMUL_F16` flags **bit5 (0x20) = W4TS** (w4 tile stream)。
   emit 条件 (全满足才标): M<32 && 权重槽为 Q4 tile (len==(K/32)*(N/32)*640)
   && K%32==0 && N%32==0 && !batched && 无 bias && t1。op_annot engine=HVX。
2. **执行器新内核 `exec_matmul_w4_stream`** (oplist_exec.c 内部 static,
   不拆 TU — 013 拆分 NO-GO 前科): M<32 分发段 `w_q4 && (flags&0x20)` 优先命中,
   未标 bit5 的 Q4 槽走现 `exec_matmul_q4_0_gemv` 零扰动。
   - 激活: f16 → `hvx_vec_f16_to_f32` → `quantize_row_f32_q8_0_tiled` → VTCM (一次/op)。
   - 权重: VTCM ring 8 槽 × n_k*640B; `dma_desc_submit` 菊花链预排 +
     `dma_desc_is_done` 晚 poll; 槽 desc 静态复用 (无逐 op heap)。
   - 计算: 逐 ct `tiled_vec_dot_q4_0_32x1` → f32 32-lane 出 VTCM → 末 f32→f16 写 y。
3. **VTCM**: hmx_runtime arena 自划 (同 HMX 先例); 需求闭式 K≤8192 →
   ring 8×160KB + 激活 288KB + 输出 ≤ 2MB, 纳入编译期 arena 预算。
4. **host 参考**: `#else` 标量分支 (tile 直解 q4×q8, f32 累加) 供 host_run 对拍。

## 数值声明 (诚实)

q8_0 激活量化 (per-32-block 对称, llama.cpp 同构) 引入 f16 激活基线之外的误差;
rmpy int8 累加 ≠ f32 逐元素累加。**对基线非位级为预期**, 精度门用 cos 不用 md5。

## 门

- G1 合成 op 设备门: 同权重同激活, w4_stream vs q4_0_gemv 输出 cos ≥ 0.9999
  + max_rel_err 呈报 + 实测 GB/s (生死门: ≥30GB/s 才继续 E2E)。
- G2 host 对拍: host 标量参考 vs host 基线 cos ≥ 0.9999。
- G3 0.6B decode E2E: 186.4ms → **目标 ≤100ms**; HF 金标 cos ≥ 0.9986
  (Q4 线口径), top1 变化呈报用户裁决。
- G4 0.8B decode E2E: 293.8ms → 呈报; f16 decode 线零回归 (位级同)。
- G5 零回归: 无 bit5 的 f16/Q4 blob 输出位级同; ctest 全绿。

## 非目标

- f16 权重 decode (f16 线 118.2ms 基线不动; W4 是本线唯一载体 — 用户 2026-10-28 裁决)。
- HMX 引擎化 (例 50 三臂已裁 M=1 HMX NO-GO)。
- 服务层批处理 (M 维, [[gehtp-decode-batching-feasibility]] 另一条线)。
- W4A16 int4 HMX 供给 (prefill 线, M%256 闸不动)。
