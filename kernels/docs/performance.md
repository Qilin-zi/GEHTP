# 性能基准

设备: 52f67807 (V81 gen5_gvm FUSA). 计时: `hmx_perf_now_us` (HAP qtimer).
方法: 每算子 warmup 1 次 (不计时), 然后 **500 次取平均**. tile = 32×32×32.

重新生成: `./verify_libs.sh bench_all` (用 `hvxhmx_kernel/test/test_bench_all.c`).

## HMX GEMM 族 (32×32×32)

| 算子 | 精度 (act×wgt→out) | 路径 | us/call |
|------|--------------------|------|---------|
| `convbbb` | u8×u8→u8 | HVX | 9.29 |
| `convbbh` | u8×u8→u16 | HVX | 9.51 |
| `convbcb` | u8×i16→u8 | HVX | 16.52 |
| `convbnb` | u8×i16→u8 | HVX | 16.45 |
| **`convf16`** | **fp16×fp16** | **真 HMX** | **124.89** |
| `convhbh` | u8×i8→u16 | HVX | 6.42 |
| `convhhh` | u8×i8→u16 | HVX | 6.42 |
| `convhch` | u8×i16→u16 | HVX | 11.22 |
| `convhnh` | u8×i16→u16 | HVX | 11.22 |

> 观察: i8 权重族 (`convhbh`/`convhhh`) 最快 (6.4us); i16 权重族 (`convbcb`/`convbnb`/
> `convhch`/`convhnh`) 因双倍带宽较慢; fp16 `convf16` 走 HMX 但单次含 VTCM 打包开销,
> 32×32×32 小 tile 上比 HVX 慢 — 大 tile (M/N/K≫32) 时 HMX 优势才显现.

## HMX depthwise / elementwise

| 算子 | us/call |
|------|---------|
| `dwconvbbb` 3×3 u8 | 13.17 |
| `dwconvf16` 3×3 fp16 | 71.91 |
| `add` fp16 32×32 | 101.99 |

## HVX divide (n=1024)

| 算子 | us/call |
|------|---------|
| `divide_u8` | 0.57 |
| `divide_u16` | 2.18 |
| `divide_flat_i32` | 8.73 |
| `floor_divide_u8` | 0.58 |
| `floor_divide_u16` | 2.19 |

## HVX activation (n=1024)

| 算子 | us/call |
|------|---------|
| `hardswish_flat_u16` | 0.11 |
| `hardswish_crouton_u16` (1 batch) | 0.12 |
| `prelu_u8` | 0.05 |

## HVX lookup / unpack (n=1024)

| 算子 | us/call |
|------|---------|
| `table_lookup_flat_u8` | 2.77 |
| `table_lookup_crouton_u8` (1 batch) | 2.78 |
| `unpack_weights` | 2.72 |
| `unpack_custom_weights` | 2.68 |

## HVX reduction (hw=32, d=32)

| 算子 | us/call |
|------|---------|
| `argminmax_depth_crouton_b` | 9.64 |
| `argminmax_depth_crouton_h` | 11.01 |
| `argminmax_depth_dLE32_crouton_b` | 9.62 |
| `argminmax_depth_flat_h` | 10.91 |
| `argminmax_depth_short_b` | 9.59 |
| `find_max_in_depth_b` | 5.27 |
| `find_max_in_depth_h` | 5.84 |
| `top1_qu8_dLE32_cr2flt` | 5.13 |
| `reducesum_depth_u8` | 1.33 |
| `reduce_sum_u8_case_4` | 0.14 |
| `reduce_sum_u16_case_4` | 0.15 |

## 关键结论

1. **HVX elementwise 极快**: activation/lookup/divide(u8) 都在亚微秒到几微秒 (n=1024).
2. **int8 GEMM 看 wgt 精度**: i8 wgt (6.4us) ≪ i16 wgt (11-16us). 能用 i8 就别用 i16.
3. **fp16 HMX 小 tile 不占优**: 32×32×32 上 convf16 (125us) 比 HVX int8 慢, 因为 VTCM
   打包 + crouton 建立开销固定. fp16 优势在 K≫32 的深 GEMM (HMX 32×32 systolic 一次
   装满 K 维才摊薄).
4. **depthwise/add 是标量路径**: 性能不高 (网络中占比小, 可接受).
5. **i32 除法最慢** (8.7us): 4-cycle shift-subtract, 不可避免. u8/u16 倒数法快得多.

---

## 性能现实: 公开 wrapper vs HMX 硬件峰值

> 这一节回答一个常被问的问题: **"README 说 HMX 峰值 20.4 TFLOPS, 为什么我调 `hmx_convf16` 只有零点几个 GFLOPS?"**
> 答案: 20.4 TFLOPS 是 HMX **硬件峰值** (systolic array 满载), 公开 `hmx_convf16` 是
> **correctness-first 的参考实现**, 每 tile 重新 gather+pack, 把大部分时间花在 CPU 搬数据上, HMX 被饿死.
> 要逼近峰值, 必须走"低级/高级路径" —— 预打包 + 裸 K-loop.

实测同一 GEMM (M=32, N=32, K=256, fp16), 4 级路径的吞吐阶梯 (52f67807, 2026-08-10):

| # | 路径 | 说明 | 实测吞吐 |
|---|------|------|----------|
| 1 | `hmx_convf16` 公开 wrapper | 每 tile 重新 gather+pack | **~1.9 GFLOPS** |
| 2 | `hmx_phase0_gemm_fp16_core` 单 tile | 一次 32×32×32, 启停全流程 | ~141 GFLOPS |
| 3 | 裸 K-loop, acquire 在循环外 | NK tile 背靠背, 单次 clracc/cvt | ~1.11 TFLOPS (含 acquire 摊销) |
| 4 | 裸 K-loop, 纯 compute (NK=8) | 同上但计时只含 MAC+clracc+cvt | **12.34 TFLOPS** |
| — | HMX 理论峰 (NK→大) | systolic 满载, 无建立开销 | ~20.4 TFLOPS |

### 为什么差 4 个数量级 (路径 1 vs 路径 4)

`hmx_convf16` (公开 wrapper) 对每个 (mt, nt, kt) tile 做:

1. 标量 gather 2048 个 fp16 → 临时行主序 buffer
2. `hmx_pack_act_fp16` 把行主序 → crouton 布局 (VTCM 内)
3. `hmx_pack_wgt_fp16` 同上
4. **1 条** HMX MAC 指令
5. `hmx_unpack_out_fp16` crouton → 行主序

步骤 1-3+5 占 ~99.6% 时间, 步骤 4 (真 HMX 算) 只占 ~0.4%. HMX 阵列几乎全程空转等 CPU 喂数据.

### 怎么逼近峰值 (路径 3/4, 高级路径)

关键: **把 gather+pack 移出热循环, 一次打包, 反复 MAC.**

```c
/* 预打包 NK 个 act/wgt 32×32 slice 进 VTCM crouton (一次性, 计时外) */
for (int kk = 0; kk < NK; kk++) {
    hmx_pack_act_fp16(act_cr[kk], act_tile[kk]);
    hmx_pack_wgt_fp16(wgt_cr[kk], wgt_tile[kk]);
}
/* 裸 K-loop: acc 跨 NK tile 持续累加, 只 clracc/cvt 一次 */
asm volatile("mxclracc.hf");
asm volatile("bias = mxmem2(%0)" :: "r"(sca));
for (int kk = 0; kk < NK; kk++) {
    asm volatile("{ activation.hf = mxmem(%0,%1):deep\n  weight.hf = mxmem(%2,%3) }"
                 :: "r"(act_cr[kk]), "r"(HMX_ALIGN_2KB),
                    "r"(wgt_cr[kk]), "r"(HMX_ALIGN_2KB));
}
asm volatile("cvt.hf = acc(%0)\n mxmem(%1,%2) = cvt"
             :: "r"(2), "r"(out_cr), "r"(0));
```

完整可跑代码见 [examples/14_hmx_peak_gemm/main.c](../examples/14_hmx_peak_gemm/main.c).

### 什么时候用哪条路径

| 场景 | 推荐 |
|------|------|
| 正确性验证 / 小 GEMM / 一次性计算 | 路径 1 (`hmx_convf16`) — 数值完全相同, 只是慢 |
| 想看 HMX 硬件是否真的工作 | 路径 2 (`phase0_gemm_fp16_core`) |
| **大 GEMM 吞吐敏感** (K≫32, 反复乘同一批权重) | **路径 3/4 (预打包 + 裸 K-loop)** |
| 网络推理 batch 化 (权重预打包一次, activation 流过) | 路径 3/4, 权重常驻 VTCM |

⚠️ 低级路径需要手工管理 VTCM 布局 + crouton pack/unpack + acquire/release 生命周期.
非吞吐敏感场景不值得. 详见 [api_lowlevel.md](api_lowlevel.md) 和 [examples/14_hmx_peak_gemm](../examples/14_hmx_peak_gemm/).

> **关于"20.4 TFLOPS"**: 这是 HMX 硬件峰值 (systolic 满载, 理论), 不是公开 API 的稳态速率.
> 路径 4 在 NK=8 时实测 12.34 TFLOPS (~60% 峰值, 受 clracc/cvt 摊销限制); 加大 NK 可继续逼近.
