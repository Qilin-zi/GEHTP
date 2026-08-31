# 数据布局与内存约束

调用 hvxhmx 库的算子前必须理解这些约束. 违反会导致 CDSP crash、silent 错误结果、
或栈溢出. 这是所有算子共享的底层规则.

## 1. 对齐

所有输入/输出 buffer **必须 128 字节对齐** (HVX 向量宽度 = 128B).

```c
static __fp16 act[1024] __attribute__((aligned(128)));   /* ✓ */
static __fp16 out[1024]   __attribute__((aligned(128)));
uint8_t *p = malloc(n);                                  /* ✗ 可能只 8/16B 对齐 */
```

HMX VTCM buffer 额外要求 **2KB 对齐** (`HMX_VTCM_ALIGN = 2048`). runtime 申请的 VTCM
指针已满足, 调用方无需处理.

## 2. 禁用 VLA (变长数组)

DSP 栈很小 (典型几十 KB). VLA 在栈上分配, 稍大的 tile 会立刻栈溢出 crash.

```c
void bad(uint32_t n) {
    uint8_t buf[n];            /* ✗ VLA, n 大就炸栈 */
}
void good(void) {
    static uint8_t buf[4096];  /* ✓ static/全局, 不占栈 */
}
```

本库所有 example/test 都用 `static __attribute__((aligned(128)))` buffer.

## 3. HMX crouton 布局

HMX 一次处理一个 32×32 "crouton" (2KB fp16 / 1KB u8). 真实物理布局是 **pair-interleave**
(不是行主序):

```
pos(row, col) = (row / 2) * 64 + 2 * col + (row & 1)
```

- fp16 crouton: 32×32×2B = 2048B, 按 pair-interleave 排列
- u8 crouton: 32×32×1B = 1024B

调用方一般不用手工打包 — `hmx_convf16` 等高层 API 内部把行主序 act/wgt 打包进 VTCM
crouton. 只有直接调 `hmx_phase0_gemm_fp16_core` 这类低级 core 才需要手工打包
(用 `hmx_pack_act_fp16` / `hmx_pack_wgt_fp16`, 见 [api_lowlevel.md](api_lowlevel.md)).

## 4. GEMM 维度约束

| 参数 | 含义 | 约束 |
|------|------|------|
| M | act 行数 (输出行) | 32 的倍数 (多 tile 时) |
| K | 收缩维 (act 列 = wgt 行) | **必须 32 的倍数** (`hmx_k_aligned(K)`) |
| N | wgt 列 (输出列) | 32 的倍数 (多 tile 时) |

单 tile GEMM = 32×32×32. M/N/K > 32 时库内部循环多 tile (K 维用 act+wgt pair 累加,
不需要 dC deep 字段). 见 [api_hmx_gemm.md](api_hmx_gemm.md).

## 5. 数据类型矩阵 (GEMM 族选型)

按 (act 精度, wgt 精度, out 精度) 三元组选族:

| 族 | act | wgt | out | 数学 | 设备路径 |
|----|-----|-----|-----|------|----------|
| `convf16` | fp16 | fp16 | fp16 | Σ act·wgt + bias | **真 HMX** (fp16) |
| `convbbb` | u8 | u8 | u8 | Σ act·wgt + bias, sat u8 | HVX int8 GEMM |
| `convbbh` | u8 | u8 | u16 | Σ act·wgt + bias | HVX int8 GEMM |
| `convbcb` | u8 | i16 | u8 | Σ act·wgt + bias, sat u8 | HVX int8 GEMM |
| `convbnb` | u8 | i16 | u8 | (+ 可选 sparsity) | HVX int8 GEMM |
| `convhbh` | u8 | i8 | u16 | Σ act·wgt + bias | HVX int8 GEMM |
| `convhhh` | u8 | i8 | u16 | (格式 :2x2) | HVX int8 GEMM |
| `convhnh` | u8 | i16 | u16 | Σ act·wgt + bias | HVX int8 GEMM |
| `convhch` | u8 | i16 | u16 | (格式 :2x2) | HVX int8 GEMM |

> **重要**: 本设备 (52f67807 gen5_gvm FUSA) 的 HMX **只支持 fp16**. int8 HMX 指令
> (`.ub`/`.b` MAC + `sat.ub` 写回) 全是 silent NOP (累加器恒 0, 输出只剩 bias).
> 所以所有 int8 族在库内部走 **HVX int8 GEMM** (`hvx_int8gemm`), 结果与 HMX int8
> 数学等价且 exact (err=0). 不要尝试直接用 HMX int8 指令.

## 6. bias 格式

| 族 | bias 类型 | 含义 |
|----|-----------|------|
| fp16 族 (`convf16`/`dwconvf16`/`add`) | `__fp16 *` | per-output-channel fp16 偏置 |
| int8 族 (其余) | `int32_t *` | per-output-channel int32 偏置 (累加后再 sat) |

bias 数组长度 = N (输出通道数).

## 7. 容差 (验证用)

| 精度 | 容差 (max abs err) |
|------|--------------------|
| u8 / i8 / u16 / i16 (整数族) | **0** (exact) |
| fp16 (convf16/dwconvf16/add) | **1 ULP** (≤1, 浮点舍入) |

fp16 容差来源: HMX fp16 累加器是 37-bit fp32 mantissa, 输出 `cvt.hf` 截断.
输入避开 ±1.0 边界 (denormal) 即可保证 ≤1 ULP.

## 8. VTCM 生命周期 (HMX 专用)

HMX 指令访问的 act/wgt/out/scales 必须在 **VTCM** (Vector Tightly-Coupled Memory)
里. `hmx_runtime_setup(vtcm_size)` 一次性申请一块 VTCM, 内部 kernel 把行主序输入
DMA/copy 进去再发 HMX 指令.

- 推荐 `vtcm_size = 2*1024*1024` (2MB, 够多 tile)
- setup 内部 memset 清零申请到的 VTCM — **这一步必须做**, 否则 CDSP CX_FAULT
  (VTCM 残留脏数据触发保护异常)
- 幂等: 重复调 setup 安全 (内部判已在用)
- teardown 归还 VTCM + HMX 下电

HVX 族 (divide/activation/...) 不需要 VTCM, 直接访 DDR; 但也建议先 setup (顺带
给 HVX 上电, 性能更稳).

## 9. 静态链接规避 CDSP 库缓存

CDSP 按 **文件名** 缓存 .so. 同名 push 新版可能仍跑旧版 (差分测试大坑).
本库的 verify/examples 把 `libhvxhmx.so` 部署到独立目录 `/data/local/tmp/hvxhmx_libs/`
(与历史 `hvxhmx_verify/` 隔离), 规避缓存命中. 若仍怀疑缓存, 换文件名或 reboot CDSP.
