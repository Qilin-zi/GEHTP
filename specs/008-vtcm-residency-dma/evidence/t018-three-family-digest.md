# T018 三族 HVX 化收口 digest — T019 门裁决 (1010)

**源**：`.scratch_008/t20.op_ts.bin`（T018 全量收口, fault=0, ver=4, 11504 op）
+ `.scratch_008/qwen35_08b_t19.wtop.manifest.json`。
**锚**：md5 `f87461e2a5089c48413e70463f4aa7dc`（v4 金标, 逐次板跑位级对拍恒 PASS）。

## 门裁决：PASS

| 门 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 三族合计 | ≤2s | **0.560s** | ✓ PASS |
| exec wall | ≤8s | **6.31s** | ✓ PASS |
| 位级 | md5==f87461e2 | PASS | ✓ |

三族明细（t20, 板跑）：

| family | n | dur | %exec | vs T018 前 |
|---|---|---|---|---|
| STRIDED_SLICE_F16 | 2373 | 0.082s | 1.3% | 3.05s → 0.082s（**37×**） |
| TRANSPOSE_GEN_F16 | 175 | 0.143s | 2.3% | 1.28s → 0.143s（**9×**） |
| UNARY_F16 | 225 | 0.335s | 5.3% | 0.34s（不变） |
| **三族合计** | **2773** | **0.560s** | **8.9%** | **7.25s → 0.56s（13×）** |

## 落地手法（T018 三条 commit）

1. **STRIDED_SLICE（7b98b94）**：最内轴 stride==1 → 输出每行是输入连续字节段，
   逐行 `hvhx_v2_copy_f16` 块拷贝（odometer 免 div/mod）；emit `op_strided_slice`
   对 `ns[nrk-1]==1` 置 HVX 注解（孪生判据）。不变量 PASS 无 slice 回落。

2. **TRANSPOSE_GEN 块拷贝（425a04d）**：最内轴固定 `p[rk-1]==rk-1`（96/175 op）→
   逐"外块" `hvhx_v2_copy_f16`（pinv 映回输入坐标, odometer 免 div/mod）；emit
   `op_transpose` 对 `np[nrk-1]==nrk-1` 置 HVX 注解（孪生）。

3. **TRANSPOSE_GEN 标量回落 odometer（eaa9a72）**：最内轴真置换（64↔128 36op）走
   输出线性 odometer（连续写 + 跨步读），免 per-element 4 div + 4 mod；纯置换位级等价。

## 残余（T018c 记残留, 非门阻塞）

- **UNARY 0.335s**（86 scalar op）：SQRT 36 / SOFTPLUS 18 为负输入 NaN 守卫 + 公式
  差异（HVX `hvhx_v2_softplus_f32` 用 `log(1+exp)` 简单式, 标量用稳定化三分段）
  **刻意保持标量**；NEG 30 可 HVX（`hvhx_v2_scale_f32(-1.0)` 位级精确）但收益
  ~0.04s 忽略；SIN/COS 各 1（RoPE）精度保留标量。已 HVX 子型 = EXP/SIGMOID/TANH/
  SILU/COPY。
- 剩余大杠杆已不在 US3 范围：MATMUL 2.44s（38.7%）/ BINARY 1.07s / GATHER 0.67s
  = US4 per-op 裁决/记账 + 算子级，非"残余搬运"。

## 前缘（US3 后, 供 US4 参考）

exec 6.31s 主力：MATMUL_F16 2.44s（38.7%）、BINARY_F16 1.07s（16.9%）、
GATHER_F16 0.67s（10.6%）、MATMUL_SPLIT 0.43s（6.9%）、UNARY 0.34s、CONV1D_SSM
0.33s、REDUCE 0.32s、RMSNORM2 0.24s。
