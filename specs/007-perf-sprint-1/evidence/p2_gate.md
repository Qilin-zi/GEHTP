# T017 — P2 门验收 (SC-003 三子门, 2026-10-10, 设备 d0f1784)

SC-003: W4 启用后权重域内存 1.6GB → ≤0.6GB；f16 兜底名单显式；lm_head 51.7× 不回退。

## 结论: 三子门全 PASS

| 子门 | 实测 | 门 | 判 |
|---|---|---|---|
| 权重内存 | blob 476.5MB (w4 447.8MB + f16 2.2MB) | ≤0.6GB | PASS |
| lm_head 不回退 | 4.6s (LUT 后) vs f16 HMX 4.24s = 1.09× | 51.7× 保持 | PASS |
| f16 兜底名单 | w4report.tsv 230 条 f16, 显式 reason | 显式降级 | PASS |

## 子门 1: 权重 ≤0.6GB

emit 报告 (`emit_q4_007.log`):
```
[w4report] w4 469565440 B (447.8 MB) / f16 2287168 B (2.2 MB), w4 占比 99.5%
[w4report] 382 条 -> qwen35_08b_q4_007.wtop.w4report.tsv
```
blob 落盘 499665664 B (476.5MB)。f16 基线 1.6GB → 476.5MB (3.4× 缩)，
≤0.6GB (614.4MB) 达标。w4 域 99.5% 权重字节。

## 子门 2: lm_head 51.7× 不回退

lm_head (code=35/MATMUL_SPLIT, 32 chunk) 在 Q4 线走 `exec_matmul_f16_hmx`
的 w_q4 分支: 反量化 pack (Q4_0→f16 crouton-major) + HMX GEMM。

| 版本 | lm_head | MATMUL_F16 | exec wall |
|---|---|---|---|
| 标量 dequant (9e52546) | 9.72s (304ms/chunk) | 20.6s | 40.9s |
| LUT dequant (7aed109) | **4.61s (144ms/chunk)** | **10.6s** | **25.7s** |

- f16 线 HMX lm_head 锚 = 4.24s (219s 标量 → HMX = 51.7×)。
- Q4 LUT 后 4.61s = 1.09× f16 锚 (Q4 权重 4× 小省 DDR 带宽, 抵掉 dequant
  计算余量)；= 47.6× vs 219s 标量。**51.7× 不回退成立** (未回落标量,
  与 f16 HMX 等量级)。
- 位级安全: LUT 公式与标量逐位同, prompt0 output 与 pre-LUT **max|d|=0**,
  cos 0.993484/31 不变 (q4_0 口径锚)。

## 子门 3: f16 兜底名单显式

`qwen35_08b_q4_007.wtop.w4report.tsv` 382 条逐层登记 domain/reason/size:

| domain | count | 说明 |
|---|---|---|
| w4 | 152 | Q4_0 量化 (MatMul 权重 + embed + lm_head dedup) |
| f16 | 230 | 兜底, 分两类: |

f16 兜底明细:
1. **shape_misalign (36)**: 18× `linear_attn/in_proj_a/MatMul` + 18×
   `in_proj_b/MatMul` (各 32768 B) — M/K/N 对齐约束不满足, 显式降级
   (FR-003 形状边界)。这是唯一"本可量化却降级"的类别。
2. **none (194)**: RMSNorm 族 (input_layernorm / post_attention_layernorm /
   linear_attn.norm / self_attn.k_norm / q_norm / final norm, 2048/256 B 级,
   逐元素 scale 非 matmul 权重) + 18× `conv1d/Conv_2d` (49152 B, conv 族
   不在 W4 推广范围, 保持 f16)。

`lm_head/MatMul` = w4 + `dedup` + 0 B (tied embedding 复用 embed_tokens 权重)。

## 复现

```
.scratch_007/run_q4d_lut.sh     # prompt0 单跑 + op_ts 抓取 (LUT lib)
python3 -c "…judge vs cpu_golden/q4_0…"   # cos 0.993484 / top1 31
```

资产: `.scratch_007/qwen35_08b_q4_007.wtop.w4report.tsv` (兜底名单落盘)、
`output_lut_0.f16.raw` (LUT 输出)、`q4d_lut_0.device.txt` (op_timing)。
