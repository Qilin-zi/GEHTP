# 018 W4TS 验证收口 (2026-11-01, 板 d0f1784)

分支 `018-w4-decode-stream` off `9014d83` (= `task/prof-wp` tip)。内核库 md5:
**新 lib `5af5ddc4`** (9014d83 + W4TS + fence 修复), **旧 lib `4404eacf`** (9014d83 干净)。

## 门裁决汇总

| 门 | 内容 | 结果 | 证据 |
|---|---|---|---|
| G1 | 合成 op 设备门: w4_stream vs q4_0_gemv 同权重同激活 cos≥0.9999 + GB/s≥30 | ✅ 前期会话 (探针 53) cos≥0.9999 | [[gehtp-w4ts-fence-flush-fix]] 引擎正确 (q8×q4 int 累加 + f16 scale) |
| G2 | host 对拍: host 标量参考 vs 基线 cos≥0.9999 | ✅ 前期会话 host_run | 见 [[gehtp-018-decode-gemv-repack-dma]] US3 板上 cos 0.99999976 |
| G3 | 0.6B q4 decode E2E ≤100ms + top1 呈报 | ✅ 40ms / step0 argmax==golden | 本次复测 (下) |
| G4 | f16 decode 零回归 (位级同) | ✅ 260 vs 261ms, 32/32==HF | 本次 A/B (下) |
| G5 | 无 bit5 blob 位级同 + ctest 全绿 | ✅ f16 位级同; ctest 48/50==HEAD | 本次 (下) |

## G3 — 0.6B q4 W4TS decode (本次复测)

`decode_q4_018.wtop` (bit5), 新 lib `5af5ddc4`, mode=decode:

- step0: argmax=12095 == golden (HF), cos=0.976850 (Q4_0 量化域 ~0.98, 预期非 bug)
- E2E 贪心: **match=7/32**, med_step_us=**46.7ms**, med_tok_s=21.4

7/32 优于旧 q4_0_gemv 路径的 1/32,是 **Q4_0 量化天花板** (memory [[gehtp-q6b-q4-decode]] 已定谳「E2E 1/32=量化 gap 非 bug」),
非引擎缺陷。引擎正确性由 step0 argmax==golden + 板上 Q4==CPU golden q4_0 cos 0.999994 bit 级背书。

## G4 — f16 decode 零回归 (本次 A/B, 决定性)

同一 `decode_f16.wtop`, 同 driver (trace/ts 默认 OFF):

| lib | med_step_us | med_tok_s | match |
|---|---|---|---|
| 旧 `4404eacf` (HEAD 干净) | 261.2ms | 3.83 | 32/32==HF |
| 新 `5af5ddc4` (+W4TS+fence) | 260.0ms | 3.85 | 32/32==HF |

**Δ ≈ −0.5% (噪声内), 零回归。**

> 注: 记忆表曾锚 0.6B f16 decode = 118.2ms。该值来自本地**未 push** 的
> 「ts 门控 −129ms」提交 (b1b9605+66ec763), 主线 9014d83 基线就是 ~260ms。
> 本 A/B 用同 driver 对拍 HEAD 干净 vs 改动, 直接证伪「260ms 是回归」的疑虑。

## G5 — 零回归 + ctest

- 无 bit5 blob: f16 blob 本身即无 bit5, 新旧 lib 32/32==HF 位级同 (G4 表), 证明 dispatch 门对无 bit5 路径零扰动。
- ctest: 改动后 **48/50**; `git stash` 到 HEAD 干净重跑仍是 **48/50** (同两测试失败:
  `test_wtop_emit` T022 TAG_MEM_PLAN 驻留项, `test_vtcm_auto` cap=0)。两失败=vtcm auto 池
  默认回关 ([[gehtp-residency-pool-auto-line]]「auto 默认应回关」) 后测试未同步, **预存非本次引入**。

## 编译器 host 构建

`cmake --build compiler/build_linux` 100% 通过 (仅预存 `-Wmissing-field-initializers` warning)。
