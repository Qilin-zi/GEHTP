# Quickstart: 全图 VTCM 驻留 + 显式 DMA 流水

**Created**: 2026-10-09 | 验证 runbook（端到端可跑场景；实现细节见 plan/tasks）

## 前置

- 设备 d0f1784（getenforce Permissive 预检；pgrep run_main 互斥；板上私有名纪律沿 003/004）。
- 库指纹与锚：跑前记录 `kernels/build_libs.sh` 产物指纹 + 基线 blob（007 交卷锚或 v4 35.31s 口径）。
- 构建环境：compiler 容器 `gehtp-toolchain:24.04`；kernels 先 source env.sh（构建环境记忆）。

## 场景 1 — 预算基线冻结（R1，US1 前置）

```bash
# 现编译器重发 0.8B 同图（003 发射配方照抄），只看 stderr 两行
<emit 命令（v4 同图同配方）> 2>emit.log
grep '\[ddr\]' emit.log
# 期望: TAG_MEM_PLAN: cap=<DDR池> vtcm_cap=<VTCM占用> n_temps=<N> n_spills=<S>
#       TEMPOFF slot: cap=... vtcm_cap=... n_temps=... (vtcm=<V>)
```

**判据**：两行出现（v4 旧日志无此行 = 旁路实锤被修复）；数字落 `evidence/budget-baseline.md` 冻结。

## 场景 2 — US1 驻留扩面端到端

1. emit 加 `--vtcm-budget 16777216 --residency-report report.json`
2. **位级门**：新 blob 板跑输出 vs 基线 blob **逐字节一致**（md5 比对；判分 judge_logits.py f16 列 cos 0.998642 族不回归）
3. **覆盖门**：report.json coverage ≥80%、spill 名单非空时逐项有 reason
4. **性能门**：wall 对比基线（首锚记录，门值校准后冻结）
5. **预算门（负例）**：`--vtcm-budget 1048576` 重发 → emit 硬报错退出 + 三方数值文本

## 场景 3 — US2 DMA 双缓冲（fence micro-test 先行）

1. fence micro-test：设备端 HMX matmul 与 dc_dma 并发 ≥100 轮 → 逐字节一致 + 无挂死（证据入 evidence/fence.txt）
2. `--dma-runlist` 全图 + 双缓冲发射 → 板跑
3. op_ts：`dma_us > 0`；HMX 族（lm_head chunk/线性层）dur 显著降（SC-002 ≤50% 锚）
4. 位级门同场景 2（纯时序变化）；回落路径负例：构造 staging 失败 → deviation 记录 + 串行继续（偏差报告不变量 PASS）

## 场景 4 — US3/US4 收口

1. US1 后重跑 census（`gehtp_prof.py --csv`）：三族（STRIDED_SLICE/UNARY/TRANSPOSE_GEN）合计 ≤2s，digest 落盘
2. `--op-ts-sample 8` 跑通 + 报告口径标注；默认模式 op_ts 与现库逐条一致
3. 终局：ctest 全绿 + 003 五场景（devA/devB/w4v3/t2l/ver1-legacy）不回归

## 期望终态数字（SC 总门）

全链 wall ≤8s（007 锚口径）／coverage ≥80%／f16 位级不动／W4 期 q4_0 门不降／per-op ≤0.5s。
