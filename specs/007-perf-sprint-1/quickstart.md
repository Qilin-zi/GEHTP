# Quickstart: Short-Term Performance Sprint 1

**Feature**: [spec.md](spec.md)
**Created**: 2026-10-08

验证三条性能兑现各自独立可跑、可关，且 003 契约不回归。前置：d0f1784 可达（getenforce Permissive、pgrep run_main 互斥）、库指纹 d33e31b3 基准、gohtp 容器 `gehtp-toolchain:24.04`。

## 0. 基线对照（改动前先采一次）

```bash
# 003 v4 锚：wall 35.31s / cos 0.998642 / top1 30/32 / 14188 op / broadcast 族允许回落 1023
cd /disk1/gehtp && python3 scripts/gehtp_prof.py <v4 op_ts>  # 族耗时+回落计数基线
```

## 1. P1 — broadcast HVX + 计划对齐

```bash
# 设备库（改 oplist_exec.c/hvx 件，多文件 → all）
cd /disk1/gehtp/kernels && ./build_libs.sh all
# 编译器（emit assign_engine 形态分派）
cd /disk1/gehtp/compiler && cmake --build build_linux -j && cd build_linux && ctest --output-on-failure
# 板跑 v17/prefill（007_ 私有名，推前 df）
# 门：回落 1023→0；broadcast 族耗时 ≤1s；wall ≤28s；cos 0.998642 逐字节不动（judge hf 口径）
python3 scripts/judge_logits.py --golden test_models/qwen35_08b/cpu_golden/hf ...
```

预期：形态 1/2 op 计划 HVX 实跑 HVX（dev=0）；形态 3 计划 scalar（合法，不出现在回落列）。

## 2. P2 — W4A16 全图

```bash
# 供给链：全量 match 表（编译期预检 GGUF 源存在）
python3 scripts/gguf_wtlink.py --gguf /disk2/aiDAPTIV/Qwen35-0.8B.gguf --out <repo 内 007_match.tsv>
# 编译：v4 配方 + 追加 --gguf/--match → emit 报告看逐层权重域 + f16 兜底名单
# 板跑判分：q4_0 口径双门
python3 scripts/judge_logits.py --golden test_models/qwen35_08b/cpu_golden/q4_0 ...
# 门：cos ≥0.9986 同量级 + top1 ≥29/32（首跑校准后冻结）；权重 ≤0.6GB；lm_head 51.7× 不回退
```

## 3. P3 — 融合准入（逐条）

```bash
# 每条规则独立：统计 → 数值门 → 过门才进默认集
./build_linux/hnnx_compile --enable-pass fusion_matmul_add --opt-stats ...   # 命中数
# 三门：合成图位级一致；0.8B 全图 cos Δ<1e-5；top1 不降
./build_linux/hnnx_compile --no-pass fusion_matmul_add ...                   # 逃生门恒可用
# 8 条全开 op 数仍 >9000 → op_ts 热点评估补规则（条件项，不阻塞 P1/P2）
# 末步：P3 放行后复跑 P1 验证（融合吸收 broadcast 物化的交互确认）
```

## 4. 契约回归兜底（每阶段收口必跑）

```bash
cd /disk1/gehtp/compiler/build_linux && ctest --output-on-failure   # 47/47
# 003 blob-v3 五场景验收（.scratch_003_rebuild/prof_v4/ 证据链同口径）
```

## 逃生门汇总

- P1：emit 计划回退旧判据（n%32）+ 设备旧执行体 = 现状路径，无 CLI 开关依赖。
- P2：去掉 `--gguf/--match` 即回 v4 全 f16 配方。
- P3：`--no-pass fusion_*` 逐条关；默认集翻转可回滚（仅 `enabled` 位）。
