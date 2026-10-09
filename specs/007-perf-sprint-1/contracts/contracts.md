# Contracts: Short-Term Performance Sprint 1

**Feature**: [spec.md](spec.md)
**Created**: 2026-10-08

## CLI 契约

### 编译器（hnnx_compile / wtop_emit）

- `--gguf <path>` — GGUF Q4_0 直读源；启用时编译期 MUST 预检文件存在性，缺失即显式失败（非静默全 f16）。
- `--match <match.tsv>` — wtlink 匹配表（`scripts/gguf_wtlink.py --gguf X.gguf --out ...` 生成）；P2 期 v4 配方仅追加此二者，无既有 CLI 语义变化。
- `--enable-pass <name>` / `--no-pass <name>` — 既有机制（005 注册制）；P3 逐条 `fusion_*` 实验放行与逃生门。
- `--opt-stats` — 输出各 pass 匹配/改写计数（P3 准入统计源）。

### 设备库构建（kernels/build_libs.sh）

- 改多文件用 `all` 勿用 `one`（单 .o 重编 + setup 见 .signed.so 跳过重建 → 陈旧 arity 陷阱链，[[gehtp-device-lib-stale-arity]]）。
- 本线板上资产前缀 `007_`（全套私有名隔离，d0f1784 多会话共享）；推前 df 检查。

## emit 报告契约

- **权重域统计（P2 新增）**：逐层 WeightDomainRecord + w4 占比汇总 + f16 兜底名单（含 fallback_reason）。
- **引擎计划（P1 变更）**：Eltwise_Broadcast 按形态分派（类 1/2→HVX、类 3→scalar），与设备侧同一形状判定函数语义一致（两处注释互指，防漂移）。

## 偏差报告不变量（003 契约，不可破）

- planned/actual/deviation 三元组语义不变：planned=HVX 而 actual=scalar 记允许回落；planned=scalar 且实跑 scalar 为 dev=0（合法）。
- P1 后 broadcast 族允许回落 MUST 为 0；类 3 计划 scalar 不计入回落（不得为凑指标硬塞 HVX gather）。
- carve footer 对拍、ver1 legacy 逐字节、blob-v3 五场景验收语义不变。

## 判分契约

- f16 期：judge_logits.py hf 口径，门 cos 0.998642 逐字节不动（P1 是纯索引重排，无浮点语义变化）。
- W4 期：切 cpu_golden/q4_0 口径，门 cos ≥0.9986 同量级 + top1 ≥29/32，首跑校准后冻结；与 hf 口径双列登记，不可直比。
- 融合准入三门：合成图位级一致 + 全图 cos Δ<1e-5 + top1 不降；任一失败保持默认关并登记证据。

## 序列化契约

- 本线**不新增 opcode、不改 slot 布局**（P1 复用既有 Eltwise_Broadcast op；P2 复用既有 `OP_MATMUL_W4A16`；P3 仅 `p.enabled` 翻转）。
- 判据/证据文件入库（specs/007-perf-sprint-1/ 或 .scratch 内 repo 侧），不入 /tmp（易失）。

## 共享资源契约

- `oplist_exec.c` exec_broadcast 段与 003 T007/004 reconcile 段同文件不同段：改前 `git status`，见他线未提交 diff 即停协调。
- 板 `/data/local/tmp/ggml08b_003/` 属 ggml 对比线，勿混用。
- 设备 d0f1784：getenforce Permissive 预检、pgrep run_main 互斥、库指纹 d33e31b3 为当前基准（改库全量重建后指纹必变，须在协调表登记）。
