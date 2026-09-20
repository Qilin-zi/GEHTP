# GEHTP 未来工作计划（ROADMAP v1.1）

> 生成日期 2026-09-20。基线：/disk1/gehtp/GEHTP @ a098509，task/prof-wp 为主家。
> v1.1 修正：收编 M1/M2 已落地（44d5ab8/db7d2ee）——§2.1 改为 M3 退役 + M4 真 DMA runlist；新增 §2.4 合并 DMA/spill-fill/cache-flush 三份既有计划。
> 本文是三条主线（用户点名）+ 正确性/工程化收口的整合计划，供跨服务器协作参照。
> 铁律沿用：判据纪律（PORTAL §3）、设备纪律（PORTAL §4 + RUNBOOK §5/§7）、
> 接口冻结点登记（TASKBOOK §2.1）。

---

## 0. 现状锚点（2026-09-20 实证）

- **架构**：host `compiler/`（逆向 libHtpPrepare 前端+IR+ST-Cut 调度+内存规划+序列化）
  → `wtop_emit`（tagged runlist → WTOP blob）→ 设备 `kernels/`（oplist_parse/exec + HMX/HVX kernel）。
- **已实证**：conv_add / L3 / spill 变体 byte-exact；probe cos=1.0 top1 256/256；
  44 个 ctest；新板 d0f1784 CDSP(domain 3) 实测可用。
- **三条结构性债务**（本次计划主攻）：
  1. **多物理树并存**（4090 / /disk2 / /disk1）+ 死路径 + 无环境自检 → 换服务器无法 bootstrap。
  2. **静态规划已收编**（M1 拓扑定稿 44d5ab8 + M2 静态规划收编 db7d2ee 已落地，`TAG_MEM_PLAN` 序列化）；
     剩余 M3 旧链路退役 + M4 算法升级（真 DMA runlist / cp_solver 默认化 / 跨组复用）。
  3. **pass 双轨**：PassManager 注册框架已建但只注册 2 个 pass（const/shape fold）；
     8 条融合规则 + 8 相位阈值分发仍是遗留结构，未走注册。

---

## 1. P0 — Git 树整理 + 多服务器可移植（用户第 1 点）

**目标**：换一台服务器，`git clone` + 一条 bootstrap 命令，即了解当前任务、拉齐依赖、编译上板。

### 1.1 单树收敛与分支纪律
- /disk1 为唯一主家；4090 树、/disk2 树转只读（`chmod` + 公告）；4090 毒 skel patch 随回合带上。
- 删除 prunable /tmp worktree（gehtp_devtbl / gehtp_m41c / gwt_bisect）。
- 一个分支一个战役（沿用 gehtp-08b / task/prof-wp / task/c4-minicpm5-2b）；`main` 只放已验证里程碑 tag。
- 提交纪律入仓：`AGENTS.md`（<80 行）固化设备纪律 5 条 + 判据纪律 + 接口冻结点登记 + commit 带任务号。
- 完成门：`git worktree list` 单真源；`git status` 无 scratch 噪音。

### 1.2 环境可移植（多服务器上板的真正门槛）
- 五件套路径（SDK/PY/HEX/SWIV/设备）环境变量化 + 默认值，收敛到 `env.sh`。
- 死路径清零：`gguf_wtlink.py:24`、`device_run.sh:17`、`conv_add/gen_all.sh`、`test_paths.hpp` 的 `/disk2`/`C:\Users`。
- 设备号参数化：域号 3 硬编码三处（gehtp:180 / conv_add_pipeline.sh:77 / device_run.sh:71）→ `--device`/`--domain`。
- 完成门：干净 shell 下 `gehtp doctor` 逐项全绿；`grep -rn "4090disk2\|/disk2/\|C:\\Users"` 归零。

### 1.3 一条命令 bootstrap
- 新增 `scripts/bootstrap.sh`：clone → env.sh 模板 → `gehtp doctor` → 编译 compiler+kernels → ctest + conv_add byte-exact 冒烟。
- 完成门：未配过的新目录 15 分钟内从零到可上板。

### 1.4 版本握手
- `libhvxhmx_v23.so` 加 `GEHTP_LIB_VERSION` 符号，runner 启动校验不匹配即拒跑（杜绝 opcode28 旧 lib+新 runner 静默垃圾）。
- 完成门：旧 lib + 新 runner 启动即报错。

---

## 2. P0 — 规划/调度整合进编译阶段 + tiling 决策（用户第 2 点）

### 2.1 收编已落地（M1/M2），余下 M3 退役 + M4 真 DMA runlist
M1 拓扑定稿（44d5ab8）+ M2 静态规划收编（db7d2ee）已在 task/prof-wp 落地并门全绿：
`do_prepare2_late` 定稿 Kahn 序 + 调 `compute_ddr_offsets` + spill 集合，经 `TAG_MEM_PLAN`
序列化进 tagged.bin；`wtop_emit` 读 bin 照抄（shadow 期保留重算 + warn）。

- **M3 旧链路退役**：删 `wtop_emit.cpp:389-401` `spill_fill_recs()` 量池段；编译器贪心 spill 重定语义为成本模型观察记录；文档状态更新。
- **M4 真 DMA runlist**（DMA/spill/fill 的核心）：收编 matmul 内嵌的 `cpu_to_vtcm`/`dc_dma_once` 为显式 runlist 条目；cp_solver 默认化（现 env `HNNX_VTCM_ALLOCATOR=cp*` 门控）；跨组复用（RuntimeAllocator 语义）。
- 完成门：M4 后 spill/fill 是 runlist 显式 op，非标量 memcpy。（M3 已达成：emit 无重算分支，只读 TAG_MEM_PLAN）

### 2.2 tiling 决策：移植 QNN 的"注册式 DSL 结构"，不移植公式
逆向结论（见 compiler/DISASM_PLAN.md M36/M36c）：QNN 的 tiling 是**声明式注册 DSL**
（`tiler.h` DSL / `tiling_registration.cc` / `TileShapeBase`），关键机制：

```
declare_tiling_rule(id, name, holder, shape_fn)   ← 每 op 注册一条规则
   └─ shape_fn(op, tcm_size_for_tiling) → tile 计划   ← 预算驱动，不硬编码
tcm_size_for_tiling = get_vtcm_tile_size() = VTCM × 3/4   ← "3/4 规则"指令级证实
两阶段 apply：tile shape apply / tile size apply            ← 形状与大小分开决策
```

**结论**：
- **可移植 = 结构**：注册模式 + shape_fn + 预算选项解耦 + 两阶段 + `minimize_tiling` 旗标。
- **不可移植 = 每 op 的具体切块公式**（51419 行 rule dump 未解；且是给 QNN 自家 HVX kernel 调的）。
- **`create_supertiles`（图级分组）已完整移植 + 42 测试**，收编时一并纳入规划序列化出口。

**决策（现在做 vs 以后做）**：
- **现在（收编时）**：加一个**"tiling 规则注册表"接缝**，默认 shape_fn = "1 tile"（恒等），
  镜像 QNN 的 `declare_tiling_rule` + `tcm_size_for_tiling` 预算查询。allocator 请求粒度从"张量"改为"tile 流"，今天 tile=整张量。
- **以后（M7 性能）**：按 cp_solver 代价模型，只在峰值内存/VTCM 驻留收益 > 阈值的 op 上
  写**我们自己的 shape_fn**（大 matmul 切 VTCM 尺寸块、长上下文注意力切 seq 维），非全面重写。
- **现在起步（host 原型）**：conv（compute_conv_tiles）+ MatMul（输出分块）两个 shape_fn 落进注册表骨架，
  host 侧分 tile vs 不分 tile byte-exact 对拍，不碰设备；设备接线仍留 M4/M7。
- **与第 3 点的关系**：QNN 用同一个 `GraphOptInfo` 注册 pass 和 tiling——我们的 pass 注册框架和
  tiling 规则注册表应**收敛到同一套"注册式图变换"机制**，而非两套。

### 2.3 三方一致性验证器
- tagged.bin（plan_order+mem_plan）↔ blob（发射序+TEMPOFF）↔ 设备 optrace（执行序）自动对拍。
- 完成门：故意改 1 字节偏移必被检出。

### 2.4 DMA / spill-fill / cache-flush 收口（合并三份既有计划）
cache 侧已实现：`fence.c`（U9 方向对偶决策表）+ `dc_parts.c`（dc_dma_fence/clean/invalidate）
+ oplist_exec 的 flush（"四铁律"源自 V2.2 api_v22_overview.md）。剩余工作分散在三处，本节合并：

| 项 | 现状 | 归口 |
|---|---|---|
| SFCD spill/fill 写侧（RE） | 104 只有读侧 388 行 vs 779 行 | BACKPORT_PLAN §2（P0，M7 阶段二必需） |
| OP_SPILL/OP_FILL 快路径 | 设备标量 memcpy | TASKBOOK B2：UserDMA + fence 决策表 |
| 真 DMA runlist 算子 | matmul 内嵌 cpu_to_vtcm/dc_dma_once | MEMPLAN M4（见 §2.1） |
| 编译器 spill 决策 | 已 TAG_MEM_PLAN 序列化 | MEMPLAN M2（已落地） |

- 依赖：tiling（§2.2）产出 tile 流 → allocator 按 tile 请求 → spill/fill/preload 变显式 runlist op。
- 完成门：spill 变体 byte-exact 不变 + 大 spill（≥1MB）耗时下降入报告（B2 门）。

---

## 3. P1 — 前端 pass 注册化（用户第 3 点）

现状已核实：`PassManager`（add_pass/disable/enable/run_phase）+ "每 pass 一文件 + `register_pass_*`"
已就位，但只有 2 个 pass 走它；8 条融合规则 + 8 相位阈值分发是遗留结构。

### 3.1 完成 PassManager 迁移
- 8 条融合规则拆成每规则一文件，转 `GraphPass{matcher, rewrite}` 注册。
- 遗留 `GraphOptInfo`/8 相位 RE 常量下沉为文档（保留 RE 证据，不进执行路径）。
- 完成门：`run_optimize_passes_single_registry` 里的 8 相位 `if(node_count<=threshold)` 死代码删除。

### 3.2 每 pass 独立开发 + 独立测试 + 一行注册（标准工作流）

```
1. 新建 compiler/src/opt/pass_<name>.cpp
   → 定义 matcher + rewrite，文件尾 register_pass_<name>() 内 add_pass(...)   ← 注册行
2. 新建 compiler/tests/test_pass_<name>.cpp
   → 构造最小触发图 + 负例图，断言 rewrite 前后 op 数/形状/数值
3. 独立跑 test_pass_<name> 到全绿（不碰其它 pass）
4. 确认 OK 后在 pass_manager.cpp 的 register_builtin_passes() 加一行
   register_pass_<name>();                                                  ← 全局启用
5. ctest 全量 + conv_add/L3 blob byte-exact 回归（pass 不得改变既有产出）
```

- 默认全开 + `--no-pass=<name>` 可关单 pass（`PassManager::disable` 已支持）。
- pass 必须"结构匹配 → 才 rewrite"，图不匹配 = 图不动（框架已保证）→ 测试覆盖正例 + 负例。
- 每 pass 独立 commit + 独立门，便于 bisect 定位数值漂移。

### 3.3 pass 测试 harness
- 补黄金驱动 harness：同一 net.json，pass 前后 dump 图 + host 对拍数值不变。
- 首批 2 个真实 pass：常量折叠补全、Reshape/Transpose 消解（直接改善 0.8B 全模型 runlist 质量）。
- 完成门：2 个 pass 各带正负例测试，走完整 5 步工作流，byte-exact 回归不变。

---

## 4. 收口项（补充）

### 4.1 正确性命门（与 1/2 可并行）
- **ScatterNd 真语义（A3）**：0.8B 1154 处 / 4B 1538 处，恒等拷贝=数值死刑，host/设备分叉。
- **host execute 直通治理收尾（A6）**：154 op 只有 ~55 有语义，其余直通 → 假绿源。
- **broadcast opcode 补齐**（换板关键路径，runner 停点）。

### 4.2 收敛假绿源
- 路径A 金样隔离（19 步硬编码 + 内嵌字节模板 → tests/fixtures/，产品路径只留 `--format tagged`）。
- `graph_prepare.cpp` 4819 行拆四文件（prepare/execute_host/extra_extractor/serialize）。
- 测试基建换血：`test_paths.hpp` 死路径清零 + 合成 net.json 入仓 + SKIP 改显式（资产缺失=FAIL 红）。
- `adapt_weights_bin` 尺寸猜测 → qairt-converter "名字→偏移" manifest。

### 4.3 闭环与 CI
- `gehtp judge` 子命令（run 后自动对拍出红绿）。
- 推送完整性 sha256（`wt_sha256.c` 接线）。
- CI smoke（无设备：ctest + conv_add byte-exact + op_inventory 预检）。

### 4.4 性能（P2，正确性稳定后）
- B1 快速件接线（标量循环 → HMX/HVX）。
- D4 cost model 用 optrace 实测回标定。
- 在此落地 2.2 预留的 tiling 规则注册表：按代价模型定向写 shape_fn。

---

## 5. 依赖与顺序总图

```
1 Git/可移植 ──▶ 2 M3退役+M4真DMA(+tiling规则注册表接缝)
     │                    │
     │            ┌───────┴───────┐
     ▼            ▼               ▼
4.2 收敛     2.3 三方验证    4.1 正确性命门（可立即并行）
     │                           │
     └──────────▶ 3 pass注册化 ◀─┘
                     │
                     ▼
              4.3 闭环CI ──▶ 4.4 性能/真tiling
```

- 1 与 4.1 可立即并行（scripts/ vs ops/wtop_ops/，文件零重叠）。
- 2 与 3 不要并行（都动 graph_prepare.cpp / tagged.bin 格式，会撞车）：收编先，pass 后。
- 4.4 的 tiling 必须等 2.2 的注册表接缝。

---

## 6. 首周落地顺序（可立即开工）

1. **阶段 0 锚点**（当天）：三树合一 + 清 worktree + 未提交改动处置。
2. **1.2 环境变量化**（1 天）：env.sh + gehtp doctor + 死路径清零。
3. **1.4 版本握手**（0.5 天）：GEHTP_LIB_VERSION 符号。
4. **4.1.1 ScatterNd 真语义**（2-3 天，.scratch_a3 已积累）：正确性最大杠杆。
5. **M3 旧链路退役**（0.5 天）：删 emit 重算/量池段，行为保持，byte-exact 回归做门。

---

*本文档是活文档：每闭合一项，标注 ✅ + commit/test 名，并同步 PORTAL/TASKBOOK/RUNBOOK。*
