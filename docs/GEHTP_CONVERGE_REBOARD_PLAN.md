# GEHTP 收敛 + 换板计划（CONVERGE & REBOARD）

> 生成日期 2026-09-18。基线树：**/disk1/gehtp/GEHTP**（task/prof-wp，C1 主家），HEAD=a098509。
> 依据：2026-09-18 全工程评审（compiler/kernels/scripts 源码通读 + 六份文档）+ 新板 d0f1784 (SA8797P) NPU 探针通过。
> 与 PARALLEL_TASKBOOK 的关系：**不替代**。本计划是「换板 + 工程收敛」专线，原任务书 A/B/C/D 线继续有效；本计划 Phase 0/1 是所有模型线的前置。
> 铁律沿用：判据纪律（PORTAL §3）、设备纪律（PORTAL §4 + RUNBOOK §5/§7）、接口冻结点登记（TASKBOOK §2.1）。

---

## 0. 战役目标

1. **P0 换板**：GEHTP 全链在新板 d0f1784 跑通（老板 52f67807 CDSP 楔死未愈，单设备风险必须解除）。
2. **P0 收敛**：多树合一 + 消灭假绿源（路径A金样、SKIP 兜底、直通、调试残留）。
3. **P1 闭环**：judge 进门、推送完整性、版本握手、CI smoke。

## 1. 事实底座（2026-09-18 实证）

- 新板 d0f1784（SA8797P, Android 16 GVM）：QNN HTP 路径**全通**（Inception 子图 2 次推理过）；NSP 域 nsp1000~1003 在线（QNN 日志 domain 16 / effective 1600）。**domain 3（CDSP）亦实测可用**（2026-09-18 晚：GEHTP runner 在本板执行数百 op rc=0，停在未实现 broadcast opcode——`fastrpc_tests_apps -d 3` 的 0x72 系该工具缺自有测试 skel，曾误读为"域未分配"）。详见 RUNBOOK §7。
- 老板 52f67807：CDSP 楔死（0x39，boot_cdsp 60s 超时），恢复依赖板方物理上电。
- GEHTP runner 域号 **3 硬编码三处**：`scripts/gehtp:180`、`scripts/conv_add_pipeline.sh:77`、`scripts/device_run.sh:71`。
- 多树现状：4090 树（旧基线+陈旧 skel 已通报未修）、/disk2 树（转只读候选）、本树（C1 主家）。opcode 28 ABI 双树分裂已实锤（TASKBOOK §8.9）。
- 评审确认的假绿源：`if(true)` fprintf（graph_prepare.cpp:915）、15 个测试硬编码路径 SKIP 兜底、host execute 直通残余（A6 已治主脉）、adapt_weights_bin 尺寸猜测（两轮证伪）。
- 推送去重只看尺寸（gehtp:139-146），`wt_sha256.c` 闲置未接线。

---

## 2. Phase 0 — 即日清账（当天完成，零风险）

| # | 任务 | 范围 | 门 |
|---|---|---|---|
| 0.1 | RUNBOOK §7 入库 | `docs/GEHTP_DEVICE_RUNBOOK.md` 未提交改动 commit | git log 可见 |
| 0.2 | AGENTS.md 落地 | 仓库根新建：设备纪律 5 条 + 判据纪律 + 完成门 + 接口冻结点，从 PORTAL/RUNBOOK 摘录，<80 行 | 新会话首条指令即读得到 |
| 0.3 | 双 C1 树合一执行 | task/prof-wp 为唯一主家；/disk2、4090 树转只读（chmod/公告）；4090 毒 skel 修正 patch 随回合带上 | `git worktree list` 单真源 |

## 3. Phase 1 — 换板（P0，一切模型线的前置）

> 目标：d0f1784 上 `gehtp setup && gehtp_quickstart.sh` 全绿。
> 2026-09-18 晚更新：**域号探针已闭合**——domain 3 在新板实测可用（GEHTP runner 已在其上执行数百 op），无需改 NSP 域。当前换板关键路径 = broadcast opcode 补齐（runner 停点）+ C1/C4 设备门。

| # | 任务 | 范围/文件 | 门 |
|---|---|---|---|
| 1.1 | ~~NSP 域探针~~【已闭合】 | 实证：domain 3 可用（RUNBOOK §7.4，optrace 为证）；NSP 域（16/1600）留档为未来多核分流选项 | runner 已实跑 |
| 1.2 | broadcast opcode 补齐【关键路径】 | runner 停点：broadcast 形态未覆盖。归 A 线 emit 覆盖墙：定位具体 op（optrace 尾部 code=12 后续），emit handler + 设备执行体 | runner 跑过停点；conv_add/L3 回归绿 |
| 1.3 | gehtp setup 适配新板 | 四件套已就位（09-18）；核对 skel 正典 md5（防 third_party 毒件）、`--device d0f1784` 全流程 | setup 读回校验绿 |
| 1.4 | 设备队列协议升双板 | PORTAL §4 / TASKBOOK §2.3 登记表加板卡列；互斥检查命令按板参数化 | 文档+quickstart 检查同步 |
| 1.5 | **C1/C4 设备门补跑** | 0.8B（A3②真 scatter 重编 blob，qwen35_08b_scatter.wtop 已上板）+ minicpm（5.15GB v2 blob）在新板跑设备门 | judge_logits / judge_minicpm 判据达标，战役日志更新 |
| 1.6 | 域号参数化（降级，可选） | job.txt 加 `domain` 键；`--domain` 旗标默认 3；三处硬编码清除（gehtp:180、conv_add_pipeline.sh:77、device_run.sh:71）。目的=双板/未来多核，不再阻塞换板 | 双板 conv_add byte-exact |

**换板战役总门**：quickstart `=== SAMPLE ALL GREEN ===` 在 d0f1784 复现 + C1/C4 设备输出达标。

## 4. Phase 2 — 收敛（P0 假绿源，与 Phase 1 可并行，文件零重叠）

| # | 任务 | 范围/文件 | 门 |
|---|---|---|---|
| 2.1 | **路径A隔离** | `scheduler.cpp:694-735` 19 步硬编码 + `context_binary_writer.cpp` 内嵌字节模板 + `hnnx_compile.cpp:458-489` 写死权重 → 整体移 `tests/fixtures/`；产品路径只留 `--format tagged`，`--format qnn` 加显式「金样重放」警告 | conv_add/L3 回归 byte-exact 不变；ctree 全绿 |
| 2.2 | **graph_prepare.cpp 拆分** | 4819 行 → prepare / execute_host / extra_extractor / serialize 四文件；机械移动不改语义 | blob byte-exact 回归 |
| 2.3 | 调试残留清缴 | 删 `if(true)` fprintf（:915）、op dump 清单（:953/:986-990）；29 处 getenv 收编 `hnnx/debug.hpp` 单点；编码统一 UTF-8 无 BOM + LF（72 处 U+FFFD 修复）；删死代码 `mid_level_ir.hpp` | ctest 44/44；`grep -c "getenv" src/` 收敛 |
| 2.4 | **测试基建换血** | `test_paths.hpp` 硬编码 `/disk2`、`C:\Users` 全删；测试资产换小尺寸合成 net.json 入仓（`compiler/test_models/`）；SKIP 改显式（资产缺失=FAIL 红，除非 `--allow-skip`） | 干净机器 clone 后 ctest 无假绿 |
| 2.5 | adapt_weights_bin 换契约 | 尺寸猜测启发式 → qairt-converter 产出「名字→偏移」manifest；旧启发式留 `--legacy-weight-match` 一个版本 | 0.8B/4B 编译产物 byte-exact |

## 5. Phase 3 — 闭环（P1，工程化收尾）

| # | 任务 | 范围/文件 | 门 |
|---|---|---|---|
| 3.1 | `gehtp judge` 子命令 | manifest 增 `gold` 路径键；run 拉回输出后自动调 judge_minicpm/judge_logits 逻辑出红绿 | 四模型 run 即出判定 |
| 3.2 | 推送完整性 sha256 | `wt_sha256.c` 接入 gehtp run 去重判定，替尺寸猜测 | 人为改 1 字节 blob 必被重推 |
| 3.3 | runner↔lib 版本握手 | `libhvxhmx_v23.so` 加 `GEHTP_LIB_VERSION` 符号；runner 启动校验不匹配即拒跑 | 旧 lib+新 runner 启动即报错（不复现 opcode28 静默垃圾） |
| 3.4 | CI smoke（无设备） | ctest + conv_add compile byte-exact + 预检回归；材料：本计划 §2.4 的合成资产 | CI 配置入仓 + 一次全绿记录 |
| 3.5 | 供给槽显式化 | MATMUL bias/atbl/otbl 尺寸猜测（oplist_exec.c:336-342）→ blob v3 reserved 字段显式 slot id | 双同尺寸槽合成图不歧义 |

## 6. 接续关系（原任务书不动，以下时机衔接）

- **A3②③（ScatterNd 真语义设备门）**：依赖 Phase 1 换板（老板未愈）→ 新板 C1 设备门一并收口。
- **A5 预检门禁**：与 2.4 合成资产天然同批做。
- **B1 快速件接线（性能）**：Phase 1 后进行，设备段在新板排期。
- **D4 cost model 标定**：新板 optrace 路径硬编码问题（oplist_exec.c:71）随 3.x 顺手修。

## 7. 风险与回退

- **broadcast 补齐比预期深**：runner 停点若牵出 broadcast 语义族（rank 对齐/双槽广播），工作量升 A 线中任务；先定位 optrace 尾部具体 opcode 再排期。
- **路径A隔离触碰字节对拍基建**：fixtures 迁移保留 `--format qnn` 入口（仅测试引用），回归门照旧 byte-exact。
- **多树合一通报遗漏**：合一公告须显式点名 4090 毒 skel patch（TASKBOOK §8.10），否则 104 侧 setup 再毒化设备。

## 8. 时间盒建议

- Phase 0：当天。
- Phase 1：1~2 个工作日（关键路径 = 1.2 broadcast opcode 补齐；1.5 取决于 blob 大小与设备窗口）。
- Phase 2：2~3 个工作日，可拆给平行会话（2.1/2.2 编译器文件与 1.x 设备侧零重叠）。
- Phase 3：穿插进行，每项独立 commit 独立门。

**总验收门**：双板 quickstart 全绿 + C1/C4 新板设备判据达标 + 干净机器 ctest 无假绿 + `gehtp judge` 四模型闭环 + CI smoke 全绿。
