# 008 锚与设备登记（T002）

**日期**: 2026-10-09 | **登记人**: 008 会话（worktree `/disk1/gehtp-wt/008-vtcm-residency-dma`，分支 `008-vtcm-residency-dma` @ `7deb63b`，同 007 基点）

## 设备

| 项 | 值 |
|---|---|
| 设备 | d0f1784（online，`getenforce` = Permissive） |
| 板上占用 | pgrep run_main_on_hexagon = 空（BOARD_FREE） |
| DEVROOT | /data/local/tmp/hvxhmx23 |
| 板上 lib md5 | `61049952`（协调表口径 d33e31b3 系，004 于 1008 推） |
| 本地主树 kernels/lib md5 | `3da98052`（**与板上不一致**——008 任何上板前必须从本 worktree 重编+推签名 lib，不得引用主树残留） |

## 基线锚

| 口径 | 值 | 备注 |
|---|---|---|
| 主口径（007 交卷锚） | 待 007 收口（≤20s） | **007 未交卷**（tasks T001-T027 实施中） |
| 回退口径（v4） | wall 35.31s / cos 0.998642 / top1 30/32 / 14188 op / 允许回落 1023 | 当前库最快，位级对拍基准 blob = `test_assets/qwen35_08b/qwen35_08b_v4.wtop` |
| f16 精度门 | 同库锚不回归（0.998642 族） | US1 验收 = 新 blob 输出与 v4 逐字节 md5 同 |
| W4 期口径 | 沿 007 冻结值（未出） | 本线 Phase 2 前不消费 |

## 编译输入（T003 用，沿 0.8B 资产线）

- net.json: `/disk1/gehtp/.scratch_a3/qwen35_08b_v2.wtop.work/qwen35_08b_v2_net.json`
- weights: `/disk1/gehtp/.scratch_a3/qwen35_08b_v2.wtop.work/model.params.bin`（2.04GB f16）
- 输入 raw: ids_from_golden.i32.raw / mask_ones.f16.raw / pos_arange.i32.raw（同目录）
- 关键旗标：`--ddr-budget <n>`（0=不规划）+ `--plan-vtcm-budget <n>`（静态 VTCM 驻留池字节）
- 编译时长参考：~70 分钟（ tagged.bin ~4GB）

## 关键机制事实（research D1 复核，T003 前）

静态规划门 = `graph_prepare.cpp:591` `if (plan_ddr_budget_ > 0)`；产物经 TAG_MEM_PLAN 进 tagged.bin（`:575-580` 注释「wtop_emit 照抄不再自算」）。v4 的 tagged.bin 编译时预算缺省 0 → `has_mem_plan()==false` → emit 侧 `if (gp.has_mem_plan())`（wtop_emit.cpp:117）整块旁路 → 无 `[ddr]` stderr 行、无 TEMPOFF 槽。**T003 = 带 `--ddr-budget 2147483648 --plan-vtcm-budget 16777216` 重跑 hnnx_compile**。
