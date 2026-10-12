# T025 quickstart 四场景复跑收口 digest (1010)

**源**：quickstart.md 四场景 = US1-US4 各阶段板跑/emit 的 runbook；本线各 US 已按序完成，
场景证据已归档 `specs/008-vtcm-residency-dma/evidence/`（不入 /tmp）。本 digest 做四场景
→ 证据映射 + 终态数字核对（归档收口，非新增复跑——各场景在对应 US 阶段已实测）。

## 四场景 → 证据映射

| 场景 | 判据 | 实测 | 证据 |
|---|---|---|---|
| 1 预算基线冻结 | `[ddr]` 两行出现 + 数字落盘 | TAG_MEM_PLAN/TEMPOFF 出现；coverage 0.8401→0.8395；n_spills=0 | `budget-baseline.md` + `ddr_offsets_p0..p3.tsv` |
| 2 US1 驻留扩面 | 位级 md5 / coverage≥80% / 预算负例 | md5 f87461e2；coverage **0.8395**；`--vtcm-budget` 超限硬报错（`error: VTCM 三方预算门超限`） | `budget-baseline.md`（GATE1 位级 + GATE2 覆盖 + 负例） |
| 3 US2 DMA 双缓冲 | fence micro-test + dma_us>0 + HMX 降 | fence 1000 轮并发逐字节一致；Σdma_us=22.53ms Σdma=1.4GB 66.7GB/s；T013 双缓冲隐藏上界 | `fence-microtest.txt` + `t014-dual-timeline-overlap.txt` + `t015-staging-fallback-boardrun.txt` |
| 4 US3/US4 收口 | 三族≤2s / op-ts-sample / ctest + 003 五场景 | 三族 0.560s；sample 幂等；ctest 48/48；五场景逐位 | `t018-three-family-digest.md` + `t022-us4-optsample-gate.md` + `t023-final-board-gate.md` + `t024-regression-003-five-scenarios.md` |

## 终态数字（SC 总门核对）

| SC | 目标 | 实测 | 判定 |
|---|---|---|---|
| SC-003 全链 wall | ≤8s（007 锚口径） | exec **6.31s**（007 锚 24.24s，3.84×） | ✓ |
| SC-001 coverage | ≥80% | **0.8395**（1630/1631 物化 temp VTCM 驻留） | ✓ |
| SC-004 f16 位级 | 逐字节不动 | md5 **f87461e2** 恒等 | ✓ |
| SC-004 W4 q4_0 | 沿 007 冻结门不降 | 0.98/20（Phase 1 纯 f16 未消费） | ✓ |
| SC-005 per-op 开销 | ≤0.5s | ~0（16-18ms 簿记） | ✓ |
| SC-005 残余三族 | ≤2s | 0.560s | ✓ |

## 结论

四场景在对应 US 阶段已全部实测通过，证据归档齐备。终态数字六门全绿，无遗漏复跑项。
**008 线 Phase 1（US1-US4 + Polish）收口完成。**

## 落地资产

- 证据目录：`specs/008-vtcm-residency-dma/evidence/`（20 文件，见 t024 digest 清单）
- 板跑脚本：`.scratch_008/run_008t{19,20,21}_board.sh` + `run_008_t024_scenarios.sh`（gitignore）
