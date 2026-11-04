# 020 tasks

- [ ] T001 契约: oplist_parse.h v3 新增 DMA schedule footer 格式 (记录 {op_idx, src_offset u64, len, vtcm_dst}) + `wt_blob` 增 `dma_base/n_dma` + 解析
- [ ] T002 编译器: wtop_emit.cpp 在 `hmx_arena_max_of` bit5 分支生成 footer (扫描 bit5 matmul, 摊平权重组, src_offset 取 slot 内偏移)
- [ ] T003 执行器 DMA 重放层: oplist_exec.c 全局 VTCM 双缓冲 + footer 读入 + 提前 N 组菊花链流水 (去 per-op `dma_wait_for_idle`)
- [ ] T004 执行器消费层: `exec_matmul_w4_stream` 改消费全局流 (激活量化仍 per-op; 权重从双缓冲取 tile 算, 算完不 drain)
- [ ] T005 host 参考: host 标量镜像 footer 路径 vs 基线 cos ≥ 0.9999 (G2)
- [ ] T006 零回归: 无 bit5 blob `off_check` md5 == 主线 (位级同) + ctest 全绿 (G1)
- [ ] T007 E2E 0.6B: 重 emit `decode_q4.wtop` (footer) 上板 — 精度 cos≥0.9999/argmax 同 + 性能 37.9→~36.5ms 呈报 (G2/G3, 实测 DDR 争用净负与否)
- [x] T008 2 NSP: 编译期双域 footer 生成 + 双序列吞吐 ~2× 位级同单域 (G4) — 2026-11-04 上板 PASS (device 1.998×/host-wall 1.98×, 位级同单域, footer 单份双域兼容编译期零改动)
- [ ] T009 收口: evidence 落盘 + memory 回写 + 衔接 021 批处理立项

依赖链: T001→T002→(T003,T004 并行)→T005→T006→T007→T008→T009。
见 plan.md 021 依赖链。
