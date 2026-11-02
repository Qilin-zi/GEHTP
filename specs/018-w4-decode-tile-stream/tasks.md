# 018 tasks

- [x] T001 契约: oplist_parse.h flags bit5 (0x20) W4TS 注释 + op_matmul.cpp emit 判别
- [x] T002 执行器: exec_matmul_w4_stream (激活量化 + DMA ring + rmpy matvec + 收尾) + M<32 分发 + host 标量参考
- [x] T003 host 对拍: 0.6B decode_q4 blob host_run — 新路径 vs 基线 cos ≥ 0.9999 (G2)
- [x] T004 合成门: pack_w4stream_ab + 例 53 上板 — cos ≥ 0.9999 + GB/s ≥ 30 (G1 生死门)
- [x] T005 E2E 0.6B: 重 emit decode_q4.wtop (bit5) 上板 — ≤100ms + top1 呈报 (G3)
- [x] T006 E2E 0.8B + f16 零回归 + ctest 全绿 (G4/G5)
- [x] T007 收口: evidence 落盘 + memory 回写 + push 主线裁决

见 evidence/verification.md (2026-11-01 收口)。
