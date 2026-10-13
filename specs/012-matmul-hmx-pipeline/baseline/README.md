# 012 基线锚（T001/T002）

日期: 2026-10-13 | 设备 d0f1784 | HEAD c77665a (merge/010-into-mainline)

## 库与 runner

- 内核库: `kernels/lib/libhvxhmx_v23.signed.so` (md5 da2ca7e4, 10-13 03:41 构建) → 私有名 `libhvxhmx_012.so`
- runner: `gehtp_runner_012.so` (DT_NEEDED libhvxhmx_012.so, vgather=0, SWIV 签名)
- blob: `.scratch_009/qwen35_08b_fused.wtop` (2013522176B, gdn_block 全开 400/18)
- 输入: `.scratch_a3/qwen35_08b_v2.wtop.work/ids_from_golden.i32.raw`
- 跑板脚本: `.scratch_012/run_012_board.sh <tag>`

## 基线数值 (baseline1 / baseline2 两次)

| 指标 | 值 |
|---|---|
| wall (op_ts) | 14121.5ms / 14120.0ms (差 0.01%, 可复现) |
| 输出 md5 | `02c69d93d0ca077be7e1f360a75d083b` (两次位级同; = 009 时代输出) |
| cos / top1 | 0.998639 / 29/32 (= 009 T026 gdn 全开锚; judge 门 0.99864 为四舍五入假 FAIL) |
| Σdur | 14103.9ms, 缝隙 17.6ms |

## 逐 opcode 基线 (改前)

| opcode | n | total_us | %wall | avg_us | max_us |
|---|---|---|---|---|---|
| MATMUL_F16 | 181 | 8024468 | 56.5% | 44334 | 104034 |
| OP35 (F16_SPLIT) | 32 | 4362574 | 30.7% | 136330 | 140626 |
| OP37 (GDN) | 18 | 640262 | 4.5% | 35570 | 37066 |
| UNARY_F16 | 191 | 372190 | 2.6% | | |

matmul+FC 合计 ≈ 87.2% wall。目标 SC-001: wall −20% (→ ~11.3s)。

产物: `baseline1.op_ts.bin` / `baseline2.op_ts.bin` / `output_012_baseline{1,2}.f16.raw` 在 `.scratch_012/`;
pre.op_ts.bin (009 时代参照) 在本目录。

---

## 012-10-13 更新: 新锚 (ctrl/pp, 见 results.md)

- ctrl = 011 工具重 emit 无 pp: wall 13575.5ms, cos 0.999930/top1 32/32, md5 9d1c0877 (fold=92, ops 884)
- pp = ctrl + --dma-runlist ([pp]182): wall 2755.5ms (−79.7%), 输出对 ctrl 位级同
- 老 009 blob 锚 (14120ms/0.998639) 保留为历史参照; 老 blob 已知漏 --dma-runlist
- blob: .scratch_012/qwen35_08b_fused_{ctrl,pp}.wtop; 库: libhvxhmx_012 (hoist-only)
