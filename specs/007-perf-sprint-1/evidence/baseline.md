# 007 基线锚（T003）

**Date**: 2026-10-09
**Source**: 003 v4 产物（主树 `test_assets/qwen35_08b/qwen35_08b_v4.wtop` + `.scratch_003_rebuild/prof_v4/v4.op_ts.bin`，设备 d0f1784，库 d33e31b3）
**Parser**: `scripts/gehtp_prof.py`（判据入库；CSV 快照 `/tmp/v4_prof.csv` 为易失中间品，数字以本文为准）

## 总锚（与跨会话协调表一致）

| 指标 | 值 |
|---|---|
| ops | 14188（blob_ver=3，ts_ops 齐平） |
| 设备 exec wall | 31778.2ms（sum_op 31757.9ms，缝隙 0.1%） |
| 全链 wall（judge 口径） | 35.31s（003 v4 锚） |
| cos / top1 | 0.998642 / 30/32（hf 口径，f16 锚） |
| 偏差报告 | 按计划 13165 \| **允许回落 1023** \| 中止 0；不变量 PASS |

## broadcast 族聚合（opcode 26 = BROADCAST_F16，emit 名 Eltwise_Binary）

| 切片 | n | total_us | %wall | avg_us |
|---|---|---|---|---|
| BROADCAST_F16 全族 | 1622 | 7,841,308 | **24.7%** | 4834.3 |
| └ planned=HVX → actual=scalar（dev=1） | **918** | **4,966,565** | 15.6% | 5410 |
| └ planned=scalar（dev=0，emit 现判据 n%32 已排） | 704 | 2,874,743 | 9.0% | 4083 |

## 1023 允许回落全构成（dev=1 按 opcode）

| opcode_name | planned→actual | n | total_us |
|---|---|---|---|
| BROADCAST_F16 | HVX→scalar | 918 | 4,966,565 |
| UNARY_F16 | HVX→scalar | 86 | 159,206 |
| ADD_F16 | HVX→scalar | 18 | 10,524 |
| BINARY_F16 | HVX→scalar | 1 | 693 |

**口径修正（对 spec SC-001/FR-002）**：broadcast 族回落真值 = **918**（非 1023——1023 是全图总数，含 UNARY 86/ADD 18/BINARY 1 三笔非 broadcast 尾数）。P1 门 = broadcast 族 dev=1 918→0；全图剩余 105 笔非 P1 范围（UNARY 尾数建议 P3 普查时顺带看）。

## 其余 top 族（供 T018 热点排序与 T004 交叉核）

| opcode | n | total_us | %wall |
|---|---|---|---|
| MATMUL_F16 | 325 | 9,440,055 | 29.7% |
| OP35(FullyConnected HMX) | 32 | 4,248,976 | 13.4% |
| STRIDED_SLICE_F16 | 2373 | 3,062,491 | 9.6% |
| UNARY_F16 | 2909 | 3,049,771 | 9.6%（138 已跑 HVX dev=0） |
| TRANSPOSE_GEN_F16 | 175 | 1,290,927 | 4.1% |
| BINARY_F16 | 1655 | 1,068,922 | 3.4%（1005 已跑 HVX） |

## P1 收益预期（实证更新）

- 消 918 回落：形态 1/2 HVX 化后该 4.97s 子集直接受益（现 avg 5.41ms/op 标量 → HVX 块拷贝）。
- 704 planned-scalar 子集（2.87s）待 T004 形态普查定形态归属：若多为类 1/2（n%32 误排），P1 后计划改 HVX 还有增量收益；若多类 3 则维持 scalar 合法。
- 原 -7.3s 估计对应全族 7.84s 全部受益的乐观口径；保守口径（仅 918 笔 + 类 1/2 比例）由 T004 census 定。
