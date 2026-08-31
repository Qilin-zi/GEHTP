# src/tiling/ — tiling 系统

对应真实 `tiler.cc` / `simple_tiler.cc` / `supertile.cc`。把大张量切成 VTCM 友好的小块。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `tiler.cpp` | `simple_tiler.cc` / `supertile.cc` 等 | SimpleTiler, **ConvTiler**, **MatMulTiler** (读真实 OutputDef), Supertiler (合并相邻), TileDistributor (round-robin), TileExtractor/Conformer, compute_conv_tile_cost |

## Tiler 对比

| Tiler | 切分策略 | 数据来源 |
|-------|----------|----------|
| SimpleTiler | 固定 224x224 按 config 切 | 硬编码默认 |
| ConvTiler | 按 H/W 切 (batch/channel 不切) | op->cached_out_def 真实 N,H,W,C |
| MatMulTiler | 按 M/N 切 (K 不切) | op->cached_out_def 真实 M,N |
| Supertiler | SimpleTiler + 合并相邻 | — |

## conform (对齐)

按 dtype 对齐到 HVX 128B:
- Int8/UInt8: 128 元素
- Int16/Float16: 64 元素
- Int32/Float32: 32 元素
