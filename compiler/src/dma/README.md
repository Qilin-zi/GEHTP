# src/dma/ — DMA 操作

对应真实 `spill_fill.cc` / `op_emitter.cc`。VTCM 溢出/回填与 op 发射。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `spill_fill.cpp` | `insert_spillfill.cc` / `grdep_spillfill.cc` | SpillFill: 当张量超 VTCM 时插入 spill (写回 DDR) / fill (重新载入) 操作 |
| `op_emitter.cpp` | `op_emitter.cc` | OpEmitter: 发射 op 指令, insert_preload_op (预取下一块) |

## 机制

- **spill/fill**: 张量生命周期内若 VTCM 不够, 先 spill 到 DDR, 用时再 fill 回来
- **preload**: 当前块执行时 DMA 预取下一块 (双缓冲, 隐藏延迟)
