# U21 btrack — 写跟踪 + 定向 flush + 派生格式缓存 (GENERIC_B 问题书实现)

> 源: `docs(外部)/GENERIC_B_WRITE_TRACKING_CACHE_PROBLEM.md`
> 单元: `include/btrack.h` + `include/bflush.h` + `include/dcache.h`
> 实现: `src/runtime/btrack.c` + `src/runtime/bflush.c` + `src/runtime/dcache.c`
> 例: `examples/35_btrack` (54 门, 设备 52f67807 全绿)

## 1. 架构: 一个事实来源, 两个消费者

```
CPU 写 ──► bt_mark_cpu_write ──┐
DMA 写 ──► bt_mark_dma_write ──┤ B1 位图 (唯一事实来源) ──► dirty 快照 ──► B2 bf_boundary
对等写 ──► bt_mark_peer_write ─┘        │                                    (定向 flush 决策)
                                        └── ver(buf) ──► B3 dc_get_or_convert
                                                         (版本化派生格式缓存)
```

## 2. B1 btrack — API 与语义

```c
bt_create(mem_bytes, blk_bytes, concurrency_mode);   /* LOCK / SHARD / ATOMIC */
bt_register_buffer(bt, base, size, &buf_id);         /* 粗粒度版本索引 */
bt_mark_cpu_write / bt_mark_peer_write(bt, addr, size);
bt_mark_dma_write(bt, addr, size, &token);           /* 铁律: 先标记后提交 */
bt_dma_complete(bt, token);                          /* 引擎 done 后关闭记账 */
bt_version(bt, buf_id);                              /* 批量区间一版本 */
bt_snapshot_dirty(bt, &snap);  bt_snapshot_free(&snap);
bt_clear / bt_clear_all_flushed(bt, snap, ...);      /* 仅清已快照过的位 */
bt_merge(bt, &external);                             /* 硬件脏页位并入 */
```

- **快照超集性质** (安全性地基): 允许重报已清位 (慢), 禁止漏报 (AX1 破坏)。
  LOCK/SHARD = stop-the-world 拷贝; ATOMIC = 两遍读 OR
  (mark 只 OR + clear 只清已快照位 ⇒ 并集 ⊇ 第一遍开始时刻脏集)。
- **F-B4 回调丢失**: 位保守保留 + `bt_dma_pending()` 审计可见, 正确性无损。
- 测试钩子: `bt_set_ver_bits` (回绕注入), `bt_debug_bump_version`,
  `bt_flag_suspect`/`bt_is_suspect` (canary 置疑 → B2 永久 FULL)。

## 3. B2 bflush — 决策器

```
bf_boundary: 快照 → 按 DECIDE_GRAN(4KiB) 合并相邻粒 → 区间集 R
  est = T_START·|R| + Σ bytes(r)/64B × cost_per_blk
  est·(1+MARGIN) < F_ALL → 定向逐区间 flush_range_inval
  否则 / 置疑 / 快照失败 → FULL (AX3 保守回退, 原因分类可查)
执行后 bt_clear_all_flushed(snap)。
```

参数默认 (附录 A): F_ALL=50µs, T_START=1µs, cost=20ns/64B, MARGIN=10%。
`bf_set_blk_cost` 供 F-B9 扰动; `bf_recalibrate` 更新 F_ALL;
`bf_get_stats`: n_full/n_directed/saved_ns/n_by_reason[4]。

**实现坑 (build 一次修复)**: 同一粒度内的多个脏 bit 不得各开新区间
(曾 128 bit → 127 区间 → est 爆表误选 FULL); 正确逻辑 =
同粒跳过 / 相邻粒合并 / 否则新区间。

## 4. B3 dcache — seqlock 形态缓存

```
读者: v0=ver(buf) → 查 (buf, v0, fmt) → 命中则复核 ver 未变
      未命中: 拷贝源快照 → convert (纯函数, 只读快照) → 复核 ver==v0
      → 原子发布 (数据完整填充后才置 valid, AX5); 期间版本变 → 返回 -1 重试
写者: bt_mark_* 原子推进版本 → 旧 key 永远查不到 (尸体由替换回收)
```

- key 用**全宽 64 位 ver**; `bt_set_ver_bits(8)` 注入回绕时
  掩码撞车但全宽不等 → `wrap_misses++` 保守未命中 (F-B5 零错命中);
- 替换: 尸体优先 → LRU / 直接映射; 双限 max_entries + max_total_bytes;
- 产物写库私有区 → 不产生新脏块 (F-B7: 开/关缓存 dirty 快照逐位一致);
- 命名注意: 失效 API 叫 `dcache_invalidate` (V2.2 dc_sync.c 已占
  `dc_invalidate` 符号, 链接器多定义)。

## 5. 例 35 门 (设备全绿 54/54)

| 门 | 内容 | 关键数字 (设备=host) |
|---|---|---|
| G1/G2/G6 | 金丝雀: 单写单区间 / 同块 100 写版本+100 / 空边界零调用 | — |
| G3 | DMA 先标后写 AX1 + F-B4 丢失保守 + token 审计 | pending 1→0 |
| G4 | 转换期间写 → -1 → 重试命中 | retries≥1 |
| G5 | VER_BITS=8 回绕 → 保守未命中零错命中 | wrap_misses≥1 |
| G7 | sparse_13b 13 边界定向 | 28.5% ≤ 30% (V2) |
| G8 | dense_13b 脏 90% → 全选 FULL | 13/13 |
| G9 | reuse3 命中 66.7% ≥ 50%; noreuse 0% ≤ 5% (V3) | — |
| G10 | F-B1 影子审计检出 → 置疑永久 FULL + 零损坏 (V6) | — |
| G11 | F-B2 标后未清 → 重复 flush 正确性无损 | — |
| G12 | F-B6 交织双写者快照超集 (ATOMIC+SHARD) | — |
| G13 | F-B7 缓存不扰 B2 逐位一致 | — |
| G14 | F-B8 逐出命中完整 + F-B9 ±50% 扰动 AX1 仍立 | evictions≥4 |
| G15 | 统计对账 (V7): 边界/转换/命中/省额 | — |

## 6. 使用纪律

1. **三条写路径全部挂钩** — 任何绕钩子的写 = F-B1 静默损坏,
   生产环境靠 canary 抽检 + `bt_flag_suspect` 永久回退 FULL 兜底;
2. DMA **先标记后提交** (描述符提交前调 mark, done 后调 complete);
3. 批量写用区间接口 (一次区间一版本), 高频碎写会伤 B3 命中率;
4. `bt_snapshot` 只许通过 `bt_clear*` 消费, 不许手改位图。
