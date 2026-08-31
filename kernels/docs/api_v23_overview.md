# api_v23_overview — V2.3 总览 (V2.2 全量 + 13 个新单元)

**V2.3 = V2.2 全部功能 + src 布局融合 + 13 个新单元 (U9-U21)**, 单库
`lib/libhvxhmx_v23.so` (SWIV 签名), umbrella 头 `include/hvxhmx_v23.h`。

## V2.2 单元 (U1-U8, 手册不变)

wtcache / dcmem / dcthread / w4a16 / gdnsm / oplist / dualdom / smallm —
见 `api_v22_*.md`。源已从 `src/v22/` **融合进原目录** (V2.3 起无 v22 子目录):

```
src/runtime  wtcache_impl dc_parts dc_sync dc_threads dd_worker dma_utils
             oplist_parse oplist_exec wt_sha256 wt_w3
             + fence arena wpool kvcache harness gemm_dispatch   ← V2.3 新
src/hmx      conv 族 + w4a16_driver_dc (+.inc)
src/hvx      V2.1 算子 + gdn_kern gdn_ref + gdn_tree pxbridge     ← V2.3 新
src/compat   V2.1 版本兼容层
```

## V2.3 新单元 (U9-U21)

| # | 单元 | 一句话 | 手册 | 例 (门) |
|---|------|--------|------|---------|
| U9 | fence | 方向对偶 cache fence 决策表 (四铁律收敛) | [api_v23_fence.md](api_v23_fence.md) | 23 (7) |
| U10 | arena | DDR/VTCM 双池对齐 arena (free+合并) | [api_v23_arena.md](api_v23_arena.md) | 24 (4) |
| U11 | harness | 对拍框架 (ex_*+sha256 case 流) | [api_v23_harness.md](api_v23_harness.md) | 25 (13) |
| U12 | wpool | 常驻 worker 池 + hmx_lock 交接 | [api_v23_wpool.md](api_v23_wpool.md) | 26 (5) |
| U13 | pxbridge | f32↔f16↔INT16 桥 (unipolar 契约) | [api_v23_pxbridge.md](api_v23_pxbridge.md) | 27 (20) |
| U14 | gdntree | 树形 GDN 闭式 oracle + f16 kernel | [api_v23_gdntree.md](api_v23_gdntree.md) | 28 (8) |
| U15 | kvcache | KV 槽缓存 (append/scatter/posmap) | [api_v23_kvcache.md](api_v23_kvcache.md) | 29 (8) |
| U16 | graphstep | oplist 整步/分段执行 + 统计 | [api_v23_graphstep.md](api_v23_graphstep.md) | 30 (8) |
| U17 | gemmdispatch | MatMul 三路由决策 + 执行体 | [api_v23_gemmdispatch.md](api_v23_gemmdispatch.md) | 31 (8) |
| U18 | rbr | recurrent 状态部分接受回退 (快照→回拨→重放) | [api_v23_rbr.md](api_v23_rbr.md) | 32 (8) |
| U19 | bledger | Buffer Ledger 数据流溯源审计 (断链消费即报错) | [api_v23_bledger.md](api_v23_bledger.md) | 33 (9) |
| U20 | dmaring | DMA 预取环: 引擎 FSM 仿真 + Law1-8 断言 + canonical submit | [api_v23_dmaring.md](api_v23_dmaring.md) | 34 (26) |
| U21 | btrack | 写跟踪位图 + 定向 flush 决策 + 版本化派生格式缓存 | [api_v23_btrack.md](api_v23_btrack.md) | 35 (54) |

依赖: fence/arena/harness/pxbridge/rbr/bledger 无依赖; wpool←fence; kvcache←fence;
gdntree←fence+arena+pxbridge; graphstep←wpool; gemmdispatch←例 17/18 引擎。

## 验收状态 (52f67807, 2026-08-17)

`./build_examples.sh all` → **35/35 例全绿, 261 门 [PASS] 0 FAIL**
(另例 15 内嵌 49 项 PASS; U20 例 34 = 26 门, U21 例 35 = 54 门)。
host 模拟 `host/gtest_v23.c` 18/18; 例 34/35 host x86 冒烟 26+54 全绿后上机,
设备数字与 host 逐位一致 (sparse 28.5% / reuse3 66.7% / 门铃比 1.000)。
性能与完整判据: [PERF_REPORT.md](../PERF_REPORT.md)。

## 上机纪律 (V2.2 沿用)

- 设备仅 **52f67807**; skel 用 hvxhmx_libs 2026-08-10 版 (build_examples 自动 cp)。
- 静态门全绿 (vgather=0, UNDEF 白名单) → SWIV 签名 → 才推设备。
- cache 协议四铁律见 [api_v22_overview.md](api_v22_overview.md); 新代码一律走
  U9 `fence_handoff`, 不再手写 `qurt_mem_cache_clean`。
