/* hvxhmx_v23.h — V2.3 单元功能库 umbrella 头
 * =====================================================================
 * 一行 include 拉入 V2.3 全部公共 API:
 *   #include "hvxhmx_v23.h"
 *
 * V2.3 = V2.2 全量 (v22 8 单元) + 9 个新工程单元 (src 已融合, 无 v22 目录):
 *
 *   单元            头                源                        例
 *   --------------  ----------------  ------------------------  ----------
 *   U9  fence       fence.h           runtime/fence.c           23
 *   U10 arena       arena.h           runtime/arena.c           24
 *   U11 harness     harness.h         runtime/harness.c         25
 *   U12 wpool       wpool.h           runtime/wpool.c           26
 *   U13 pxbridge    pxbridge.h        hvx/pxbridge.c            27
 *   U14 gdntree     gdn_tree.h        hvx/gdn_tree.c(+gdn_ref)  28
 *   U15 kvcache     kvcache.h         runtime/kvcache.c         29
 *   U16 graphstep   oplist_exec.h     runtime/oplist_exec.c     30
 *   U17 gemmdisp    gemm_dispatch.h   runtime/gemm_dispatch.c   31
 *   U18 rbr         rbr.h             runtime/rbr.c             32
 *   U19 bledger     bledger.h         runtime/bledger.c         33
 *   U20 dmaring     ring_policy.h     runtime/ring_sim.c(+policy) 34
 *   U21 btrack      btrack.h          runtime/btrack.c(+bflush+dcache) 35
 *
 * cache 协议四铁律同 V2.2 (docs/api_v22_overview.md); fence 单元把
 * (writer,reader,mem) → 决策表固化成 API, 混访代码直接调 fence_handoff。
 */
#ifndef HVXHMX_V23_H
#define HVXHMX_V23_H

/* ---- V2.2 全量 ---- */
#include "hvxhmx_v22.h"

/* ---- V2.3 工程单元 ---- */
#include "fence.h"          /* U9:  cache 决策表 + handoff */
#include "arena.h"          /* U10: 双池 boundary-tag 分配器 */
#include "harness.h"        /* U11: 金标对拍框架 (sha 钉字节) */
#include "wpool.h"          /* U12: 常驻 job 工人池 + hmx 门 */
#include "pxbridge.h"       /* U13: f32↔f16↔INT16 对称量化桥 */
#include "gdn_tree.h"       /* U14: 树形 GDN 串/并执行 */
#include "kvcache.h"        /* U15: append/scatter KV 管理 */
#include "gemm_dispatch.h"  /* U17: M/K/N 三路由决策 */
#include "rbr.h"           /* U18: recurrent 状态部分接受回退 */
#include "bledger.h"       /* U19: Buffer Ledger 数据流溯源审计 */
#include "ring_policy.h"  /* U20: DMA 预取环 (engine_adapter + 策略库) */
#include "bflush.h"       /* U21: 写跟踪 + 定向 flush (含 btrack.h) */
#include "dcache.h"       /* U21: 版本化派生格式缓存 */

#endif /* HVXHMX_V23_H */
