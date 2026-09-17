/* hvxhmx_v22.h — V2.2 单元功能库 umbrella 头
 * =====================================================================
 * 一行 include 拉入 V2.2 全部公共 API:
 *   #include "hvxhmx_v22.h"
 *
 * V2.2 = V2.1 算子层 (hvhx_v2_* 60 符号) + 8 个已设备闭合的工程单元:
 *
 *   单元            头                源模块 (设备 PASS 存档)
 *   --------------  ----------------  ----------------------------------
 *   U1 wtcache      wtcache.h         wtcache_pin_v81 (T1-T9 9/9)
 *   U2 dcmem        dc_parts.h        dualcore_v81/4C (arena+DMA+W4 引擎)
 *   U3 dcthread     dc_threads.h      dualcore_v81/4C (QURT 线程+同步)
 *   U4 w4a16        dc_parts.h        t10 (w4a16_invoke, M%256 约束)
 *                    ↑ htpw4a16_v81 的 kernel .inc 同源 (37 LSB 规范值)
 *   U5 gdnsm        gdn_sm.h          gdn_sm_v81 (G1-G9)
 *   U6 oplist       oplist_parse.h    wt_repack_v81 (W1-W5, blob v1)
 *                   oplist_exec.h
 *   U7 dualdom      dd_worker.h       dualdomain_v81 (D1-D7, 2.001×)
 *   U8 smallm       (U4 的 pad-256 用法, 见 examples/18 + docs)
 *
 * cache 协议四铁律 (任何混访代码上机前过一遍, docs/api_v22_overview.md):
 *   ① CPU 写完 DDR 给 DMA bypass 读之前 dc_clean_ddr
 *   ② wtcache_open 已全 VTCM FLUSH (V2.2 起内置, 原 memset 脏零行坑)
 *   ③ DMA 写完的 DDR CPU 读之前 QURT_MEM_CACHE_INVALIDATE
 *   ④ 退出必 wtcache_close / wt_exec_shutdown (PASS/FAIL 两路)
 */
#ifndef HVXHMX_V22_H
#define HVXHMX_V22_H

/* ---- V2.1 算子层 (全量) ---- */
#include "hvxhmx_v2.h"

/* ---- V2.2 工程单元 ---- */
#include "wtcache.h"        /* U1: pin/ring VTCM 权重缓存 */
#include "dc_threads.h"     /* U3: QURT 线程/同步原语 */
#include "dc_parts.h"       /* U2+U4: arena/DMA 流/dc_w4 W4A16 引擎 */
#include "gdn_sm.h"         /* U5: conv1d/delta-rule/solve-tri 状态机 */
#include "oplist_parse.h"   /* U6: blob v1 解析 (host/设备同源) */
#include "oplist_exec.h"    /* U6: op 执行引擎 */
#include "dd_worker.h"      /* U7: 双域 step-list 分片执行 */

#endif /* HVXHMX_V22_H */
