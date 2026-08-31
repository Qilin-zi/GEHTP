/* oplist_exec.h — V2.2 oplist 执行器公共接口 (源: wt_repack_v81 MODULE D)
 *
 * 设备侧 blob 执行引擎: 解析产物 wt_blob → 顺序执行 op 表 (NOP/PIN/
 * MATMUL_W4A16/RMSNORM_F16), temp id 0..7 跨 op 传递中间结果。
 *
 * 生命周期铁律 (V81 cache 协议, 见 docs/api_v22_oplist.md):
 *   1. blob 用 dc_read_file 读入后必须 dc_clean_ddr (CPU 写, DMA bypass 要读)
 *   2. wt_exec_run 完必须 wt_exec_shutdown (PASS/FAIL 两路都要), 否则
 *      VTCM/HMX 占死域, 下一进程连文件都建不出
 *   3. 引擎 MATMUL 面全部 2KB 对齐 (HMX mxmem 约束), carve 已内置
 */
#ifndef HVXHMX_V22_OPLIST_EXEC_H
#define HVXHMX_V22_OPLIST_EXEC_H

#include <stdint.h>
#include <stddef.h>
#include "oplist_parse.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WT_EXEC_MAX_TEMPS 8

/* 顺序执行 b->ops。engine_m 回填引擎形状 (0=无 MATMUL op)。
 * op_us (可 NULL) 回填每 op 耗时表数组 (长度必须 >= n_ops)。
 * err/errn (可 NULL) 出错时写人类可读原因。
 * 返回 0 = 全部 op 成功; 负 = 失败 (err 有内容)。 */
int wt_exec_run(const struct wt_blob* b, uint32_t* engine_m,
                int64_t* op_us, char* err, size_t errn);

/* temp id 读取 (MATMUL 写 crouton16 面 / RMSNORM 写 f16 面) */
uint8_t*  wt_exec_temp(uint32_t id);
uint32_t  wt_exec_temp_bytes(uint32_t id);

/* 收尾: wtcache_close + temps 释放。任何路径退出前必须调。 */
void wt_exec_shutdown(void);

/* ---- V2.3 U16: 分段执行 + 统计 ----
 * run_range 只执行 ops[first, first+count): 整步下发 vs 逐算子下发共用
 * 同一执行体 (fused=run 一次; split=按 op 逐段), 输出必须恒等。 */
struct wt_exec_stats {
    uint32_t ops, nop, matmul, rmsnorm, silu, pin, pin_skipped;
    uint32_t im2col, conv2d, add, spill, fill, transpose;  /* GEHTP 阶段9 */
};
int  wt_exec_run_range(const struct wt_blob* b, uint32_t first, uint32_t count,
                       uint32_t* engine_m, int64_t* op_us, char* err, size_t errn);
void wt_exec_get_stats(struct wt_exec_stats* st);

/* GEHTP 阶段9 (Level 1 输入注入): 与 wt_exec_run 同语义, 但
 *   in_ptr  : 外部输入缓冲 (addr==WT_SLOT_EXT_IN 的 slot 从此读)
 *   out_ptr : 输出缓冲 (执行完后把 temps[out_temp] 拷到这里)
 *   out_temp: 输出 temp id (由 manifest 记录)
 * in_ptr/out_ptr 可 NULL (对应方向不注入/不回传)。 */
int  wt_exec_run_io(const struct wt_blob* b, const void* in_ptr, void* out_ptr,
                    uint32_t out_temp,
                    uint32_t* engine_m, int64_t* op_us, char* err, size_t errn);

/* W3 解析报告 (源: wt_w3.c)。emit 逐行收到 JSON 行; host wt_inspect 与
 * 设备输出共用此函数, 行逐字节一致。 */
void wt_w3_report(const char* blob_name, const uint8_t* buf, size_t size,
                  const struct wt_blob* w,
                  void (*emit)(const char* line, void* ud), void* ud);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V22_OPLIST_EXEC_H */
