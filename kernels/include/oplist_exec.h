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

/* 0.8B 全模型 blob temp id 实测至 1403+(输出 temp 1401), 256 会死在首个
 * temp≥256 的 op (broadcast ref fail, 2026-09-22 v3 复跑实锤) → 4096 对齐 WT_MAX_SLOTS */
#define WT_EXEC_MAX_TEMPS 4096

/* 顺序执行 b->ops。engine_m 回填引擎形状 (0=无 MATMUL op)。
 * op_us (可 NULL) 回填每 op 耗时表数组 (长度必须 >= n_ops)。
 * err/errn (可 NULL) 出错时写人类可读原因。
 * 返回 0 = 全部 op 成功; 负 = 失败 (err 有内容)。 */
int wt_exec_run(const struct wt_blob* b, uint32_t* engine_m,
                int64_t* op_us, char* err, size_t errn);

/* temp id 读取 (MATMUL 写 crouton16 面 / RMSNORM 写 f16 面) */
uint8_t*  wt_exec_temp(uint32_t id);
uint32_t  wt_exec_temp_bytes(uint32_t id);
/* 每 temp 最后写入字节数 (temp_bytes=分配大小, 复用只扩不缩可能大于实际) */
uint32_t  wt_exec_temp_last_bytes(uint32_t id);

/* 收尾: wtcache_close + temps 释放。任何路径退出前必须调。 */
void wt_exec_shutdown(void);

/* 第7步阶段一插槽: 编译期静态 temp 池 + 偏移表。
 * base=NULL && offsets=NULL 恢复运行时 bump 路径(默认)。
 * base!=NULL: 池由调用方提供, static_cap = 表内静态区大小,
 *   表内 temp 地址 = base + offsets[id](哨兵 0xFFFFFFFF = 表外,
 *   回落在 [static_cap, total_cap) bump; 偏移正确性由编译期重叠
 *   检查器保证, 设备只做边界检查)。 */
void wt_exec_pool_init(uint8_t* base, uint32_t total_cap, uint32_t static_cap,
                       const uint32_t* offsets);

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

/* 路线B: 外部权重区注入(run/run_io 前调一次; 权重 slot addr==WT_SLOT_EXT_WGT
 * 时从此基址+slot.offset 读)。NULL=禁用。与 run_io 解耦(权重常驻)。 */
void wt_exec_set_ext_weights(const void* wgt_ptr);
/* ---- PROF 战役 W-P2: 逐 op 时间戳对 + optrace 开关 ----
 * wt_op_ts: 每 op 的 {start_us, dur_us, dma_us, dma_bytes, engine}, start_us 相对
 * run_range 入口 (u32 us, 约 71min 回绕, 单 run 不会超)。注册后 run/run_range 自动
 * 回填; 索引 = 全局 op 序号, 缓冲须 >= n_ops (run_range 分段调用时按绝对序号写)。
 * 不注册 = 零开销旧行为。 */
enum wt_engine { WT_ENG_SCALAR = 0, WT_ENG_HVX = 1, WT_ENG_HMX = 2 };
struct wt_op_ts {
    uint32_t start_us;    /* 相对 run_range 入口 */
    uint32_t dur_us;      /* 本 op 总耗时 */
    uint32_t dma_us;      /* 本 op 内 UserDMA 累计耗时 */
    uint32_t dma_bytes;   /* 本 op 内 UserDMA 累计搬运字节 */
    uint32_t engine;      /* WT_ENG_* 主计算引擎 */
};
void wt_exec_set_ts(struct wt_op_ts* buf, uint32_t cap);

/* optrace 逐 op 行开关。默认开(=旧行为, 存量 runner 零回归); 性能测量前
 * 显式 set_trace(0) 关 = 无落盘污染。只控 run_range 内 per-op pre/post
 * fprintf; run 级 rtrace 里程碑常开 (M4.2 崩溃取证命脉)。句柄铁律不动:
 * 仍只有 g_rtrace 一个 FILE*, 开关只控 fopen 与否, 绝不新建路径。 */
void wt_exec_set_trace(int on);

/* 逐 op 张量 dump 钩子 (精度排查): spec = "opidx:temp,opidx:temp,...", 执行到
 * op idx (run_range 局部序号, 与 optrace 行号一致) 且 rc==0 后, 把 temp 按
 * g_last_bytes 落盘 hook_<idx>_<temp>.f16.raw (设备 /data/local/tmp/hrt/gehtp/,
 * host /tmp/)。NULL/空串 = 关 (零开销)。host 未 set 时回退 GEHTP_HOOK env
 * (旧 hostsim 行为保持); 设备经 runner job.txt "hook" 键下发。run_io 前调一次。 */
void wt_exec_set_hook(const char* spec);

/* W3 解析报告 (源: wt_w3.c)。emit 逐行收到 JSON 行; host wt_inspect 与
 * 设备输出共用此函数, 行逐字节一致。 */
void wt_w3_report(const char* blob_name, const uint8_t* buf, size_t size,
                  const struct wt_blob* w,
                  void (*emit)(const char* line, void* ud), void* ud);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V22_OPLIST_EXEC_H */
