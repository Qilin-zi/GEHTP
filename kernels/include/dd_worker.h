/* dd_worker.h — V2.2 双域 step-list 执行器 (源: dualdomain_v81 MODULE C)
 *
 * 把 "N 步 W4A16 GEMM (权重共享, 每步独立 act)" 的分片执行封装成单元:
 * 一进程一段 [start, start+len)。双域 = 同一二进制在 dom3/dom4 各起一个进程,
 * host 侧并发拉起 (build_examples.sh 20 号用例), 模块内不做跨域通信。
 *
 * 切分等价性契约 (MODULE C D2 已设备闭合): 第 k 步输出只由 (k, 权重) 决定,
 * 与域/并发/次数无关 — serial[0..48) 与 halfA[0..24)+halfB[24..48) 逐步
 * byte-exact。业务侧切分策略: 对半切拿全部收益 (2.001×), 不均衡=大半边墙钟。
 *
 * cache 协议 (4 铁律全内置): 预载后 dc_clean_ddr / wtcache_open 已全 VTCM
 * FLUSH / 读回槽 INVALIDATE / 退出必 wtcache_close。
 */
#ifndef HVXHMX_V22_DD_WORKER_H
#define HVXHMX_V22_DD_WORKER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dd_cfg {
    const char* asset_dir;  /* packed_weight/folded_bias/act_table/out_table 所在目录 */
    uint32_t m, k, n;       /* GEMM 形状 (m 必须 256 倍数, HMX m_t=8 硬约束) */
    uint32_t chunk;         /* 每预载块步数 (0 → 默认 8) */
    uint32_t n_act_files;   /* >0: asset_dir/acts/dd_act_%d.raw 按 k%n 取;
                               =0: asset_dir/act_variants/v%d.raw 按 k%4 循环 */
    int dump;               /* 1 = 每步输出 dump 到 out_dir/dd_<tag>_step<k>.raw */
};

struct dd_stats {
    uint64_t wall_us;       /* 纯 compute 环 (DMA→invoke→读回), 文件 I/O 不入环 */
    uint64_t e2e_us;        /* 端到端 (含预载读) */
    int64_t  first_step_us; /* 首步 (冷效应参考) */
    uint32_t steps;
    double   per_step_us;
};

/* 执行 steps [start, start+len)。返回 0 或负错 (err/errn 可 NULL)。 */
int dd_run(const struct dd_cfg* c, const char* tag, int start, int len,
           const char* out_dir, struct dd_stats* st, char* err, size_t errn);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V22_DD_WORKER_H */
