/* dd_worker.c — V2.2 双域 step-list 执行器
 *
 * 忠实移植 submoudles/dualdomain_v81/dsp/dd_main.c 的 ddwork 主环
 * (D1-D7 设备闭合版), 参数化 asset_dir/形状/输出目录; cache 四铁律全内置。
 * 计时口径与 MODULE C 一致: wall_us 只含 [DMA→HMX invoke→UserDMA 读回发布槽]。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt.h>

#include "dd_worker.h"
#include "dc_parts.h"
#include "wtcache.h"

static int64_t now_us(void) { return HAP_perf_get_time_us(); }

static void cpu_to_vtcm(uint8_t* dst, const uint8_t* src, uint32_t bytes) {
    memcpy(dst, src, bytes);
    qurt_mem_cache_clean((qurt_addr_t)dst, bytes,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

static uint8_t* read_act(const struct dd_cfg* c, int k, uint32_t* bytes) {
    char p[512];
    if (c->n_act_files > 0)
        snprintf(p, sizeof(p), "%s/acts/dd_act_%d.raw", c->asset_dir,
                 k % (int)c->n_act_files);
    else
        snprintf(p, sizeof(p), "%s/act_variants/v%d.raw", c->asset_dir, k % 4);
    return dc_read_file(p, bytes);
}

static int dump_file(const char* out_dir, const char* name, const void* data,
                     size_t bytes) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", out_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(data, 1, bytes, f);
    fclose(f);
    return (w == bytes) ? 0 : -2;
}

int dd_run(const struct dd_cfg* c, const char* tag, int start, int len,
           const char* out_dir, struct dd_stats* st, char* err, size_t errn) {
    struct wtcache_ctx* wc = NULL;
    struct dc_arena arena;
    struct dc_w4 e;
    uint8_t *wt = NULL, *bias = NULL, *at = NULL, *ot = NULL;
    uint8_t *act_buf = NULL, *out_buf = NULL;
    struct dc_dma d_act, d_out;
    dc_mutex_t mu;
    int rc = -1;
    uint32_t b_wt = 0, b_bias = 0, b_at = 0, b_ot = 0;
    const uint32_t M = c->m, K = c->k, N = c->n;
    const uint32_t ACT_B = M * K * 2u, OUT_B = M * N * 2u, WT_B = K * N / 2u;
    const uint32_t CHUNK = c->chunk ? c->chunk : 8u;

#define DD_BAIL(fmt, ...) do { snprintf(err, errn, fmt, ##__VA_ARGS__); goto out; } while (0)

    if (!c->asset_dir || M % 256 || K % 32 || N % 32) {
        snprintf(err, errn, "bad shape %lu %lu %lu", (unsigned long)M, (unsigned long)K, (unsigned long)N);
        return -1;
    }
    memset(st, 0, sizeof(*st));
    memset(&d_act, 0, sizeof(d_act));
    memset(&d_out, 0, sizeof(d_out));

    if (wtcache_open(&wc, 4096) != WTC_OK) DD_BAIL("wtcache_open");
    {   /* arena 从 pin 区上限之后切 (与 dd_main sess_open 同布局) */
        void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        uint32_t off = (pc + 2047u) & ~2047u;
        dc_arena_init(&arena, (uint8_t*)vb + off, vs - off);
    }

    {
        char p[512];
        snprintf(p, sizeof(p), "%s/packed_weight.raw", c->asset_dir);
        wt = dc_read_file(p, &b_wt);
        snprintf(p, sizeof(p), "%s/folded_bias.raw", c->asset_dir);
        bias = dc_read_file(p, &b_bias);
        snprintf(p, sizeof(p), "%s/act_table.raw", c->asset_dir);
        at = dc_read_file(p, &b_at);
        snprintf(p, sizeof(p), "%s/out_table.raw", c->asset_dir);
        ot = dc_read_file(p, &b_ot);
    }
    if (!wt || !bias || !at || !ot || b_wt != WT_B) DD_BAIL("assets wt=%p %lu", wt, (unsigned long)b_wt);
    dc_clean_ddr(wt, b_wt); dc_clean_ddr(bias, b_bias);   /* CPU 写 → DMA 可见 */

    if (dc_w4_carve(&e, &arena, M, K, N, at, ot)) DD_BAIL("carve");
    cpu_to_vtcm(e.wt, wt, b_wt);
    cpu_to_vtcm(e.bias, bias, b_bias);

    dc_mutex_init(&mu);
    act_buf = memalign(128, (size_t)CHUNK * ACT_B);
    out_buf = memalign(128, (size_t)CHUNK * OUT_B);
    if (!act_buf || !out_buf) DD_BAIL("chunk alloc");

    /* warmup (首步 act, 不入墙钟) */
    {
        uint32_t ab = 0;
        uint8_t* a0 = read_act(c, start, &ab);
        if (!a0 || ab != ACT_B) { free(a0); DD_BAIL("act %d", start); }
        memcpy(act_buf, a0, ACT_B);
        free(a0);
    }
    if (dc_dma_init(&d_act, act_buf, e.act, ACT_B, &mu) ||
        dc_dma_init(&d_out, e.out, out_buf, OUT_B, &mu))
        DD_BAIL("dma_init");
    dc_clean_ddr(act_buf, ACT_B);
    if (dc_dma_once(&d_act) || dc_w4_invoke(&e)) DD_BAIL("warmup");

    int64_t t_e2e0 = now_us();
    int64_t t_first = 0;
    uint64_t wall = 0;
    for (int c0 = 0; c0 < len; c0 += (int)CHUNK) {
        int clen = (len - c0 < (int)CHUNK) ? (len - c0) : (int)CHUNK;
        for (int j = 0; j < clen; j++) {                 /* 块首预载 (不计时) */
            int k = start + c0 + j;
            if (j == 0) continue;                        /* 首 act 已在 warmup 载入 */
            uint32_t ab = 0;
            uint8_t* ak = read_act(c, k, &ab);
            if (!ak || ab != ACT_B) { free(ak); DD_BAIL("act %d", k); }
            memcpy(act_buf + (size_t)j * ACT_B, ak, ACT_B);
            free(ak);
        }
        dc_clean_ddr(act_buf, (uint32_t)clen * ACT_B);
        int64_t t0 = now_us();                           /* 计时环 */
        for (int j = 0; j < clen; j++) {
            int k = start + c0 + j;
            d_act.src = act_buf + (size_t)j * ACT_B;
            if (dc_dma_once(&d_act)) DD_BAIL("dma %d", k);
            if (dc_w4_invoke(&e)) DD_BAIL("invoke %d", k);
            d_out.dst = out_buf + (size_t)j * OUT_B;
            if (dc_dma_once(&d_out)) DD_BAIL("rb %d", k);
            /* DMA 写落 DDR, INVALIDATE 丢驻留旧行再读 (T4c 教训) */
            qurt_mem_cache_clean((qurt_addr_t)(out_buf + (size_t)j * OUT_B), OUT_B,
                                 QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            if (c0 == 0 && j == 0) t_first = now_us() - t0;
        }
        wall += (uint64_t)(now_us() - t0);
        if (c->dump) {
            for (int j = 0; j < clen; j++) {
                int k = start + c0 + j;
                char name[64];
                snprintf(name, sizeof(name), "dd_%s_step%d.raw", tag, k);
                if (dump_file(out_dir, name, out_buf + (size_t)j * OUT_B, OUT_B))
                    DD_BAIL("dump %d", k);
            }
        }
    }
    st->wall_us = wall;
    st->e2e_us = (uint64_t)(now_us() - t_e2e0);
    st->first_step_us = t_first;
    st->steps = (uint32_t)len;
    st->per_step_us = len ? (double)wall / len : 0.0;
    rc = 0;

out:
    if (d_act.desc || d_out.desc) { dc_dma_destroy(&d_act); dc_dma_destroy(&d_out); }
    free(act_buf); free(out_buf);
    free(wt); free(bias); free(at); free(ot);
    if (wc) wtcache_close(wc);                            /* 铁律④: 任何路径必关 */
    if (rc) FARF(ALWAYS, "dd_run[%s] FAIL: %s", tag, err);
    return rc;
#undef DD_BAIL
}
