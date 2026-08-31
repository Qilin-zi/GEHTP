/* gemm_dispatch.c — U17 MatMul 三路由决策 (见 include/gemm_dispatch.h) */
#include "gemm_dispatch.h"
#include "gdn_sm.h"
#include <qurt.h>
#include <stdlib.h>
#include <string.h>

int gemm_route_for(uint32_t m, uint32_t k, uint32_t n) {
    (void)k; (void)n;
    if (m >= 256u && (m % 256u) == 0u) return GR_W4A16;
    if (m < 32u) return GR_SMALLM;
    return GR_DENSE_F16;
}

const char* gemm_route_name(int r) {
    switch (r) {
    case GR_W4A16:    return "w4a16";
    case GR_SMALLM:   return "smallm_pad256";
    case GR_DENSE_F16: return "dense_f16";
    default:          return "?";
    }
}

void gemm_f16_dense(const int16_t* a, const int16_t* w, int16_t* c,
                    uint32_t m, uint32_t k, uint32_t n) {
    /* f16 位型经 gdn_sm 软转换 (库内同源, RNE) */
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < k; ki++)
                acc += gdn_f16_to_f32(a[(size_t)mi * k + ki]) *
                       gdn_f16_to_f32(w[(size_t)ki * n + ni]);
            c[(size_t)mi * n + ni] = gdn_f32_to_f16(acc);
        }
    }
}

void gemm_crouton_encode(const uint16_t* lin, uint16_t* surf,
                         uint32_t rows, uint32_t cols) {
    uint32_t n_m32 = rows / 32u, n_kt = cols / 32u, out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++)
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    uint16_t* p = surf + out;
                    for (uint32_t c = kt * 32; c < kt * 32 + 32; c++) {
                        p[0] = lin[(size_t)row0 * cols + c];
                        p[1] = lin[(size_t)(row0 + 1) * cols + c];
                        p += 2;
                    }
                    out += 64;
                }
}

void gemm_crouton_decode(const uint16_t* surf, uint16_t* lin,
                         uint32_t rows, uint32_t cols) {
    uint32_t n_m32 = rows / 32u, n_kt = cols / 32u, out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++)
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    const uint16_t* p = surf + out;
                    for (uint32_t c = kt * 32; c < kt * 32 + 32; c++) {
                        lin[(size_t)row0 * cols + c] = p[0];
                        lin[(size_t)(row0 + 1) * cols + c] = p[1];
                        p += 2;
                    }
                    out += 64;
                }
}

/* 解码 scratch: 进程级单例 — 设备实测热路径上每次 memalign/free 128KB
 * 引入 ~38ms 分配抖动, 单例消除。smallm/w4a16_m256 的引擎路径由调用方
 * 串行化 (同引擎 invoke 本身互斥), 共享安全。 */
static uint16_t* s_dec;   /* 出面暂存 256*n */
static uint16_t* s_lin;   /* 线性暂存 256*n */
static uint32_t   s_cap;  /* 字节 */

static int scratch_ensure(uint32_t n) {
    uint32_t need = 256u * n * 2u;
    if (s_cap >= need) return 0;
    free(s_dec); free(s_lin);
    s_dec = (uint16_t*)memalign(128, need);
    s_lin = (uint16_t*)memalign(128, need);
    if (!s_dec || !s_lin) {
        free(s_dec); free(s_lin);
        s_dec = s_lin = 0; s_cap = 0u;
        return -1;
    }
    s_cap = need;
    return 0;
}

int gemm_smallm_pad256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                       int16_t* out_lin) {
    if (!e || !act_lin || !out_lin || m == 0 || m > 256u) return -1;
    uint32_t k = e->k, n = e->n;
    uint32_t face = 256u * k * 2u;
    /* 线性 pad-256 面 → crouton 编码进引擎 act 面 */
    uint16_t* lin = (uint16_t*)calloc(256u * k, 2u);
    if (!lin) return -2;
    memcpy(lin, act_lin, (size_t)m * k * 2u);
    gemm_crouton_encode(lin, (uint16_t*)e->act, 256u, k);
    free(lin);
    qurt_mem_cache_clean((qurt_addr_t)e->act, face,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    if (dc_w4_invoke(e)) return -3;
    if (scratch_ensure(n)) return -4;
    dc_w4_read_out(e, s_dec);
    /* 不能原地解码: crouton 读流 (phase 外层) 与写行 (g 外层) 顺序不同,
     * 原地会在高位行读到未解码残片 (256 面实测 32767/65536 败) */
    gemm_crouton_decode(s_dec, s_lin, 256u, n);
    memcpy(out_lin, s_lin, (size_t)m * n * 2u);
    return 0;
}

int gemm_w4a16_m256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                    int16_t* out_lin) {
    if (!e || !act_lin || !out_lin || e->m != 256u || m == 0u || m % 256u)
        return -1;
    uint32_t k = e->k, n = e->n;
    if (scratch_ensure(n)) return -2;
    for (uint32_t b = 0, nb = m / 256u; b < nb; b++) {
        gemm_crouton_encode((const uint16_t*)act_lin + (size_t)b * 256u * k,
                            (uint16_t*)e->act, 256u, k);
        qurt_mem_cache_clean((qurt_addr_t)e->act, 256u * k * 2u,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (dc_w4_invoke(e)) return -3;
        dc_w4_read_out(e, s_dec);
        gemm_crouton_decode(s_dec,
                            (uint16_t*)out_lin + (size_t)b * 256u * n, 256u, n);
    }
    return 0;
}
