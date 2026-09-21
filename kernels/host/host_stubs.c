/* host_stubs.c — wt_host_exec 的 QURT/HAP/dc 桩 (标量 blob 路径不触达) */
#include <stdint.h>
#include <stdlib.h>

/* qurt_mem_cache_clean: host 无 DSP cache, 空操作 */
void qurt_mem_cache_clean(uint64_t addr, uint32_t size, int op, int type) {
    (void)addr; (void)size; (void)op; (void)type;
}

/* wtcache: VTCM 驻留路径 (第7步阶段二) host 假池 —— 数值等价: VTCM 在 host
 * 上就是一块 malloc 内存, 懒初始化时 wtcache_open 给 8MB 假 VTCM
 * (TEMPOFF 槽 vtcm_cap ≤ 8MB 时够; 更大 blob 需调 VTCM_FAKE_MB 环境变量)。
 * 假池基址+布局: vb=假池, vs=假池大小, pb=假池 (bump 区同源), pc=0。 */
#define WTC_HOST_OK 0
struct wtcache_ctx { uint8_t* fake; uint32_t fake_sz; };
int  wtcache_open(struct wtcache_ctx** wc, uint32_t cap) {
    (void)cap;
    static struct wtcache_ctx ctx;
    const char* e = getenv("VTCM_FAKE_MB");
    uint32_t mb = e ? (uint32_t)strtoul(e, NULL, 10) : 16u;
    ctx.fake_sz = mb * 1024u * 1024u;
    ctx.fake = malloc(ctx.fake_sz);
    *wc = ctx.fake ? &ctx : NULL;
    return ctx.fake ? 0 : -1;
}
void wtcache_layout(struct wtcache_ctx* wc, void** vb, uint32_t* vs, void** pb, uint32_t* pc) {
    if (vb) *vb = wc ? wc->fake : NULL;
    if (vs) *vs = wc ? wc->fake_sz : 0;
    if (pb) *pb = wc ? wc->fake : NULL;
    if (pc) *pc = 0;
}
void wtcache_close(struct wtcache_ctx* wc) { if (wc) { free(wc->fake); wc->fake = NULL; } }

/* ---- host 功能化桩: W4A16 标量参考 (与设备 dc_parts 同契约) ---- */
#include <math.h>
#include "dc_parts.h"

struct h_arena { uint8_t* base; uint32_t size; uint32_t used; };
void dc_arena_init(struct dc_arena* a, uint8_t* base, uint32_t size) {
    struct h_arena* h = (struct h_arena*)a;
    h->base = base; h->size = size; h->used = 0;
    return (base && size) ? 0 : -1;
}
uint8_t* dc_arena_alloc(struct dc_arena* a, uint32_t bytes, uint32_t align) {
    struct h_arena* h = (struct h_arena*)a;
    uint32_t off = (h->used + align - 1) & ~(align - 1);
    if (off + bytes > h->size) return NULL;
    h->used = off + bytes;
    return h->base + off;
}
struct h_dma { uint8_t* src; uint8_t* dst; uint32_t bytes; };
int  dc_dma_init(struct dc_dma* d, uint8_t* src, uint8_t* dst, uint32_t bytes, dc_mutex_t* mu) {
    (void)mu; struct h_dma* h = (struct h_dma*)d;
    h->src = src; h->dst = dst; h->bytes = bytes;
    return (src && dst) ? 0 : -1;
}
int  dc_dma_once(struct dc_dma* d) {
    struct h_dma* h = (struct h_dma*)d;
    memcpy(h->dst, h->src, h->bytes);
    return 0;
}
void dc_dma_destroy(struct dc_dma* d) { (void)d; }
void dc_dma_clean_src(struct dc_dma* d) { (void)d; }
void dc_mutex_init(dc_mutex_t* mu) { (void)mu; }
void cpu_to_vtcm(uint8_t* dst, const uint8_t* src, uint32_t bytes) { memcpy(dst, src, bytes); }

/* host 标量 W4A16: carve 与设备同布局 (wt 区 = kernel 格式 K*N/2,
 * 设备 crouton 契约的差异在设备 cut 解决) */
int  dc_w4_carve(struct dc_w4* e, struct dc_arena* a, uint32_t M, uint32_t K, uint32_t N,
                 const uint8_t* atbl, const uint8_t* otbl) {
    memset(e, 0, sizeof(*e));
    e->m = M; e->k = K; e->n = N;
    e->act  = dc_arena_alloc(a, M * K * 2, 2048);
    e->out  = dc_arena_alloc(a, M * N * 2, 2048);
    e->wt   = dc_arena_alloc(a, K * N / 2, 2048);  /* kernel 格式 (闭包 k4-lohi) */
    e->bias = dc_arena_alloc(a, (N / 32) * 512, 2048);
    e->atbl = dc_arena_alloc(a, 8 * (K / 32) * 4, 128);
    e->otbl = dc_arena_alloc(a, 8 * (N / 32) * 4, 128);
    e->mask = dc_arena_alloc(a, 32, 128);
    e->extra = dc_arena_alloc(a, 16, 128);
    e->atbl_ddr = atbl;
    e->otbl_ddr = otbl;
    if (!e->act || !e->out || !e->wt || !e->bias || !e->atbl || !e->otbl ||
        !e->mask || !e->extra) return 0xD300;
    return 0;
}

/* kernel 格式权重解码 + f32 GEMM (host 标量参考; 纯数学语义 — 设备
 * 固定域 ÷7/cvt 细节由设备 cut 的黄金对拍承担, 见任务书 ④)。
 * 布局 (闭包 pack_w4_kblock32_nmajor_k4_lohi):
 *   kb(外) × N32 × kg(4) × n × kr(4) lohi 字节。
 * 存储字节 = ((w+8)&0xF | ...) ^ 0x88 = w 的 4-bit 补码本身
 * (闭包: nib=(w+8)&0xF 再整区 XOR 0x88 — w≥0: w+8^8=w; w<0: w+16)。
 * 解码 = 直接补码 (v≥8 → v-16), 不要再 XOR!
 * 列 scale f16 在 e->scale_ddr; out[m][n] = Σ_k act[m][k]·wq[k][n]·S[n]/7
 * (折叠 bias 是设备 acc 定点修正项, host 纯数学不消费) */
int  dc_w4_invoke(struct dc_w4* e) {
    uint32_t M = e->m, K = e->k, N = e->n;
    int8_t* wq = (int8_t*)malloc(sizeof(int8_t) * K * N);
    float* S = (float*)malloc(sizeof(float) * N);
    if (!wq || !S) { free(wq); free(S); return -1; }
    size_t o = 0;
    for (uint32_t kb = 0; kb < K / 32; kb++)
        for (uint32_t n_base = 0; n_base < N; n_base += 32)
            for (uint32_t kg = 0; kg < 4; kg++) {
                uint32_t k_base = kb * 32 + kg * 8;
                for (uint32_t n = n_base; n < n_base + 32; n++)
                    for (uint32_t kr = 0; kr < 4; kr++) {
                        uint8_t b = e->wt[o++];
                        int lo = (int)(b & 0xF), hi = (int)(b >> 4);
                        wq[(size_t)(k_base + kr) * N + n] = (int8_t)(lo >= 8 ? lo - 16 : lo);
                        wq[(size_t)(k_base + kr + 4) * N + n] = (int8_t)(hi >= 8 ? hi - 16 : hi);
                    }
            }
    for (uint32_t n = 0; n < N; n++) {
        uint16_t s16;
        memcpy(&s16, e->scale_ddr + (size_t)n * 2, 2);
        float sv = 0.0f;
        { /* f16 → f32 */
            uint32_t sign = (uint32_t)(s16 & 0x8000u) << 16;
            uint32_t ex = (s16 >> 10) & 0x1F, mn = s16 & 0x3FF;
            uint32_t u;
            if (ex == 0) u = sign;
            else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
            else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
            memcpy(&sv, &u, 4);
        }
        S[n] = sv;  /* scale 槽已含 /7 (max|col|/7); 真值 = wq·S, 勿再除 */
    }
    const uint16_t* act = (const uint16_t*)e->act;
    uint16_t* out = (uint16_t*)e->out;
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; k++) {
                float av;
                uint16_t ah = act[(size_t)m * K + k];
                { /* f16 → f32 */
                    uint32_t sign = (uint32_t)(ah & 0x8000u) << 16;
                    uint32_t ex = (ah >> 10) & 0x1F, mn = ah & 0x3FF;
                    uint32_t u;
                    if (ex == 0) u = sign;
                    else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
                    else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
                    memcpy(&av, &u, 4);
                }
                acc += (double)av * (double)wq[(size_t)k * N + n] * (double)S[n];
            }
            float r = (float)acc;
            uint16_t rh;
            { /* f32 → f16 RNE */
                uint32_t u;
                memcpy(&u, &r, 4);
                uint32_t sgn = (u >> 16) & 0x8000u;
                int32_t ex = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mn = u & 0x7FFFFFu;
                if (((u >> 23) & 0xFF) == 0xFF) { rh = (uint16_t)(sgn | 0x7C00u | (mn ? 0x200u : 0u)); out[(size_t)m * N + n] = rh; continue; }
                if (((u >> 23) & 0xFF) == 0) { rh = (uint16_t)sgn; out[(size_t)m * N + n] = rh; continue; }
                if (ex >= 31) { rh = (uint16_t)(sgn | 0x7C00u); out[(size_t)m * N + n] = rh; continue; }
                uint32_t half;
                if (ex <= 0) {
                    if (ex < -10) { rh = (uint16_t)sgn; out[(size_t)m * N + n] = rh; continue; }
                    mn |= 0x800000u;
                    int32_t sh = 14 - ex;
                    half = mn >> sh;
                    uint32_t rm = mn & ((1u << sh) - 1);
                    half += (rm > (1u << (sh - 1))) || (rm == (1u << (sh - 1)) && (half & 1));
                    rh = (uint16_t)(sgn | half);
                } else {
                    half = (mn >> 13) & 0x3FFu;
                    uint32_t rm = mn & 0x1FFFu;
                    half += (rm > 0x1000u) || (rm == 0x1000u && (half & 1));
                    if (half == 0x400u) { ex++; half = 0; }
                    if (ex >= 31) { rh = (uint16_t)(sgn | 0x7C00u); out[(size_t)m * N + n] = rh; continue; }
                    rh = (uint16_t)(sgn | ((uint32_t)ex << 10) | half);
                }
            }
            out[(size_t)m * N + n] = rh;
        }
    }
    free(wq);
    free(S);
    return 0;
}

/* dc_w4_run — host 实现: 与设备同语义但纯 f32 数学 (act 不量化, 直读 DDR;
 * 设备 a16 量化 + >>8 截断噪声由 ④ 容差门承担)。与设备 dc_w4_run 同签名:
 *   out_f16 = Σ_k act[m,k]·wq[k,n]·S[n]  (S=列 scale, 已含 /7) */
int dc_w4_run(struct dc_w4* e, const uint8_t* act_ddr, uint8_t* out_ddr,
              uint32_t m, uint32_t k, uint32_t n, const uint8_t* scale_ddr,
              uint32_t out_row_bytes, float wq_rms, float f_fixed) {
    (void)wq_rms; (void)f_fixed;  /* host 纯数学, 无固定域, f 不参与 */
    if (!e || !act_ddr || !out_ddr || !scale_ddr) return -1;
    if (m % 32 || k % 32 || n % 32) return -2;
    uint32_t M = m, K = k, N = n;
    int8_t* wq = (int8_t*)malloc(sizeof(int8_t) * K * N);
    float* S = (float*)malloc(sizeof(float) * N);
    if (!wq || !S) { free(wq); free(S); return -1; }
    size_t o = 0;
    for (uint32_t kb = 0; kb < K / 32; kb++)
        for (uint32_t n_base = 0; n_base < N; n_base += 32)
            for (uint32_t kg = 0; kg < 4; kg++) {
                uint32_t k_base = kb * 32 + kg * 8;
                for (uint32_t nn = n_base; nn < n_base + 32; nn++)
                    for (uint32_t kr = 0; kr < 4; kr++) {
                        uint8_t b = e->wt[o++];
                        int lo = (int)(b & 0xF), hi = (int)(b >> 4);
                        wq[(size_t)(k_base + kr) * N + nn] = (int8_t)(lo >= 8 ? lo - 16 : lo);
                        wq[(size_t)(k_base + kr + 4) * N + nn] = (int8_t)(hi >= 8 ? hi - 16 : hi);
                    }
            }
    for (uint32_t nn = 0; nn < N; nn++) {
        uint16_t s16;
        memcpy(&s16, scale_ddr + (size_t)nn * 2, 2);
        float sv = 0.0f;
        { /* f16 → f32 */
            uint32_t sign = (uint32_t)(s16 & 0x8000u) << 16;
            uint32_t ex = (s16 >> 10) & 0x1F, mn = s16 & 0x3FF;
            uint32_t u;
            if (ex == 0) u = sign;
            else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
            else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
            memcpy(&sv, &u, 4);
        }
        S[nn] = sv;  /* scale 槽已含 /7 */
    }
    const uint16_t* act = (const uint16_t*)act_ddr;
    for (uint32_t mm = 0; mm < M; mm++) {
        uint16_t* out = (uint16_t*)((uint8_t*)out_ddr + (size_t)mm * out_row_bytes);
        for (uint32_t nn = 0; nn < N; nn++) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < K; kk++) {
                float av;
                uint16_t ah = act[(size_t)mm * K + kk];
                { /* f16 → f32 */
                    uint32_t sign = (uint32_t)(ah & 0x8000u) << 16;
                    uint32_t ex = (ah >> 10) & 0x1F, mn = ah & 0x3FF;
                    uint32_t u;
                    if (ex == 0) u = sign;
                    else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
                    else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
                    memcpy(&av, &u, 4);
                }
                acc += (double)av * (double)wq[(size_t)kk * N + nn] * (double)S[nn];
            }
            float r = (float)acc;
            uint16_t rh;
            { /* f32 → f16 RNE */
                uint32_t u;
                memcpy(&u, &r, 4);
                uint32_t sgn = (u >> 16) & 0x8000u;
                int32_t ex = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mn = u & 0x7FFFFFu;
                if (((u >> 23) & 0xFF) == 0xFF) { out[nn] = (uint16_t)(sgn | 0x7C00u | (mn ? 0x200u : 0u)); continue; }
                if (((u >> 23) & 0xFF) == 0) { out[nn] = (uint16_t)sgn; continue; }
                if (ex >= 31) { out[nn] = (uint16_t)(sgn | 0x7C00u); continue; }
                uint32_t half;
                if (ex <= 0) {
                    if (ex < -10) { out[nn] = (uint16_t)sgn; continue; }
                    mn |= 0x800000u;
                    int32_t sh = 14 - ex;
                    half = mn >> sh;
                    uint32_t rm = mn & ((1u << sh) - 1);
                    half += (rm > (1u << (sh - 1))) || (rm == (1u << (sh - 1)) && (half & 1));
                    out[nn] = (uint16_t)(sgn | half);
                } else {
                    half = (mn >> 13) & 0x3FFu;
                    uint32_t rm = mn & 0x1FFFu;
                    half += (rm > 0x1000u) || (rm == 0x1000u && (half & 1));
                    if (half == 0x400u) { ex++; half = 0; }
                    if (ex >= 31) { out[nn] = (uint16_t)(sgn | 0x7C00u); continue; }
                    out[nn] = (uint16_t)(sgn | ((uint32_t)ex << 10) | half);
                }
            }
        }
    }
    free(wq);
    free(S);
    return 0;
}

/* HAP perf stub */
unsigned long long HAP_perf_get_time_us(void) { return 0; }

