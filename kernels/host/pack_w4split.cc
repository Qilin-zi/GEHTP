/* pack_w4split.cc — 018 双域 N 维分片 W4TS 单 op blob 组包器 (ver=3 + annot)
 * =====================================================================
 * 裁决问题 (specs/018 双域加速): W4TS (exec_matmul_w4_stream) 是 MLP/latency-
 * bound (op_ts 实测 lm_head 3.6GB/s / 算力 0.09%), 双核 N 维分片应 ~2×。
 * 复用 executor 已原生的 OP_MATMUL_F16_SPLIT (opcode 35) N 列分片 —
 * 每 chunk 只算列区间 [n0, n0+N_chunk), 写共享 out temp 的列子区间。
 * 发射 3 个 blob (同权重同激活, 仅 n0/N_chunk 两标量不同):
 *   full: n0=0        N_chunk=N         (单域基线)
 *   c0  : n0=0        N_chunk=N/2       (dom3)
 *   c1  : n0=N/2      N_chunk=N/2       (dom4)
 *
 * blob 契约 (ver=3): 24B slot 记录 + op 表 + annot footer (n_ops×4B,
 * 值 = wt_annot_pack(HVX=1,fb=1,0)|wt_annot_set_q4 = 0x105) + 128B 对齐
 * weight 区。Q4 tile 布局 = repack_q4_0_tiles 契约 (640B/tile, A 序
 * nibble: byte[i*32+r] lo=值(2i) hi=值(2i+1); scale f16 @ tile+512+2r;
 * pad 64B 全零)。确定性生成器与 pack_w4stream_ab.cc 逐位一致。
 *
 * flags = bit1(t1) | bit5(w4ts) = 2 | 0x20 = 0x22。
 *
 * 用法: ./pack_w4split --out <dir>
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
#include "oplist_parse.h"
#include "wt_sha256.h"
}

static void wr_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(x & 0xff); v.push_back(x >> 8);
}
static void wr_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
    v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> (shift + 1);
        uint32_t rest = man & ((1u << shift) - 1u);
        if ((man >> shift) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}

static void gen_act(uint16_t* a, uint32_t K) {
    for (uint32_t k = 0; k < K; k++)
        a[k] = f32_to_f16((float)((int)(k % 29u) - 14) * 0.25f);
}

static void gen_w_q4(uint8_t* w, uint32_t K, uint32_t N) {
    uint32_t n_dot = K / 32u, n_ct = N / 32u;
    for (uint32_t ct = 0; ct < n_ct; ct++)
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            for (uint32_t i = 0; i < 16u; i++)
                for (uint32_t r = 0; r < 32u; r++)
                    tile[i * 32u + r] = (uint8_t)(ct * 31u + kt * 17u + i * 7u + r * 11u + i * r);
            for (uint32_t r = 0; r < 32u; r++) {
                uint16_t d = f32_to_f16(0.02f + (float)((ct + r) % 7u) * 0.005f);
                tile[512u + 2u * r] = (uint8_t)(d & 0xffu);
                tile[512u + 2u * r + 1u] = (uint8_t)(d >> 8);
            }
            for (uint32_t p = 512u + 64u; p < 640u; p++) tile[p] = 0;
        }
}

/* OP_MATMUL_F16_SPLIT (35): [a_ref,w_s,out_t,M,K,N_full,n0,N_chunk,flags] arity 9 */
static std::vector<uint8_t> build_blob(const uint16_t* a, uint32_t K,
                                       const uint8_t* w, uint32_t w_bytes,
                                       uint32_t N_full, uint32_t n0,
                                       uint32_t N_chunk, uint32_t flags) {
    const uint32_t a_bytes = K * 2u;
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), {'W', 'T', 'O', 'P'});
    wr_u16(blob, 3u);                 /* ver=3 (annot footer) */
    wr_u16(blob, WT_ENDIAN_CHK);
    wr_u32(blob, 2u);                 /* 2 slots */
    wr_u32(blob, 1u);                 /* 1 op */
    uint32_t off1 = (a_bytes + 127u) & ~127u;
    wr_u32(blob, a_bytes); wr_u32(blob, 1u); wr_u32(blob, 0u);
    wr_u32(blob, 0u); wr_u32(blob, 0u); wr_u32(blob, 0u);
    wr_u32(blob, w_bytes); wr_u32(blob, 1u); wr_u32(blob, off1);
    wr_u32(blob, 0u); wr_u32(blob, 0u); wr_u32(blob, 0u);
    /* op: OP_MATMUL_F16_SPLIT */
    wr_u16(blob, 35u); wr_u16(blob, 9u);
    wr_u32(blob, 0x8000u | 0u); wr_u32(blob, 0x8000u | 1u); wr_u32(blob, 0u);
    wr_u32(blob, 1u); wr_u32(blob, K);
    wr_u32(blob, N_full); wr_u32(blob, n0); wr_u32(blob, N_chunk); wr_u32(blob, flags);
    /* annot footer: HVX=1 fb=1 q4=1 */
    wr_u32(blob, 0x105u);
    while (blob.size() % 128u) blob.push_back(0);
    const uint8_t* ab = (const uint8_t*)a;
    blob.insert(blob.end(), ab, ab + a_bytes);
    while (blob.size() % 128u) blob.push_back(0);
    blob.insert(blob.end(), w, w + w_bytes);
    return blob;
}

int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    if (out.empty()) { fprintf(stderr, "usage: %s --out <dir>\n", argv[0]); return 1; }

    const uint32_t K = 1024u;
    const uint32_t Ns[2] = {16384u, 151936u};   /* 小形状(对拍) + lm_head 词表形状(计时) */
    const struct { const char* role; uint32_t n0; } roles[3] = {
        {"full", 0u}, {"c0", 0u}, {"c1", 0u},   /* n0 在循环里按 role 填 */
    };
    char sha[65];
    for (uint32_t N : Ns) {
        std::vector<uint16_t> a(K);
        gen_act(a.data(), K);
        uint32_t w_bytes = (K / 32u) * (N / 32u) * 640u;
        std::vector<uint8_t> w(w_bytes);
        gen_w_q4(w.data(), K, N);
        for (const auto& r : roles) {
            uint32_t n0 = 0u, chunk = N;
            if (!strcmp(r.role, "c1")) { n0 = N / 2u; chunk = N / 2u; }
            else if (!strcmp(r.role, "c0")) { chunk = N / 2u; }
            std::vector<uint8_t> blob = build_blob(a.data(), K, w.data(), w_bytes,
                                                   N, n0, chunk, 0x22u);
            char name[160];
            snprintf(name, sizeof(name), "%s/split_n%u_%s.wtop", out.c_str(), N, r.role);
            std::ofstream f(name, std::ios::binary);
            f.write((const char*)blob.data(), (std::streamsize)blob.size());
            f.close();
            printf("[pack] %s: %zu bytes sha=%s (n0=%u chunk=%u)\n", name, blob.size(),
                   wt_sha256_hex(blob.data(), blob.size(), sha), n0, chunk);
        }
    }
    return 0;
}
