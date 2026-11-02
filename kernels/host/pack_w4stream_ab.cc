/* pack_w4stream_ab.cc — 018 W4TS 两臂单 op blob 组包器 (ver=3 + annot)
 * =====================================================================
 * 裁决问题 (specs/018-w4-decode-tile-stream G1 生死门):
 *   Q4_0 640B tile 槽 + M=1 decode 形态, 「DMA 菊花链流式 + q8 激活 +
 *   rmpy 行并行 matvec」(flags bit5, exec_matmul_w4_stream) vs 基线
 *   (flags=2, exec_matmul_q4_0_gemv, HVX l2fetch 散读 DDR) —
 *   同确定性权重/激活, 输出 cos 门 + 实测 GB/s。
 *
 * 两臂 (同数据同形状, 仅 flags 差 bit5):
 *   A: flags=2      → exec_matmul_q4_0_gemv (f16 激活, HVX 散读)
 *   B: flags=2|0x20 → exec_matmul_w4_stream (q8 激活, DMA 流式)
 * 形状两档: K=1024/N=2048 (64 ct) + K=1024/N=16384 (512 ct 长链)。
 *
 * blob 契约 (ver=3): 24B slot 记录 + op 表 + annot footer (n_ops×4B,
 * 值 = wt_annot_pack(HVX=1,fb=1,0)|wt_annot_set_q4 = 0x105) + 128B 对齐
 * weight 区。Q4 tile 布局 = repack_q4_0_tiles 契约 (640B/tile, A 序
 * nibble: byte[i*32+r] lo=值(2i) hi=值(2i+1); scale f16 @ tile+512+2r;
 * pad 64B 全零)。确定性生成器与例 53 main.c 逐位一致。
 *
 * 用法: ./pack_w4stream_ab --out <dir>
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

/* IEEE 754 binary16 (RNE), 与设备同算法 */
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

/* 确定性激活: a[k] = ((k%29)-14) * 0.25 ∈ [-3.5, 3.5] (f16 精确可表) */
static void gen_act(uint16_t* a, uint32_t K) {
    for (uint32_t k = 0; k < K; k++)
        a[k] = f32_to_f16((float)((int)(k % 29u) - 14) * 0.25f);
}

/* 确定性 Q4 tile 权重: tile(ct,kt) @ (ct*n_dot+kt)*640。
 * quants byte[i*32+r] = hash(ct,kt,i,r) (A 序任意字节合法);
 * scale f16 @ tile+512+2r = 0.02 + ((ct+r)%7)*0.005; pad 64B 零。 */
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

static std::vector<uint8_t> build_blob(const uint16_t* a, uint32_t K,
                                       const uint8_t* w, uint32_t w_bytes,
                                       uint32_t N, uint32_t flags) {
    const uint32_t a_bytes = K * 2u;
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), {'W', 'T', 'O', 'P'});
    wr_u16(blob, 3u);                 /* ver=3 (annot footer) */
    wr_u16(blob, WT_ENDIAN_CHK);
    wr_u32(blob, 2u);                 /* 2 slots */
    wr_u32(blob, 1u);                 /* 1 op */
    /* 24B slot 记录 ×2: {len, count, off_lo, off_hi, addr, rsv} */
    uint32_t off1 = (a_bytes + 127u) & ~127u;
    wr_u32(blob, a_bytes); wr_u32(blob, 1u); wr_u32(blob, 0u);
    wr_u32(blob, 0u); wr_u32(blob, 0u); wr_u32(blob, 0u);
    wr_u32(blob, w_bytes); wr_u32(blob, 1u); wr_u32(blob, off1);
    wr_u32(blob, 0u); wr_u32(blob, 0u); wr_u32(blob, 0u);
    /* op: OP_MATMUL_F16 [a,w,out,M,K,N,flags] */
    wr_u16(blob, 24u); wr_u16(blob, 7u);
    wr_u32(blob, 0x8000u | 0u); wr_u32(blob, 0x8000u | 1u); wr_u32(blob, 0u);
    wr_u32(blob, 1u); wr_u32(blob, K); wr_u32(blob, N); wr_u32(blob, flags);
    /* annot footer: wt_annot_pack(HVX=1,fb=1,0) | set_q4 = 0x105 */
    wr_u32(blob, 0x105u);
    while (blob.size() % 128u) blob.push_back(0);
    /* weight 区 */
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

    struct Shape { uint32_t K, N; const char* tag; };
    const Shape shapes[2] = {
        {1024u, 2048u, "k1024n2048"},
        {1024u, 16384u, "k1024n16384"},
    };
    char sha[65];
    for (const auto& s : shapes) {
        std::vector<uint16_t> a(s.K);
        gen_act(a.data(), s.K);
        uint32_t w_bytes = (s.K / 32u) * (s.N / 32u) * 640u;
        std::vector<uint8_t> w(w_bytes);
        gen_w_q4(w.data(), s.K, s.N);
        for (int arm = 0; arm < 2; arm++) {
            uint32_t flags = 2u | (arm ? 0x20u : 0u);
            std::vector<uint8_t> blob = build_blob(a.data(), s.K, w.data(), w_bytes, s.N, flags);
            char name[160];
            snprintf(name, sizeof(name), "%s/blob_w4s_%s_%s.wtop",
                     out.c_str(), s.tag, arm ? "stream" : "base");
            std::ofstream f(name, std::ios::binary);
            f.write((const char*)blob.data(), (std::streamsize)blob.size());
            f.close();
            printf("[pack] %s: %zu bytes sha=%s\n", name, blob.size(),
                   wt_sha256_hex(blob.data(), blob.size(), sha));
        }
        printf("[pack] %s: K=%u N=%u W=%uB (q4 tile, %u ct)\n",
               s.tag, s.K, s.N, w_bytes, s.N / 32u);
    }
    return 0;
}
