/* pack_gemv_ab.cc — 50_q6b_gemv_ab 三臂单 op blob 组包器 (q_proj 形状)
 *
 * 裁决问题: 0.6B decode 253 个 M=1 GEMV 全在 HVX ~2GB/s (op_ts 91.5% wall),
 * "VTCM/tiling 规划" (= 编译期 crouton 布局 + HMX + 线性 DMA 驻场) 是否是解。
 * 三臂同数据同形状 (K=1024, N=2048, W=4MB f16), 仅 op 形态不同:
 *   A: blob_gemv_m1.wtop    M=1  flags=2   → 现状: M<32 → HVX t1 gemv
 *   B: blob_gemv_m32.wtop   M=32 flags=2   → HMX + 运行时标量散 pack
 *   C: blob_gemv_m32pp.wtop M=32 flags=10  → HMX + 预打包线性 DMA (bit3 w_pp)
 *
 * 取值确定性整数 (∈[-18,18]) → f16/f32/FP37/HMX-fp32-acc 各侧全部精确,
 * 跨引擎 byte-exact 可比 (同 pack_matmul_split 契约)。
 * C 臂权重 = crouton 镜像布局, 与 exec_matmul_f16_hmx 散 pack 目标序逐位同:
 *   tile = wgt_cr[(c*n_dot+kt)*1024]; tile[hmx_crouton_pos(k,n)] = w_t1[(c*32+n)*K + kt*32+k]
 *   hmx_crouton_pos(r,c) = (r/2)*64 + 2*c + (r&1)   (include/hmx_crouton.h)
 *
 * 用法: ./pack_gemv_ab --out <dir>
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

/* IEEE 754 binary16 (RNE), 与 oplist_exec.c f32_to_f16 同算法 */
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

static std::string f16_bytes(const std::vector<uint16_t>& h) {
    std::string s;
    s.reserve(h.size() * 2);
    for (uint16_t v : h) { s.push_back((char)(v & 0xff)); s.push_back((char)(v >> 8)); }
    return s;
}

static std::vector<uint8_t> build_blob(const std::string& a, const std::string& w,
                                       const std::vector<std::pair<uint16_t, std::vector<uint32_t>>>& ops) {
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), {'W', 'T', 'O', 'P'});
    wr_u16(blob, WT_BLOB_VER);
    wr_u16(blob, WT_ENDIAN_CHK);
    wr_u32(blob, 2);                  /* 2 slots */
    wr_u32(blob, (uint32_t)ops.size());
    std::vector<uint32_t> lens = {(uint32_t)a.size(), (uint32_t)w.size()};
    uint32_t off0 = 0, off1 = ((uint32_t)a.size() + 127u) & ~127u;
    wr_u32(blob, lens[0]); wr_u32(blob, 1); wr_u32(blob, off0); wr_u32(blob, 0);
    wr_u32(blob, lens[1]); wr_u32(blob, 1); wr_u32(blob, off1); wr_u32(blob, 0);
    for (const auto& op : ops) {
        wr_u16(blob, op.first);
        wr_u16(blob, (uint16_t)op.second.size());
        for (uint32_t v : op.second) wr_u32(blob, v);
    }
    while (blob.size() % 128) blob.push_back(0);
    uint32_t weight_off = (uint32_t)blob.size();
    while (blob.size() < weight_off + off0) blob.push_back(0);
    blob.insert(blob.end(), a.begin(), a.end());
    while (blob.size() < weight_off + off1) blob.push_back(0);
    blob.insert(blob.end(), w.begin(), w.end());
    return blob;
}

static unsigned cr_pos(unsigned row, unsigned col) {
    return (row / 2u) * 64u + 2u * col + (row & 1u);
}

int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    if (out.empty()) { fprintf(stderr, "usage: %s --out <dir>\n", argv[0]); return 1; }

    const uint32_t K = 1024, N = 2048;
    /* 激活: 平铺索引取值 (A 的 [0,1024) 与 M32 臂行 0 逐位同) */
    std::vector<uint16_t> a1(K), a32(32u * K);
    for (uint32_t i = 0; i < K; i++) a1[i] = f32_to_f16((float)((int)(i % 29) - 14));
    std::memcpy(a32.data(), a1.data(), (size_t)K * 2);
    for (size_t i = K; i < a32.size(); i++) a32[i] = 0;  /* pad 行 1..31 = 0 */
    /* 权重 [N,K] 行主 (t1), 确定性整数 */
    std::vector<uint16_t> w((size_t)N * K);
    for (size_t i = 0; i < w.size(); i++) w[i] = f32_to_f16((float)((int)(i % 37) - 18));
    /* C 臂: 同权重 crouton 镜像 (c=oc 块, kt=k 块, tile 内 pair-interleave) */
    const uint32_t n_dot = K / 32u, n_col = N / 32u;
    std::vector<uint16_t> wcr((size_t)n_col * n_dot * 1024u);
    for (uint32_t c = 0; c < n_col; c++)
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            uint16_t* tile = wcr.data() + ((size_t)c * n_dot + kt) * 1024u;
            for (uint32_t k = 0; k < 32u; k++)
                for (uint32_t n = 0; n < 32u; n++)
                    tile[cr_pos(k, n)] = w[(size_t)(c * 32u + n) * K + (kt * 32u + k)];
        }

    auto ab1 = f16_bytes(a1), ab32 = f16_bytes(a32);
    auto wb = f16_bytes(w), wcb = f16_bytes(wcr);

    auto write = [&](const char* name, const std::vector<uint8_t>& b) {
        std::ofstream f(out + "/" + name, std::ios::binary);
        f.write((const char*)b.data(), (std::streamsize)b.size());
    };
    write("blob_gemv_m1.wtop", build_blob(ab1, wb, {
        {OP_MATMUL_F16, {0x8000u | 0u, 0x8000u | 1u, 0, 1u, K, N, 2u}},
    }));
    write("blob_gemv_m32.wtop", build_blob(ab32, wb, {
        {OP_MATMUL_F16, {0x8000u | 0u, 0x8000u | 1u, 0, 32u, K, N, 2u}},
    }));
    write("blob_gemv_m32pp.wtop", build_blob(ab32, wcb, {
        {OP_MATMUL_F16, {0x8000u | 0u, 0x8000u | 1u, 0, 32u, K, N, 2u | 8u}},
    }));

    char sha[65];
    for (const char* nm : {"blob_gemv_m1.wtop", "blob_gemv_m32.wtop", "blob_gemv_m32pp.wtop"}) {
        std::ifstream f(out + "/" + nm, std::ios::binary);
        std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        printf("[pack] %s: %zu bytes sha=%s\n", nm, b.size(),
               wt_sha256_hex(b.data(), b.size(), sha));
    }
    printf("[pack] K=%u N=%u (W=%u B f16; pp 臂 crouton 镜像 %u tiles)\n",
           K, N, N * K * 2u, n_col * n_dot);
    return 0;
}
