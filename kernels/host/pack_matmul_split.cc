/* pack_matmul_split.cc — 002 US1 设备要害门 (T017) 的合成 blob 组包器
 *
 * 生成两个 f16 matmul blob, 数据/权重完全一致, 仅 op 形态不同:
 *   blob_fc_whole.wtop   1 × OP_MATMUL_F16     (M=32,K=1024,N=2048,flags=2 FC)
 *   blob_fc_chunk.wtop   2 × OP_MATMUL_F16_SPLIT (N_full=2048, n0=0/1024, Nc=1024)
 * 二者 temp0 必须逐字节一致 = 分块不破坏 HMX 逐字节恒等 (plan.md US1 验证门3)。
 *
 * slot 表:
 *   0 = input  f16 [32×1024]   (确定性整数取值, f16 精确)
 *   1 = weight f16 [2048×1024] (确定性整数取值, f16 精确)
 * 取值选整数 (∈[-18,18]) 保证 f16/f32/FP37 三侧累加全部精确, 任何 n0 错位
 * 都会改乘积流 → memcmp 可见。engine 侧由设备 runner 经 wt_exec_set_ts 断言
 * WT_ENG_HMX, 排除「双侧静默回落标量」的假阳性。
 *
 * 用法: ./pack_matmul_split --out <dir>
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

/* 组 blob: slots(0=input,1=weight) + ops */
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

int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    if (out.empty()) { fprintf(stderr, "usage: %s --out <dir>\n", argv[0]); return 1; }

    const uint32_t M = 32, K = 1024, N_full = 2048, Nc = 1024;
    /* 确定性整数取值 (f16 精确) */
    std::vector<uint16_t> a((size_t)M * K), w((size_t)N_full * K);
    for (size_t i = 0; i < a.size(); i++) a[i] = f32_to_f16((float)((int)(i % 29) - 14));
    for (size_t i = 0; i < w.size(); i++) w[i] = f32_to_f16((float)((int)(i % 37) - 18));
    std::string ab = f16_bytes(a), wb = f16_bytes(w);

    /* 整算版: OP_MATMUL_F16 [a_ref,w_s,out_t,M,K,N,flags] */
    std::vector<uint8_t> whole = build_blob(ab, wb, {
        {OP_MATMUL_F16, {0x8000u | 0u, 0x8000u | 1u, 0, M, K, N_full, 2u}},
    });
    /* 分块版: 2 × OP_MATMUL_F16_SPLIT [a_ref,w_s,out_t,M,K,N_full,n0,N_chunk,flags] */
    std::vector<uint8_t> chunk = build_blob(ab, wb, {
        {OP_MATMUL_F16_SPLIT, {0x8000u | 0u, 0x8000u | 1u, 0, M, K, N_full, 0, Nc, 2u}},
        {OP_MATMUL_F16_SPLIT, {0x8000u | 0u, 0x8000u | 1u, 0, M, K, N_full, Nc, Nc, 2u}},
    });

    auto write = [&](const std::string& name, const std::vector<uint8_t>& b) {
        std::ofstream f(out + "/" + name, std::ios::binary);
        f.write((const char*)b.data(), (std::streamsize)b.size());
    };
    write("blob_fc_whole.wtop", whole);
    write("blob_fc_chunk.wtop", chunk);
    { std::ofstream f(out + "/a.f16.raw", std::ios::binary); f.write(ab.data(), (std::streamsize)ab.size()); }
    { std::ofstream f(out + "/w.f16.raw", std::ios::binary); f.write(wb.data(), (std::streamsize)wb.size()); }

    char sha[65];
    printf("[pack] blob_fc_whole.wtop: %zu bytes sha=%s\n", whole.size(),
           wt_sha256_hex(whole.data(), whole.size(), sha));
    printf("[pack] blob_fc_chunk.wtop: %zu bytes sha=%s\n", chunk.size(),
           wt_sha256_hex(chunk.data(), chunk.size(), sha));
    printf("[pack] M=%u K=%u N_full=%u Nc=%u (2 chunks)\n", M, K, N_full, Nc);
    return 0;
}
