/* pack_oplist.cc — host 端 blob v1 组包器 (C++, 复用 vendored weight_pack)
 *
 * 用法:
 *   ./pack_oplist --t10 <t10 s2560 assets dir> --out <dir> --tag w4|w5
 * 产物:
 *   <out>/blob_<tag>.wtop      权重+oplist 单文件 blob
 *   <out>/rms_w.f16.raw        slot5 (rmsnorm 权重) 独立落盘 (host 参考用)
 *   <out>/manifest_<tag>.json  W3 逐字段对拍清单 (设备解析结果必须与它一致)
 *
 * slot 表 (固定, 下标即 slot id):
 *   0 act_surface  1 packed_weight  2 folded_bias  3 act_table  4 out_table
 *   5 rms_w (f16×2560, 本工具确定性生成)
 *   6 q8_0 crouton (vendored weight_pack 打包, 随车运输证明)
 * op 表:
 *   w4: NOP; PIN(1); PIN(2); MATMUL(0,1,temp0,256,2560,2560)
 *   w5: 上者 + RMSNORM(temp0,5,temp1,2560)
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "weight_pack.h"
extern "C" {
#include "oplist_parse.h"
#include "wt_sha256.h"
}

using qnn::htp::block_q8_0;
using qnn::htp::CrateEntry;

static std::string read_file(const std::string& p, uint32_t* bytes) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    *bytes = (uint32_t)s.size();
    return s;
}

static void wr_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(x & 0xff); v.push_back(x >> 8);
}
static void wr_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
    v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
}

int main(int argc, char** argv) {
    std::string t10, out, tag;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--t10") && i + 1 < argc) t10 = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--tag") && i + 1 < argc) tag = argv[++i];
    }
    if (t10.empty() || out.empty() || (tag != "w4" && tag != "w5")) {
        fprintf(stderr, "usage: %s --t10 <dir> --out <dir> --tag w4|w5\n", argv[0]);
        return 1;
    }

    struct SlotDef { std::string name; uint32_t len; std::string bytes; };
    std::vector<SlotDef> slots;
    const char* faces[5] = {"act_surface.raw", "packed_weight.raw", "folded_bias.raw",
                            "act_table.raw", "out_table.raw"};
    for (int i = 0; i < 5; i++) {
        uint32_t n = 0;
        std::string b = read_file(t10 + "/" + faces[i], &n);
        slots.push_back({faces[i], n, b});
    }
    /* slot5: rms_w f16×2560 */
    {
        std::string b;
        for (uint32_t i = 0; i < 2560; i++) {
            float v = 1.0f + 0.3f * sinf(0.37f * (float)i);
            uint16_t h = qnn::htp::f32_to_f16_bits(v);
            b.push_back((char)(h & 0xff));
            b.push_back((char)(h >> 8));
        }
        std::ofstream f(out + "/rms_w.f16.raw", std::ios::binary);
        f.write(b.data(), (std::streamsize)b.size());
        slots.push_back({"rms_w.f16", (uint32_t)b.size(), b});
    }
    /* slot6: vendored q8_0 crouton (随机块 seed 7, ne0=ne1=64 → 4 tile) */
    {
        const uint32_t ne0 = 64, ne1 = 64;
        std::vector<block_q8_0> src((size_t)ne1 * (ne0 / 32));
        uint32_t st = 7;
        auto rnd = [&st]() { st = st * 1664525u + 1013904223u; return st; };
        for (auto& blk : src) {
            blk.d = qnn::htp::f32_to_f16_bits(0.05f);
            for (int j = 0; j < 32; j++) blk.qs[j] = (int8_t)(rnd() & 0xff);
        }
        size_t sz = qnn::htp::tiled_size_q8_0(ne0, ne1);
        std::string b(sz, '\0');
        qnn::htp::repack_q8_0_crouton(src.data(), (uint8_t*)b.data(), ne0, ne1);
        slots.push_back({"q8_crouton_64x64", (uint32_t)sz, b});
    }

    /* op 表 */
    struct OpDef { uint16_t opcode; std::vector<uint32_t> args; };
    std::vector<OpDef> ops = {
        {OP_NOP, {}},
        {OP_PIN, {1}},
        {OP_PIN, {2}},
        {OP_MATMUL_W4A16, {0, 1, 0, 256, 2560, 2560}},
    };
    if (tag == "w5") {
        ops.push_back({OP_RMSNORM_F16, {0, 5, 1, 2560}});
        ops.push_back({OP_PIN, {1}}); /* 引擎已建立后的真 pin (幂等 restage) */
    }

    /* 组 blob */
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), {'W', 'T', 'O', 'P'});
    wr_u16(blob, WT_BLOB_VER);
    wr_u16(blob, WT_ENDIAN_CHK);
    wr_u32(blob, (uint32_t)slots.size());
    wr_u32(blob, (uint32_t)ops.size());
    /* weight 区布局先定 (offset 相对 weight 区) */
    std::vector<uint32_t> offs;
    uint32_t cur = 0;
    for (auto& s : slots) { offs.push_back(cur); cur += s.len; cur = (cur + 127u) & ~127u; }
    for (size_t i = 0; i < slots.size(); i++) {
        wr_u32(blob, slots[i].len); wr_u32(blob, 1);
        wr_u32(blob, offs[i]);      wr_u32(blob, 0);
    }
    for (auto& op : ops) {
        wr_u16(blob, op.opcode); wr_u16(blob, (uint16_t)op.args.size());
        for (uint32_t a : op.args) wr_u32(blob, a);
    }
    while (blob.size() % 128) blob.push_back(0);
    uint32_t weight_off = (uint32_t)blob.size();
    for (size_t i = 0; i < slots.size(); i++) {
        while (blob.size() < weight_off + offs[i]) blob.push_back(0);
        blob.insert(blob.end(), slots[i].bytes.begin(), slots[i].bytes.end());
    }

    /* manifest */
    char sha_buf[65];
    std::string path = out + "/blob_" + tag + ".wtop";
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*)blob.data(), (std::streamsize)blob.size());
    }
    std::stringstream man;
    man << "{\n";
    man << " \"tag\": \"" << tag << "\",\n";
    man << " \"ver\": " << WT_BLOB_VER << ",\n";
    man << " \"endian_chk\": " << WT_ENDIAN_CHK << ",\n";
    man << " \"n_slots\": " << slots.size() << ",\n";
    man << " \"n_ops\": " << ops.size() << ",\n";
    man << " \"blob_bytes\": " << blob.size() << ",\n";
    man << " \"blob_sha256\": \"" << wt_sha256_hex(blob.data(), blob.size(), sha_buf) << "\",\n";
    man << " \"weight_off\": " << weight_off << ",\n";
    man << " \"weight_bytes\": " << (blob.size() - weight_off) << ",\n";
    man << " \"weight_sha256\": \""
        << wt_sha256_hex(blob.data() + weight_off, blob.size() - weight_off, sha_buf) << "\",\n";
    man << " \"slots\": [\n";
    for (size_t i = 0; i < slots.size(); i++) {
        man << "  {\"i\":" << i << ",\"name\":\"" << slots[i].name << "\",\"len\":" << slots[i].len
            << ",\"count\":1,\"offset\":" << offs[i] << ",\"addr\":0,\"sha256\":\""
            << wt_sha256_hex(slots[i].bytes.data(), slots[i].bytes.size(), sha_buf) << "\"}"
            << (i + 1 < slots.size() ? "," : "") << "\n";
    }
    man << " ],\n \"ops\": [\n";
    for (size_t i = 0; i < ops.size(); i++) {
        man << "  {\"i\":" << i << ",\"opcode\":" << ops[i].opcode
            << ",\"n_args\":" << ops[i].args.size() << ",\"args\":[";
        for (size_t a = 0; a < ops[i].args.size(); a++)
            man << ops[i].args[a] << (a + 1 < ops[i].args.size() ? "," : "");
        man << "]}" << (i + 1 < ops.size() ? "," : "") << "\n";
    }
    man << " ]\n}\n";
    {
        std::ofstream f(out + "/manifest_" + tag + ".json", std::ios::binary);
        f << man.str();
    }
    printf("[pack] %s: %zu bytes, %zu slots, %zu ops\n",
           path.c_str(), blob.size(), slots.size(), ops.size());
    return 0;
}
