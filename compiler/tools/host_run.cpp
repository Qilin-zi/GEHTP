// host_run: GEHTP host 参考执行器 (net.json → prepare → execute_host)
//
// 用法:
//   host_run --net-json <net.json> [--weights-bin <params.bin|TAR>]
//            [--input-f32 <in.f32.raw>] --out <out.f32.raw>
//
// 对拍链枢纽(M3c 机制): execute_host 按 op_id dump /tmp/host_id_<oid>.f32.raw,
// 与设备 ex39 的 dump_<idx>.f16.raw(blob 序 ≡ manifest op_ids 序)逐 op 对拍。
// 输入文件按图输入元素数读 f32(raw); 缺省全零。
// 2.48 params.bin(静态张量按 net.json 声明序 f16 拼接)自动识别并转 TAR。
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/opt/pass_manager.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <unistd.h>  // getpid

using namespace hnnx;

static float widen_f16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            u = sign | ((uint32_t)(127 + 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static bool looks_like_tar(const std::vector<uint8_t>& buf) {
    if (buf.size() < 512) return false;
    // name: 以字母/下划线/点开头(2.48 params.bin 开头是权重字节,
    // 恰好含可打印字符曾造成误判)
    bool name_ok = (buf[0] >= 'A' && buf[0] <= 'Z') || (buf[0] >= 'a' && buf[0] <= 'z') ||
                   buf[0] == '_' || buf[0] == '.';
    if (!name_ok) return false;
    for (int i = 1; i < 100; i++)
        if (buf[i] == 0) break;
        else if (buf[i] < 0x20 || buf[i] >= 0x7F) return false;
    // size: 跳过空格/NUL 后连续 >= 4 位八进制(单个 '0'-'7' 权重字节
    // 不再误判)
    int ndig = 0;
    bool started = false;
    for (int i = 124; i < 136; i++) {
        char c = (char)buf[i];
        if (c >= '0' && c <= '7') { started = true; ndig++; continue; }
        if ((c == ' ' || c == 0) && !started) continue;
        break;
    }
    return started && ndig >= 4;
}

static void tar_append(std::vector<uint8_t>& tar, const std::string& name,
                       const uint8_t* data, size_t len) {
    tar.resize(tar.size() + 512, 0);
    uint8_t* h = tar.data() + tar.size() - 512;
    std::memset(h, 0, 512);
    std::memcpy(h, name.c_str(), std::min<size_t>(name.size(), 99));
    std::memcpy(h + 100, "0000644", 7);
    std::memcpy(h + 108, "0000000", 7);
    std::memcpy(h + 116, "0000000", 7);
    char sz[12];
    std::snprintf(sz, sizeof(sz), "%011lo", (unsigned long)len);
    std::memcpy(h + 124, sz, 11);
    std::memset(h + 136, '0', 11);
    std::memset(h + 148, ' ', 8);
    h[156] = '0';
    std::memcpy(h + 257, "ustar", 5);
    std::memcpy(h + 263, "00", 2);
    uint32_t sum = 0;
    for (int i = 0; i < 512; i++) sum += h[i];
    std::snprintf(sz, sizeof(sz), "%06o", sum);
    std::memcpy(h + 148, sz, 6);
    h[154] = 0; h[155] = ' ';
    tar.insert(tar.end(), data, data + len);
    while (tar.size() % 512 != 0) tar.push_back(0);
}

/* 2.48 params.bin → TAR(与 hnnx_compile 同款适配, 顺序分配按 net.json 出现序) */
static std::string adapt_weights_bin(const std::string& net_json_path,
                                     const std::string& weights_bin_path) {
    std::vector<uint8_t> raw;
    {
        std::ifstream f(weights_bin_path, std::ios::binary);
        if (!f) return "";
        raw.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    if (looks_like_tar(raw)) return weights_bin_path;

    struct Cand { std::string name; size_t bytes; uint32_t dtype; };
    std::vector<Cand> cands;
    {
        std::string js;
        {
            std::ifstream f(net_json_path, std::ios::binary);
            if (!f) return "";
            js.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        }
        size_t tpos = js.find("\"tensors\"");
        if (tpos == std::string::npos) return "";
        size_t i = js.find('{', tpos) + 1;
        if (i == std::string::npos) return "";
        size_t depth = 1;
        std::string cur_name;
        bool cur_static = false;
        uint32_t cur_dtype = 0x0232;  // 缺省 f32(仅静态张量用它定宽)
        auto parse_dims = [&](size_t from) -> size_t {
            size_t lb = js.find('[', from);
            size_t rb = js.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) return from;
            std::string dimstr = js.substr(lb + 1, rb - lb - 1);
            /* 字节/元素按 data_type: int32 变体(0x0132/0x0032/0x0432/
             * 306)=4, f32(0x0232)=4, bool(0x0508)=1, 其余 f16=2
             * (2.48 params.bin 实测: 静态张量按声明序紧凑拼接,
             *  字节宽 = 张量自身 dtype, 非全 f16) */
            size_t eb = 2;
            if (cur_dtype == 0x0232 || cur_dtype == 0x0132 || cur_dtype == 0x0032 ||
                cur_dtype == 0x0432 || cur_dtype == 306 || cur_dtype == 0x0532 ||
                cur_dtype == 0x0632)
                eb = 4;
            else if (cur_dtype == 0x0508)
                eb = 1;
            size_t bytes = eb;
            bool any = false;
            size_t start = 0;
            while (start < dimstr.size()) {
                size_t comma = dimstr.find(',', start);
                std::string tok = dimstr.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                long v = std::atol(tok.c_str());
                if (v > 0) { bytes *= (size_t)v; any = true; }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (any && cur_static && !cur_name.empty())
                cands.push_back({cur_name, bytes, cur_dtype});
            return rb;
        };
        while (i < js.size()) {
            char c = js[i];
            if (c == '{') { depth++; i++; continue; }
            if (c == '}') { depth--; if (depth == 0) break; i++; continue; }
            if (c == '"') {
                size_t q2 = js.find('"', i + 1);
                if (q2 == std::string::npos) break;
                std::string key = js.substr(i + 1, q2 - i - 1);
                if (depth == 1) { cur_name = key; cur_static = false; cur_dtype = 0x0232; }
                else if (depth == 2 && key == "dims") { i = parse_dims(q2 + 1); continue; }
                else if (depth == 2 && key == "type") {
                    size_t colon = js.find(':', q2);
                    size_t comma2 = js.find_first_of(",}\n", colon);
                    if (colon != std::string::npos)
                        cur_static = (std::atoi(js.substr(colon + 1, comma2 - colon - 1).c_str()) == 4);
                } else if (depth == 2 && key == "data_type") {
                    size_t colon = js.find(':', q2);
                    size_t comma2 = js.find_first_of(",}\n", colon);
                    if (colon != std::string::npos)
                        cur_dtype = (uint32_t)std::strtoul(js.substr(colon + 1, comma2 - colon - 1).c_str(), nullptr, 0);
                }
                i = q2 + 1;
                continue;
            }
            i++;
        }
    }
    if (cands.empty()) return "";
    std::vector<Cand> uniq;
    for (const auto& c : cands) {
        bool dup = false;
        for (const auto& u : uniq) if (u.name == c.name) dup = true;
        if (!dup) uniq.push_back(c);
    }
    cands = uniq;

    std::vector<uint8_t> tar;
    size_t off = 0;
    size_t n_hits = 0;
    for (const auto& c : cands) {
        if (off + c.bytes > raw.size() || c.bytes == 0) continue;
        const bool d4 = (c.dtype == 0x0232 || c.dtype == 0x0132 || c.dtype == 0x0032 ||
                         c.dtype == 0x0432 || c.dtype == 306 || c.dtype == 0x0532 ||
                         c.dtype == 0x0632);
        const bool d1 = (c.dtype == 0x0508);
        std::vector<float> f32;
        if (d4) {
            /* f32/int32 → 4B 直拷贝(位模式保留, int32 消费者按位读) */
            f32.resize(c.bytes / 4);
            std::memcpy(f32.data(), raw.data() + off, c.bytes);
        } else if (d1) {
            /* bool → float(0/1) */
            f32.resize(c.bytes);
            for (size_t k = 0; k < f32.size(); k++) f32[k] = (float)raw[off + k];
        } else {
            /* f16 → f32 拓宽 */
            const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data() + off);
            f32.resize(c.bytes / 2);
            for (size_t k = 0; k < f32.size(); k++) f32[k] = widen_f16(h[k]);
        }
        tar_append(tar, c.name + ".raw",
                   reinterpret_cast<const uint8_t*>(f32.data()), f32.size() * 4);
        off += c.bytes;
        n_hits++;
    }
    if (n_hits == 0) return "";
    tar.resize(tar.size() + 1024, 0);

    char tmpname[256];
    std::snprintf(tmpname, sizeof(tmpname), "/tmp/host_run_weights_%d.tar", (int)getpid());
    {
        std::ofstream f(tmpname, std::ios::binary);
        if (!f) return "";
        f.write(reinterpret_cast<const char*>(tar.data()), tar.size());
    }
    std::fprintf(stderr, "[weights] 2.48 params.bin adapted: %zu tensors -> %s\n", n_hits, tmpname);
    return tmpname;
}

int main(int argc, char** argv) {
    std::string net_json, weights_bin, bin_path, in_path, out_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--net-json" && i + 1 < argc) net_json = argv[++i];
        else if (a == "--weights-bin" && i + 1 < argc) weights_bin = argv[++i];
        else if (a == "--bin" && i + 1 < argc) bin_path = argv[++i];
        else if (a == "--input-f32" && i + 1 < argc) in_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--no-pass" && i + 1 < argc) PassManager::instance().disable(argv[++i]);
        else { std::fprintf(stderr, "usage: host_run --net-json N [--weights-bin W] [--input-f32 I] --out O\n"); return 1; }
    }
    if (out_path.empty() || (net_json.empty() && bin_path.empty())) {
        std::fprintf(stderr, "usage: host_run --net-json N [--weights-bin W] [--input-f32 I] --out O\n");
        return 1;
    }

    register_all_ops();
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    if (!bin_path.empty()) {
        /* 从 tagged.bin 反序列化图(与 wtop_emit 同图对象, 逐 op 对拍用) */
        std::ifstream bf(bin_path, std::ios::binary);
        if (!bf) { std::fprintf(stderr, "Error: cannot open %s\n", bin_path.c_str()); return 1; }
        std::vector<uint8_t> bin((std::istreambuf_iterator<char>(bf)),
                                 std::istreambuf_iterator<char>());
        if (!gp.deserialize(bin.data(), bin.size())) {
            std::fprintf(stderr, "Error: deserialize failed\n");
            return 1;
        }
        std::fprintf(stderr, "graph from %s: %zu ops\n", bin_path.c_str(), bin.size());
    } else if (!weights_bin.empty()) {
        std::string tar = adapt_weights_bin(net_json, weights_bin);
        if (!tar.empty()) {
            loader.set_weights(loader.load_weight_bin(tar));
            if (tar.rfind("/tmp/host_run_weights_", 0) == 0) std::remove(tar.c_str());
        } else {
            std::fprintf(stderr, "Warning: weight adaptation failed, proceeding without weights\n");
        }
    }
    if (bin_path.empty()) {
        uint32_t op_count = loader.load_net_json(net_json);
        if (op_count == 0) { std::fprintf(stderr, "Error: no ops loaded\n"); return 1; }
    }
    HexagonNNEnv env;
    GraphStatus st = gp.prepare(env);
    if (st != GraphStatus::Success) { std::fprintf(stderr, "Error: prepare=%d\n", (int)st); return 1; }
    std::fprintf(stderr, "prepare ok: input=%llu output=%llu plan=%zu\n",
                 (unsigned long long)gp.get_input_node_id(),
                 (unsigned long long)gp.get_output_node_id(),
                 gp.plan_order().size());

    // 输入: --input-f32 逗号分隔多文件(按图输入声明序); 缺省全零
    // (全注意力层 4 输入: hidden/causal_mask/cos/sin)
    std::vector<std::vector<float>> ins;
    if (!in_path.empty()) {
        size_t pos = 0;
        while (pos <= in_path.size()) {
            size_t comma = in_path.find(',', pos);
            std::string p = in_path.substr(pos, comma == std::string::npos
                                                   ? std::string::npos : comma - pos);
            std::ifstream f(p, std::ios::binary);
            if (!f) { std::fprintf(stderr, "Error: cannot open %s\n", p.c_str()); return 1; }
            f.seekg(0, std::ios::end);
            size_t nbytes = (size_t)f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<float> v(nbytes / 4, 0.0f);
            f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)nbytes);
            ins.push_back(std::move(v));
            std::fprintf(stderr, "input[%zu]: %zu f32 from %s\n", ins.size() - 1,
                         ins.back().size(), p.c_str());
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    auto r = gp.execute_host(ins.empty() ? std::vector<float>() : ins[0],
                             ins.empty() ? nullptr : &ins);
    if (!r.ok) { std::fprintf(stderr, "Error: execute_host failed\n"); return 1; }
    std::fprintf(stderr, "host: ok, out=%zu f32\n", r.output.size());
    {
        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(r.output.data()),
                (std::streamsize)(r.output.size() * 4));
    }
    std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    return 0;
}
