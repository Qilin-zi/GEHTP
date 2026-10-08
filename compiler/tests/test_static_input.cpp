// test_static_input: --static-input 导入期物化 (P2)
//
// 最小双输入图(net.json): input(f32 [1,4]) + mask(f32 [1,4])
//   → Eltwise_Binary(ADD) → out。
// 物化 mask 后: Input 节点同 id 换 const(f16 语义, 池内 f32 拓宽);
// prepare/serialize 通过; wtop_emit 只喂主输入文件即可(P1 门不再索要
// mask 文件); blob 中出现 f16 常量槽(字节 = 物化值)。
// 负例: 主输入 / 未知名 / 坏尺寸 / .i32.raw 全部硬错。
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ops/ops.hpp"
#include "oplist_parse.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool write_file(const std::string& path, const void* data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)len);
    return f.good();
}

static bool load_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

static const char* NET_JSON =
    "{\"graph\":{\"tensors\":{"
    "\"input\":{\"id\":1,\"type\":0,\"data_type\":562,\"dims\":[1,4]},"
    "\"mask\":{\"id\":2,\"type\":0,\"data_type\":562,\"dims\":[1,4]},"
    "\"out\":{\"id\":3,\"type\":1,\"data_type\":562,\"dims\":[1,4]}},"
    "\"nodes\":{"
    "\"/add\":{\"type\":\"Eltwise_Binary\",\"input_names\":[\"input\",\"mask\"],"
    "\"output_names\":[\"out\"]}}}}";

// 折叠链图: mask → EQ(mask,mask) → SELECT(eq, mask, zeros) → ADD(input, sel)。
// sel/out dtype=f16(0x216): 终折 const 池内 f16-native → 发射走原样入槽分支。
// 预期: EQ/SELECT 两 op 被 const_fold 折掉, blob 只剩 1 个计算 op 且含
// f16 2.0×4 常量槽。
static const char* NET_JSON_FOLD =
    "{\"graph\":{\"tensors\":{"
    "\"input\":{\"id\":1,\"type\":0,\"data_type\":562,\"dims\":[1,4]},"
    "\"mask\":{\"id\":2,\"type\":0,\"data_type\":562,\"dims\":[1,4]},"
    "\"zc\":{\"id\":5,\"type\":4,\"data_type\":562,\"dims\":[1,4]},"
    "\"eqo\":{\"id\":6,\"type\":3,\"data_type\":562,\"dims\":[1,4]},"
    "\"selo\":{\"id\":7,\"type\":3,\"data_type\":534,\"dims\":[1,4]},"
    "\"out\":{\"id\":8,\"type\":1,\"data_type\":534,\"dims\":[1,4]}},"
    "\"nodes\":{"
    "\"/eq\":{\"type\":\"Eltwise_Binary\",\"input_names\":[\"mask\",\"mask\"],"
    "\"output_names\":[\"eqo\"],\"scalar_params\":{\"operation\":{\"50\":3}}},"
    "\"/sel\":{\"type\":\"Eltwise_Ternary\",\"input_names\":[\"eqo\",\"mask\",\"zc\"],"
    "\"output_names\":[\"selo\"]},"
    "\"/add\":{\"type\":\"Eltwise_Binary\",\"input_names\":[\"input\",\"selo\"],"
    "\"output_names\":[\"out\"],\"scalar_params\":{\"operation\":{\"50\":0}}}}}}";

int main() {
    std::printf("=== test_static_input (P2 import-time materialization) ===\n\n");
    register_all_ops();

    const std::string dir = "/tmp/static_input_test/";
    std::system(("mkdir -p " + dir).c_str());
    const std::string nj = dir + "mini_net.json";
    CHECK(write_file(nj, NET_JSON, std::strlen(NET_JSON)), "write mini_net.json");

    // mask 值 = 2.0 ×4 (f32)
    float mv[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    const std::string mask_f32 = dir + "mask_2.f32.raw";
    CHECK(write_file(mask_f32, mv, sizeof(mv)), "write mask f32 file");
    // f16 形式 (0x4000 ×4)
    uint16_t mh[4] = {0x4000, 0x4000, 0x4000, 0x4000};
    const std::string mask_f16 = dir + "mask_2.f16.raw";
    CHECK(write_file(mask_f16, mh, sizeof(mh)), "write mask f16 file");
    // 坏尺寸 / i32 文件
    const std::string bad = dir + "bad.f32.raw";
    CHECK(write_file(bad, mv, 6), "write bad-size file");
    const std::string i32f = dir + "mask.i32.raw";
    CHECK(write_file(i32f, mv, sizeof(mv)), "write .i32.raw file");
    // 主输入文件 (f32 → 1.0×4)
    float iv[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const std::string in_f32 = dir + "in_1.f32.raw";
    CHECK(write_file(in_f32, iv, sizeof(iv)), "write main input file");

    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_net_json(nj);
    CHECK(ops == 1, "load net.json: 1 op node");
    CHECK(gp.get_op_at(1) && gp.get_op_at(1)->name_tag &&
          std::strcmp(gp.get_op_at(1)->name_tag->name(), "Input") == 0,
          "input(id1) 是 Input 节点");

    std::string err;
    CHECK(!loader.materialize_static_input("nope", mask_f32, err) &&
          err.find("not found") != std::string::npos, "neg: 未知名硬错");
    CHECK(!loader.materialize_static_input("input", mask_f32, err) &&
          err.find("main input") != std::string::npos, "neg: 主输入不可物化");
    CHECK(!loader.materialize_static_input("mask", bad, err) &&
          err.find("elems*2/4") != std::string::npos, "neg: 坏尺寸硬错");
    CHECK(!loader.materialize_static_input("mask", dir + "missing.f32.raw", err) &&
          err.find("cannot open") != std::string::npos, "neg: 缺文件硬错");
    CHECK(!loader.materialize_static_input("mask", i32f, err) &&
          err.find("i32") != std::string::npos, "neg: .i32.raw 拒绝");

    CHECK(loader.materialize_static_input("mask", mask_f32, err), "mask 物化成功(f32 文件)");
    const OpDef* mc = gp.get_op_at(2);
    CHECK(mc && mc->const_data_size == 16 && !mc->is_dead(), "op2 变 const(池 16B f32)");
    CHECK(mc && mc->output_def.dtype == (uint32_t)DType::Float16 &&
          mc->output_def.element_size == 2, "物化 od = f16 语义");
    // f16 文件路径: 拓宽进池后再洗回应位级还原 → 同一 const
    GraphPrepare gp2;
    QnnIRLoader loader2(gp2);
    loader2.load_net_json(nj);
    CHECK(loader2.materialize_static_input("mask", mask_f16, err), "mask 物化(f16 文件)");
    const OpDef* mc2 = gp2.get_op_at(2);
    CHECK(mc2 && mc2->const_data_size == 16 && mc && mc2 &&
          std::memcmp(gp.const_pool().data() + mc->const_data_offset,
                      gp2.const_pool().data() + mc2->const_data_offset, 16) == 0,
          "f16/f32 文件池 payload 位级一致");

    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    CHECK(gp.prepare(env) == GraphStatus::Success, "prepare after materialize");

    std::vector<uint8_t> bin(1u << 18, 0);
    size_t bin_len = 0;
    CHECK(gp.serialize(bin.data(), bin.size(), bin_len) && bin_len > 0, "serialize .bin");
    bin.resize(bin_len);
    const std::string binp = dir + "mini.bin";
    CHECK(write_file(binp, bin.data(), bin.size()), "write .bin");

    // emit: 只喂主输入文件 — mask 已物化, 不再是图输入(P1 门应放行)
    const std::string blobp = dir + "mini.wtop";
    std::string cmd = std::string(HNNX_COMPILE_PATH) + " --net-json " + nj +
                      " --static-input mask=" + mask_f32 +
                      " --format tagged --output " + binp + " >" + dir + "c.log 2>&1";
    CHECK(std::system(cmd.c_str()) == 0, "hnnx_compile --static-input 退出 0");

    cmd = std::string(WTOP_EMIT_PATH) + " --bin " + binp + " --input-f16 " + in_f32 +
          " --out " + blobp + " 2>" + dir + "e.log";
    CHECK(std::system(cmd.c_str()) == 0, "emit 只喂主输入退出 0(物化输入免文件)");
    std::vector<uint8_t> elog;
    load_file(dir + "e.log", elog);
    CHECK(std::string((char*)elog.data(), elog.size()).find("has no input file") ==
              std::string::npos, "emit 无缺文件报错");

    std::vector<uint8_t> blob;
    CHECK(load_file(blobp, blob), "read blob");
    wt_blob* wb = new wt_blob{};
    CHECK(wt_parse(blob.data(), blob.size(), wb) == WT_OK, "wt_parse OK");
    bool found_mask_slot = false;
    for (uint32_t s = 0; s < wb->n_slots; s++) {
        const wt_slot& sl = wb->slots[s];
        if (sl.len != 8) continue;
        const uint8_t* p = wb->weight_base + sl.offset;
        bool all2 = true;
        for (int i = 0; i < 4; i++) {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            if (h != 0x4000) { all2 = false; break; }
        }
        if (all2) { found_mask_slot = true; break; }
    }
    CHECK(found_mask_slot, "blob 含 f16 2.0×4 常量槽(8B)");

    // ---- 折叠链: materialize → const_fold 吃链 → f16-native 池 const 发射 ----
    const std::string njf = dir + "fold_net.json";
    CHECK(write_file(njf, NET_JSON_FOLD, std::strlen(NET_JSON_FOLD)), "write fold_net.json");
    const std::string binf = dir + "fold.tagged.bin";
    std::string cmdf = std::string(HNNX_COMPILE_PATH) + " --net-json " + njf +
                       " --static-input mask=" + mask_f32 +
                       " --format tagged --output " + binf + " >" + dir + "cf.log 2>&1";
    CHECK(std::system(cmdf.c_str()) == 0, "fold 链 compile 退出 0");
    const std::string blobf = dir + "fold.wtop";
    std::string cmde = std::string(WTOP_EMIT_PATH) + " --bin " + binf + " --input-f16 " +
                       in_f32 + " --out " + blobf + " 2>" + dir + "ef.log";
    CHECK(std::system(cmde.c_str()) == 0, "fold 链 emit 退出 0");
    std::vector<uint8_t> blob2;
    CHECK(load_file(blobf, blob2), "read fold blob");
    wt_blob* wb2 = new wt_blob{};
    CHECK(wt_parse(blob2.data(), blob2.size(), wb2) == WT_OK, "fold blob wt_parse OK");
    CHECK(wb2->n_ops == 1, "EQ/SELECT 已折, 仅剩 1 个计算 op");
    bool found_fold_slot = false;
    for (uint32_t s = 0; s < wb2->n_slots; s++) {
        const wt_slot& sl = wb2->slots[s];
        if (sl.len != 8) continue;
        const uint8_t* p = wb2->weight_base + sl.offset;
        bool all2 = true;
        for (int i = 0; i < 4; i++) {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            if (h != 0x4000) { all2 = false; break; }
        }
        if (all2) { found_fold_slot = true; break; }
    }
    CHECK(found_fold_slot, "折后 f16-native 池 const 原样入槽(2.0×4)");

    std::printf("\n%s\n", failed ? "FAILED" : "ALL PASSED");
    return failed ? 1 : 0;
}
