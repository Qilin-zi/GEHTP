// test_vtcm_auto: --plan-vtcm-budget 缺省 auto 解析 (VTCM 驻留默认开)
//
// 三态: 缺省=auto(16MB − HMX scratch 峰值 − 余量) / 显式 N / 显式 0=关。
// 锚点: 大 matmul M=32 K=1024 N=6144 → pool = 16777216 − 13043712 − 524288
// = 3209216 逐字节钉死镜像分块公式(graph_prepare/op_matmul/assign_engine
// 三处同口径拷贝的同步绊线)。验证走 compile → emit → wt_parse TEMPOFF 槽。
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "oplist_parse.h"

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool write_file(const std::string& p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write((const char*)d, (std::streamsize)n);
    return f.good();
}
static bool load_file(const std::string& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

// 迷你 ADD 图: input[1,64] + const c[1,64] → out
static const char* NET_A =
    "{\"graph\":{\"tensors\":{"
    "\"input\":{\"id\":1,\"type\":0,\"data_type\":562,\"dims\":[1,64]},"
    "\"c\":{\"id\":2,\"type\":4,\"data_type\":562,\"dims\":[1,64]},"
    "\"out\":{\"id\":3,\"type\":1,\"data_type\":562,\"dims\":[1,64]}},"
    "\"nodes\":{"
    "\"/add\":{\"type\":\"Eltwise_Binary\",\"input_names\":[\"input\",\"c\"],"
    "\"output_names\":[\"out\"]}}}}";

// 大 matmul 锚: input[32,1024] @ w[1024,6144] → out[32,6144]
// scratch = need_mm(32,1024,6144) = 13043712 (< 16MB 不分块)
static const char* NET_B =
    "{\"graph\":{\"tensors\":{"
    "\"input\":{\"id\":1,\"type\":0,\"data_type\":562,\"dims\":[32,1024]},"
    "\"w\":{\"id\":2,\"type\":4,\"data_type\":562,\"dims\":[1024,6144]},"
    "\"out\":{\"id\":3,\"type\":1,\"data_type\":562,\"dims\":[32,6144]}},"
    "\"nodes\":{"
    "\"/mm\":{\"type\":\"MatMul\",\"input_names\":[\"input\",\"w\"],"
    "\"output_names\":[\"out\"]}}}}";

struct TempoffInfo { bool present = false; uint32_t cap = 0, vtcm_cap = 0, n = 0, n_vtcm = 0; };

static TempoffInfo inspect_tempoff(const std::string& blob_path) {
    TempoffInfo ti;
    std::vector<uint8_t> blob;
    if (!load_file(blob_path, blob)) return ti;
    wt_blob* wb = new wt_blob{};
    if (wt_parse(blob.data(), blob.size(), wb) != WT_OK) { delete wb; return ti; }
    for (uint32_t s = 0; s < wb->n_slots; s++) {
        if (wb->slots[s].addr != 0xFFFFFFFDu) continue;  // WT_SLOT_TEMPOFF
        const uint8_t* p = wb->weight_base + wb->slots[s].offset;
        uint32_t cap, reserve, n;
        std::memcpy(&cap, p, 4); std::memcpy(&reserve, p + 4, 4); std::memcpy(&n, p + 8, 4);
        ti.present = true;
        ti.cap = cap;
        ti.vtcm_cap = (reserve >> 16) * 1024u;
        ti.n = n;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t tid;
            std::memcpy(&tid, p + 12 + i * 12, 4);
            if (tid & 0x4000u) ti.n_vtcm++;
        }
        break;
    }
    delete wb;
    return ti;
}

// compile+emit 一趟, 返回 blob 路径; 失败返回空
static std::string build_blob(const std::string& dir, const std::string& tag,
                              const std::string& net_json, const std::string& in_file,
                              const std::string& extra_flags, std::string* log) {
    const std::string binp = dir + "/" + tag + ".bin";
    const std::string blobp = dir + "/" + tag + ".wtop";
    std::string cmd = std::string(HNNX_COMPILE_PATH) + " --net-json " + net_json +
                      (extra_flags.empty() ? "" : " " + extra_flags) +
                      " --format tagged --output " + binp + " >" + dir + "/" + tag + ".c.log 2>&1";
    if (std::system(cmd.c_str()) != 0) { if (log) *log = "compile rc!=0"; return ""; }
    cmd = std::string(WTOP_EMIT_PATH) + " --bin " + binp + " --input-f16 " + in_file +
          " --out " + blobp + " 2>" + dir + "/" + tag + ".e.log";
    if (std::system(cmd.c_str()) != 0) { if (log) *log = "emit rc!=0"; return ""; }
    return blobp;
}

int main() {
    std::printf("=== test_vtcm_auto (VTCM 驻留缺省 auto) ===\n\n");
    const std::string dir = "/tmp/vtcm_auto_test/";
    std::system(("mkdir -p " + dir).c_str());

    float ia[64];
    for (int i = 0; i < 64; i++) ia[i] = 1.0f;
    CHECK(write_file(dir + "in_a.f32.raw", ia, sizeof(ia)), "write in_a");
    float ib[32 * 1024];
    for (size_t i = 0; i < 32 * 1024; i++) ib[i] = 0.01f;
    CHECK(write_file(dir + "in_b.f32.raw", ib, sizeof(ib)), "write in_b");
    CHECK(write_file(dir + "net_a.json", NET_A, std::strlen(NET_A)), "write net_a");
    CHECK(write_file(dir + "net_b.json", NET_B, std::strlen(NET_B)), "write net_b");

    // 1. 迷你图缺省 auto: scratch_max=0 → pool = 16MB − 512KB = 16252928
    {
        std::string log;
        std::string blob = build_blob(dir, "a_auto", dir + "net_a.json", dir + "in_a.f32.raw", "", &log);
        CHECK(!blob.empty(), "a_auto compile+emit 通过");
        if (!blob.empty()) {
            TempoffInfo ti = inspect_tempoff(blob);
            CHECK(ti.present, "a_auto: TEMPOFF 槽存在(驻留默认开)");
            CHECK(ti.vtcm_cap == 16252928u, "a_auto: vtcm_cap == 16MB-512KB = 16252928");
            CHECK(ti.n_vtcm >= 1, "a_auto: ≥1 个 VTCM 驻留项");
            std::printf("  [info] cap=%u vtcm_cap=%u n=%u n_vtcm=%u\n", ti.cap, ti.vtcm_cap, ti.n, ti.n_vtcm);
        } else std::fprintf(stderr, "  (%s)\n", log.c_str());
    }

    // 2. 显式 0 = 关(debug 旧口径): 无 TEMPOFF
    {
        std::string log;
        std::string blob = build_blob(dir, "a_off", dir + "net_a.json", dir + "in_a.f32.raw",
                                      "--plan-vtcm-budget 0", &log);
        CHECK(!blob.empty(), "a_off compile+emit 通过");
        if (!blob.empty()) {
            TempoffInfo ti = inspect_tempoff(blob);
            CHECK(!ti.present, "a_off: 显式 0 → 无 TEMPOFF(旧口径)");
        } else std::fprintf(stderr, "  (%s)\n", log.c_str());
    }

    // 3. 显式 N 优先于 auto
    {
        std::string log;
        std::string blob = build_blob(dir, "a_64k", dir + "net_a.json", dir + "in_a.f32.raw",
                                      "--plan-vtcm-budget 65536", &log);
        CHECK(!blob.empty(), "a_64k compile+emit 通过");
        if (!blob.empty()) {
            TempoffInfo ti = inspect_tempoff(blob);
            CHECK(ti.present && ti.vtcm_cap == 65536u, "a_64k: 显式值照用 vtcm_cap==65536");
        } else std::fprintf(stderr, "  (%s)\n", log.c_str());
    }

    // 4. 大 matmul 锚: 镜像公式钉死 pool = 16777216 − 13043712 − 524288
    {
        std::string log;
        std::string blob = build_blob(dir, "b_auto", dir + "net_b.json", dir + "in_b.f32.raw", "", &log);
        CHECK(!blob.empty(), "b_auto compile+emit 通过");
        if (!blob.empty()) {
            TempoffInfo ti = inspect_tempoff(blob);
            CHECK(ti.present, "b_auto: TEMPOFF 槽存在");
            CHECK(ti.vtcm_cap == 3209216u,
                  "b_auto: vtcm_cap == 3209216 (scratch_max=13043712 镜像公式锚)");
            std::printf("  [info] cap=%u vtcm_cap=%u n=%u n_vtcm=%u\n", ti.cap, ti.vtcm_cap, ti.n, ti.n_vtcm);
        } else std::fprintf(stderr, "  (%s)\n", log.c_str());
    }

    std::printf("\n%s\n", failed ? "FAILED" : "ALL PASSED");
    return failed ? 1 : 0;
}
