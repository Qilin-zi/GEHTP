// test_scheduler_integration: Scheduler 接 ST-Cut(阶段 5)
//
// 验收:
//   1. do_prepare2 计算 plan_order_(ST-Cut 计划序), 非空且覆盖全部图 op
//   2. Scheduler::schedule(gp).op_order == gp.plan_order()
//   3. serialize 的 op 记录发射序 == plan_order_(含 TAG_PLAN_ORDER 记录)
//   4. deserialize 恢复 plan_order_; re-serialize 两次字节全同
//   5. 既有 ST-Cut 测试保持绿(ctest 全量)
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/schedule/scheduler.hpp"
#include "hnnx/serialize/serializer.hpp"  // encode_bin_tag

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static OutputDef make_od4(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    OutputDef od{};
    od.rank = 4;
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    return od;
}

// 从 .bin 中提取 op 记录发射序(按 TAG_OP_RECORD 出现序取 op_id)
static std::vector<op_id_t> parse_emit_order(const std::vector<uint8_t>& buf) {
    std::vector<op_id_t> order;
    uint32_t enc_op = encode_bin_tag(0x4F50);
    size_t i = 0;
    while (i + 8 <= buf.size()) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v == enc_op) {
            // [encoded][word_count][?][payload: name_len u32 ... op_id u64 @ 4+name_padded]
            uint32_t name_len;
            std::memcpy(&name_len, buf.data() + i + 12, 4);  // 假定记录头 12B
            uint32_t name_padded = (name_len + 3) & ~3u;
            uint64_t op_id;
            std::memcpy(&op_id, buf.data() + i + 12 + 4 + name_padded, 8);
            order.push_back(static_cast<op_id_t>(op_id));
            i += 8;
        } else {
            i++;
        }
    }
    return order;
}

int main() {
    std::printf("=== scheduler ST-Cut integration test ===\n\n");

    register_all_ops();

    GraphPrepare gp;
    auto od_nchw = make_od4(1, 32, 32, 32);
    gp.append_node("Input", 1, nullptr, 0, &od_nchw, 1, nullptr);

    const int32_t perm_in[4] = {0, 2, 3, 1};
    const int32_t perm_out[4] = {0, 3, 1, 2};
    OutputDef perm_od{};
    perm_od.rank = 1; perm_od.dims[0] = 4; perm_od.element_size = 4;
    perm_od.dtype = static_cast<uint32_t>(DType::Int32);
    gp.append_const_node(2, perm_od, reinterpret_cast<const uint8_t*>(perm_in), sizeof(perm_in));
    gp.get_op_at(2)->name_tag = string_tag_t::map_str("X_perm");

    InputDef t1in[2] = {{1, 0}, {2, 0}};
    gp.append_node("Transpose", 3, t1in, 2, &od_nchw, 1, nullptr);

    auto w_od = make_od4(3, 3, 32, 32);
    std::vector<float> Wq(32 * 32 * 3 * 3, 0.5f);
    gp.append_const_node(4, w_od,
                         reinterpret_cast<const uint8_t*>(Wq.data()), Wq.size() * sizeof(float));
    gp.get_op_at(4)->name_tag = string_tag_t::map_str("W");

    OutputDef b_od{};
    b_od.rank = 1; b_od.dims[0] = 32; b_od.element_size = 4;
    b_od.dtype = static_cast<uint32_t>(DType::Float32);
    std::vector<float> Bv(32, 0.1f);
    gp.append_const_node(5, b_od,
                         reinterpret_cast<const uint8_t*>(Bv.data()), Bv.size() * sizeof(float));
    gp.get_op_at(5)->name_tag = string_tag_t::map_str("B");

    InputDef cin[3] = {{3, 0}, {4, 0}, {5, 0}};
    gp.append_node("Conv2d", 6, cin, 3, &od_nchw, 1, nullptr);

    InputDef ain[2] = {{6, 0}, {3, 0}};
    gp.append_node("Eltwise_Binary", 7, ain, 2, &od_nchw, 1, nullptr);

    gp.append_const_node(8, perm_od, reinterpret_cast<const uint8_t*>(perm_out), sizeof(perm_out));
    gp.get_op_at(8)->name_tag = string_tag_t::map_str("Z_perm");
    InputDef t2in[2] = {{7, 0}, {8, 0}};
    gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);

    InputDef oin[1] = {{9, 0}};
    gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);

    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    GraphStatus st = gp.prepare(env);
    CHECK(st == GraphStatus::Success, "prepare");

    // 1. plan_order_ 非空且覆盖全部存活 op
    const auto& po = gp.plan_order();
    CHECK(!po.empty(), "plan_order 非空(ST-Cut 计划序已计算)");
    std::set<op_id_t> expect;
    for (op_id_t id : {3, 6, 7, 9}) expect.insert(id);  // 四个计算 op(不含 const/IO)
    // 宽松: plan_order 至少包含全部计算 op(可能含更多)
    bool covers = true;
    for (op_id_t id : expect)
        covers &= (std::find(po.begin(), po.end(), id) != po.end());
    CHECK(covers, "plan_order 覆盖全部计算 op");

    // 2. Scheduler::schedule == gp.plan_order()
    Scheduler sched;
    Scheduler::Plan plan = sched.schedule(gp);
    bool same = (plan.op_order.size() == po.size());
    if (same)
        for (size_t i = 0; i < po.size(); i++)
            same &= (plan.op_order[i] == static_cast<uint32_t>(po[i]));
    CHECK(same, "Scheduler::schedule(gp).op_order == gp.plan_order()");

    // 3. serialize 发射序 == plan_order_
    std::vector<uint8_t> buf(1u << 18, 0);
    size_t out_size = 0;
    bool ok = gp.serialize(buf.data(), buf.size(), out_size);
    CHECK(ok && out_size > 0, "serialize");
    buf.resize(out_size);

    std::vector<op_id_t> emit = parse_emit_order(buf);
    // emit 序的前缀应等于 plan_order_ 中出现在记录里的 id 序列
    std::vector<op_id_t> expect_seq;
    for (op_id_t id : po) {
        if (gp.get_op_at(id) && gp.get_op_at(id)->is_enabled() && !gp.get_op_at(id)->is_dead())
            expect_seq.push_back(id);
    }
    bool order_match = (emit.size() == expect_seq.size());
    if (order_match)
        for (size_t i = 0; i < emit.size(); i++)
            order_match &= (emit[i] == expect_seq[i]);
    CHECK(order_match, "serialize 发射序 == plan_order_");

    // 4. TAG_PLAN_ORDER 记录存在 + deserialize 恢复 + re-serialize 全同
    size_t pos = 0;
    bool has_pl = false;
    for (size_t i = 0; i + 4 <= buf.size(); i++) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v == encode_bin_tag(0x504C)) { has_pl = true; pos = i; break; }
    }
    CHECK(has_pl, ".bin 含 TAG_PLAN_ORDER 记录");

    GraphPrepare gp2;
    CHECK(gp2.deserialize(buf.data(), buf.size()), "deserialize");
    CHECK(gp2.plan_order().size() == po.size(), "deserialize 恢复 plan_order 尺寸");
    bool same2 = (gp2.plan_order().size() == po.size());
    if (same2)
        for (size_t i = 0; i < po.size(); i++)
            same2 &= (gp2.plan_order()[i] == po[i]);
    CHECK(same2, "deserialize 恢复 plan_order 内容一致");

    std::vector<uint8_t> buf2(1u << 18, 0);
    size_t out_size2 = 0;
    ok = gp2.serialize(buf2.data(), buf2.size(), out_size2);
    CHECK(ok && out_size2 == out_size &&
          std::memcmp(buf.data(), buf2.data(), out_size) == 0,
          "re-serialize 两次字节全同");

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
