// test_opt_passes —— 图优化 pass 单元测试(PassManager 结构匹配框架)
// =====================================================================
// 覆盖:
//   const_fold: 正例(全 const 输入折叠 + 数值正确)/反例(非全 const、非白名单)
//               /统一使能开关(disable 后不折)
//   shape_fold: 恒等 Reshape 删除(A)/不同 dims 不折(A 反例)
//               Reshape 链合并(B)/链首多消费者不折(B 反例)
//               恒等 Transpose 删除(C)
//   stats: 匹配/重写计数
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ops/ops.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace hnnx { void register_all_ops(); }

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { std::printf("ok: %s\n", msg); } \
} while (0)

static OutputDef f32_def(uint64_t d0 = 1, uint64_t d1 = 1,
                         uint64_t d2 = 1, uint64_t d3 = 1) {
    OutputDef od{};
    od.rank = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    return od;
}

static op_id_t add_input(GraphPrepare& gp, op_id_t id, const OutputDef& def) {
    return gp.append_node("Input", id, nullptr, 0, &def, 1, nullptr);
}
static op_id_t add_output(GraphPrepare& gp, op_id_t id, op_id_t src) {
    InputDef in{};
    in.rank = src;  // InputDef.rank 槽位携带 src_id(项目约定)
    in.dtype = 0;   // out_idx
    return gp.append_node("Output", id, &in, 1, nullptr, 0, nullptr);
}
static op_id_t add_const_f32(GraphPrepare& gp, op_id_t id, const OutputDef& def,
                             const std::vector<float>& vals) {
    std::vector<uint8_t> data(vals.size() * 4);
    std::memcpy(data.data(), vals.data(), data.size());
    return gp.append_const_node(id, def, data.data(), data.size());
}
static op_id_t add_unary_op(GraphPrepare& gp, op_id_t id, const char* name,
                            const std::vector<op_id_t>& srcs, const OutputDef& out_def) {
    std::vector<InputDef> ins;
    for (op_id_t s : srcs) {
        InputDef in{};
        in.rank = s;
        in.dtype = 0;
        ins.push_back(in);
    }
    return gp.append_node(name, id, ins.data(), ins.size(), &out_def, 1, nullptr);
}

int main() {
    using namespace hnnx;
    register_all_ops();
    HexagonNNEnv env;

    std::printf("=== [1] const_fold 正例: 全 const 输入折叠 + 数值 ===\n");
    {
        GraphPrepare gp;
        auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
        auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
        auto add = add_unary_op(gp, 10, "Eltwise_Binary", {c1, c2}, f32_def());
        add_output(gp, 20, add);
        gp.run_optimize_passes(env);

        OpDef* a = gp.get_op_at(add);
        CHECK(a && a->is_const(), "const_fold: Eltwise_Binary 折叠成 const");
        CHECK(a && a->const_data_size == 4, "const_fold: const 数据 4B");
        if (a) {
            bool in_pool = a->const_data_offset + a->const_data_size <= gp.const_pool().size();
            CHECK(in_pool, "const_fold: 数据在 const pool 内");
            if (in_pool) {
                float v;
                std::memcpy(&v, gp.const_pool().data() + a->const_data_offset, 4);
                CHECK(std::fabs(v - 5.0f) < 1e-6f, "const_fold: 2+3=5 数值正确");
            }
        }
    }

    std::printf("=== [2] const_fold 反例: 非全 const 输入不折 ===\n");
    {
        GraphPrepare gp;
        auto in = add_input(gp, 5, f32_def());
        auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
        auto add = add_unary_op(gp, 10, "Eltwise_Binary", {in, c2}, f32_def());
        add_output(gp, 20, add);
        gp.run_optimize_passes(env);
        OpDef* a = gp.get_op_at(add);
        CHECK(a && !a->is_const() && a->is_enabled(), "const_fold: 非全 const 不折");
    }

    std::printf("=== [3] const_fold 反例: 白名单外(MatMul)不折 ===\n");
    {
        GraphPrepare gp;
        auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
        auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
        auto mm = add_unary_op(gp, 10, "MatMul", {c1, c2}, f32_def());
        add_output(gp, 20, mm);
        gp.run_optimize_passes(env);
        OpDef* m = gp.get_op_at(mm);
        CHECK(m && !m->is_const() && m->is_enabled(), "const_fold: MatMul 白名单外不折");
    }

    std::printf("=== [4] 统一使能管理: disable(const_fold) 后不折 ===\n");
    {
        PassManager& pm = PassManager::instance();
        CHECK(pm.disable("const_fold"), "pass_mgr: disable(const_fold) 生效");
        GraphPrepare gp;
        auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
        auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
        auto add = add_unary_op(gp, 10, "Eltwise_Binary", {c1, c2}, f32_def());
        add_output(gp, 20, add);
        gp.run_optimize_passes(env);
        OpDef* a = gp.get_op_at(add);
        CHECK(a && !a->is_const(), "const_fold: disable 后不折");
        pm.enable("const_fold");
        CHECK(pm.is_enabled("const_fold"), "pass_mgr: enable 恢复");
    }

    std::printf("=== [5] shape_fold A: 恒等 Reshape 删除 + 消费者改线 ===\n");
    {
        GraphPrepare gp;
        auto in = add_input(gp, 5, f32_def(1, 1, 2, 3));
        auto r = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 1, 2, 3));
        auto out = add_output(gp, 20, r);
        gp.run_optimize_passes(env);
        OpDef* ro = gp.get_op_at(r);
        CHECK(!ro || ro->is_dead(), "shape_fold: 恒等 Reshape 删除");
        OpDef* oo = gp.get_op_at(out);
        CHECK(oo && oo->inputs.size() == 1 && oo->inputs[0].src_id == in,
              "shape_fold: Output 改线到 Input");
    }

    std::printf("=== [6] shape_fold A 反例: 变 dims Reshape 不折 ===\n");
    {
        GraphPrepare gp;
        auto in = add_input(gp, 5, f32_def(1, 1, 2, 3));
        auto r = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 1, 3, 2));
        add_output(gp, 20, r);
        gp.run_optimize_passes(env);
        OpDef* ro = gp.get_op_at(r);
        CHECK(ro && !ro->is_dead(), "shape_fold: 变 dims Reshape 保留");
    }

    std::printf("=== [7] shape_fold B: Reshape 链合并 ===\n");
    {
        GraphPrepare gp;
        auto in = add_input(gp, 5, f32_def(1, 1, 2, 3));
        auto r1 = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 1, 6, 1));
        auto r2 = add_unary_op(gp, 11, "Reshape", {r1}, f32_def(1, 1, 1, 6));
        add_output(gp, 20, r2);
        gp.run_optimize_passes(env);
        OpDef* r1o = gp.get_op_at(r1);
        OpDef* r2o = gp.get_op_at(r2);
        CHECK(!r1o || r1o->is_dead(), "shape_fold: 链首 R1 删除");
        CHECK(r2o && r2o->inputs.size() == 1 && r2o->inputs[0].src_id == in,
              "shape_fold: 链尾 R2 改线到 Input");
        CHECK(r2o && r2o->output_def.dims[0] == 1 && r2o->output_def.dims[1] == 1 &&
              r2o->output_def.dims[2] == 1 && r2o->output_def.dims[3] == 6,
              "shape_fold: 链尾目标 dims 不变");
    }

    std::printf("=== [8] shape_fold B 反例: 链首多消费者不折 ===\n");
    {
        GraphPrepare gp;
        auto in = add_input(gp, 5, f32_def(1, 1, 2, 3));
        auto r1 = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 1, 6, 1));
        auto r2 = add_unary_op(gp, 11, "Reshape", {r1}, f32_def(1, 1, 1, 6));
        auto r3 = add_unary_op(gp, 12, "Reshape", {r1}, f32_def(1, 1, 2, 3));
        add_output(gp, 20, r2);
        add_output(gp, 21, r3);
        gp.run_optimize_passes(env);
        OpDef* r1o = gp.get_op_at(r1);
        CHECK(r1o && !r1o->is_dead(), "shape_fold: 多消费者链首保留");
    }

    std::printf("=== [9] shape_fold C: 恒等 Transpose 删除 ===\n");
    {
        GraphPrepare gp;
        // perm = [0,1,2,3] int32 const, 经 tensor_param_ids 挂在 Transpose 上
        OutputDef perm_def{};
        perm_def.rank = 1;
        perm_def.dtype = static_cast<uint32_t>(DType::Int32);
        perm_def.dims[0] = 4;
        perm_def.element_size = 4;
        int32_t perm[4] = {0, 1, 2, 3};
        auto perm_id = gp.append_const_node(1, perm_def,
                    reinterpret_cast<const uint8_t*>(perm), sizeof(perm));
        auto in = add_input(gp, 5, f32_def(1, 1, 2, 3));
        auto tr = add_unary_op(gp, 10, "Transpose", {in}, f32_def(1, 1, 2, 3));
        {
            OpDef* tod = gp.get_op_at(tr);
            CHECK(tod != nullptr, "shape_fold: Transpose 建图");
            if (tod) tod->tensor_param_ids.push_back(perm_id);
        }
        add_output(gp, 20, tr);
        gp.run_optimize_passes(env);
        OpDef* tro = gp.get_op_at(tr);
        CHECK(!tro || tro->is_dead(), "shape_fold: 恒等 Transpose 删除");
    }

    std::printf("=== [10] stats: 匹配/重写计数与前后条目数 ===\n");
    {
        PassManager& pm = PassManager::instance();
        pm.reset_stats();
        GraphPrepare gp;
        auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
        auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
        auto add = add_unary_op(gp, 10, "Eltwise_Binary", {c1, c2}, f32_def());
        add_output(gp, 20, add);
        gp.run_optimize_passes(env);
        std::string s = pm.stats_line();
        CHECK(s.find("const_fold=") != std::string::npos, "stats: const_fold 段存在");
        CHECK(s.find("shape_fold=") != std::string::npos, "stats: shape_fold 段存在");
        CHECK(s.find("before=") != std::string::npos &&
              s.find("after=") != std::string::npos, "stats: before/after 存在");
        std::printf("      %s\n", s.c_str());
    }

    std::printf("=== [11] fusion 注册制: 默认关 vs enable 后 run_optimize 触发 ===\n");
    {
        PassManager& pm = PassManager::instance();
        // 图 A: 默认关, Relu 不被融合(契约: 融合非设备准入, 主路径不开启)
        {
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 224, 224, 3));
            auto w    = add_const_f32(gp, 1, f32_def(32, 3, 3, 3), std::vector<float>(864, 0.01f));
            auto conv = add_unary_op(gp, 10, "Conv", {in, w}, f32_def(1, 222, 222, 32));
            auto relu = add_unary_op(gp, 20, "Relu", {conv}, f32_def(1, 222, 222, 32));
            add_output(gp, 30, relu);
            pm.disable("fusion_conv_relu");
            gp.run_optimize_passes(env);
            OpDef* ro = gp.get_op_at(relu);
            std::string nm = (ro && ro->name_tag) ? ro->name_tag->name() : "";
            CHECK(nm == "Relu", "fusion: 默认关, Relu 保持不融合");
        }
        // 图 B: enable 后经注册制 run_phase(PHASE_3) 触发融合
        {
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 224, 224, 3));
            auto w    = add_const_f32(gp, 1, f32_def(32, 3, 3, 3), std::vector<float>(864, 0.01f));
            auto conv = add_unary_op(gp, 10, "Conv", {in, w}, f32_def(1, 222, 222, 32));
            auto relu = add_unary_op(gp, 20, "Relu", {conv}, f32_def(1, 222, 222, 32));
            add_output(gp, 30, relu);
            pm.enable("fusion_conv_relu");
            gp.run_optimize_passes(env);
            OpDef* ro = gp.get_op_at(relu);
            OpDef* co = gp.get_op_at(conv);
            std::string nm = (ro && ro->name_tag) ? ro->name_tag->name() : "";
            bool conv_dead = (co == nullptr || co->is_dead());
            CHECK(nm == "ConvActivations", "fusion: enable 后 Relu -> ConvActivations");
            CHECK(conv_dead, "fusion: enable 后 Conv producer dead");
            pm.disable("fusion_conv_relu");
        }
    }

    std::printf("=== [12] fc_bias_fold: FC→Reshape→Add(const bias) 折叠 ===\n");
    {
        PassManager& pm = PassManager::instance();
        // 图 A: 正例 (M=32,K=1024,N=16 — N%32!=0 非 HMX 形, 应折叠)
        {
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 32, 1024, 1));
            auto w    = add_const_f32(gp, 1, f32_def(16, 1024, 1, 1), std::vector<float>(16384, 0.01f));
            auto bias = add_const_f32(gp, 2, f32_def(16, 1, 1, 1), std::vector<float>(16, 0.5f));
            auto fc   = add_unary_op(gp, 10, "FullyConnected", {in, w}, f32_def(1, 1, 32, 16));
            auto rs   = add_unary_op(gp, 11, "Reshape", {fc}, f32_def(1, 32, 16, 1));
            auto add  = add_unary_op(gp, 12, "Eltwise_Binary", {rs, bias}, f32_def(1, 32, 16, 1));
            add_output(gp, 30, add);
            gp.run_optimize_passes(env);
            OpDef* fo = gp.get_op_at(fc);
            OpDef* ao = gp.get_op_at(add);
            OpDef* oo = gp.get_op_at(30);
            CHECK(fo && fo->inputs.size() == 3, "fc_bias: FC 收到 inputs[2]=bias");
            CHECK(fo && fo->inputs.size() == 3 && fo->inputs[2].src_id == bias,
                  "fc_bias: inputs[2] 指向 bias const");
            {
                bool mark = false;
                if (fo) {
                    auto sp = unpack_scalar_params(fo->op_data);
                    const ScalarParam* p = scalar_get(sp, "has_bias");
                    mark = p && p->is_numeric && p->value_num != 0;
                }
                CHECK(mark, "fc_bias: op_data 写入 has_bias=1 可信标记");
            }
            CHECK(!ao || ao->is_dead(), "fc_bias: Add 标 dead");
            CHECK(oo && oo->inputs.size() == 1 && oo->inputs[0].src_id == rs,
                  "fc_bias: Output 改线到 Reshape");
        }
        // 图 D: 注入形态 — FC 已带 [wscale($Const id≥0x10000), quant_marker],
        //        折叠 = 替换 inputs[2] 为 bias(真实管线 do_prepare1 注入后形状)
        {
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 32, 1024, 1));
            auto w    = add_const_f32(gp, 1, f32_def(16, 1024, 1, 1), std::vector<float>(16384, 0.01f));
            auto bias = add_const_f32(gp, 2, f32_def(16, 1, 1, 1), std::vector<float>(16, 0.5f));
            auto wsc  = add_const_f32(gp, 0x10000, f32_def(1, 1, 1, 16), std::vector<float>(16, 0.0f));
            auto qm   = add_const_f32(gp, 0x10001, f32_def(1, 1, 1, 1), std::vector<float>(1, 0.0f));
            auto fc   = add_unary_op(gp, 10, "FullyConnected", {in, w, wsc, qm}, f32_def(1, 32, 16, 1));
            auto add  = add_unary_op(gp, 12, "Eltwise_Binary", {fc, bias}, f32_def(1, 32, 16, 1));
            add_output(gp, 30, add);
            gp.run_optimize_passes(env);
            OpDef* fo = gp.get_op_at(fc);
            OpDef* ao = gp.get_op_at(add);
            CHECK(fo && fo->inputs.size() == 4 && fo->inputs[2].src_id == bias,
                  "fc_bias: 注入形态 inputs[2] 由 wscale 替换为 bias");
            CHECK(fo && fo->inputs[3].src_id == qm, "fc_bias: quant_marker 不动");
            CHECK(!ao || ao->is_dead(), "fc_bias: 注入形态 Add 标 dead");
        }
        // 图 B: HMX 门 — M/K/N 全 32 倍数 (N=64) 不折(内核 v1 仅标量路径应用 bias)
        {
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 32, 1024, 1));
            auto w    = add_const_f32(gp, 1, f32_def(64, 1024, 1, 1), std::vector<float>(65536, 0.01f));
            auto bias = add_const_f32(gp, 2, f32_def(64, 1, 1, 1), std::vector<float>(64, 0.5f));
            auto fc   = add_unary_op(gp, 10, "FullyConnected", {in, w}, f32_def(1, 32, 64, 1));
            auto add  = add_unary_op(gp, 12, "Eltwise_Binary", {fc, bias}, f32_def(1, 32, 64, 1));
            add_output(gp, 30, add);
            gp.run_optimize_passes(env);
            OpDef* fo = gp.get_op_at(fc);
            OpDef* ao = gp.get_op_at(add);
            CHECK(fo && fo->inputs.size() == 2, "fc_bias: HMX 形不折, FC 保持 2 输入");
            CHECK(ao && !ao->is_dead(), "fc_bias: HMX 形不折, Add 保留");
        }
        // 图 C: disable 后不折
        {
            pm.disable("fc_bias_fold");
            GraphPrepare gp;
            auto in   = add_input(gp, 5, f32_def(1, 32, 1024, 1));
            auto w    = add_const_f32(gp, 1, f32_def(16, 1024, 1, 1), std::vector<float>(16384, 0.01f));
            auto bias = add_const_f32(gp, 2, f32_def(16, 1, 1, 1), std::vector<float>(16, 0.5f));
            auto fc   = add_unary_op(gp, 10, "FullyConnected", {in, w}, f32_def(1, 32, 16, 1));
            auto add  = add_unary_op(gp, 12, "Eltwise_Binary", {fc, bias}, f32_def(1, 32, 16, 1));
            add_output(gp, 30, add);
            gp.run_optimize_passes(env);
            OpDef* fo = gp.get_op_at(fc);
            OpDef* ao = gp.get_op_at(add);
            CHECK(fo && fo->inputs.size() == 2, "fc_bias: disable 后 FC 保持 2 输入");
            CHECK(ao && !ao->is_dead(), "fc_bias: disable 后 Add 保留");
            pm.enable("fc_bias_fold");
        }
    }


    // ============================================================
    // [13] view_fold(005 P1): SS∘SS 合并 / Reshape 吸收 / T∘T 恒等
    // ============================================================
    {
        // ranges const 造件: rank2 [R,3] Int32, 名含 "ranges"(find 口径)
        auto add_ranges_const = [](GraphPrepare& gp, op_id_t id,
                                   const std::vector<int32_t>& triples) -> op_id_t {
            OutputDef od{};
            od.rank = 2;
            od.dtype = static_cast<uint32_t>(DType::Int32);
            od.dims[0] = triples.size() / 3; od.dims[1] = 3;
            od.element_size = 4;
            std::vector<uint8_t> data(triples.size() * 4);
            std::memcpy(data.data(), triples.data(), data.size());
            op_id_t rid = gp.append_const_node(id, od, data.data(), data.size());
            OpDef* c = gp.get_op_at(rid);
            if (c) c->name_tag = string_tag_t::map_str("Slice_ranges");
            return rid;
        };
        auto attach_ranges = [&](OpDef* ss, op_id_t rid) {
            ss->tensor_param_ids.push_back(rid);
        };
        auto read_ranges_i32 = [&](GraphPrepare& gp, OpDef* ss) -> std::vector<int32_t> {
            std::vector<int32_t> v;
            for (op_id_t tpid : ss->tensor_param_ids) {
                OpDef* pc = gp.get_op_at(tpid);
                if (!pc || !pc->name_tag || !std::strstr(pc->name_tag->name(), "ranges")) continue;
                if (pc->const_data_offset + pc->const_data_size > gp.const_pool().size()) break;
                v.resize(pc->const_data_size / 4);
                std::memcpy(v.data(), gp.const_pool().data() + pc->const_data_offset,
                            pc->const_data_size);
                break;
            }
            return v;
        };

        std::printf("=== [13a] view_fold R1 正例: SS∘SS 区间合成 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 8, 1));            // [1,1,8,1] 8 元素
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,1,1, 2,6,1, 0,1,1});
            auto rg2 = add_ranges_const(gp, 2, {0,1,1, 0,1,1, 1,3,1, 0,1,1});
            auto ss1 = add_unary_op(gp, 10, "StridedSlice", {in}, f32_def(1, 1, 4, 1));
            auto ss2 = add_unary_op(gp, 11, "StridedSlice", {ss1}, f32_def(1, 1, 2, 1));
            attach_ranges(gp.get_op_at(ss1), rg1);
            attach_ranges(gp.get_op_at(ss2), rg2);
            add_output(gp, 20, ss2);
            gp.run_optimize_passes(env);

            OpDef* o1 = gp.get_op_at(ss1);
            OpDef* o2 = gp.get_op_at(ss2);
            CHECK(!o1 || o1->is_dead(), "view_fold R1: 内层 SS 标 dead");
            CHECK(o2 && !o2->is_dead(), "view_fold R1: 外层 SS 存活");
            CHECK(o2 && o2->inputs.size() >= 1 && o2->inputs[0].src_id == in,
                  "view_fold R1: 外层 SS 直连 Input");
            auto rg = o2 ? read_ranges_i32(gp, o2) : std::vector<int32_t>{};
            CHECK(rg == std::vector<int32_t>({0,1,1, 0,1,1, 3,5,1, 0,1,1}),
                  "view_fold R1: 合成 ranges = [3,5) 步长 1(2+1*1 起, cnt=2)");
        }

        std::printf("=== [13b] view_fold R1 恒等: 双全区间 → 整体旁路 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 8, 1));
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,1,1, 0,8,1, 0,1,1});
            auto rg2 = add_ranges_const(gp, 2, {0,1,1, 0,1,1, 0,8,1, 0,1,1});
            auto ss1 = add_unary_op(gp, 10, "StridedSlice", {in}, f32_def(1, 1, 8, 1));
            auto ss2 = add_unary_op(gp, 11, "StridedSlice", {ss1}, f32_def(1, 1, 8, 1));
            attach_ranges(gp.get_op_at(ss1), rg1);
            attach_ranges(gp.get_op_at(ss2), rg2);
            auto out = add_output(gp, 20, ss2);
            gp.run_optimize_passes(env);

            CHECK(!gp.get_op_at(ss1) || gp.get_op_at(ss1)->is_dead(), "view_fold: 恒等链内层删");
            CHECK(!gp.get_op_at(ss2) || gp.get_op_at(ss2)->is_dead(), "view_fold: 恒等链外层删");
            OpDef* oo = gp.get_op_at(out);
            CHECK(oo && oo->inputs.size() == 1 && oo->inputs[0].src_id == in,
                  "view_fold: 恒等链 Output 直连 Input");
        }

        std::printf("=== [13c] view_fold R1 反例: 中间张量旁路不合并 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 8, 1));
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,1,1, 2,6,1, 0,1,1});
            auto rg2 = add_ranges_const(gp, 2, {0,1,1, 0,1,1, 1,3,1, 0,1,1});
            auto ss1 = add_unary_op(gp, 10, "StridedSlice", {in}, f32_def(1, 1, 4, 1));
            auto ss2 = add_unary_op(gp, 11, "StridedSlice", {ss1}, f32_def(1, 1, 2, 1));
            attach_ranges(gp.get_op_at(ss1), rg1);
            attach_ranges(gp.get_op_at(ss2), rg2);
            auto c = add_const_f32(gp, 3, f32_def(), {1.0f});
            auto eb = add_unary_op(gp, 12, "Eltwise_Binary", {ss2, ss1, c},
                                   f32_def(1, 1, 2, 1));              // ss1 旁路消费者
            add_output(gp, 20, eb);
            gp.run_optimize_passes(env);

            OpDef* o1 = gp.get_op_at(ss1);
            OpDef* o2 = gp.get_op_at(ss2);
            CHECK(o1 && !o1->is_dead(), "view_fold 反例: 旁路时内层保留");
            CHECK(o2 && o2->inputs.size() >= 1 && o2->inputs[0].src_id == ss1,
                  "view_fold 反例: 外层仍接内层");
        }

        std::printf("=== [13d] view_fold R2: 纯 1 维增删 Reshape 吸收 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 2, 4, 1));            // [1,2,4,1]
            auto rs = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 2, 1, 4)); // [1,2,1,4] 中插 1
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,2,1, 0,1,1, 1,3,1});
            auto ss = add_unary_op(gp, 11, "StridedSlice", {rs}, f32_def(1, 2, 1, 2));
            attach_ranges(gp.get_op_at(ss), rg1);
            add_output(gp, 20, ss);
            gp.run_optimize_passes(env);

            OpDef* rso = gp.get_op_at(rs);
            OpDef* sso = gp.get_op_at(ss);
            CHECK(!rso || rso->is_dead(), "view_fold R2: Reshape 被吸收(标 dead)");
            CHECK(sso && !sso->is_dead(), "view_fold R2: SS 存活");
            CHECK(sso && sso->inputs.size() >= 1 && sso->inputs[0].src_id == in,
                  "view_fold R2: SS 直连底层 Input");
            auto rg = sso ? read_ranges_i32(gp, sso) : std::vector<int32_t>{};
            CHECK(rg == std::vector<int32_t>({0,1,1, 0,2,1, 1,3,1, 0,1,1}),
                  "view_fold R2: 落底层帧 ranges(新增 1 轴消解, 尾 1 轴补全)");
        }

        std::printf("=== [13e] view_fold R2 反例: 跨维重排 Reshape 不吸收 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 2, 4, 1));            // [1,2,4,1]
            auto rs = add_unary_op(gp, 10, "Reshape", {in}, f32_def(1, 4, 2, 1)); // 跨维重排
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,4,1, 0,2,1, 0,1,1});
            auto ss = add_unary_op(gp, 11, "StridedSlice", {rs}, f32_def(1, 4, 2, 1));
            attach_ranges(gp.get_op_at(ss), rg1);
            add_output(gp, 20, ss);
            gp.run_optimize_passes(env);

            OpDef* rso = gp.get_op_at(rs);
            CHECK(rso && !rso->is_dead(), "view_fold R2 反例: 跨维重排保留");
        }

        std::printf("=== [13f] view_fold R3: Transpose∘Transpose 恒等对 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 2, 3, 1));            // [1,2,3,1]
            auto p1 = add_ranges_const(gp, 1, {1,0,2,3});
            auto p2 = add_ranges_const(gp, 2, {1,0,2,3});
            auto t1 = add_unary_op(gp, 10, "Transpose", {in}, f32_def(1, 2, 3, 1));
            // perm [1,0,2,3] 对 [1,2,3,1]: 轴 1/2 互换 → [1,3,2,1]
            OpDef* t1o = gp.get_op_at(t1);
            t1o->output_def = f32_def(1, 3, 2, 1);
            auto t2 = add_unary_op(gp, 11, "Transpose", {t1}, f32_def(1, 2, 3, 1));
            // 再换回来 → [1,2,3,1]
            for (op_id_t pid : {p1, p2}) {
                OpDef* pc = gp.get_op_at(pid);
                if (pc) pc->name_tag = string_tag_t::map_str("Transpose_perm");
            }
            gp.get_op_at(t1)->tensor_param_ids.push_back(p1);
            gp.get_op_at(t2)->tensor_param_ids.push_back(p2);
            auto out = add_output(gp, 20, t2);
            gp.run_optimize_passes(env);

            CHECK(!gp.get_op_at(t1) || gp.get_op_at(t1)->is_dead(), "view_fold R3: 内层 T 删");
            CHECK(!gp.get_op_at(t2) || gp.get_op_at(t2)->is_dead(), "view_fold R3: 外层 T 删");
            OpDef* oo = gp.get_op_at(out);
            CHECK(oo && oo->inputs.size() == 1 && oo->inputs[0].src_id == in,
                  "view_fold R3: Output 直连 Input");
        }

        std::printf("=== [13g] view_fold 幂等: 二跑零改写 + 逃生门 ===\n");
        {
            PassManager& pm = PassManager::instance();
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 8, 1));
            auto rg1 = add_ranges_const(gp, 1, {0,1,1, 0,1,1, 2,6,1, 0,1,1});
            auto rg2 = add_ranges_const(gp, 2, {0,1,1, 0,1,1, 1,3,1, 0,1,1});
            auto ss1 = add_unary_op(gp, 10, "StridedSlice", {in}, f32_def(1, 1, 4, 1));
            auto ss2 = add_unary_op(gp, 11, "StridedSlice", {ss1}, f32_def(1, 1, 2, 1));
            attach_ranges(gp.get_op_at(ss1), rg1);
            attach_ranges(gp.get_op_at(ss2), rg2);
            add_output(gp, 20, ss2);
            gp.run_optimize_passes(env);
            gp.run_optimize_passes(env);                               // 二跑
            std::string st = pm.stats_line();
            CHECK(st.find("view_fold=0/0") != std::string::npos,
                  "view_fold 幂等: 二跑零改写");

            // 逃生门: --no-pass view_fold 等价 disable 后链保持
            pm.disable("view_fold");
            GraphPrepare gp2;
            auto in2 = add_input(gp2, 5, f32_def(1, 1, 8, 1));
            auto r21 = add_ranges_const(gp2, 1, {0,1,1, 0,1,1, 2,6,1, 0,1,1});
            auto r22 = add_ranges_const(gp2, 2, {0,1,1, 0,1,1, 1,3,1, 0,1,1});
            auto s21 = add_unary_op(gp2, 10, "StridedSlice", {in2}, f32_def(1, 1, 4, 1));
            auto s22 = add_unary_op(gp2, 11, "StridedSlice", {s21}, f32_def(1, 1, 2, 1));
            attach_ranges(gp2.get_op_at(s21), r21);
            attach_ranges(gp2.get_op_at(s22), r22);
            add_output(gp2, 20, s22);
            gp2.run_optimize_passes(env);
            OpDef* o21 = gp2.get_op_at(s21);
            CHECK(o21 && !o21->is_dead(), "view_fold 逃生: disable 后链保持");
            pm.enable("view_fold");
        }
    }


    // ============================================================
    // [14] eltwise_const_fold(005 P2): 同型合并 + Div→Mul
    // ============================================================
    {
        auto set_eb_oper = [](OpDef* od, int code) {
            auto sp = unpack_scalar_params(od->op_data);
            bool found = false;
            for (auto& p : sp)
                if (p.name == "operation") {
                    p.is_numeric = true; p.value_num = (double)code; p.value_str.clear();
                    found = true;
                }
            if (!found) {
                ScalarParam p; p.name = "operation"; p.is_numeric = true;
                p.value_num = (double)code;
                sp.push_back(std::move(p));
            }
            od->op_data = pack_scalar_params(sp);
        };
        auto get_eb_oper = [](const OpDef* od) -> int {
            auto sp = unpack_scalar_params(od->op_data);
            const ScalarParam* p = scalar_get(sp, "operation");
            return p && p->is_numeric ? (int)p->value_num : 0;
        };
        auto const_val_at = [](GraphPrepare& gp, op_id_t cid) -> float {
            OpDef* c = gp.get_op_at(cid);
            if (!c || c->const_data_size < 4) return -1e30f;
            float v;
            std::memcpy(&v, gp.const_pool().data() + c->const_data_offset, 4);
            return v;
        };

        std::printf("=== [14a] 同型 ADD∘ADD 合并(c1+c2) ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 4, 1));
            auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
            auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
            auto a1 = add_unary_op(gp, 10, "Eltwise_Binary", {in, c1}, f32_def(1, 1, 4, 1));
            auto a2 = add_unary_op(gp, 11, "Eltwise_Binary", {a1, c2}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp.get_op_at(a1), 0);
            set_eb_oper(gp.get_op_at(a2), 0);
            add_output(gp, 20, a2);
            gp.run_optimize_passes(env);

            OpDef* o1 = gp.get_op_at(a1);
            OpDef* o2 = gp.get_op_at(a2);
            CHECK(!o1 || o1->is_dead(), "eltwise: 内层 ADD 标 dead");
            CHECK(o2 && !o2->is_dead(), "eltwise: 外层 ADD 存活");
            float merged = (o2 && o2->inputs.size() == 2) ? const_val_at(gp, o2->inputs[1].src_id) : -1e30f;
            CHECK(std::fabs(merged - 5.0f) < 1e-6f, "eltwise: 合成 const = 2+3=5");
            CHECK(o2 && o2->inputs.size() == 2 && o2->inputs[0].src_id == in,
                  "eltwise: 外层数据输入直连 Input");
        }

        std::printf("=== [14b] Div→Mul(1/c, f32 口径; 默认关需显式开) ===\n");
        {
            PassManager::instance().enable("eltwise_div_to_mul");
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 4, 1));
            auto c = add_const_f32(gp, 1, f32_def(), {4.0f});
            auto dv = add_unary_op(gp, 10, "Eltwise_Binary", {in, c}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp.get_op_at(dv), 2);                    // DIVIDE
            add_output(gp, 20, dv);
            gp.run_optimize_passes(env);
            PassManager::instance().disable("eltwise_div_to_mul");

            OpDef* d = gp.get_op_at(dv);
            CHECK(d && !d->is_dead() && get_eb_oper(d) == 13,
                  "eltwise: DIVIDE → MULTIPLY(13)");
            float inv = (d && d->inputs.size() == 2) ? const_val_at(gp, d->inputs[1].src_id) : -1e30f;
            CHECK(std::fabs(inv - 0.25f) < 1e-6f, "eltwise: const 4 → 1/4=0.25");
        }

        std::printf("=== [14c] 反例: 除数零 / 旁路 / int32 / 默认关 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 4, 1));
            auto c0 = add_const_f32(gp, 1, f32_def(), {0.0f});
            auto dv = add_unary_op(gp, 10, "Eltwise_Binary", {in, c0}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp.get_op_at(dv), 2);
            add_output(gp, 20, dv);
            gp.run_optimize_passes(env);
            OpDef* d = gp.get_op_at(dv);
            CHECK(d && get_eb_oper(d) == 2, "eltwise 反例: 除数 0 不转(显式开后)");

            GraphPrepare gp0;
            auto in0 = add_input(gp0, 5, f32_def(1, 1, 4, 1));
            auto c40 = add_const_f32(gp0, 1, f32_def(), {4.0f});
            auto dv0 = add_unary_op(gp0, 10, "Eltwise_Binary", {in0, c40}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp0.get_op_at(dv0), 2);
            add_output(gp0, 20, dv0);
            gp0.run_optimize_passes(env);
            OpDef* d0 = gp0.get_op_at(dv0);
            CHECK(d0 && get_eb_oper(d0) == 2, "eltwise 反例: 默认关时 Div 不转 Mul");
            PassManager::instance().enable("eltwise_div_to_mul");

            GraphPrepare gp2;
            auto in2 = add_input(gp2, 5, f32_def(1, 1, 4, 1));
            auto cb = add_const_f32(gp2, 1, f32_def(), {1.0f});
            auto b1 = add_unary_op(gp2, 10, "Eltwise_Binary", {in2, cb}, f32_def(1, 1, 4, 1));
            auto b2 = add_unary_op(gp2, 11, "Eltwise_Binary", {b1, cb}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp2.get_op_at(b1), 0);
            set_eb_oper(gp2.get_op_at(b2), 0);
            auto c3 = add_const_f32(gp2, 3, f32_def(), {7.0f});
            auto b3 = add_unary_op(gp2, 12, "Eltwise_Binary", {b1, c3}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp2.get_op_at(b3), 0);
            add_output(gp2, 21, b2);
            add_output(gp2, 22, b3);
            gp2.run_optimize_passes(env);
            OpDef* o1 = gp2.get_op_at(b1);
            CHECK(o1 && !o1->is_dead(), "eltwise 反例: 旁路消费者时内层保留");
        }

        std::printf("=== [14e] const 解包: Reshape 包装(前导 1) ===\n");
        {
            // 外层 const 经 [1]→[1,1] Reshape 包装(0.8B 实图形态): 解包后合并
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 4, 1));
            auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
            auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
            auto r = add_unary_op(gp, 3, "Reshape", {c2}, f32_def(1, 1, 1, 1));
            auto a1 = add_unary_op(gp, 10, "Eltwise_Binary", {in, c1}, f32_def(1, 1, 4, 1));
            auto a2 = add_unary_op(gp, 11, "Eltwise_Binary", {a1, r}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp.get_op_at(a1), 0);
            set_eb_oper(gp.get_op_at(a2), 0);
            add_output(gp, 20, a2);
            gp.run_optimize_passes(env);

            OpDef* o1 = gp.get_op_at(a1);
            OpDef* o2 = gp.get_op_at(a2);
            OpDef* rr = gp.get_op_at(r);
            CHECK(o1 && o1->is_dead(), "unwrap: 内层 ADD 标 dead");
            CHECK(o2 && !o2->is_dead(), "unwrap: 外层 ADD 存活");
            float merged = (o2 && o2->inputs.size() == 2) ? const_val_at(gp, o2->inputs[1].src_id) : -1e30f;
            CHECK(std::fabs(merged - 5.0f) < 1e-6f, "unwrap: 合成 const = 5(透过 Reshape)");
            {
                OpDef* cin = gp.get_op_at(o2->inputs[1].src_id);
                std::string cn = (cin && cin->name_tag) ? cin->name_tag->name() : "";
                CHECK(cn != "Reshape" || !cin->name_tag, "unwrap: 外层 const 输入不再指向 Reshape");
            }
            (void)rr;

            // 反例: [4]→[4,1] 改变右对齐广播语义, 不解包不合并
            GraphPrepare gp2;
            auto in2 = add_input(gp2, 5, f32_def(1, 1, 4, 1));
            auto d1 = add_const_f32(gp2, 1, f32_def(), {2.0f});
            auto d2 = add_const_f32(gp2, 2, f32_def(4), {3.0f, 3.0f, 3.0f, 3.0f});
            auto r2 = add_unary_op(gp2, 3, "Reshape", {d2}, f32_def(4, 1));
            auto e1 = add_unary_op(gp2, 10, "Eltwise_Binary", {in2, d1}, f32_def(1, 1, 4, 1));
            auto e2 = add_unary_op(gp2, 11, "Eltwise_Binary", {e1, r2}, f32_def(4, 1));
            set_eb_oper(gp2.get_op_at(e1), 0);
            set_eb_oper(gp2.get_op_at(e2), 0);
            add_output(gp2, 21, e2);
            gp2.run_optimize_passes(env);
            OpDef* i1 = gp2.get_op_at(e1);
            CHECK(i1 && !i1->is_dead(), "unwrap 反例: [4]→[4,1] 不解包不合并");
        }

        std::printf("=== [14d] 幂等: 二跑零改写 ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, f32_def(1, 1, 4, 1));
            auto c1 = add_const_f32(gp, 1, f32_def(), {2.0f});
            auto c2 = add_const_f32(gp, 2, f32_def(), {3.0f});
            auto a1 = add_unary_op(gp, 10, "Eltwise_Binary", {in, c1}, f32_def(1, 1, 4, 1));
            auto a2 = add_unary_op(gp, 11, "Eltwise_Binary", {a1, c2}, f32_def(1, 1, 4, 1));
            set_eb_oper(gp.get_op_at(a1), 0);
            set_eb_oper(gp.get_op_at(a2), 0);
            add_output(gp, 20, a2);
            gp.run_optimize_passes(env);
            gp.run_optimize_passes(env);
            std::string st = PassManager::instance().stats_line();
            CHECK(st.find("eltwise_const_fold=0/0") != std::string::npos,
                  "eltwise 幂等: 二跑零改写");
        }
    }

    // [15] fc_bias_fold US3a: MatMul 链头(非对齐折/对齐挡/batched 挡)
    {
        auto def_r = [](std::initializer_list<uint32_t> dims) {
            OutputDef od{};
            od.rank = (uint32_t)dims.size();
            od.dtype = static_cast<uint32_t>(DType::Float32);
            uint32_t i = 0;
            for (uint32_t d : dims) od.dims[i++] = d;
            od.element_size = 4;
            return od;
        };
        auto set_eb_oper = [](OpDef* od, int code) {
            auto sp = unpack_scalar_params(od->op_data);
            bool found = false;
            for (auto& p : sp)
                if (p.name == "operation") {
                    p.is_numeric = true; p.value_num = (double)code; p.value_str.clear();
                    found = true;
                }
            if (!found) {
                ScalarParam p; p.name = "operation"; p.is_numeric = true;
                p.value_num = (double)code;
                sp.push_back(std::move(p));
            }
            od->op_data = pack_scalar_params(sp);
        };
        auto mark_has_bias = [](GraphPrepare& gp, op_id_t id) -> bool {
            OpDef* od = gp.get_op_at(id);
            if (!od) return false;
            auto sp = unpack_scalar_params(od->op_data);
            const ScalarParam* p = scalar_get(sp, "has_bias");
            return p && p->is_numeric && p->value_num != 0;
        };

        std::printf("=== [15a] 非对齐 MatMul+bias 折叠(M=3) ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, def_r({3, 100}));
            auto w = add_const_f32(gp, 1, def_r({100, 64}), std::vector<float>(100 * 64, 0.5f));
            auto b = add_const_f32(gp, 2, def_r({64}), std::vector<float>(64, 0.25f));
            auto mm = add_unary_op(gp, 30, "MatMul", {in, w}, def_r({3, 64}));
            auto ad = add_unary_op(gp, 31, "Eltwise_Binary", {mm, b}, def_r({3, 64}));
            set_eb_oper(gp.get_op_at(ad), 0);
            add_output(gp, 20, ad);
            gp.run_optimize_passes(env);

            OpDef* m = gp.get_op_at(mm);
            OpDef* a = gp.get_op_at(ad);
            CHECK(a && a->is_dead(), "matmul-bias: Add 标 dead");
            CHECK(m && !m->is_dead() && m->inputs.size() == 3 &&
                      m->inputs[2].src_id == b,
                  "matmul-bias: bias 并入 MatMul inputs[2]");
            CHECK(mark_has_bias(gp, mm), "matmul-bias: op_data has_bias=1 标记");
        }

        std::printf("=== [15d] 数值门: 折/不折 execute_host 三方对拍 ===\n");
        {
            const uint32_t M = 3, K = 100, N = 64;
            std::vector<float> x(M * K), w(K * N), b(N);
            for (uint32_t i = 0; i < x.size(); ++i) x[i] = -1.0f + 2.0f * ((i * 7 + 3) % 97) / 97.0f;
            for (uint32_t i = 0; i < w.size(); ++i) w[i] = -0.5f + ((i * 13 + 5) % 89) / 89.0f;
            for (uint32_t i = 0; i < b.size(); ++i) b[i] = -2.0f + 4.0f * ((i * 11 + 1) % 71) / 71.0f;

            auto build = [&](GraphPrepare& gp) {
                auto in = add_input(gp, 5, def_r({M, K}));
                auto wc = add_const_f32(gp, 1, def_r({K, N}), w);
                auto bc = add_const_f32(gp, 2, def_r({N}), b);
                auto mm = add_unary_op(gp, 30, "MatMul", {in, wc}, def_r({M, N}));
                auto ad = add_unary_op(gp, 31, "Eltwise_Binary", {mm, bc}, def_r({M, N}));
                set_eb_oper(gp.get_op_at(ad), 0);
                add_output(gp, 20, ad);
                return ad;
            };

            GraphPrepare gp_nofold;
            build(gp_nofold);
            PassManager::instance().disable("fc_bias_fold");
            gp_nofold.run_optimize_passes(env);
            PassManager::instance().enable("fc_bias_fold");
            auto r0 = gp_nofold.execute_host(x);

            GraphPrepare gp_fold;
            build(gp_fold);
            gp_fold.run_optimize_passes(env);
            auto r1 = gp_fold.execute_host(x);

            // numpy 同式参考: X[M,K] @ W[K,N] + B[N]
            std::vector<float> ref(M * N);
            for (uint32_t i = 0; i < M; ++i)
                for (uint32_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (uint32_t p = 0; p < K; ++p) acc += x[i * K + p] * w[p * N + j];
                    ref[i * N + j] = acc + b[j];
                }

            CHECK(r0.ok && r1.ok && r0.output.size() == M * N,
                  "matmul-bias 数值: execute_host 双跑成功");
            double max_d = 0.0;
            uint32_t n_fold_diff = 0;
            for (uint32_t i = 0; i < M * N; ++i) {
                double d1 = std::fabs((double)r1.output[i] - ref[i]);
                if (d1 > max_d) max_d = d1;
                if (r0.output[i] != r1.output[i]) n_fold_diff++;
            }
            double scale = 1.0;
            for (double v : ref) scale = std::max(scale, std::fabs((double)v));
            CHECK(max_d / scale < 1e-5,
                  "matmul-bias 数值: 折后 vs 手算参考 (1e-5 相对)");
            CHECK(n_fold_diff == 0,
                  "matmul-bias 数值: 折/不折 execute_host 位级一致");
        }

        std::printf("=== [15b] 反例: 32 对齐不折(HMX 门) ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, def_r({64, 64}));
            auto w = add_const_f32(gp, 1, def_r({64, 64}), std::vector<float>(64 * 64, 0.5f));
            auto b = add_const_f32(gp, 2, def_r({64}), std::vector<float>(64, 0.25f));
            auto mm = add_unary_op(gp, 30, "MatMul", {in, w}, def_r({64, 64}));
            auto ad = add_unary_op(gp, 31, "Eltwise_Binary", {mm, b}, def_r({64, 64}));
            set_eb_oper(gp.get_op_at(ad), 0);
            add_output(gp, 20, ad);
            gp.run_optimize_passes(env);
            OpDef* m = gp.get_op_at(mm);
            OpDef* a = gp.get_op_at(ad);
            CHECK(a && !a->is_dead(), "matmul-bias 反例: 32 对齐不折(防 HMX 倒退)");
            CHECK(m && m->inputs.size() == 2, "matmul-bias 反例: bias 未并入");
        }

        std::printf("=== [15c] 反例: batched BMM 不折(设备硬错防御) ===\n");
        {
            GraphPrepare gp;
            auto in = add_input(gp, 5, def_r({2, 3, 100}));
            auto w = add_const_f32(gp, 1, def_r({2, 100, 64}), std::vector<float>(2 * 100 * 64, 0.5f));
            auto b = add_const_f32(gp, 2, def_r({64}), std::vector<float>(64, 0.25f));
            auto mm = add_unary_op(gp, 30, "MatMul", {in, w}, def_r({2, 3, 64}));
            auto ad = add_unary_op(gp, 31, "Eltwise_Binary", {mm, b}, def_r({2, 3, 64}));
            set_eb_oper(gp.get_op_at(ad), 0);
            add_output(gp, 20, ad);
            gp.run_optimize_passes(env);
            OpDef* m = gp.get_op_at(mm);
            OpDef* a = gp.get_op_at(ad);
            CHECK(a && !a->is_dead(), "matmul-bias 反例: batched 不折");
            CHECK(m && m->inputs.size() == 2, "matmul-bias 反例: batched bias 未并入");
        }
    }

    std::printf("\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
