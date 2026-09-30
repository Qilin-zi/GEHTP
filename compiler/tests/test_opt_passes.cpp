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

    std::printf("\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
