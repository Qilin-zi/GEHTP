#include "hnnx/ir/types.hpp"
#include "hnnx/ir/tensor.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/mcast/mcast_optimizer.hpp"
#include "hnnx/scheduler/dp_sequencer.hpp"
#include "hnnx/tiling/tiler.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace hnnx;

    // Test 1: Fibonacci hash
    auto h = fibonacci_hash(0x1234567890ABCDEF);
    assert(h != 0);
    std::cout << "Fibonacci hash: 0x" << std::hex << h << std::dec << "\n";

    // Test 2: OpDef_Const construction
    GraphPrepare gp;
    OutputDef od{};
    od.rank = 1;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    od.dims[0] = 1;
    OpDef_Const const_op(gp, 42, od, nullptr, 0);
    assert(const_op.op_id == 42);
    assert(const_op.is_const());
    assert(const_op.is_enabled());
    std::cout << "OpDef_Const: id=" << const_op.op_id
              << " flags=0x" << std::hex << const_op.flags << std::dec << "\n";

    // Test 3: Optimization phase thresholds
    assert(PHASE_0 == 3000);
    assert(PHASE_1 == 10190);
    assert(PHASE_2 == 11892);
    assert(PHASE_3 == 12492);
    assert(PHASE_4 == 21101);
    assert(PHASE_5 == 22000);
    assert(PHASE_TERM == 0xFFFFFFFF);
    std::cout << "Phase thresholds verified\n";

    // Test 4: Serializer tag encoding
    // (tag & 0xFFFF | tag << 16) ^ 0xFFFF
    uint32_t tag = 0xEF4D;
    uint32_t encoded = ((tag & 0xFFFF) | (tag << 16)) ^ 0xFFFF;
    std::cout << "Tag 0x" << std::hex << tag << " -> encoded 0x" << encoded << std::dec << "\n";

    // Test 5: Cost model
    costbased::CostSource cs;
    assert(cs.init_for_soc("v75"));
    std::cout << "CostSource initialized\n";

    // Test 6: DP Sequencer config
    SequencerConfig sc;
    assert(sc.svf_en);
    assert(!sc.external_sequencer);
    std::cout << "Sequencer config verified\n";

    // Test 7: VtcmCacheInstance
    VtcmCacheInstance vtcm(0, 8 * 1024 * 1024);
    assert(vtcm.size() == 8 * 1024 * 1024);
    assert(vtcm.nsp_id() == 0);
    std::cout << "VtcmCacheInstance: " << vtcm.size() << " bytes\n";

    // Test 8: Tiling registry
    auto& tiling_reg = TilingRegistry::instance();
    tiling_reg.register_tiler("Conv2d", std::make_unique<SimpleTiler>());
    auto* tiler = tiling_reg.get_tiler("Conv2d");
    assert(tiler != nullptr);
    assert(std::string(tiler->name()) == "SimpleTiler");
    std::cout << "TilingRegistry: registered SimpleTiler for Conv2d\n";

    // Test 9: Multicast optimizer
    McastOptimizer mcast_opt;
    auto result = mcast_opt.optimize({}, 0);
    assert(result.empty());
    std::cout << "McastOptimizer: basic test passed\n";

    // Test 10: Op emitter
    OpEmitter emitter(&gp);
    emitter.insert_preload_op(100, 50);
    std::cout << "OpEmitter: preload op inserted\n";

    // ===== Phase 1.3: GCP 2.x field tests =====

    // Test 11: StorageClass enum values
    assert(DDR == 0);
    assert(VTCM == 1);
    assert(CONST == 2);
    assert(VTCM_PERSISTENT == 3);
    std::cout << "StorageClass enum: DDR=0, VTCM=1, CONST=2, VTCM_PERSISTENT=3\n";

    // Test 12: OpDef GCP 2.1 default values
    {
        OpDef opdef;
        assert(opdef.op_type == 0);
        assert(opdef.sub_type == 0);
        assert(opdef.phase_id == 0);
        assert(opdef.quant_count == 0);
        assert(opdef.quant_array == nullptr);
        assert(opdef.crouton_from_vtcm == 0);
        assert(opdef.crouton_to_vtcm == 0);
        assert(opdef.tag_bitmap == 0);
        assert(opdef.priority == 0);
        assert(opdef.inputs_vec.empty());
        assert(opdef.outputs_vec.empty());
        std::cout << "OpDef GCP 2.1: all defaults verified\n";
    }

    // Test 13: OpDef GCP 2.1 field set/get
    {
        OpDef opdef;
        opdef.op_type = 0;  // MatMul
        opdef.sub_type = 1;
        opdef.phase_id = PHASE_2;
        opdef.quant_count = 4;
        opdef.crouton_from_vtcm = 1;
        opdef.crouton_to_vtcm = 1;
        opdef.priority = 100;
        assert(opdef.op_type == 0);
        assert(opdef.sub_type == 1);
        assert(opdef.phase_id == PHASE_2);
        assert(opdef.quant_count == 4);
        assert(opdef.crouton_from_vtcm == 1);
        assert(opdef.crouton_to_vtcm == 1);
        assert(opdef.priority == 100);
        std::cout << "OpDef GCP 2.1: field set/get verified\n";
    }

    // Test 14: OpDef OP_MIGRATED flag
    {
        OpDef opdef;
        opdef.flags = OP_ENABLED | OP_MIGRATED;
        assert(opdef.is_enabled());
        assert(opdef.is_migrated());
        assert(!opdef.is_dead());
        // OP_MIGRATED and OP_SLICED share bit 0x40
        assert(OP_MIGRATED == OP_SLICED);
        std::cout << "OpDef OP_MIGRATED: flag test passed\n";
    }

    // Test 15: Graph GCP 2.3 fields
    {
        Graph graph{};
        assert(graph.ops.empty());
        assert(graph.tensors.empty());
        assert(graph.graph_deps_ptr == nullptr);
        assert(graph.state_machine == 1);  // CONSTRUCTION
        assert(graph.graph_dirty == false);
        // Test mutation
        graph.state_machine = 3;  // COMPILED
        graph.graph_dirty = true;
        assert(graph.state_machine == 3);
        assert(graph.graph_dirty == true);
        std::cout << "Graph GCP 2.3: defaults and mutation verified\n";
    }

    // ===== Phase 1.4: Tensor compute tests =====

    // Test 16: Tensor::get_element_size() for each DType
    {
        hnnx::Tensor t{};
        t.dtype = static_cast<uint32_t>(DType::Float32);
        assert(t.get_element_size() == 4);
        t.dtype = static_cast<uint32_t>(DType::Float16);
        assert(t.get_element_size() == 2);
        t.dtype = static_cast<uint32_t>(DType::BFloat16);
        assert(t.get_element_size() == 2);
        t.dtype = static_cast<uint32_t>(DType::Int8);
        assert(t.get_element_size() == 1);
        t.dtype = static_cast<uint32_t>(DType::UInt8);
        assert(t.get_element_size() == 1);
        t.dtype = static_cast<uint32_t>(DType::Int16);
        assert(t.get_element_size() == 2);
        t.dtype = static_cast<uint32_t>(DType::Int32);
        assert(t.get_element_size() == 4);
        t.dtype = static_cast<uint32_t>(DType::Bool);
        assert(t.get_element_size() == 1);
        t.dtype = static_cast<uint32_t>(DType::FP8_E4M3);
        assert(t.get_element_size() == 1);
        std::cout << "Tensor::get_element_size: all DTypes verified\n";
    }

    // Test 17: Tensor::num_elements() with dims vector
    {
        hnnx::Tensor t{};
        t.dims = {1, 1, 4, 8};  // rank-4 padded MatMul shape
        assert(t.num_elements() == 32);
        t.dims = {2, 3, 4};
        assert(t.num_elements() == 24);
        t.dims = {7};
        assert(t.num_elements() == 7);
        t.dims = {};
        t.data[0] = 0;  // rank 0 -> scalar
        assert(t.num_elements() == 1);  // scalar = 1
        std::cout << "Tensor::num_elements: dims vector path verified\n";
    }

    // Test 18: Tensor::num_elements() legacy data[] fallback
    {
        hnnx::Tensor t{};
        t.dims.clear();  // force fallback to data[]
        t.data[0] = 2;   // rank = 2
        t.data[1] = 4;   // dim[0] = 4
        t.data[2] = 8;   // dim[1] = 8
        assert(t.num_elements() == 32);

        t.data[0] = 4;   // rank = 4
        t.data[1] = 1;   // dim[0] = 1
        t.data[2] = 1;   // dim[1] = 1
        t.data[3] = 4;   // dim[2] = 4
        t.data[4] = 8;   // dim[3] = 8
        assert(t.num_elements() == 32);
        std::cout << "Tensor::num_elements: legacy data[] fallback verified\n";
    }

    // Test 19: Tensor::memory_cost() and getSize()
    {
        hnnx::Tensor t{};
        t.dims = {1, 1, 4, 8};
        t.dtype = static_cast<uint32_t>(DType::Float32);  // 4 bytes
        assert(t.num_elements() == 32);
        assert(t.memory_cost() == 128);  // 32 * 4
        assert(t.getSize() == 128);

        t.dtype = static_cast<uint32_t>(DType::Int8);  // 1 byte
        assert(t.memory_cost() == 32);   // 32 * 1
        assert(t.getSize() == 32);

        t.dtype = static_cast<uint32_t>(DType::Float16);  // 2 bytes
        assert(t.memory_cost() == 64);   // 32 * 2
        assert(t.getSize() == 64);
        std::cout << "Tensor::memory_cost/getSize: Float32/Int8/Float16 verified\n";
    }

    // Test 20: Tensor GCP 2.2 fields and isVTCM()
    {
        hnnx::Tensor t{};
        assert(t.storage_class == DDR);
        assert(!t.isVTCM());
        t.storage_class = VTCM;
        assert(t.isVTCM());
        t.storage_class = VTCM_PERSISTENT;
        assert(t.isVTCM());
        assert(t.bank_mask == 0);
        assert(t.vtcm_offset == 0);
        assert(t.lifetime_start == 0);
        assert(t.lifetime_end == 0);
        assert(t.producers.empty());
        assert(t.consumers.empty());
        assert(!t.is_output);
        assert(!t.has_side_effect);
        // Test mutation
        t.id = 42;
        t.bank_mask = 0x03;
        t.vtcm_offset = 2048;
        t.lifetime_start = 5;
        t.lifetime_end = 10;
        t.producers = {1, 2};
        t.consumers = {3, 4, 5};
        assert(t.id == 42);
        assert(t.bank_mask == 0x03);
        assert(t.vtcm_offset == 2048);
        assert(t.lifetime_start == 5);
        assert(t.lifetime_end == 10);
        assert(t.producers.size() == 2);
        assert(t.consumers.size() == 3);
        std::cout << "Tensor GCP 2.2: fields and isVTCM() verified\n";
    }

    // Test 21: Large tensor memory cost (realistic Conv weight)
    {
        hnnx::Tensor t{};
        t.dims = {64, 3, 3, 3};  // Conv2D weight: 64 filters, 3x3x3
        t.dtype = static_cast<uint32_t>(DType::Float32);
        assert(t.num_elements() == 64 * 3 * 3 * 3);
        assert(t.memory_cost() == 64 * 3 * 3 * 3 * 4);  // 6912 bytes
        std::cout << "Tensor: Conv weight [64,3,3,3] Float32 = "
                  << t.memory_cost() << " bytes\n";
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}


