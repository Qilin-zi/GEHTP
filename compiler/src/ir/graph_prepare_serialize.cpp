#include <cstdio>
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/pass_view_fold.hpp"  // 005: dump_view_census
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/host_decode_state.hpp"
#include "hnnx/mcast/mcast_optimizer.hpp"
#include "hnnx/scheduler/dp_sequencer.hpp"
#include "hnnx/vtcm/dp_group_graph.hpp"
#include "hnnx/vtcm/supertile.hpp"
#include "hnnx/ir/graph_deps.hpp"
#include "hnnx/schedule/scheduler.hpp"  // Scheduler(ST-Cut 计划序, do_prepare2 计算)
#include "hnnx/tiling/conv_tiling.hpp"  // 阶段6: conv 空间分块(halo 推导)
#include <algorithm>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <cstring>


namespace hnnx {

// Serialization
// Source: graph_prepare.cc:6751
bool GraphPrepare::serialize(uint8_t* buf, size_t buf_size, size_t& out_size) const {
    // Check: serialized_loaded_ (+0x45ac) != 0 -> error
    // "Cannot serialize a graph loaded from serialization" (graph_prepare.cc:6751)
    if (serialized_loaded_) return false;

    // Check: force_barrel_ (+0x6008) != 0 -> error
    // "Cannot apply serialize_force_barrel in serialize-to-memory" (graph_prepare.cc:6755)
    if (force_barrel_) return false;

    // If buf_size == 0, use internal buffer
    // If buf_size < 0x40 or not 4-byte aligned, error
    if (buf_size != 0 && (buf_size < 0x40 || (reinterpret_cast<uintptr_t>(buf) & 3) != 0)) {
        return false;
    }

    // 真实 .so 的 GraphPrepare ctor 必持 Allocator& (FancyAllocator ctor
    // @0xf3f2c0 ← fa::RuntimeAllocator(hnnx::Allocator::Mode, Graph&)), 此处不为
    // 空; 本仓近似 GraphPrepare 的 allocator_ 无供给路径 (恒 nullptr) —— 取
    // serializer.cpp 的兜底分配器 (空实现, 仅满足 ctor @0x12f14ab 的
    // dynamic_cast<FancyAllocator&>(*alloc))。M35/M36 接真供给后移除。
    hnnx::Allocator *ser_alloc = static_cast<hnnx::Allocator *>(allocator_);
    if (ser_alloc == nullptr) ser_alloc = hnnx::default_serializer_allocator();

    // 容量安全(M1): Serializer 内存模式无边界检查,大图单缓冲直接堆溢出
    // (0.8B 图 1MB 缓冲 ASAN 实锤)。两遍: Prescan 计长 → 不足则返回所需值。
    // 两遍各用全新 Serializer(内部状态机 f_188 跨遍残留会改变输出)。
    size_t required = 0;
    {
        // 计量遍: measure_only 不受 do_serialize 内部 mode 翻转影响
        Serializer ser_prescan(*const_cast<GraphPrepare*>(this), ser_alloc,
                               reinterpret_cast<char*>(buf), buf_size);
        ser_prescan.set_measure_only(true);
        if (!do_serialize(ser_prescan)) return false;
        required = ser_prescan.current_position();
    }
    // out_size 最终对齐到 8, 计量值同步对齐, 调用方按此分配
    required = (required + 7) & ~size_t(7);
    if (buf_size == 0) {
        // 无缓冲 = 仅计量(调用方按 out_size 再分配)
        out_size = (required + 7) & ~size_t(7);
        return true;
    }
    if (required > buf_size) {
        out_size = required;  // 告诉调用方所需字节
        return false;
    }

    // 实写(全新 Serializer; do_serialize 内部自管 IO prescan)
    Serializer ser(*const_cast<GraphPrepare*>(this), ser_alloc,
                   reinterpret_cast<char*>(buf), buf_size);
    bool ok = do_serialize(ser);
    if (ok) {
        out_size = ser.current_position();
        // Align to 8 bytes: (out_size + 7) & ~7
        out_size = (out_size + 7) & ~size_t(7);
        // Fill remaining with 0x55
        if (out_size < buf_size) {
            std::memset(buf + out_size, 0x55, buf_size - out_size);
        }
    }
    return ok;
}

// do_serialize: full serialization
// Source: graph_prepare.cc, 3629 bytes at 0xF64DB0
bool GraphPrepare::do_serialize(Serializer& ser) const {
    // Check: serialized_loaded_ (+0x45ac) != 0 -> error
    // "Cannot serialize graph loaded from serialization!" (graph_prepare.cc:517)
    if (serialized_loaded_) {
        ser.set_error("Cannot serialize graph loaded from serialization!");
        return false;
    }

    // Check multi-NSP: graph_deps +0x7468, +0x160 != 0 and ser +0x10c == 0
    // "cannot use serialize method on multi-nsp graph" (graph_prepare.cc:526)

    // Write BinHeader (大端! 从真�?.bin 样本确认: 计数+偏移表格�?
    // Source: hexagon_nn_deserialize_graph 反汇�?+ test_minimal.serialized.bin
    // 无文件级 BEEF/FA00 magic; 头部�?num_graphs + num_records + 偏移
    {
        BinHeader hdr{};
        hdr.num_graphs = 1;
        hdr.num_records = static_cast<uint32_t>(const_extents_.size());
        hdr.reserved_08 = 0;
        hdr.flags = 1;
        hdr.offset_table = 0;
        hdr.block_size = const_pool_.size();
        hdr.base_address = 0;
        ser.write_tagged_record(0x4845, &hdr, sizeof(hdr));  // 'HE' header
    }

    // Write config records (tagged records):
    // Each record: write_tagged_record(tag, data, data_size)
    // Source: FUN_00f82370 (tagged_record_writer)

    // io_dma_bypass (0xEF4D) if +0x5c8c != 0
    {
        uint32_t dma_bypass = 0; // *(uint32_t*)(this + 0x5c8c)
        if (dma_bypass != 0 && dma_bypass <= 3) {
            ser.write_tagged_record(TAG_IO_DMA_BYPASS, &dma_bypass, 4);
        }
    }

    // spill_fill_instead (0x4453) if +0x58d8 != 0 and +0x62c0 >= 2
    {
        uint32_t spill_fill = 0; // *(uint32_t*)(this + 0x58d8)
        uint64_t multicast_count = 0; // *(uint64_t*)(this + 0x62c0)
        if (spill_fill != 0 && multicast_count >= 2) {
            uint32_t val = 0;
            ser.write_tagged_record(TAG_SPILL_FILL_INSTEAD, &val, 4);
        }
    }

    // extended_udma (0xD446) if +0x6144 != 0
    {
        uint8_t extended_udma = 0; // *(uint8_t*)(this + 0x6144)
        if (extended_udma != 0) {
            uint32_t val = 0;
            ser.write_tagged_record(TAG_EXTENDED_UDMA, &val, 4);
        }
    }

    // io_tensors_config (0xE347)
    {
        uint32_t io_config = 0; // *(uint32_t*)(this + 0x7470)
        ser.write_tagged_record(TAG_IO_TENSORS_CONFIG, &io_config, 4);
    }

    // extra_config (0xD352) - 0x2c bytes of extra config data
    {
        uint8_t extra_config[0x2c] = {};
        ser.write_tagged_record(TAG_EXTRA_CONFIG, extra_config, 0x2c);
    }

    // multicast_config (0xD349)
    {
        uint32_t mcast_count = 0; // calculated from op count
        ser.write_tagged_record(TAG_MULTICAST_CONFIG, &mcast_count, 4);
    }

    // serialize_io (prescan mode) - first pass to compute sizes
    uint64_t counter = 0;
    ser.set_mode(Serializer::Mode::Prescan);
    const_cast<GraphPrepare*>(this)->serialize_io(ser, counter, true);
    // Restore Write mode so config records below are actually emitted.
    // (Prescan only measures sizes; all subsequent writes must hit the buffer.)
    ser.set_mode(Serializer::Mode::Write);

    // make_plan_for_deser_by_segments if segment count > 0
    {
        uint32_t num_segments = 0; // *(uint32_t*)(this + 0x6028)
        if (num_segments > 0) {
            make_plan_for_deser_by_segments(&ser, num_segments, 0);
        }
    }

    // Write runlist counts:
    // 0x5248: vec runlist count (lVar22 = ops count)
    {
        uint32_t vec_count = 0;
        ser.write_tagged_record(TAG_NUM_SEGMENTS, &vec_count, 4);
    }
    // 0x524C: mtx runlist count (iVar24)
    {
        uint32_t mtx_count = 0;
        ser.write_tagged_record(0x524C, &mtx_count, 4);
    }
    // 0x5647: runlist segment desc (always 0x200000000)
    {
        uint64_t seg_desc = 0x200000000ULL;
        ser.write_tagged_record(TAG_RUNLIST_SEGMENT_DESC, &seg_desc, 8);
    }

    // Self-slicing config (0xC953) if +0x45ad != 0
    {
        uint8_t self_slice_loaded = 0; // *(uint8_t*)(this + 0x45ad)
        if (self_slice_loaded != 0) {
            uint32_t val = 1;
            ser.write_tagged_record(TAG_SELF_SLICING, &val, 4);
        }
    }
    // 18000 (0x4650): self-slicing mode if +0x61ed != 0
    {
        uint8_t slicing_mode = 0; // *(uint8_t*)(this + 0x61ed)
        if (slicing_mode != 0) {
            uint32_t val = slicing_mode;
            ser.write_tagged_record(18000, &val, 4);
        }
    }

    // Pass registry config (0xC955, 0xCF55) �?verified tag IDs
    // 0xC955: pass registry size (variable length)
    {
        // *(uint64_t*)(this + 0x5db0) to *(uint64_t*)(this + 0x5db8) range
        ser.write_tagged_record(TAG_PASS_REGISTRY, nullptr, 0);
    }
    // 0xCF55: another registry (variable length)
    {
        ser.write_tagged_record(TAG_ANOTHER_REGISTRY, nullptr, 0);
    }

    // Profiling config �?not using 0x5350 (that's IO counts in real .bin)
    // Profiling uses a different tag in real lib; omitted here (prof_count=0)
    {
        uint32_t prof_count = 0; // *(uint32_t*)(this + 0x6508)
        if (prof_count != 0) {
            uint64_t prof_data[2] = {prof_count, 0};
            ser.write_tagged_record(0x6508, prof_data, 8);
        }
    }

    // MC cacheable shared (0x5453) if +0x62d6 != 0
    {
        uint8_t mc_cacheable = 0; // *(uint8_t*)(this + 0x62d6)
        if (mc_cacheable != 0) {
            uint32_t mc_size = 0; // calculated from cacheable blocks
            ser.write_tagged_record(TAG_RUNLIST_AUX, &mc_size, 4);
        }
    }

    // IO counts (0x5350): [uint32 input_count][uint32 output_count] = 8 bytes
    // Source: do_serialize @ 0xf654b0 (verified tag)
    {
        uint32_t in_count = (input_node_id_ != 0) ? 1u : 0u;
        uint32_t out_count = (output_node_id_ != 0) ? 1u : 0u;
        uint32_t io_counts[2] = {in_count, out_count};
        ser.write_tagged_record(TAG_IO_COUNTS, io_counts, sizeof(io_counts));
    }

    // Graph header: counts so the deserializer can pre-allocate tables.
    // Record payload: [uint32 op_count][uint32 input_count][uint32 output_count]
    {
        uint32_t op_count = 0;
        for (auto& [id, opdef] : opdef_map_) {
            if (opdef && opdef->is_enabled() && !opdef->is_dead()) op_count++;
        }
        uint32_t in_count = (input_node_id_ != 0) ? 1u : 0u;
        uint32_t out_count = (output_node_id_ != 0) ? 1u : 0u;
        uint32_t hdr[3] = {op_count, in_count, out_count};
        ser.write_tagged_record(TAG_GRAPH_HEADER, hdr, sizeof(hdr));
    }

    // Separator 0xFA0000FA
    ser.write_uint32(0xFA0000FA);

    // Const pool: 先写 extent 描述符表 (TAG_CONST_EXTENT)，再写常量数据块�?    // �?ConstExtentDesc 结构: [op_id][offset][size][tensor_type][reserved] = 32 字节
    // Source: Deserializer::extract_const_extent_table / auxdata_read_const_extent_descriptor
    if (!const_extents_.empty()) {
        std::vector<uint8_t> ext_buf(const_extents_.size() * sizeof(ConstExtentDesc), 0);
        size_t off = 0;
        for (const auto& e : const_extents_) {
            ConstExtentDesc desc{};
            desc.op_id = e.op_id;
            desc.offset = e.offset;
            desc.size = e.size;
            desc.tensor_type = 0;  // PlainFloat 占位 (真实�?tensor 类型�?
            desc.reserved = 0;
            std::memcpy(ext_buf.data() + off, &desc, sizeof(desc));
            off += sizeof(desc);
        }
        ser.write_tagged_record(TAG_CONST_EXTENT, ext_buf.data(),
                                static_cast<int>(ext_buf.size()));
        // 常量数据�?(整块 4 字节对齐); 路线B 权重外置时跳过
        // (19.48GB 超单记录 u32 词数限, 权重字节走独立文件)
        if (!const_pool_.empty() && !const_pool_external_) {
            ser.write_tagged_record(0xCF56, const_pool_.data(),
                                    const_pool_.size());
        }
    }

    // Block table: VTCM block_id -> (pool_id, offset, size) 映射�?    // 每个 entry: [uint32 block_id][uint32 pool_id][uint64 offset][uint64 size]
    if (!block_table_.empty()) {
        std::vector<uint8_t> bt_buf(block_table_.size() * 24, 0);
        size_t off = 0;
        for (const auto& b : block_table_) {
            std::memcpy(bt_buf.data() + off, &b.block_id, 4); off += 4;
            std::memcpy(bt_buf.data() + off, &b.pool_id, 4); off += 4;
            std::memcpy(bt_buf.data() + off, &b.offset, 8); off += 8;
            std::memcpy(bt_buf.data() + off, &b.size, 8); off += 8;
        }
        ser.write_tagged_record(0x4254, bt_buf.data(),  // 'BT' block table
                                static_cast<int>(bt_buf.size()));
    }

    // Segment plan: record segment offsets for incremental loading.
    if (!segment_plans_.empty()) {
        std::vector<uint8_t> sp_buf(segment_plans_.size() * 24, 0);
        size_t off = 0;
        for (const auto& s : segment_plans_) {
            std::memcpy(sp_buf.data() + off, &s.segment_index, 4); off += 4;
            std::memcpy(sp_buf.data() + off, &s.op_count, 4); off += 4;
            std::memcpy(sp_buf.data() + off, &s.byte_offset, 8); off += 8;
            off += 8;  // reserved
        }
        ser.write_tagged_record(0x5347, sp_buf.data(),  // 'SG' segment plan
                                static_cast<int>(sp_buf.size()));
    }

    // serialize_io (write mode) - actual write
    // Mode already restored to Write after the prescan above.
    counter = 0;
    const_cast<GraphPrepare*>(this)->serialize_io(ser, counter, false);

    // Write runlist ops
    // For each op in runlist: serialize_internal
    // Source: serialize_oplist.cc, Op::serialize_internal
    // We serialize from the OpDef (graph structure); the execution Op is
    // regenerated by the op factory at load time. Iterate in topological
    // order when available, otherwise fall back to map order.
    // 确定性发射: 按 op_id 升序(unordered_map 迭代序随进程 ASLR 变化,
    // deserialize 后重建的 map 序也可能不同 → round-trip 字节确定性
    // 必须显式排序; 设备侧执行序由 wtop_emit/引擎的拓扑处理保证)。
    // 发射序 = Scheduler 计划序(ST-Cut, do_prepare2 计算); 未调度(空)则
    // 回退 op_id 升序(确定性)。计划序随后以 TAG_PLAN_ORDER 记录写出,
    // deserialize 回读 → re-serialize 字节确定性。
    std::vector<op_id_t> emit_order = plan_order_;
    if (emit_order.empty()) {
        for (auto& [id, opdef] : opdef_map_) {
            if (opdef && opdef->is_enabled() && !opdef->is_dead())
                emit_order.push_back(id);
        }
        std::sort(emit_order.begin(), emit_order.end());
    }
    for (op_id_t id : emit_order) {
        auto it = opdef_map_.find(id);
        if (it == opdef_map_.end() || !it->second) continue;
        if (!it->second->is_enabled() || it->second->is_dead()) continue;
        serialize_opdef(ser, *it->second);
    }
    // TAG_PLAN_ORDER: [u32 count][u32 ids...](count=0 也写出, 显式"未调度")
    {
        std::vector<uint8_t> pl(4 + emit_order.size() * 4, 0);
        uint32_t cnt = static_cast<uint32_t>(emit_order.size());
        std::memcpy(pl.data(), &cnt, 4);
        for (size_t i = 0; i < emit_order.size(); i++) {
            uint32_t v = static_cast<uint32_t>(emit_order[i]);
            std::memcpy(pl.data() + 4 + i * 4, &v, 4);
        }
        ser.write_tagged_record(TAG_PLAN_ORDER, pl.data(), static_cast<int>(pl.size()));
    }

    // TAG_MEM_PLAN(M2 内存规划收编): 编译器定稿的静态内存规划, wtop_emit 照抄。
    // [u32 ver=1][u32 ddr_cap][u32 vtcm_cap][u32 bump_reserve][u32 pin_cap][u32 n_entries]
    //   n×20B {u32 op_id}{u32 out_idx}{u32 offset}{u32 size}{u32 flags bit0=in_vtcm}
    // [u32 n_spills] n×16B {u32 op_id}{u32 spill_off}{u32 size}{u32 pad}
    if (has_mem_plan_) {
        const MemPlan& mp = mem_plan_;
        std::vector<uint8_t> buf(24 + mp.entries.size() * 20 + 4 + mp.spills.size() * 16, 0);
        size_t o = 0;
        auto w32 = [&](uint32_t v) { std::memcpy(buf.data() + o, &v, 4); o += 4; };
        w32(1);
        w32(static_cast<uint32_t>(mp.ddr_cap));
        w32(static_cast<uint32_t>(mp.vtcm_cap));
        w32(static_cast<uint32_t>(mp.bump_reserve));
        w32(static_cast<uint32_t>(mp.pin_cap));
        w32(static_cast<uint32_t>(mp.entries.size()));
        for (const auto& e : mp.entries) {
            w32(static_cast<uint32_t>(e.op_id));
            w32(e.out_idx);
            w32(static_cast<uint32_t>(e.offset));
            w32(static_cast<uint32_t>(e.size));
            w32(e.in_vtcm ? 1u : 0u);
        }
        w32(static_cast<uint32_t>(mp.spills.size()));
        for (const auto& s : mp.spills) {
            w32(static_cast<uint32_t>(s.op_id));
            w32(static_cast<uint32_t>(s.offset));
            w32(static_cast<uint32_t>(s.size));
            w32(0);
        }
        ser.write_tagged_record(TAG_MEM_PLAN, buf.data(), static_cast<int>(buf.size()));
    }

    // Spill/fill(阶段7): 溢出张量 → 0x4453 配置记录 + 每张量 0x5346 DMA 记录。
    // M3 起重定语义为「成本模型观察记录」: 真实溢出决策已收编进 TAG_MEM_PLAN
    // (compute_ddr_offsets → mp.spills), 本节仍序列化 greedy allocator spill 集
    // (vtcm_allocations_ 的 spilled 标记 + tcm_migration SPILL_TO_DDR=0x40 标记),
    // 供成本模型活跃集观察与 round-trip 确定性, wtop_emit 不再消费为发射语义。
    // DDR 池偏移按 op_id 升序确定性分配(128B 对齐, 起 0x1000)。
    // 反序列化回读 spill_fill_recs_, 优先复用(round-trip 确定性)。
    {
        std::vector<SpillFillRec> recs;
        if (!spill_fill_recs_.empty()) {
            recs = spill_fill_recs_;
        } else {
            auto tensor_size = [&](op_id_t id) -> uint64_t {
                const OpDef* od = get_op_at(id);
                if (!od) return 0;
                uint64_t sz = 1;
                for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d)
                    if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
                return sz * (od->output_def.element_size ? od->output_def.element_size : 4);
            };
            for (const auto& [id, a] : vtcm_allocations_) {
                if (a.spilled) recs.push_back({id, a.block_id, 0, tensor_size(id)});
            }
            constexpr uint32_t SPILL_TO_DDR = 0x40;
            for (auto& [id, opdef] : opdef_map_) {
                if (!opdef || !(opdef->flags2 & SPILL_TO_DDR)) continue;
                bool dup = false;
                for (const auto& r : recs) dup |= (r.op_id == id);
                if (!dup) recs.push_back({id, 0, 0, tensor_size(id)});
            }
            std::sort(recs.begin(), recs.end(),
                      [](const SpillFillRec& a, const SpillFillRec& b) { return a.op_id < b.op_id; });
            uint64_t cur = 0x1000;
            for (auto& r : recs) {
                r.ddr_offset = cur;
                cur += (r.size + 127) & ~uint64_t(127);
            }
        }
        if (!recs.empty()) {
            uint32_t cnt = static_cast<uint32_t>(recs.size());
            ser.write_tagged_record(TAG_SPILL_FILL_INSTEAD, &cnt, 4);
            // 28B 固定布局(显式逐字段写入, 避免 struct padding 歧义;
            // 设备侧同构解析): [u64 op_id][u32 block_id][u64 ddr_offset][u64 size]
            for (const auto& r : recs) {
                uint8_t payload[28];
                size_t o = 0;
                auto w64 = [&](uint64_t v) { std::memcpy(payload + o, &v, 8); o += 8; };
                auto w32 = [&](uint32_t v) { std::memcpy(payload + o, &v, 4); o += 4; };
                w64(r.op_id); w32(r.block_id); w64(r.ddr_offset); w64(r.size);
                ser.write_tagged_record(TAG_SPILL_FILL_OP, payload, 28);
            }
        }
    }

    // End separator 0xBEEFF00D
    ser.write_uint32(0xBEEFF00D);

    // Log: "RUNLIST: total %d ops, %d in runlist, %d in vec runlist, %d in mtx runlist, %d in elt runlist"
    // (graph_prepare.cc:736)

    // Allocator setup -> setup_heap_info
    // adjust_heap_stats
    // check_total_allocation (memory limit check)

    return true;
}

// ---------------------------------------------------------------------------
// per-op 参数 schema 注册(通用性落地点): 新算子 = 注册 extractor 函数。
// serialize_opdef 在固定结构字段后尾随 [u32 extra_len][extra bytes];
// deserialize 按 length 回读并原样保存(serialized_extra), re-serialize 原样
// 重发 → round-trip 字节确定性。旧流(无尾随字段)天然兼容(remaining < 4)。
// extractor 从图中推导参数(权重 const 形状 + 注入的 stride/pad/dilation
// const 数据), 不依赖任何模型特判。
// ---------------------------------------------------------------------------

// 005 US3a: pass_fc_bias 复用同源形状推导(声明前置, 定义在匿名 ns 之后)
std::vector<uint8_t> extract_matmul_extra(const GraphPrepare& gp, const OpDef& opdef);

namespace {
using ExtraInfoFn = std::vector<uint8_t>(*)(const GraphPrepare& gp, const OpDef& opdef);

const OpDef* find_param_const(const GraphPrepare& gp, const OpDef& opdef, const char* key) {
    for (const auto& c : opdef.inputs) {
        const OpDef* s = gp.get_op_at(c.src_id);
        if (!s || !s->name_tag) continue;
        const char* n = s->name_tag->name();
        if (n && std::strstr(n, key)) return s;
    }
    /* tensor_param 不一定在 inputs(ranges 等留在 tensor_param_ids) */
    for (op_id_t tpid : opdef.tensor_param_ids) {
        const OpDef* s = gp.get_op_at(tpid);
        if (!s || !s->name_tag) continue;
        const char* n = s->name_tag->name();
        if (n && std::strstr(n, key)) return s;
    }
    return nullptr;
}

// 从注入 const(在 const_pool_ 中)读 u32 数组
std::vector<uint32_t> read_param_u32(const GraphPrepare& gp, const OpDef* p, size_t max_n) {
    std::vector<uint32_t> v;
    if (!p || p->const_data_size < 4) return v;
    size_t n = std::min(max_n, p->const_data_size / 4);
    const auto& pool = gp.const_pool();
    if (p->const_data_offset == 0 || p->const_data_offset + p->const_data_size > pool.size())
        return v;
    v.resize(n);
    std::memcpy(v.data(), pool.data() + p->const_data_offset, n * 4);
    return v;
}

std::vector<uint8_t> extract_conv_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraConv e;
    // kh/kw 来自权重 const 的 output_def (inputs[1] = W, dims [Kh,Kw,Cin,Cout])
    if (opdef.inputs.size() > 1) {
        e.weight_src = opdef.inputs[1].src_id;
        const OpDef* w = gp.get_op_at(opdef.inputs[1].src_id);
        if (w && w->output_def.rank >= 4) {
            e.kh = static_cast<uint32_t>(w->output_def.dims[0]);
            e.kw = static_cast<uint32_t>(w->output_def.dims[1]);
        }
    }
    if (opdef.inputs.size() > 2) e.bias_src = opdef.inputs[2].src_id;

    if (const OpDef* st = find_param_const(gp, opdef, "stride")) {
        auto v = read_param_u32(gp, st, 2);
        if (v.size() == 2) { e.sh = v[0]; e.sw = v[1]; }
    }
    if (const OpDef* pd = find_param_const(gp, opdef, "pad_amount")) {
        auto v = read_param_u32(gp, pd, 4);
        if (v.size() == 4) { e.ph_begin = v[0]; e.ph_end = v[1]; e.pw_begin = v[2]; e.pw_end = v[3]; }
    }
    if (const OpDef* dl = find_param_const(gp, opdef, "dilation")) {
        auto v = read_param_u32(gp, dl, 2);
        if (v.size() == 2) { e.dh = v[0]; e.dw = v[1]; }
    }
    // 位置回退(loader 建的 tensor_param const 无名字, 按 inject_htp_prepare_inputs
    // 注入序契约: inputs[3]=stride, [4]=pad_amount, [5]=dilation, 各为 u32 数组)
    if (e.sh == 1 && e.sw == 1 && e.ph_begin == 0 && e.ph_end == 0 &&
        e.pw_begin == 0 && e.pw_end == 0 && e.dh == 1 && e.dw == 1) {
        auto pos_param = [&](size_t idx) -> const OpDef* {
            if (idx >= opdef.inputs.size()) return nullptr;
            const OpDef* c = gp.get_op_at(opdef.inputs[idx].src_id);
            return (c && c->is_const()) ? c : nullptr;
        };
        if (const OpDef* st = pos_param(3)) {
            auto v = read_param_u32(gp, st, 2);
            if (v.size() == 2) { e.sh = v[0]; e.sw = v[1]; }
        }
        if (const OpDef* pd = pos_param(4)) {
            auto v = read_param_u32(gp, pd, 4);
            if (v.size() == 4) { e.ph_begin = v[0]; e.ph_end = v[1]; e.pw_begin = v[2]; e.pw_end = v[3]; }
        }
        if (const OpDef* dl = pos_param(5)) {
            auto v = read_param_u32(gp, dl, 2);
            if (v.size() == 2) { e.dh = v[0]; e.dw = v[1]; }
        }
    }

    // 几何推导兜底: net.json 路径的 tensor_param 值在 NETP 二进制中(B线尚未
    // 解析, loader 零填充)。当参数值缺失且 out==in 尺寸(3x3 s1 same 契约签名)
    // 时按 same-pad 推导; 否则保持 0, 由 wtop_emit 受支持几何门诚实报错。
    if (e.sh == 1 && e.sw == 1 && e.ph_begin == 0 && e.ph_end == 0 &&
        e.pw_begin == 0 && e.pw_end == 0 && e.dh == 1 && e.dw == 1) {
        uint32_t in_h = 0, in_w = 0, out_h = 0, out_w = 0;
        if (!opdef.inputs.empty()) {
            const OpDef* act = gp.get_op_at(opdef.inputs[0].src_id);
            if (act && act->output_def.rank >= 4) {
                in_h = (uint32_t)act->output_def.dims[1];
                in_w = (uint32_t)act->output_def.dims[2];
            }
        }
        if (opdef.output_def.rank >= 4) {
            out_h = (uint32_t)opdef.output_def.dims[1];
            out_w = (uint32_t)opdef.output_def.dims[2];
        }
        if (in_h > 0 && in_w > 0 && out_h == in_h && out_w == in_w &&
            (e.kh & 1) && (e.kw & 1)) {
            uint32_t ph = e.kh / 2, pw = e.kw / 2;
            e.ph_begin = e.ph_end = ph;
            e.pw_begin = e.pw_end = pw;
        }
    }

    std::vector<uint8_t> out(sizeof(ExtraConv));
    std::memcpy(out.data(), &e, sizeof(ExtraConv));

    // Tiling 段(阶段6, 通用): [u32 tile_h][u32 tile_w][u32 co_per_tile]
    //                          [u32 num_tiles][ConvTileDesc × num_tiles]
    // 配置来源: gp.tiling_config()(conv_height/width/channel_tiling = 切分数)
    {
        uint32_t in_h = 0, in_w = 0, cin = 0;
        if (!opdef.inputs.empty()) {
            const OpDef* act = gp.get_op_at(opdef.inputs[0].src_id);
            if (act && act->output_def.rank >= 4) {
                in_h = static_cast<uint32_t>(act->output_def.dims[1]);
                in_w = static_cast<uint32_t>(act->output_def.dims[2]);
                cin = static_cast<uint32_t>(act->output_def.dims[3]);
            }
        }
        uint32_t out_h = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[1]) : 0;
        uint32_t out_w = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[2]) : 0;
        uint32_t cout = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[3]) : 0;

        const TilingConfig& cfg = gp.tiling_config();
        uint32_t ht = cfg.conv_height_tiling > 0 ? cfg.conv_height_tiling : 1;
        uint32_t wt = cfg.conv_width_tiling > 0 ? cfg.conv_width_tiling : 1;
        uint32_t ct = cfg.conv_channel_tiling > 0 ? cfg.conv_channel_tiling : 1;
        uint32_t tile_h = out_h > 0 ? (out_h + ht - 1) / ht : 0;
        uint32_t tile_w = out_w > 0 ? (out_w + wt - 1) / wt : 0;
        uint32_t co_per_tile = cout > 0 ? (cout + ct - 1) / ct : 0;

        std::vector<ConvTileDesc> tiles = compute_conv_tiles(
            in_h, in_w, cin, out_h, out_w, cout,
            e.kh, e.kw, e.sh, e.sw, e.ph_begin, e.pw_begin,
            tile_h, tile_w, co_per_tile);

        std::vector<uint8_t> sec(16 + tiles.size() * sizeof(ConvTileDesc));
        uint32_t hdr[4] = {tile_h, tile_w, co_per_tile,
                           static_cast<uint32_t>(tiles.size())};
        std::memcpy(sec.data(), hdr, 16);
        std::memcpy(sec.data() + 16, tiles.data(), tiles.size() * sizeof(ConvTileDesc));
        out.insert(out.end(), sec.begin(), sec.end());
    }
    return out;
}

// ---- 20 型 extractor(M2): 参数从 op_data(scalar_params)+ tensor_param const 推导 ----

std::vector<uint8_t> raw_extra(const void* p, size_t n) {
    std::vector<uint8_t> out(n);
    std::memcpy(out.data(), p, n);
    return out;
}

// 从 opdef.op_data 取 scalar(数值)
double scalar_num(const OpDef& opdef, const char* key, double dflt) {
    auto sp = unpack_scalar_params(opdef.op_data);
    const ScalarParam* p = scalar_get(sp, key);
    return (p && p->is_numeric) ? p->value_num : dflt;
}

// 从 opdef.op_data 取 scalar(字符串)
std::string scalar_str(const OpDef& opdef, const char* key) {
    auto sp = unpack_scalar_params(opdef.op_data);
    const ScalarParam* p = scalar_get(sp, key);
    return p ? p->value_str : "";
}

// 从 tensor_param const 读 f64 数组
std::vector<double> read_param_f64(const GraphPrepare& gp, const OpDef* p, size_t max_n) {
    std::vector<double> v;
    if (!p || p->const_data_size < 8) return v;
    size_t n = std::min(max_n, p->const_data_size / 8);
    const auto& pool = gp.const_pool();
    if (p->const_data_offset == 0 || p->const_data_offset + p->const_data_size > pool.size())
        return v;
    v.resize(n);
    std::memcpy(v.data(), pool.data() + p->const_data_offset, n * 8);
    return v;
}

uint32_t op_to_u32(const std::string& s, uint32_t dflt) {
    if (s == "ADD") return 0;
    if (s == "SUB") return 1;
    if (s == "MUL") return 2;
    if (s == "DIV") return 3;
    return dflt;
}

uint32_t neuron_to_u32(const std::string& s, uint32_t dflt) {
    if (s == "SWISH") return 0;
    if (s == "SIGMOID") return 1;
    if (s == "TANH") return 2;
    if (s == "GELU") return 3;
    if (s == "RELU") return 4;
    return dflt;
}

uint32_t unary_to_u32(const std::string& s, uint32_t dflt) {
    if (s == "NEG") return 0;
    if (s == "EXP") return 1;
    if (s == "SQRT") return 2;
    if (s == "RSQRT") return 3;
    if (s == "LOG") return 4;
    if (s == "ABS") return 5;
    if (s == "SIN") return 6;
    if (s == "COS") return 7;
    return dflt;
}

// 通用: 从 opdef 输入侧 const 的 output_def 拿形状乘积
size_t const_elem_count(const GraphPrepare& gp, const OpDef& opdef, size_t input_idx) {
    if (opdef.inputs.size() <= input_idx) return 0;
    const OpDef* s = gp.get_op_at(opdef.inputs[input_idx].src_id);
    if (!s) return 0;
    size_t n = 1;
    for (uint32_t i = 0; i < s->output_def.rank && i < 5; ++i) n *= s->output_def.dims[i];
    return n;
}

std::vector<uint8_t> extract_eltwise_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraEltwise e{};
    /* 2.48 scalar 为数字枚举(NEG=8/MULTIPLY=13 等) — 直读数字;
     * 旧字符串形态仅作兼容回退 */
    const std::string os = scalar_str(opdef, "operation");
    e.operation = os.empty() ? static_cast<uint32_t>(scalar_num(opdef, "operation", 0))
                             : (op_to_u32(os, static_cast<uint32_t>(scalar_num(opdef, "operation", 0))));
    e.eltwise_type = (scalar_str(opdef, "eltwise_type") == "SELECT") ? 0 : 0xFFFFFFFFu;
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_axis_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraAxis e{};
    /* Reduce 的轴在 axes tensor_param (QNN 无 scalar "axis") — 旧码读
     * scalar 恒得 -1=0xFFFFFFFF, 设备 exec_reduce d[axis] 野读 (0.8B 全图
     * 1170 个 Reduce 全灭)。与 extract_rmsnorm_extra 同源读 tensor const。 */
    const OpDef* axes = find_param_const(gp, opdef, "axes");
    auto ax = read_param_u32(gp, axes, 4);
    e.axis = ax.empty() ? static_cast<int32_t>(scalar_num(opdef, "axis", -1))
                        : static_cast<int32_t>(ax[0]);
    /* reduce_type QNN 写 "ReduceSum/Mean/Max/Min" — 旧码比 "SUM"/"AVG"
     * 恒落空得 0xFFFFFFFF (subtype 语义全错)。契约: 0=SUM 1=AVG 2=MAX 3=MIN。 */
    const std::string rt = scalar_str(opdef, "reduce_type");
    e.reduce_type = (rt == "ReduceSum" || rt == "SUM" || rt.empty()) ? 0
                  : (rt == "ReduceMean" || rt == "AVG") ? 1
                  : (rt == "ReduceMax") ? 2
                  : (rt == "ReduceMin") ? 3 : 0xFFFFFFFFu;
    e.keep_dims = static_cast<uint32_t>(scalar_num(opdef, "keep_dims", 0));
    e.exclusive = static_cast<uint32_t>(scalar_num(opdef, "exclusive", 0));
    e.reverse = static_cast<uint32_t>(scalar_num(opdef, "reverse", 0));
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_rmsnorm_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraRmsNorm e{};
    e.epsilon = scalar_num(opdef, "epsilon", 1e-6);
    // axes 是 tensor_param; 无则 -1
    const OpDef* axes = find_param_const(gp, opdef, "axes");
    auto ax = read_param_u32(gp, axes, 4);
    e.axes = ax.empty() ? 0xFFFFFFFFu : ax[0];
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_fc_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraFc e{};
    // K = 权重 const(inputs[1])的首维(2D [out,in] 时 in = dims[1])
    const OpDef* w = opdef.inputs.size() > 1 ? gp.get_op_at(opdef.inputs[1].src_id) : nullptr;
    if (w) {
        size_t n = 1;
        for (uint32_t i = 0; i < w->output_def.rank && i < 5; ++i) n *= w->output_def.dims[i];
        uint32_t r = w->output_def.rank;
        if (r >= 2) {
            e.k = w->output_def.dims[r - 1];  // 内维
            e.n = n / e.k;
        } else {
            e.k = 0; e.n = static_cast<uint32_t>(n);
        }
    }
    // M = 输入激活的元素数 / K
    size_t m = const_elem_count(gp, opdef, 0);
    e.m = (e.k > 0) ? static_cast<uint32_t>(m / e.k) : 0;
    /* has_bias 只认 pass_fc_bias 写入的 op_data scalar 标记。旧形状启发
     * (inputs.size()>2 && inputs[2] 元素数==N) 在注入形态下不可靠:
     * inject_htp_prepare_inputs 给每个无 bias FC 追加的 wscale const 恰好
     * 也是 N 元素, 会被误判成 bias(test_wtop_emit US1 实锤: 2 输入 FC 被
     * 发射成带 wscale-as-bias 的 OP_MATMUL_F16_BIAS)。 */
    e.has_bias = (scalar_num(opdef, "has_bias", 0) != 0) ? 1u : 0u;
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_strided_slice_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraStridedSlice e{};
    e.begin_mask = static_cast<uint32_t>(scalar_num(opdef, "begin_mask", 0));
    e.end_mask = static_cast<uint32_t>(scalar_num(opdef, "end_mask", 0));
    e.new_axes_mask = static_cast<uint32_t>(scalar_num(opdef, "new_axes_mask", 0));
    e.shrink_axes = static_cast<uint32_t>(scalar_num(opdef, "shrink_axes", 0));
    // ranges = tensor_param const (名含 "ranges")
    const OpDef* rg = find_param_const(gp, opdef, "ranges");
    if (rg) {
        e.rank = static_cast<uint32_t>(rg->const_data_size / 4 / 3);
        e.ranges_offset = static_cast<uint64_t>(rg->const_data_offset);
    }
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_reshape_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraReshape e{};
    // 形状 const: 名含 "shape" 的 tensor_param(loader 合成 Reshape_post_*_shape)
    const OpDef* sh = find_param_const(gp, opdef, "shape");
    if (!sh) sh = find_param_const(gp, opdef, "Reshape_post");
    if (sh) {
        auto d = read_param_u32(gp, sh, 5);
        // 形状可能存 i32/f32: 先按 u32 读, 数序取非零合理值
        e.rank = static_cast<uint32_t>(std::min<size_t>(d.size(), 5));
        for (size_t i = 0; i < d.size() && i < 5; ++i) e.dims[i] = static_cast<int64_t>(d[i]);
    } else {
        // 兜底: 输出形状
        e.rank = opdef.output_def.rank;
        for (uint32_t i = 0; i < e.rank && i < 5; ++i) e.dims[i] = opdef.output_def.dims[i];
    }
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_transpose_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraTranspose e{};
    const OpDef* pm = find_param_const(gp, opdef, "perm");
    if (pm) {
        auto d = read_param_u32(gp, pm, 5);
        e.rank = static_cast<uint32_t>(std::min<size_t>(d.size(), 5));
        for (size_t i = 0; i < d.size() && i < 5; ++i) e.perm[i] = static_cast<int64_t>(d[i]);
    } else {
        e.rank = 0;
    }
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_depthwise_conv_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraDepthwiseConv e{};
    // 权重 const QNN 布局 [1,k,1,C] (末维=通道 C)。kw = 总元素/C —— 对
    // [1,4,1,6144] 得 4; 旧码取 dims[r-2] 命中 singleton 1 (0.8B GDN 实锤:
    // 真核 k=4 被发成 k=1, 设备 conv 输出能量差 30 倍)。
    const OpDef* w = opdef.inputs.size() > 1 ? gp.get_op_at(opdef.inputs[1].src_id) : nullptr;
    if (w && w->output_def.rank >= 2) {
        uint32_t r = w->output_def.rank;
        uint64_t elems = 1;
        for (uint32_t i = 0; i < r && i < 5; i++) elems *= w->output_def.dims[i];
        uint64_t ch = w->output_def.dims[r - 1];
        uint32_t kw = ch ? (uint32_t)(elems / ch) : 1;
        e.kw = kw ? kw : 1;
        e.kh = 1;
    }
    // stride/pad 在 tensor_param(stride/pad_amount), 名匹配
    const OpDef* st = find_param_const(gp, opdef, "stride");
    auto sv = read_param_u32(gp, st, 2);
    if (sv.size() >= 2) { e.sh = sv[0]; e.sw = sv[1]; }
    const OpDef* pd = find_param_const(gp, opdef, "pad");
    auto pv = read_param_u32(gp, pd, 4);
    if (pv.size() >= 4) { e.pad_t = pv[0]; e.pad_b = pv[1]; e.pad_l = pv[2]; e.pad_r = pv[3]; }
    const OpDef* w2 = opdef.inputs.size() > 1 ? gp.get_op_at(opdef.inputs[1].src_id) : nullptr;
    if (w2 && w2->output_def.rank >= 1) {
        size_t g = 1;
        for (uint32_t i = 0; i + 1 < w2->output_def.rank && i < 4; ++i) g *= w2->output_def.dims[i];
        e.group = static_cast<uint32_t>(g);
    }
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_pad_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraPad e{};
    e.scheme = static_cast<uint32_t>(scalar_num(opdef, "scheme", 0));
    e.constant_value = scalar_num(opdef, "pad_constant_value", 0.0);
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_scatternd_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraScatterNd e{};
    const std::string rd = scalar_str(opdef, "reduction");
    e.reduction = (rd == "SUM" || rd.empty()) ? 0 : (rd == "MUL" ? 1 : 0xFFFFFFFFu);
    return raw_extra(&e, sizeof(e));
}

std::vector<uint8_t> extract_split_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraSplit e{};
    e.axis = static_cast<int32_t>(scalar_num(opdef, "axis", -1));
    e.split_index = static_cast<uint32_t>(scalar_num(opdef, "split_index", 0));
    // 段大小 tensor_param 名含 split_index
    const OpDef* si = find_param_const(gp, opdef, "split_index");
    if (!si) si = find_param_const(gp, opdef, "split");
    auto v = read_param_u32(gp, si, 8);
    e.num_splits = static_cast<uint32_t>(std::min<size_t>(v.size(), 8));
    for (size_t i = 0; i < v.size() && i < 8; ++i) e.sizes[i] = v[i];
    /* 形状推断优先: converter 2.48 对 3 段 Split 的 param 实测写错
     * ([64,64,64] 写成 [64,128] — ONNX Constant 实为三段), 一律以
     * 输入末维/输出末维 推段数与段长覆盖 */
    {
        const OpDef* src = opdef.inputs.empty() ? nullptr
                                                : gp.get_op_at(opdef.inputs[0].src_id);
        if (src) {
            uint32_t ar = src->output_def.rank;
            while (ar > 0 && src->output_def.dims[ar - 1] == 1) ar--;
            uint32_t rk = opdef.output_def.rank;
            while (rk > 0 && opdef.output_def.dims[rk - 1] == 1) rk--;
            uint64_t in_axis = (ar >= 1) ? src->output_def.dims[ar - 1] : 0;
            uint64_t out_axis = (rk >= 1) ? opdef.output_def.dims[rk - 1] : 1;
            if (in_axis > 0 && out_axis > 0 && in_axis % out_axis == 0) {
                e.num_splits = static_cast<uint32_t>(in_axis / out_axis);
                for (uint32_t i = 0; i < e.num_splits && i < 8; ++i)
                    e.sizes[i] = static_cast<uint32_t>(out_axis);
            }
        }
    }
    return raw_extra(&e, sizeof(e));
}

/* 009 T016: GdnBlock 块几何 extractor。
 * H/D/d_conv/chunk 标量口径与 ops.cpp GdnBlock host 参考分支 (T011) 一致;
 * T 从 in0 (xqkv [T,3HD]) 形状积 / 3HD 推导, 标量 "T" 可显式覆盖。
 * c = min(chunk, T) (kernel 按 chunk 循环, 尾块在内核内截; ≤GDN_C_MAX=64
 * 由设备侧 exec 校验, 004 契约)。
 * vtcm_budget_bytes 闭式 = 内核私有 scratch 保守超集 (内核可复用/别名;
 * FR-004 超 16MB 由 emit carve 门显式失败, 非 parse 层):
 *   state f32 H·D·D·4 + conv 滑窗 f32 (d_conv−1)·3HD·4
 *   + yc 暂存 f16 T·3HD·2 + y 出口 f16 T·HD·2 + beta/g f16 T·H·2·2
 *   + L/T 三角 f16 2·H·c·c·2 + chunk 缓冲 f32 5·H·c·D·4, 128 对齐。 */
std::vector<uint8_t> extract_gdn_block_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraGdnBlock e{};
    e.h = static_cast<uint32_t>(scalar_num(opdef, "H", 16));
    e.d = static_cast<uint32_t>(scalar_num(opdef, "D", 128));
    e.d_conv = static_cast<uint32_t>(scalar_num(opdef, "d_conv", 4));
    e.chunk = static_cast<uint32_t>(scalar_num(opdef, "chunk", 32));
    const uint64_t hd = (uint64_t)e.h * e.d, di3 = 3ull * hd;
    uint64_t t = (uint64_t)scalar_num(opdef, "T", 0);
    if (t == 0) {
        const OpDef* x = opdef.inputs.size() > 0 ? gp.get_op_at(opdef.inputs[0].src_id) : nullptr;
        if (x) {
            uint64_t elems = 1;
            for (uint32_t i = 0; i < x->output_def.rank; ++i) elems *= x->output_def.dims[i];
            if (di3 && elems >= di3 && elems % di3 == 0) t = elems / di3;
        }
    }
    e.t = (uint32_t)t;
    e.ntok = e.t;
    e.c = e.chunk < e.t ? e.chunk : e.t;
    const uint64_t cc = e.c;
    uint64_t budget = (uint64_t)e.h * e.d * e.d * 4ull
                    + (uint64_t)(e.d_conv ? e.d_conv - 1u : 0u) * di3 * 4ull
                    + t * di3 * 2ull + t * hd * 2ull
                    + t * e.h * 2ull * 2ull
                    + 2ull * e.h * cc * cc * 2ull
                    + 5ull * e.h * cc * e.d * 4ull;
    if (budget > 0xFFFFFFFFull) budget = 0xFFFFFFFFull;   /* 超界哨兵, emit 门显式失败 */
    e.vtcm_budget_bytes = (uint32_t)((budget + 127ull) & ~127ull);
    e.state_idx = 0xFFFFFFFFu;   /* prefill 瞬态 carve 哨兵; US5 实义化 */
    e.contract_ver = 1u;         /* numerics-contract.md contract_ver=1 */
    return raw_extra(&e, sizeof(e));
}

// 004 T018 decode 状态族: 5×u32 定长 extra 的 per-type 键表(extractor 与
// rebuild_op_data_from_extra 双侧共用; 键域与 ops.cpp dnum / wtop_emit 一一对应)
bool decode_state_keys(const char* nm, const char* (&keys)[5]) {
    const char* kt[6][6] = {
        {"DecodeFaceF16", "u16_begin", "u16_len", "", "", ""},
        {"KvAppendF16", "state_idx", "kv_heads", "head_dim", "", ""},
        {"KvGatherKT", "state_idx", "n_heads", "kv_heads", "head_dim", ""},
        {"KvGatherV", "state_idx", "n_heads", "kv_heads", "head_dim", ""},
        {"Conv1dSsmStepF16", "state_idx", "C", "d_conv", "", ""},
        {"DeltaChunkF16", "state_idx", "H", "D", "q_raw", ""},
    };
    for (auto& row : kt)
        if (!std::strcmp(nm, row[0])) {
            for (int i = 0; i < 5; i++) keys[i] = row[i + 1];
            return true;
        }
    return false;
}

// op_data(scalar_params, net.json loader 写入) → 5×u32 extra
std::vector<uint8_t> extract_decode_state_extra(const GraphPrepare& gp, const OpDef& opdef) {
    (void)gp;
    const char* nm = opdef.name_tag ? opdef.name_tag->name() : "";
    const char* keys[5];
    if (!decode_state_keys(nm, keys)) return {};
    auto sp = unpack_scalar_params(opdef.op_data);
    uint32_t v[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; i++) {
        if (!keys[i][0]) continue;
        const ScalarParam* p = scalar_get(sp, keys[i]);
        if (p) v[i] = (uint32_t)p->as_int();
    }
    if (getenv("GEHTP_DBG_T18"))
        std::fprintf(stderr, "[t18-ext] %s id=%llu op_data=%zu -> [%u,%u,%u,%u,%u]\n", nm,
                     (unsigned long long)opdef.op_id, opdef.op_data.size(),
                     v[0], v[1], v[2], v[3], v[4]);
    std::vector<uint8_t> out(sizeof v);
    std::memcpy(out.data(), v, sizeof v);
    return out;
}

std::map<std::string, ExtraInfoFn>& extra_info_registry() {
    static std::map<std::string, ExtraInfoFn> reg = [] {
        std::map<std::string, ExtraInfoFn> m;
        m["Conv2d"] = extract_conv_extra;
        // M2: 0.8B net.json 实测 20 型
        m["Eltwise_Binary"] = extract_eltwise_extra;
        m["Eltwise_Unary"] = extract_eltwise_extra;
        m["ElementWiseNeuron"] = extract_eltwise_extra;
        m["Eltwise_Ternary"] = extract_eltwise_extra;
        m["Softmax"] = extract_axis_extra;
        m["Gather"] = extract_axis_extra;
        m["Concat"] = extract_axis_extra;
        m["Reduce"] = extract_axis_extra;
        m["CumulativeSum"] = extract_axis_extra;
        m["RmsNorm"] = extract_rmsnorm_extra;
        m["FullyConnected"] = extract_fc_extra;
        m["MatMul"] = extract_matmul_extra;
        m["StridedSlice"] = extract_strided_slice_extra;
        m["Reshape"] = extract_reshape_extra;
        m["Transpose"] = extract_transpose_extra;
        m["DepthWiseConv2d"] = extract_depthwise_conv_extra;
        m["Pad"] = extract_pad_extra;
        m["ScatterNd"] = extract_scatternd_extra;
        m["Split"] = extract_split_extra;
        m["GdnBlock"] = extract_gdn_block_extra;
        // 004 T018: decode 图状态族(net.json loader op_data → extra 尾随)
        m["DecodeFaceF16"] = extract_decode_state_extra;
        m["KvAppendF16"] = extract_decode_state_extra;
        m["KvGatherKT"] = extract_decode_state_extra;
        m["KvGatherV"] = extract_decode_state_extra;
        m["Conv1dSsmStepF16"] = extract_decode_state_extra;
        m["DeltaChunkF16"] = extract_decode_state_extra;
        return m;
    }();
    return reg;
}
} // namespace

std::vector<uint8_t> extract_matmul_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ExtraMatMul e{};
    // M/K/N: a [M,K] b [K,N](transpose_in0/1 由 scalar 决定; 默认 0=不转置)
    e.transpose_in0 = static_cast<uint32_t>(scalar_num(opdef, "transpose_in0", 0));
    e.transpose_in1 = static_cast<uint32_t>(scalar_num(opdef, "transpose_in1", 0));
    const OpDef* a = opdef.inputs.size() > 0 ? gp.get_op_at(opdef.inputs[0].src_id) : nullptr;
    const OpDef* b = opdef.inputs.size() > 1 ? gp.get_op_at(opdef.inputs[1].src_id) : nullptr;
    if (a && b && a->output_def.rank >= 2 && b->output_def.rank >= 2) {
        uint32_t ar = a->output_def.rank, br = b->output_def.rank;
        /* 压尾 1 得有效 rank(QNN 前置填充 [1,1,m,k]) */
        while (ar > 0 && a->output_def.dims[ar - 1] == 1) ar--;
        while (br > 0 && b->output_def.dims[br - 1] == 1) br--;
        /* K 一致性校验: 压尾后两侧 K 不一致 ⇒ 尾 1 是真实外积 K 维
         * (RoPE inv_freq [1,64,1] × pos [1,1,16] → [1,64,16]), 回退未压秩 */
        if (ar >= 2 && br >= 2 &&
            a->output_def.dims[ar - 1] != b->output_def.dims[br - 2]) {
            ar = a->output_def.rank; br = b->output_def.rank;
        }
        uint32_t bk = e.transpose_in1 ? b->output_def.dims[br - 1] : b->output_def.dims[br - 2];
        uint32_t bn = e.transpose_in1 ? b->output_def.dims[br - 2] : b->output_def.dims[br - 1];
        (void)bk;
        e.n = bn;
        /* batched BMM(attention q·kᵀ / probs·v): 有效 rank≥3 且 A/B 批维
         * 同构 → 批维不折叠, m = 每批行数(批数由输出形状推)。
         * rank≤2 或 B 批维广播 → 普通 GEMM 折批(等价)。 */
        uint64_t mprod = 1;
        uint32_t alast = a->output_def.dims[ar - 1];
        bool same_batch = (ar >= 3 && br >= 3);
        if (same_batch)
            for (uint32_t i = 0; i + 2 < ar; ++i)
                if (a->output_def.dims[i] != b->output_def.dims[i]) same_batch = false;
        if (same_batch) {
            e.batched = 1;
            e.m = a->output_def.dims[ar - 2];
            e.k = alast;
        } else {
            for (uint32_t i = 0; i + 1 < ar; ++i) mprod *= a->output_def.dims[i];
            if (e.transpose_in0) { e.m = alast; e.k = (uint32_t)mprod; }
            else { e.m = (uint32_t)mprod; e.k = alast; }
        }
    }
    e.has_bias = (scalar_num(opdef, "has_bias", 0) != 0) ? 1u : 0u;
    return raw_extra(&e, sizeof(e));
}


// 跨 TU 的算子 extra 提取入口(序列化 serialize_opdef 与 VTCM vtcm_auto_scan
// 共用): 按 op 名查注册表现算。序列化区已拆到本文件, 主文件 vtcm_auto_scan
// 经此函数复用同一 extractor 族, 无需触及匿名 ns 内部。
std::vector<uint8_t> compute_op_extra(const GraphPrepare& gp, const OpDef& opdef) {
    const char* nm = opdef.name_tag ? opdef.name_tag->name() : "";
    auto& reg = extra_info_registry();
    auto it = reg.find(nm);
    if (it != reg.end()) return it->second(gp, opdef);
    return {};
}

// serialize_opdef: write a single op's structural record (TAG_OP_RECORD).
//   [uint32 name_len][char[] name, 4-aligned][uint64 op_id][uint16 flags]
//   [uint16 reserved][uint32 num_inputs]
//   per input: [uint64 src_id][uint32 out_idx][uint32 pad]
//   [OutputDef output_def]
// The execution Op is regenerated by the op factory at load time. Structural
// data and the per-op parameter blob must persist because host execution reads
// the latter through Op::params.
//
// [反汇编证�?libQnnHtp.so v2.48] 真实 serialize_op @0x12ec630 写入格式:
//   [0x1303EE{XX}] (4B 标记, XX=记录ID, 硬编�?0x1303ee71 起始)
//   [counter:u32]  (= tensor id, 来自 net.json 张量 id)
//   [type:u32<<24] (0x01=compute, 0x02=memory, 0x03=sync, 0x04=DMA, 0x0b=HMX)
//                  (type = (extra_info[0x18]>>6 & 0xf) << 24, HMX 修正)
//   [block_ref:u32] (0x10{idx}, VTCM 块引�? extra_info[0x24]&0xFFFFFF|(3<<24))
//   [tensor_ids...] (变长, 输入/输出张量 id 列表)
// 注意: SET/WAIT (make_dma_checkpoint_op @0xd958d0) 共享 serialize_internal @0xd969a0
//       �?context binary 字节不可区分 SET vs WAIT
// 详见 reference/docs/schedule_analysis.md
void GraphPrepare::serialize_opdef(Serializer& ser, const OpDef& opdef) const {
    const char* name = opdef.name_tag ? opdef.name_tag->name() : "";
    uint32_t name_len = static_cast<uint32_t>(std::strlen(name));
    uint32_t name_padded = (name_len + 3) & ~uint32_t(3);

    // 尾随 per-op 参数 schema(通用): 优先复用反序列化回读的字节(round-trip
    // 确定性), 否则按注册表 extractor 现算。
    std::vector<uint8_t> extra;
    if (!opdef.serialized_extra.empty()) {
        extra = opdef.serialized_extra;
    } else {
        auto it = extra_info_registry().find(name);
        if (it != extra_info_registry().end()) extra = it->second(*this, opdef);
    }

    // M2c: 记录内加 grouping(原始节点名, wtop_emit --gguf 的 match 键)
    const std::string& grp = opdef.grouping;
    uint32_t grp_len = static_cast<uint32_t>(grp.size());
    uint32_t grp_padded = (grp_len + 3) & ~uint32_t(3);

    // Compute total payload size for the tagged record header word count.
    /* T018d: tensor_param_ids(axes/perm/ranges const 的挂靠表, loader 写入、
     * execute_host 按 od2->tensor_param_ids 追加输入)此前不序列化 → --bin
     * 路径 ReduceSum 退化 reduce-all、Transpose 丢 perm 全图发散(cos 0.08)。
     * 尾随 [u32 n_tp][ids u64×n](恒写, 空=0 — 与 op_data 同为自描述尾段,
     * 否则空 tp 的 op 反序列化会把 op_data 首 4B 误读成 n_tp 破坏字节全同) */
    size_t payload =
        4 + name_padded +            // name_len + name
        4 + grp_padded +             // grp_len + grouping(M2c)
        8 + 2 + 2 + 4 +              // op_id, flags, reserved, num_inputs
        opdef.inputs.size() * 16 +   // (src_id, out_idx, pad) per input
        sizeof(OutputDef) +           // output_def
        8 + 8 +                       // const_data_offset, const_data_size
        4 + extra.size() +            // 尾随 [u32 extra_len][extra bytes]
        4 + opdef.tensor_param_ids.size() * 8 +  // 尾随 [u32 n_tp][ids u64×n](恒写, 空=0)
        4 + opdef.op_data.size();     // 尾随 [u32 op_data_len][op_data bytes]
    payload = (payload + 3) & ~size_t(3);  // 4B 对齐，匹配 tagged record 的 word_count

    // Build the payload in a local buffer, then emit as one tagged record.
    std::vector<uint8_t> buf(payload, 0);
    size_t off = 0;
    auto w32 = [&](uint32_t v) { std::memcpy(buf.data() + off, &v, 4); off += 4; };
    auto w16 = [&](uint16_t v) { std::memcpy(buf.data() + off, &v, 2); off += 2; };
    auto w64 = [&](uint64_t v) { std::memcpy(buf.data() + off, &v, 8); off += 8; };

    w32(name_len);
    std::memcpy(buf.data() + off, name, name_len);
    off += name_padded;

    w32(grp_len);
    std::memcpy(buf.data() + off, grp.data(), grp_len);
    off += grp_padded;
    w64(opdef.op_id);
    w16(opdef.flags);
    w16(0);  // reserved
    w32(static_cast<uint32_t>(opdef.inputs.size()));
    for (const auto& conn : opdef.inputs) {
        w64(conn.src_id);
        w32(conn.out_idx);
        w32(0);  // pad
    }
    std::memcpy(buf.data() + off, &opdef.output_def, sizeof(OutputDef));
    off += sizeof(OutputDef);
    w64(opdef.const_data_offset);
    w64(opdef.const_data_size);
    w32(static_cast<uint32_t>(extra.size()));
    if (!extra.empty()) {
        std::memcpy(buf.data() + off, extra.data(), extra.size());
        off += extra.size();
    }
    w32(static_cast<uint32_t>(opdef.tensor_param_ids.size()));
    for (op_id_t tpid : opdef.tensor_param_ids) w64(tpid);

    // Serialize op_data blob: per-op scalar params (e.g., has_bias for FC)
    w32(static_cast<uint32_t>(opdef.op_data.size()));
    if (!opdef.op_data.empty()) {
        std::memcpy(buf.data() + off, opdef.op_data.data(), opdef.op_data.size());
        off += opdef.op_data.size();
    }

    ser.write_tagged_record(TAG_OP_RECORD, buf.data(), static_cast<int>(payload));
}

// Cursor for reading the tagged-record stream back from a flat buffer.
struct BinReader {
    const uint8_t* p;
    const uint8_t* end;
    explicit BinReader(const uint8_t* base, size_t n) : p(base), end(base + n) {}
    bool eof() const { return p >= end; }
    size_t remaining() const { return end - p; }
    uint32_t r32() { uint32_t v = 0; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t r64() { uint64_t v = 0; std::memcpy(&v, p, 8); p += 8; return v; }
    uint16_t r16() { uint16_t v = 0; std::memcpy(&v, p, 2); p += 2; return v; }
    void read(void* dst, size_t n) { if (dst) std::memcpy(dst, p, n); p += n; }
};

/* M2d: --bin 反序列化路径重建 op_data(execute_host 的 ops.cpp 算子读
 * scalar params; serialize_opdef 只写 serialized_extra 尾随段不写 op_data
 * → Eltwise operation 退化 ADD(0)、Gather/Softmax/Concat axis 退化 0 —
 * 0.8B seq=1 --bin cos 0.076 实锤: Expand(Mul) 变 Add、And 变 Add(1+1=2.0f))。
 * 从 serialized_extra 反向 pack, 键域与 ops.cpp scalar_get 一一对应;
 * extra 缺失/长度不足的 op 不动(保持 loader 语义: 无参数走各算子启发)。 */
static void rebuild_op_data_from_extra(OpDef* od) {
    if (!od || od->serialized_extra.empty()) return;
    const char* nm = od->name_tag ? od->name_tag->name() : nullptr;
    if (!nm) return;
    const auto& ex = od->serialized_extra;
    std::vector<ScalarParam> sp;
    auto num = [&](const char* k, double v) {
        ScalarParam q; q.name = k; q.is_numeric = true; q.value_num = v;
        sp.push_back(q);
    };
    auto str = [&](const char* k, const char* v) {
        ScalarParam q; q.name = k; q.is_numeric = false; q.value_str = v;
        sp.push_back(q);
    };
    const bool eltwise = !std::strcmp(nm, "Eltwise_Binary") ||
                         !std::strcmp(nm, "Eltwise_Unary") ||
                         !std::strcmp(nm, "ElementWiseNeuron") ||
                         !std::strcmp(nm, "Eltwise_Ternary");
    if (eltwise) {
        if (ex.size() < sizeof(ExtraEltwise)) return;
        ExtraEltwise e; std::memcpy(&e, ex.data(), sizeof e);
        num("operation", (double)e.operation);
        if (e.eltwise_type == 0) str("eltwise_type", "SELECT");
    } else if (!std::strcmp(nm, "Softmax") || !std::strcmp(nm, "Gather") ||
               !std::strcmp(nm, "Concat") || !std::strcmp(nm, "CumulativeSum")) {
        if (ex.size() < sizeof(ExtraAxis)) return;
        ExtraAxis e; std::memcpy(&e, ex.data(), sizeof e);
        num("axis", (double)e.axis);
    } else if (!std::strcmp(nm, "Reduce")) {
        if (ex.size() < sizeof(ExtraAxis)) return;
        ExtraAxis e; std::memcpy(&e, ex.data(), sizeof e);
        switch (e.reduce_type) {
        case 1: str("reduce_type", "ReduceMean"); break;
        case 2: str("reduce_type", "ReduceMax"); break;
        case 3: str("reduce_type", "ReduceMin"); break;
        default: break;  // 0(Sum)/未知: ops.cpp 缺省 ReduceSum
        }
    } else if (!std::strcmp(nm, "Split")) {
        if (ex.size() < sizeof(ExtraSplit)) return;
        ExtraSplit e; std::memcpy(&e, ex.data(), sizeof e);
        num("split_index", (double)e.split_index);
    } else if (!std::strcmp(nm, "Pad")) {
        if (ex.size() < sizeof(ExtraPad)) return;
        ExtraPad e; std::memcpy(&e, ex.data(), sizeof e);
        num("scheme", (double)e.scheme);
        num("pad_constant_value", e.constant_value);
    } else if (!std::strcmp(nm, "MatMul")) {
        if (ex.size() < sizeof(ExtraMatMul)) return;
        ExtraMatMul e; std::memcpy(&e, ex.data(), sizeof e);
        num("transpose_in1", (double)e.transpose_in1);
    } else {
        /* 004 T018: decode 状态族 5×u32 定长 extra(与
         * extract_decode_state_extra 同构回读) */
        const char* dk[5];
        if (getenv("GEHTP_DBG_T18"))
            std::fprintf(stderr, "[t18-rb] %s id=%llu ex=%zu match=%d\n", nm,
                         (unsigned long long)od->op_id, ex.size(),
                         decode_state_keys(nm, dk));
        if (decode_state_keys(nm, dk) && ex.size() >= 20) {
            uint32_t v[5];
            std::memcpy(v, ex.data(), sizeof v);
            for (int i = 0; i < 5; i++)
                if (dk[i][0]) num(dk[i], (double)v[i]);
        }
    }
    if (!sp.empty()) od->op_data = pack_scalar_params(sp);
}

// deserialize: rebuild the graph from a buffer produced by serialize().
// Round-trips the structural records written by do_serialize/serialize_opdef.
bool GraphPrepare::deserialize(const uint8_t* buf, size_t buf_size) {
    if (!buf || buf_size < 4) return false;

    // Reset to a fresh construction-phase graph.
    opdef_map_.clear();
    ordering_.clear();
    ops_.clear();
    const_pool_.clear();
    const_extents_.clear();
    block_table_.clear();
    segment_plans_.clear();
    input_node_id_ = 0;
    output_node_id_ = 0;
    next_op_id_ = 1;
    construction_state_ = 1;  // allow direct opdef insertion
    serialized_loaded_ = true;

    BinReader r(buf, buf_size);
    uint32_t expected_ops = 0;

    while (!r.eof()) {
        uint32_t word0 = r.r32();

        // Single-word separators / end marker are not tagged records.
        if (word0 == SEPARATOR_NORMAL || word0 == SEPARATOR_AUX) {
            continue;  // segment / aux separator
        }
        if (word0 == SEPARATOR_END) {
            break;  // 0xBEEFF00D end-of-stream
        }

        // Otherwise word0 is an encoded tag: read word_count + third field.
        uint32_t word_count = r.r32();
        uint32_t third = r.r32();  // third field (== word_count for our writer)
        (void)third;
        size_t data_size = static_cast<size_t>(word_count) * 4;
        if (data_size > r.remaining()) {
            std::fprintf(stderr, "[deser] tag=0x%04x data_size=%zu remaining=%zu\n",
                         word0 ^ 0xFFFF, data_size, r.remaining());
            return false;
        }
        const uint8_t* rec = r.p;
        r.p += data_size;

        // Decode the tag: inverse of ((tag & 0xFFFF) | (tag << 16)) ^ 0xFFFF.
        uint32_t decoded = word0 ^ 0xFFFF;
        uint32_t tag = decoded & 0xFFFF;

        if (tag == 0x4845) {  // 'HE' BinHeader (大端计数+偏移�?
            BinHeader hdr;
            std::memcpy(&hdr, rec, sizeof(BinHeader));
            // 真实 .bin 无文件级 magic; �?num_graphs 合理性校�?
    if (hdr.num_graphs == 0 || hdr.num_graphs > 100) {
                return false;
            }
            continue;
        }
        if (tag == TAG_GRAPH_HEADER) {
            expected_ops = rec[0] | (rec[1] << 8) | (rec[2] << 16) | (rec[3] << 24);
            (void)expected_ops;
            continue;
        }
        if (tag == TAG_SPILL_FILL_OP) {
            BinReader rr(rec, data_size);
            if (rr.remaining() >= 28) {
                SpillFillRec r{};
                r.op_id = rr.r64();
                r.block_id = rr.r32();
                r.ddr_offset = rr.r64();
                r.size = rr.r64();
                spill_fill_recs_.push_back(r);
            }
            continue;
        }
        if (tag == TAG_SPILL_FILL_INSTEAD && data_size == 4) {
            // 0x4453 现在携带 spill 计数(阶段7); 旧语义(仅 flag+multicast)已被
            // 此计数形式取代 —— 记录本身无副作用, 计数由 0x5346 记录实际决定。
            continue;
        }
        if (tag == TAG_PLAN_ORDER) {
            BinReader rr(rec, data_size);
            if (rr.remaining() >= 4) {
                uint32_t cnt = rr.r32();
                plan_order_.clear();
                for (uint32_t i = 0; i < cnt && rr.remaining() >= 4; i++)
                    plan_order_.push_back(static_cast<op_id_t>(rr.r32()));
            }
            continue;
        }
        if (tag == TAG_MEM_PLAN) {
            // M2: 对称回读编译器定稿规划(字段序同写侧; 坏记录不置 has_)
            BinReader rr(rec, data_size);
            if (rr.remaining() >= 24 && rr.r32() == 1) {
                MemPlan mp;
                mp.ddr_cap = rr.r32();
                mp.vtcm_cap = rr.r32();
                mp.bump_reserve = rr.r32();
                mp.pin_cap = rr.r32();
                uint32_t ne = rr.r32();
                bool ok = true;
                for (uint32_t i = 0; i < ne; i++) {
                    if (rr.remaining() < 20) { ok = false; break; }
                    MemPlanEntry e{};
                    e.op_id = rr.r32();
                    e.out_idx = rr.r32();
                    e.offset = rr.r32();
                    e.size = rr.r32();
                    e.in_vtcm = (rr.r32() & 1) != 0;
                    mp.entries.push_back(e);
                }
                if (ok) {
                    if (rr.remaining() < 4) { ok = false; }
                    else {
                        uint32_t ns = rr.r32();
                        for (uint32_t i = 0; i < ns; i++) {
                            if (rr.remaining() < 16) { ok = false; break; }
                            MemPlanSpill s{};
                            s.op_id = rr.r32();
                            s.offset = rr.r32();
                            s.size = rr.r32();
                            (void)rr.r32();
                            mp.spills.push_back(s);
                        }
                    }
                }
                if (ok) { mem_plan_ = mp; has_mem_plan_ = true; }
            }
            continue;
        }
        if (tag == TAG_CONST_EXTENT) {
            // extent �? 每条 ConstExtentDesc = 32 字节
            // [op_id 8B][offset 8B][size 8B][tensor_type 4B][reserved 4B]
            size_t n = data_size / sizeof(ConstExtentDesc);
            const_extents_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                ConstExtentDesc desc;
                std::memcpy(&desc, pp, sizeof(desc));
                pp += sizeof(desc);
                const_extents_.push_back({desc.op_id, desc.offset, desc.size});
            }
            continue;
        }
        if (tag == 0xCF56) {
            // 常量数据�? 直接拷回 const_pool_
            const_pool_.assign(rec, rec + data_size);
            continue;
        }
        if (tag == 0x4254) {
            // block table: �?24 字节 [block_id][pool_id][offset][size]
            size_t n = data_size / 24;
            block_table_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                BlockEntry b{};
                std::memcpy(&b.block_id, pp, 4); pp += 4;
                std::memcpy(&b.pool_id, pp, 4); pp += 4;
                std::memcpy(&b.offset, pp, 8); pp += 8;
                std::memcpy(&b.size, pp, 8); pp += 8;
                block_table_.push_back(b);
            }
            continue;
        }
        if (tag == 0x5347) {
            // segment plan: �?24 字节 [seg_idx][op_count][byte_offset][reserved8]
            size_t n = data_size / 24;
            segment_plans_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                SegmentPlan s{};
                std::memcpy(&s.segment_index, pp, 4); pp += 4;
                std::memcpy(&s.op_count, pp, 4); pp += 4;
                std::memcpy(&s.byte_offset, pp, 8); pp += 8;
                pp += 8;  // reserved
                segment_plans_.push_back(s);
            }
            continue;
        }
        if (tag == TAG_IO_TENSOR_DESC) {
            // [uint32 kind][uint32 pad][uint64 op_id][OutputDef]
            BinReader rr(rec, data_size);
            uint32_t kind = rr.r32();
            (void)rr.r32();  // pad
            uint64_t op_id = rr.r64();
            // Register the Input/Output node by id; the producer/consumer
            // links are re-established when reading the op records below.
            auto opdef = std::make_unique<OpDef>();
            opdef->flags = OP_ENABLED;
            opdef->graph = this;
            opdef->op_id = op_id;
            opdef->name_tag = string_tag_t::map_str(kind == 0 ? "Input" : "Output");
            std::memcpy(&opdef->output_def, rr.p, sizeof(OutputDef));
            if (kind == 0) input_node_id_ = op_id;
            else output_node_id_ = op_id;
            opdef_map_[op_id] = std::move(opdef);
            continue;
        }
        if (tag == TAG_OP_RECORD) {
            BinReader rr(rec, data_size);
            uint32_t name_len = rr.r32();
            uint32_t name_padded = (name_len + 3) & ~uint32_t(3);
            std::string name(reinterpret_cast<const char*>(rr.p), name_len);
            rr.p += name_padded;
            // M2c: grouping(原始节点名) — 用 remaining 守卫兼容旧流
            std::string grp;
            if (rr.remaining() >= 4) {
                uint32_t grp_len = rr.r32();
                uint32_t grp_padded = (grp_len + 3) & ~uint32_t(3);
                if (rr.remaining() >= grp_padded) {
                    grp.assign(reinterpret_cast<const char*>(rr.p), grp_len);
                    rr.p += grp_padded;
                } else {
                    rr.p -= 4;
                }
            }
            uint64_t op_id = rr.r64();
            uint16_t flags = rr.r16();
            (void)rr.r16();  // reserved
            uint32_t num_inputs = rr.r32();

            auto opdef = std::make_unique<OpDef>();
            opdef->flags = flags;
            opdef->graph = this;
            opdef->op_id = op_id;
            opdef->name_tag = string_tag_t::map_str(name.c_str());
            opdef->grouping = grp;

            opdef->inputs.reserve(num_inputs);
            for (uint32_t i = 0; i < num_inputs; ++i) {
                uint64_t src_id = rr.r64();
                uint32_t out_idx = rr.r32();
                (void)rr.r32();  // pad
                opdef->inputs.push_back({src_id, out_idx});
            }
            std::memcpy(&opdef->output_def, rr.p, sizeof(OutputDef));
            rr.p += sizeof(OutputDef);
            opdef->const_data_offset = rr.r64();
            opdef->const_data_size = rr.r64();
            // 尾随 per-op 参数 schema(向后兼容: 旧流无此字段 → remaining < 4)
            if (rr.remaining() >= 4) {
                uint32_t elen = rr.r32();
                if (elen > 0 && rr.remaining() >= elen) {
                    opdef->serialized_extra.resize(elen);
                    rr.read(opdef->serialized_extra.data(), elen);
                }
            }
            /* T018d: 尾随 tensor_param_ids(axes/perm/ranges const 挂靠表;
             * 旧流无尾段跳过)。上界守卫防误读越界。 */
            if (rr.remaining() >= 4) {
                uint32_t ntp = rr.r32();
                if (ntp > 0 && ntp <= 64 && rr.remaining() >= (size_t)ntp * 8) {
                    opdef->tensor_param_ids.reserve(ntp);
                    for (uint32_t k = 0; k < ntp; k++)
                        opdef->tensor_param_ids.push_back((op_id_t)rr.r64());
                }
            }
            // 尾随 op_data blob(向后兼容: 旧流无此字段 → remaining < 4)
            bool has_od_field = rr.remaining() >= 4;
            if (has_od_field) {
                uint32_t od_len = rr.r32();
                if (od_len > 0 && rr.remaining() >= od_len) {
                    opdef->op_data.resize(od_len);
                    rr.read(opdef->op_data.data(), od_len);
                }
            }
            // 004 线旧流无尾随 op_data 字段 → 从 serialized_extra 重建;
            // 新流以写出侧原字节为准(空 op_data 也是合法态, 勿重建覆盖)。
            if (!has_od_field) rebuild_op_data_from_extra(opdef.get());

            if (name == "Input") input_node_id_ = op_id;
            else if (name == "Output") output_node_id_ = op_id;

            opdef_map_[op_id] = std::move(opdef);
            if (op_id >= next_op_id_) next_op_id_ = op_id + 1;
            continue;
        }
        // Unknown config record: already skipped via r.p advance above.
    }

    // Rebuild consumer lists from the restored input connections.
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        for (const auto& conn : opdef->inputs) {
            auto src_it = opdef_map_.find(conn.src_id);
            if (src_it != opdef_map_.end() && src_it->second) {
                src_it->second->consumers.push_back(id);
            }
        }
    }

    construction_state_ = 2;  // graph now in "loaded" state
    // The reconstructed graph is a clean, self-contained structure (not tied
    // to any external loaded buffer), so allow it to be re-serialized. This
    // lets the round-trip test verify deterministic layout.
    serialized_loaded_ = false;

    // Rebuild topological ordering so ops_ are in valid execution order.
    // Without this, iterating opdef_map_ (unordered) may produce wrong order
    // (e.g. Add before MatMul).
    order_nodes(false);

    // Build ops_ from the restored opdef_map_ so execute_host can run
    // without a separate prepare() call.  Use ordering_ for correct
    // topological execution order.
    ops_.clear();
    if (!ordering_.empty()) {
        for (op_id_t id : ordering_) {
            auto it = opdef_map_.find(id);
            if (it == opdef_map_.end() || !it->second) continue;
            if (!it->second->is_enabled() || it->second->is_dead()) continue;
            OpIoPtrs io{};
            io.graph_prepare = this;
            io.opdef_ptr = it->second.get();
            auto op = op_factory_generate(io, id);
            if (op) { ops_.push_back(std::move(op)); }
        }
    } else {
        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
            OpIoPtrs io{};
            io.graph_prepare = this;
            io.opdef_ptr = opdef.get();
            auto op = op_factory_generate(io, id);
            if (op) { ops_.push_back(std::move(op)); }
        }
    }

    return true;
}

void GraphPrepare::serialize_io(Serializer& ser, uint64_t& counter, bool is_prescan) const {
    (void)is_prescan;
    // Input node: its single output_def describes the input tensor fed at runtime
    if (input_node_id_ != 0) {
        auto it = opdef_map_.find(input_node_id_);
        if (it != opdef_map_.end() && it->second) {
            const OpDef& iod = *it->second;
            // payload: [uint32 kind=0(input)][uint64 op_id][OutputDef]
            struct IoDesc {
                uint32_t kind;     // 0 = input, 1 = output
                uint32_t pad;
                uint64_t op_id;
                OutputDef od;
            } desc{};
            desc.kind = 0;
            desc.op_id = iod.op_id;
            desc.od = iod.output_def;
            ser.write_tagged_record(TAG_IO_TENSOR_DESC, &desc, sizeof(desc));
            counter++;
        }
    }
    // Output node: its single input references the tensor produced for the host
    if (output_node_id_ != 0) {
        auto it = opdef_map_.find(output_node_id_);
        if (it != opdef_map_.end() && it->second) {
            const OpDef& iod = *it->second;
            // Resolve the producer op to grab the real output_def
            OutputDef od{};
            if (!iod.inputs.empty()) {
                auto src = opdef_map_.find(iod.inputs[0].src_id);
                if (src != opdef_map_.end() && src->second) {
                    od = src->second->output_def;
                }
            }
            struct IoDesc {
                uint32_t kind;     // 1 = output
                uint32_t pad;
                uint64_t op_id;
                OutputDef od;
            } desc{};
            desc.kind = 1;
            desc.op_id = iod.op_id;
            desc.od = od;
            ser.write_tagged_record(TAG_IO_TENSOR_DESC, &desc, sizeof(desc));
            counter++;
        }
    }
}

} // namespace hnnx
