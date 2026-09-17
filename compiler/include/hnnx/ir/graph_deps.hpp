#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace hnnx {

class GraphPrepare;

// GraphDeps: build_graph_deps 的产物 (反汇编确认 @ 0xfac220, 8216B)
// 存入 GraphPrepare::this+0x7468, 供 FancyAllocator 与 runlist 调度消费
//
// 反汇编证据 (GraphDeps::GraphDeps @ 0xfb1190, 4624B 构造):
//   +0x00: GraphPrepare* (mov [rdi],rsi; rsi=this)
//   +0x08: vtable[0x1c0] 返回值 (call [rax+0x1c0])
//   +0x10: 0 (qword)
//   +0x18: 0xffffffff (dword, 哨兵)
//   +0x50/+0xa0/+0xd8/+0x1c0/...: 0xffffffff 哨兵 (多个)
//   +0x4b0: 子对象 (call 0xfe1e30 构造)
//   +0x5d0: 子对象 (call 0xfd1540 构造)
//   +0x5e8: xnsp 标志

// DepOpDesc: 单个 op 的依赖描述符
// 反汇编确认 (OpDesc array build @ 0xfb2270): imul r14,rsi,0xd0
//   → 每元素 0xd0 = 208 字节
//
// flags 位 (反汇编确认):
//   bit 0x01 = enabled (sub3: and 0xfb 清此位 = disabled)
//   bit 0x04 = ? (sub3: and [OpDesc+1],0xfb 清 bit2, 即 +1 字节的 bit)
//   bit 0x20 = unused 标记 (sub1/sub2 设置, sub3 检查并清理)
//   bit 0x80 = pending 标记 (sub1: or esi,0x80 设置)
struct DepOpDesc {
    // +0x00: op_id (hash 插入 0xfa3630 比较 [rax]==op_id)
    op_id_t op_id = 0;
    // +0x08: 关联指针
    const void* ptr_08 = nullptr;
    // +0x10: OpDef 指针
    const OpDef* opdef_ptr = nullptr;
    // +0x18: extra_obj
    void* extra_obj = nullptr;
    // +0x20: reserved
    uint64_t reserved_20 = 0;
    // +0x28: flags (hash 插入比较 [r10+rax+0x28]: 0=空, 1=占用)
    uint8_t flags = 0;
    uint8_t reserved_29[7] = {};
    // +0x30..: 生命期 + memgroup 字段 (208B 内的剩余部分)
    uint32_t life_begin = 0;          // first_use (拓扑序 index)
    uint32_t life_end = 0;            // last_use
    uint32_t access_count = 0;        // 被访问次数
    uint32_t nsp_assignment = 0;     // NSP 分配
    // set_distances_from_constraints 产物 (反汇编确认 @ 0xfeb360, sub7)
    //   OpDesc.+0xb4 = distance, +0xc0 = predecessor, +0xc4 = successor/chain
    uint32_t distance = 0;           // +0xb4: 到起点的距离
    uint32_t predecessor = 0;        // +0xc0: 前驱 op idx (1-based, 0=无)
    uint32_t successor = 0;          // +0xc4: 后继链
    std::vector<uint32_t> input_memgroups;
    std::vector<uint32_t> output_memgroups;
    uint64_t vtcm_requirement = 0;
    uint64_t ddr_requirement = 0;
};

// MemGroup: memgroup 描述符
// 反汇编字符串: "memgroup(bytes=%zu, is_tcm=%s, gen_op='#%d'"
struct MemGroup {
    uint32_t mg_id = 0;
    uint64_t bytes = 0;
    bool is_tcm = false;
    op_id_t gen_op_id = 0;
    std::vector<op_id_t> consumers;
};

    // OpDesc hash 桶 (反汇编确认 @ 0xfa3630 fibonacci hash 开放寻址)
//   bucket = fibonacci_hash(op_id) & (size-1)
//   探测步长 = (hash>>15 & 0x1fffe) | 1
struct OpDescHashBucket {
    std::vector<DepOpDesc> slots;   // 实际存储 (size = 2^n - 1)
    uint32_t mask = 0;              // size-1
    uint32_t count = 0;             // 已用数

    void init(uint32_t capacity);
    DepOpDesc* lookup(op_id_t op_id);
    DepOpDesc* insert(op_id_t op_id);
};

// GroupEntry: group_entry 数组元素 (反汇编确认 @ 0xfd20a0, sub1 构建)
// 供 sub2 populate_groups_in_opdesc 遍历
// 真实每元素 0x38=56 字节 (imul 0x38e38e39 除法推断)
struct GroupEntry {
    std::vector<op_id_t> op_ids;      // +0x30/+0x38: op 列表 vector
    std::vector<uint32_t> group_ids;  // +0x48/+0x50: group id 列表
    uint32_t op_count = 0;            // +0x24: op 数量 (循环用)
};

// GraphDeps: 完整依赖图
class GraphDeps {
public:
    GraphDeps();
    ~GraphDeps();

    // ===== 字段 (反汇编确认偏移) =====
    // +0x00: GraphPrepare*
    GraphPrepare* gp = nullptr;
    // +0x08: vtable[0x1c0] 返回值
    void* cb_handle = nullptr;

    // OpDesc 数组 (@ +0x80, 每元素 0xd0=208B)
    std::vector<DepOpDesc> opdescs;

    // group_entry 数组 (@ +0x148, sub1 构建, sub2 遍历)
    std::vector<GroupEntry> group_entries;

    // OpDesc hash 桶 (@ +0x120, fibonacci hash 开放寻址)
    OpDescHashBucket opdesc_hash;

    // runlist (@ +0x48 头, +0x58/+0x60 计数)
    std::vector<op_id_t> runlist;
    uint32_t runlist_count = 0;

    // memgroup 表
    std::vector<MemGroup> memgroups;
    std::unordered_map<op_id_t, uint32_t> memgroup_map;

    // 生命期
    struct LifetimeInfo {
        op_id_t op_id = 0;
        uint32_t first_use = 0;
        uint32_t last_use = 0;
        uint32_t access_count = 0;
    };
    std::vector<LifetimeInfo> lifetimes;

    // xnsp 标志 (@ +0x5e8)
    bool xnsp_enabled = false;

    // set_distances_from_constraints 产物 (反汇编确认 @ 0xfeb360, sub7)
    //   GraphDeps.+0x138 = ordering 状态 (2=完成)
    //   GraphDeps.+0x13c = max_distance
    uint32_t ordering_state_ = 0;
    uint32_t max_distance_ = 0;

    // sanity_check 产物 (反汇编确认 @ 0xfff2e0, sub8)
    //   行 509: peak_tcm_use
    //   行 512: largest_tcm_op_id / largest_tcm_op_size
    uint64_t peak_tcm_use_ = 0;
    op_id_t largest_tcm_op_id_ = 0;
    uint64_t largest_tcm_op_size_ = 0;

    // ===== 方法 =====
    // 初始化 (对应 GraphDeps::GraphDeps 构造)
    void init(GraphPrepare& gp);
    // 查找 OpDesc by op_id (fibonacci hash)
    DepOpDesc* find_opdesc(op_id_t op_id);
    const DepOpDesc* find_opdesc(op_id_t op_id) const;
    // 添加 op (返回 OpDesc 指针)
    DepOpDesc* add_op(op_id_t op_id, const OpDef* opdef, const Op* op);
    // 构建 runlist (反汇编确认 @ 0xfb10a0)
    bool build_runlist();
    // sub1: prepare_group_entries (反汇编确认 @ 0xfd20a0, 528B)
    //   遍历 OpDesc 标记 flags|=0x80, 释放旧 group_entry, 构建新 group_entry
    //   供 sub2 populate_groups 遍历
    bool prepare_group_entries();
    // sub3: remove_unused_ops (反汇编确认 @ 0xfb4cc0, 1104B)
    //   遍历 OpDesc, 对 flags bit 0x20 置位的 op 清理 memgroup 关联 + 依赖边 + 标记 removed
    //   无错误日志 (sub5 才带行 925 错误检查)
    bool remove_unused_ops();
    // sub4: rebuild_hash_bucket (反汇编确认 @ 0xfc83d0, 1200B)
    //   sub3 移除 unused op 后, 清空并重建 GraphDeps+0x120 的 OpDesc hash 桶
    //   含 3 个子函数: vector 析构 + fibonacci hash 查找 + vector 析构
    bool rebuild_hash_bucket();
    // sub5: remove_unused_ops_checked (反汇编确认 @ 0xfb4ee0+0xfb5020, 560B)
    //   带错误检查的二次移除: 找仍 enabled 但无有效数据的 op, 尝试移除
    //   失败报行 925 "error removing unused op#%d" (grdep_main.cc)
    //   筛选条件: (flags&0xf)==1 && vector1/vector2 空 && vector3 非空 && !bit0x100
    bool remove_unused_ops_checked();
    // populate_groups_in_opdesc (反汇编确认 @ 0xfb3e40, 3712B, sub2)
    //   遍历 group_entry 数组, 把 memgroup 信息填入每个 OpDesc
    //   错误: 行 1161 "not all forwarded blks accounted for"
    //         行 1169 "memgroup split failed!"
    //         "populate_groups_in_opdesc: !gen_op"
    bool populate_groups_in_opdesc();
    // set_distances_from_constraints (反汇编确认 @ 0xfeb360, 1344B, sub7)
    //   grdep_ordering.cc: 带约束的 distance 传播 + loop 检测
    //   错误: 行 898 "Loop detected in set_distances_from_constraints"
    //   产物: OpDesc.+0xb4 = distance, GraphDeps.+0x13c = max_distance
    bool set_distances_from_constraints();
    // sanity_check (反汇编确认 @ 0xfff2e0, 512B, sub8)
    //   grdep_sanity.cc: 10 个子检查调度器
    //   成功: 行 509 "sanity check passed; peak tcm use = %zu"
    //         行 512 "largest TCM op is #%d, size = %zu"
    //   失败: 行 426 "failed to find ordering"
    //         行 435 "sanity check failed"
    bool sanity_check();
    // 构建 memgroup
    void build_memgroups(GraphPrepare& gp);
    // 建立依赖链
    void build_dependencies(GraphPrepare& gp);
    // 计算生命期
    void compute_lifetimes(GraphPrepare& gp);
    // 调试 dump
    void dump(const std::string& filename) const;
};

// fibonacci hash (反汇编确认 @ 0xfa3630)
//   h1 = (op_id>>32) * 0x192e2101
//   h  = ((op_id ^ h1) * 0x740f1de9) >> ... xor
uint64_t fibonacci_hash_op(op_id_t op_id);

} // namespace hnnx
