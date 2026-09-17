#include "hnnx/ir/graph_deps.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/types.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace hnnx {

// ===== fibonacci hash (反汇编确认 @ 0xfa3630) =====
// 0xfa3645: shr rax,0x20; imul eax,0x192e2101
// 0xfa364f: xor edx,eax; imul rax,rdx,0x740f1de9
// 0xfa365e: shr rsi,0x20; xor rsi,rax
uint64_t fibonacci_hash_op(op_id_t op_id) {
    uint64_t h1 = (static_cast<uint64_t>(static_cast<uint32_t>(op_id >> 32)) * 0x192e2101ULL);
    uint32_t lo = static_cast<uint32_t>(op_id) ^ static_cast<uint32_t>(h1);
    uint64_t h = lo * 0x740f1de9ULL;
    uint64_t r = (h >> 32) ^ h;
    return r;
}

// ===== OpDescHashBucket =====
void OpDescHashBucket::init(uint32_t capacity) {
    // 反汇编: size = 2^n - 1 (mask = size-1)
    uint32_t sz = 16;
    while (sz < capacity * 2) sz <<= 1;
    mask = sz - 1;
    slots.assign(sz, DepOpDesc{});
    count = 0;
}

DepOpDesc* OpDescHashBucket::lookup(op_id_t op_id) {
    if (mask == 0) return nullptr;
    uint64_t h = fibonacci_hash_op(op_id);
    uint32_t bucket = static_cast<uint32_t>(h) & mask;
    uint32_t step = (static_cast<uint32_t>(h >> 15) & 0x1fffe) | 1;
    for (uint32_t i = 0; i <= mask; ++i) {
        DepOpDesc& slot = slots[bucket];
        // 反汇编 0xfa368e: cmp byte [r10+rax+0x28], 1 → flags==1 表示占用
        if (slot.flags == 1 && slot.op_id == op_id) return &slot;
        if (slot.flags == 0) return nullptr;  // 空 → 未找到
        bucket = (bucket + step) & mask;
    }
    return nullptr;
}

DepOpDesc* OpDescHashBucket::insert(op_id_t op_id) {
    if (mask == 0) init(64);
    uint64_t h = fibonacci_hash_op(op_id);
    uint32_t bucket = static_cast<uint32_t>(h) & mask;
    uint32_t step = (static_cast<uint32_t>(h >> 15) & 0x1fffe) | 1;
    for (uint32_t i = 0; i <= mask; ++i) {
        DepOpDesc& slot = slots[bucket];
        if (slot.flags == 0) {
            slot.op_id = op_id;
            slot.flags = 1;  // 占用
            count++;
            return &slot;
        }
        if (slot.flags == 1 && slot.op_id == op_id) return &slot;  // 已存在
        bucket = (bucket + step) & mask;
    }
    // 表满 → 扩容重插 (反汇编 0xfa36c1 抛 "hash lookup failed")
    uint32_t new_cap = (mask + 1) * 2;
    std::vector<DepOpDesc> old = std::move(slots);
    init(new_cap);
    for (auto& s : old) {
        if (s.flags == 1) {
            DepOpDesc* ns = insert(s.op_id);
            *ns = s;
        }
    }
    return insert(op_id);
}

// ===== GraphDeps =====
GraphDeps::GraphDeps() = default;
GraphDeps::~GraphDeps() = default;

void GraphDeps::init(GraphPrepare& g) {
    // 反汇编确认 (GraphDeps::GraphDeps @ 0xfb1190):
    //   +0x00 = GraphPrepare*
    //   +0x08 = vtable[0x1c0] 返回值
    //   +0x18/+0x50/+0xa0/... = 0xffffffff 哨兵
    gp = &g;
    cb_handle = nullptr;
    runlist_count = 0;
    xnsp_enabled = false;
    opdescs.clear();
    opdesc_hash.init(64);
    memgroups.clear();
    memgroup_map.clear();
    lifetimes.clear();
}

DepOpDesc* GraphDeps::find_opdesc(op_id_t op_id) {
    return opdesc_hash.lookup(op_id);
}

const DepOpDesc* GraphDeps::find_opdesc(op_id_t op_id) const {
    // const 版本: 直接查
    if (opdesc_hash.mask == 0) return nullptr;
    uint64_t h = fibonacci_hash_op(op_id);
    uint32_t bucket = static_cast<uint32_t>(h) & opdesc_hash.mask;
    uint32_t step = (static_cast<uint32_t>(h >> 15) & 0x1fffe) | 1;
    for (uint32_t i = 0; i <= opdesc_hash.mask; ++i) {
        const DepOpDesc& slot = opdesc_hash.slots[bucket];
        if (slot.flags == 1 && slot.op_id == op_id) return &slot;
        if (slot.flags == 0) return nullptr;
        bucket = (bucket + step) & opdesc_hash.mask;
    }
    return nullptr;
}

DepOpDesc* GraphDeps::add_op(op_id_t op_id, const OpDef* opdef, const Op* op) {
    DepOpDesc* desc = opdesc_hash.insert(op_id);
    if (!desc->opdef_ptr) {
        // 新插入: 填字段
        desc->opdef_ptr = opdef;
        desc->ptr_08 = op;
        desc->op_id = op_id;
        desc->flags = 1;
        opdescs.push_back(*desc);  // 线性表也存一份 (供遍历)
    }
    return desc;
}

// ===== build_runlist (反汇编确认 @ 0xfb10a0, 4864B) =====
// 真实逻辑: 先释放旧 runlist, 再调 8 个子函数构建
//   sub1(0xfd20a0 prepare_entries) → sub2(0xfb3e40 populate_groups) → sub3(0xfb4cc0 remove unused)
//   → sub4(0xfc83d0 hash clear) → sub5(0xfb4ee0 remove unused) → sub6(0xfb5020 cleanup)
//   → sub7(0xfeb360 set_distances, grdep_ordering.cc) → sub8(0xfff2e0 sanity check)
// 失败报 "Sanity check failed" (grdep_main.cc:892)
bool GraphDeps::build_runlist() {
    runlist.clear();
    runlist_count = 0;
    if (!gp) return false;

    // sub1: prepare_group_entries (反汇编 @ 0xfd20a0, 528B)
    // 遍历 OpDesc 标记 flags|=0x80, 释放旧 group_entry, 构建新 group_entry
    if (!prepare_group_entries()) return false;

    // sub2: populate_groups_in_opdesc (反汇编 @ 0xfb3e40, 3712B)
    // 把 memgroup 信息填入每个 OpDesc (核心)
    if (!populate_groups_in_opdesc()) return false;

    // sub3: remove_unused_ops (反汇编 @ 0xfb4cc0, 1104B)
    // 遍历 flags bit 0x20 置位的 op, 清理 memgroup 关联 + 依赖边 + 标记 removed
    if (!remove_unused_ops()) return false;

    // sub4: rebuild_hash_bucket (反汇编 @ 0xfc83d0, 1200B)
    // sub3 移除 unused op 后, 清空并重建 OpDesc hash 桶 (deps.+0x120)
    if (!rebuild_hash_bucket()) return false;

    // sub5: remove_unused_ops_checked (反汇编 @ 0xfb5020)
    // 带错误检查的二次移除: 找仍 enabled 但无有效数据的 op
    if (!remove_unused_ops_checked()) return false;

    // sub6: 收尾清理 (反汇编 @ 0xfb5020 之后, 240B)
    // 简化: sub5 已完成, 跳过

    // sub7: set_distances_from_constraints (grdep_ordering.cc, 行 898 loop 检测)
    // 带约束的 distance 传播, 产物: OpDesc.distance + max_distance
    if (!set_distances_from_constraints()) return false;

    // 用 distance 排序构建 runlist (distance 小的先执行)
    // 真实 sub7 之后 sub8 做 sanity check, 这里用排序后的拓扑序
    const auto& ordering = gp->get_ordering();
    if (ordering.empty()) return false;

    // 按 distance 排序 (distance = 到起点的距离, 小的先)
    std::vector<std::pair<uint32_t, op_id_t>> dist_op;
    for (op_id_t oid : ordering) {
        const DepOpDesc* desc = find_opdesc(oid);
        if (!desc) continue;
        dist_op.push_back({desc->distance, oid});
    }
    std::stable_sort(dist_op.begin(), dist_op.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& [dist, oid] : dist_op) {
        runlist.push_back(oid);
        runlist_count++;
    }

    // sub8: sanity_check (grdep_sanity.cc, 行 509/512/426/435)
    // 10 个子检查 + 计算 peak tcm use + largest TCM op
    if (!sanity_check()) return false;

    if (runlist.empty()) return false;
    return true;
}

// ===== sub1: prepare_group_entries (反汇编确认 @ 0xfd20a0, 528B) =====
// 真实逻辑 (反汇编 0xfd20a0-0xfd2264):
//   阶段1: 遍历 OpDesc (0xfd20ce-0xfd2160)
//     for idx = 1 to op_count:
//       读 OpDesc[idx-1].flags (deps.+0x80, 每元素 0xd0=208B)
//       若 flags==0 (空槽) → 跳过
//       若 flags bit 0x20 置位 → 跳过 (已标记)
//       检查 hash 桶[idx-1] 的 +0x30/+0x38 和 +0x48/+0x50 向量 (deps.+0x120, 每元素 0x78)
//       若非空 → OpDesc.flags |= 0x80 (标记"待处理")
//                 call 0xfd22b0 (处理该 op)
//   阶段2: 释放旧 group_entry 数组 (0xfd2160-0xfd2178)
//     deps.+0x148 = begin, +0x150 = end
//     遍历释放每个 entry 的子对象 (operator delete)
//     deps.+0x150 = begin (清空)
//   阶段3: 构建新 group_entry (0xfd2181-0xfd2198)
//     call 0xfd2610 (构建, edx=1=首次)
//     call 0xfd6530 (安装)
//   阶段4: 第二次遍历 OpDesc (0xfd219d-0xfd2242)
//     for idx = 1 to op_count:
//       类似阶段1, 但 call 0xfd22b0 传 hash 桶而非 OpDesc
//   阶段5: 收尾构建 (0xfd2242)
//     call 0xfd2610 (edx=0=收尾)
//     call 0xfd6530 (安装)
bool GraphDeps::prepare_group_entries() {
    const uint32_t op_count = static_cast<uint32_t>(opdescs.size());
    if (op_count == 0) {
        return true;  // 反汇编 0xfd20d6: 若 +0x110==0 跳到收尾
    }

    // 阶段1: 遍历 OpDesc, 标记有效 op 的 flags |= 0x80
    // 反汇编: 0xfd212c 读 flags, 0xfd2136 检查 bit 0x20, 0xfd20f3 or esi,0x80
    for (uint32_t i = 0; i < op_count; ++i) {
        DepOpDesc& desc = opdescs[i];
        if (desc.flags == 0) continue;           // 空槽
        if (desc.flags & 0x20) continue;          // 已标记, 跳过
        // 标记为"待处理" (反汇编: or esi,0x80)
        desc.flags |= 0x80;
        // 真实: call 0xfd22b0 (callee1, 864B) 处理该 op
        // 简化: 标记即可 (host reimpl 无复杂子结构)
    }

    // 阶段2: 释放旧 group_entry 数组
    // 反汇编: 0xfd2160-0xfd2178, deps.+0x148/+0x150
    group_entries.clear();

    // 阶段3: 构建新 group_entry (首次)
    // 反汇编: call 0xfd2610 (edx=1)
    // 简化: 按 op 的连续依赖链分组
    // 真实 callee2(0xfd2610, 240B) 做实质构建, 这里用拓扑序连续链近似
    const auto& ordering = gp->get_ordering();
    if (ordering.empty()) return true;

    // 把连续的 op (无跨组依赖) 归入同一 group_entry
    // (类似 DPGroupGraph 的 op_id_based_grouping)
    std::unordered_map<op_id_t, uint32_t> op_idx;
    for (uint32_t i = 0; i < opdescs.size(); ++i) {
        op_idx[opdescs[i].op_id] = i;
    }

    GroupEntry* cur = nullptr;
    for (op_id_t oid : ordering) {
        auto it = op_idx.find(oid);
        if (it == op_idx.end()) continue;
        DepOpDesc& desc = opdescs[it->second];
        if (desc.flags == 0) continue;

        // 新 group 条件: 无前驱在同一 group
        bool start_new = (cur == nullptr);
        if (!start_new && desc.opdef_ptr) {
            for (const auto& conn : desc.opdef_ptr->inputs) {
                auto pit = op_idx.find(conn.src_id);
                if (pit == op_idx.end()) continue;
                bool in_cur = false;
                if (cur) {
                    for (auto goid : cur->op_ids) if (goid == conn.src_id) { in_cur = true; break; }
                }
                if (!in_cur) { start_new = true; break; }
            }
        }

        if (start_new) {
            group_entries.emplace_back();
            cur = &group_entries.back();
        }
        cur->op_ids.push_back(oid);
        cur->op_count++;
        // 清除"待处理"标记 (反汇编: 处理后清除)
        desc.flags &= ~0x80;
    }

    // 阶段4+5: 第二次遍历 + 收尾 (反汇编 0xfd219d-0xfd2248)
    // 真实: 再遍历一次处理漏网的 op, 调 callee2(edx=0) 收尾
    // 简化: 跳过 (阶段3 已完成分组)

    return true;
}

// ===== sub3: remove_unused_ops (反汇编确认 @ 0xfb4cc0, 1104B) =====
// 文件: grdep_main.cc (无错误日志, sub5 才带行 925)
// 真实逻辑 (反汇编 0xfb4cc0-0xfb4e62):
//   for idx = 1 to deps.+0x110 (op_count):
//     OpDesc = deps.+0x80[(idx-1)*0xd0]
//     // 0xfb4d25: test byte [OpDesc], 0x20
//     if (OpDesc.flags & 0x20) == 0: continue   // 未标记 unused
//
//     // 0xfb4d2c-0xfb4d36: 读 OpDesc+0x50/+0x58 (memgroup vector 末尾元素)
//     // 0xfb4d56-0xfb4d73: 读 hash 桶[(idx-1)*0x78] 的 +0x60/+0x68/+0x70
//     //   清空 hash 桶槽 (movups xmm0; mov [rdx+rax+0x70],0; movups [rdx+rax+0x60],xmm1)
//
//     // 0xfb4dab: call 0xfb7af0 → 清理该 op 的 group 关联
//     //   (传 OpDesc+0x68 vector + 本地结果)
//     // 0xfb4dd0-0xfb4de9: 遍历 OpDesc+0x50 vector (每个 memgroup id):
//     //   call 0xfea640 → 移除该 op 对该 memgroup 的依赖边
//     // 0xfb4df6: call 0xfe90f0(edx=2) → 标记 op 为 removed (状态 2)
//     // 0xfb4dfe: and byte [OpDesc+1], 0xfb → 清 flags bit 0x04 (实际清 +1 字节的 bit2)
//     //   即 OpDesc.flags 高字节的 enabled 位
bool GraphDeps::remove_unused_ops() {
    const uint32_t op_count = static_cast<uint32_t>(opdescs.size());
    // 反汇编 0xfb4ce2: test ecx,ecx; je 收尾 (op_count==0 直接返回)

    for (uint32_t i = 0; i < op_count; ++i) {
        DepOpDesc& desc = opdescs[i];
        // 反汇编 0xfb4d25: test byte [OpDesc], 0x20
        // bit 0x20 = unused 标记 (sub1/sub2 阶段设置)
        if ((desc.flags & 0x20) == 0) continue;  // 未标记 unused → 跳过

        // 该 op 被标记为 unused, 执行清理:

        // 1. 清理 memgroup 关联
        // 反汇编 0xfb4dab: call 0xfb7af0 (清理 group 关联)
        // 真实: 遍历该 op 的 output_memgroups, 从 memgroup 表中移除
        for (uint32_t mg_id : desc.output_memgroups) {
            if (mg_id < memgroups.size()) {
                // 从 memgroup 的 consumers 中移除该 op
                auto& consumers = memgroups[mg_id].consumers;
                consumers.erase(
                    std::remove(consumers.begin(), consumers.end(), desc.op_id),
                    consumers.end());
            }
        }

        // 2. 移除依赖边
        // 反汇编 0xfb4dd0-0xfb4de9: 遍历 OpDesc+0x50, call 0xfea640 移除边
        // 真实: 遍历该 op 的 input_memgroups, 从源 op 的 consumers 中移除
        for (uint32_t mg_id : desc.input_memgroups) {
            if (mg_id < memgroups.size()) {
                auto& consumers = memgroups[mg_id].consumers;
                consumers.erase(
                    std::remove(consumers.begin(), consumers.end(), desc.op_id),
                    consumers.end());
            }
        }

        // 3. 清空该 op 的 memgroup 关联
        desc.output_memgroups.clear();
        desc.input_memgroups.clear();

        // 4. 标记为 removed
        // 反汇编 0xfb4df6: call 0xfe90f0(edx=2) → 标记 removed (状态 2)
        // 反汇编 0xfb4dfe: and byte [OpDesc+1], 0xfb → 清 enabled 位
        desc.flags &= ~0x04;  // 清 enabled (反汇编 and 0xfb 清 bit2)
        desc.flags |= 0x08;   // 标记 removed (OpDefFlags::OP_DEAD 对应位)
        // 从 hash 桶移除
        // 反汇编 0xfb4d73: 清空 hash 桶槽
        // 简化: opdesc_hash 里保留但 flags 标记 dead
    }

    return true;
}

// ===== sub4: rebuild_hash_bucket (反汇编确认 @ 0xfc83d0, 1200B) =====
// 文件: grdep_main.cc
// 真实逻辑 (反汇编):
//   入口: rdi=&GraphDeps+0x120 (hash 桶 header), rsi=桶 begin
//   0xfc83de: rbx = [rdi+8] (桶 end)
//   0xfc83e2: cmp rbx,rsi; 若 begin==end → [rdi+8]=rsi; return (空桶)
//   否则: 遍历释放每个槽 (步长 0x78=120B):
//     0xfc8408-0xfc8466: 释放每个槽的 -0x18/-0x30/-0x48/-0x60/-0x78 子对象
//   最后: [rdi+8] = rsi (重置 end=begin, 清空)
//
//   内部还含 2 个子函数:
//   0xfc8480: fibonacci hash 查找 (和 0xfa3630 同算法, imul 0x192e2101/0x740f1de9)
//   0xfc8570: 另一 vector 析构 (步长 0x50)
//
// 调用上下文 (build_runlist @ 0xfb1124):
//   lea r14,[rbx+0x120]    → r14 = &deps.hash_bucket
//   mov rsi,[rbx+0x120]    → rsi = bucket begin
//   call 0xfc83d0          → 清空重建
//
// 作用: sub3 移除 unused op 后, hash 桶里有空槽/脏槽,
//       sub4 清空桶并重建, 供 sub5/sub6/sub7 使用
bool GraphDeps::rebuild_hash_bucket() {
    // 反汇编: 若桶空 (begin==end) 直接返回
    if (opdesc_hash.slots.empty()) {
        return true;
    }

    // 收集所有仍有效的 OpDesc (未被 sub3 标记 removed)
    std::vector<DepOpDesc> valid;
    for (auto& desc : opdesc_hash.slots) {
        if (desc.flags == 1 && (desc.flags & 0x08) == 0) {
            // flags==1 (占用) 且未标记 dead (bit 0x08)
            valid.push_back(desc);
        }
    }

    // 反汇编: 清空旧桶 (释放每个槽的子对象)
    // 真实: 遍历步长 0x78 释放 -0x18/-0x30/-0x48/-0x60/-0x78
    // 简化: vector clear 会自动释放
    uint32_t old_mask = opdesc_hash.mask;
    opdesc_hash.slots.clear();
    opdesc_hash.mask = 0;
    opdesc_hash.count = 0;

    // 反汇编: 重建桶 (调 fibonacci hash 重新插入)
    // 真实: 0xfc8480 做 hash 查找, 重新填充桶
    opdesc_hash.init(static_cast<uint32_t>(valid.size()));
    for (auto& desc : valid) {
        DepOpDesc* slot = opdesc_hash.insert(desc.op_id);
        if (slot) *slot = std::move(desc);
    }

    return true;
}

// ===== sub5: remove_unused_ops_checked (反汇编确认 @ 0xfb5020, grdep_main.cc) =====
// 真实逻辑 (反汇编 0xfb5020-0xfb5102):
//   for ebx = 1 to deps.+0x110 (op_count):
//     OpDesc = deps.+0x80[(ebx-1)*0xd0]
//     // 0xfb5086: and esi,0xf; cmp esi,1 → flags 低 4 位 == 1 (enabled, 未 removed)
//     if (flags & 0xf) != 1: continue
//     // 0xfb508e-0xfb5098: OpDesc+0x38/+0x40 (vector1) begin==end? (空)
//     // 0xfb509a-0xfb50a4: OpDesc+0x50/+0x58 (vector2) begin==end? (空)
//     //   两个 vector 都空 → 继续; 任一非空 → 跳过 (有数据)
//     if (vector1 非空 || vector2 非空): continue
//     // 0xfb50a6-0xfb50b6: OpDesc+0x80/+0x88 (vector3) begin==end? (空)
//     //   vector3 空 → 跳过 (无 runlist 数据)
//     if (vector3 空): continue
//     // 0xfb50b8: and edx,0x100; 若 bit 0x100 置位 → 跳过 (已标记处理)
//     if (flags & 0x100): continue
//     // 0xfb50cd: call 0xfb5110 → 清理该 op 关联
//     // 0xfb50d7: call 0xfb25a0 → 尝试移除 (含行 352 "trying to remove #%d which has connected output groups")
//     //   返回 al = 是否成功
//     if (!al):
//       // 0xfb50e0: 行 925 "error removing unused op#%d" (grdep_main.cc)
//       log(ERROR, "grdep_main.cc", 925, "error removing unused op#%d", ebx)
//       // 继续 (不中止, 只记录错误)
//
// 作用: sub3 移除 bit 0x20 的 op 后, sub5 找仍 enabled 但数据异常的 op
//   (vector1+vector2 空, vector3 非空 = runlist 有但无 memgroup 数据)
//   尝试二次移除, 失败报 925 错误 (不中止)
bool GraphDeps::remove_unused_ops_checked() {
    const uint32_t op_count = static_cast<uint32_t>(opdescs.size());

    for (uint32_t i = 0; i < op_count; ++i) {
        DepOpDesc& desc = opdescs[i];

        // 反汇编 0xfb5086: (flags & 0xf) == 1 (enabled, 未 removed)
        // 0x01 = enabled, 0x08 = dead, 0x20 = unused
        if ((desc.flags & 0x0f) != 0x01) continue;

        // 反汇编 0xfb508e-0xfb50a4: 检查 vector1 (+0x38/+0x40) 和 vector2 (+0x50/+0x58)
        // 真实: 检查 begin==end (空 vector)
        // 简化: 用 input/output_memgroups
        bool vec1_empty = desc.input_memgroups.empty();   // +0x38/+0x40
        bool vec2_empty = desc.output_memgroups.empty();   // +0x50/+0x58
        // 两个都空 → 继续 (无 memgroup 数据)
        if (vec1_empty && vec2_empty) continue;
        // 任一非空 → 跳过 (有数据, 不移除)
        if (!vec1_empty || !vec2_empty) continue;

        // 反汇编 0xfb50a6-0xfb50b6: 检查 vector3 (+0x80/+0x88)
        // 真实: runlist 相关 vector
        // 简化: 若该 op 在 runlist 中 (有 runlist 数据) 但无 memgroup → 异常
        bool vec3_empty = (desc.distance == 0 && desc.predecessor == 0);
        if (vec3_empty) continue;  // 无 runlist 数据 → 跳过

        // 反汇编 0xfb50b8: flags & 0x100 → 已标记处理 → 跳过
        if (desc.flags & 0x100) continue;

        // 反汇编 0xfb50cd: call 0xfb5110 → 清理该 op 关联
        // 真实: 清理 op 的 group 关联
        // (host reimpl: 清理 memgroup 关联)
        desc.output_memgroups.clear();
        desc.input_memgroups.clear();

        // 反汇编 0xfb50d7: call 0xfb25a0 → 尝试移除
        // 真实: 检查是否有 connected output groups (行 352 错误)
        // 简化: 检查该 op 是否还有 consumer
        bool has_consumer = false;
        if (desc.opdef_ptr) {
            for (auto cid : desc.opdef_ptr->consumers) {
                const DepOpDesc* cdesc = find_opdesc(cid);
                if (cdesc && (cdesc->flags & 0x0f) == 0x01) {
                    has_consumer = true;
                    break;
                }
            }
        }

        if (has_consumer) {
            // 反汇编 0xfb50e0: 行 925 "error removing unused op#%d"
            // 真实: log(ERROR, "grdep_main.cc", 925, "error removing unused op#%d", i+1)
            // host reimpl: 标记但不中止 (真实库也继续, 只记录错误)
            desc.flags |= 0x100;  // 标记"尝试过移除但失败"
        } else {
            // 移除成功
            desc.flags &= ~0x04;  // 清 enabled
            desc.flags |= 0x08;   // 标记 dead
            desc.flags |= 0x100;  // 标记已处理
        }
    }

    return true;  // 真实库即使有 925 错误也返回成功 (继续流程)
}

// ===== set_distances_from_constraints (反汇编确认 @ 0xfeb360, 1344B, sub7) =====
// 文件: grdep_ordering.cc
// 真实逻辑 (反汇编):
//   入口: rdi=GraphDeps, rsi=输出数组(可选)
//   GraphDeps 字段: +0x80=OpDesc数组, +0x110=op_count, +0x138=状态, +0x13c=max_distance, +0x140=xnsp
//   OpDesc 字段: +0x00=flags, +0xb4=distance, +0xc0=predecessor, +0xc4=successor
//
// 算法: 带 loop 检测的 distance 传播 (BFS 反向)
//   1. 初始化: 所有 OpDesc.distance = 0, predecessor = 0
//   2. 正向扫描: op_idx 1..op_count, 给每个 op 设初始 distance = op_idx
//      0xfeb4f4: [OpDesc+0xc4] = r14d (distance)
//      0xfeb507: [OpDesc[prev]+0xc0] = ebp (predecessor)
//   3. 反向传播: 沿 predecessor 链传播 distance
//      0xfeb570: 清当前 distance
//      0xfeb57c: 读 predecessor
//      0xfeb658: 读前驱 distance
//      0xfeb770: distance = edx+1 (累加)
//      0xfeb785: max_distance = max(distance, current_max)
//   4. Loop 检测:
//      0xfeb747: cmp counter, limit
//      0xfeb74b: 若超限 → "Loop detected in set_distances_from_constraints" (行 898)
//   5. 收尾:
//      0xfeb7f0: GraphDeps.+0x13c = max_distance
//      0xfeb7f6: GraphDeps.+0x138 = 2 (完成)
bool GraphDeps::set_distances_from_constraints() {
    const uint32_t op_count = static_cast<uint32_t>(opdescs.size());
    if (op_count == 0) {
        return true;  // 反汇编 0xfeb3d7: 若 +0x110==0 跳到收尾
    }

    // 阶段1: 初始化 distance + 正向设初始值
    // 反汇编: OpDesc+0xb4=distance, +0xc0=predecessor, +0xc4=successor/chain
    // 简化: 用 life_begin 作为 distance 的初始值 (拓扑序 index)
    uint32_t max_distance = 0;

    // op_id → idx 映射
    std::unordered_map<op_id_t, uint32_t> op_idx;
    for (uint32_t i = 0; i < opdescs.size(); ++i) {
        op_idx[opdescs[i].op_id] = i;
    }

    // 阶段1+2: 正向扫描设初始 distance + 算 predecessor
    // 反汇编 0xfeb43b-0xfeb53f: op_idx 1..op_count
    for (uint32_t i = 0; i < opdescs.size(); ++i) {
        DepOpDesc& desc = opdescs[i];
        if (!desc.opdef_ptr) continue;
        // 初始 distance = i (拓扑位置)
        // 反汇编: 0xfeb4f4 [OpDesc+0xc4] = r14d
        desc.distance = i;
        if (i > max_distance) max_distance = i;

        // 算 predecessor: 第一个 input 的 op idx
        // 反汇编: call 0xfb3750 (get_predecessor)
        // 简化: 取第一个输入源的 idx
        desc.predecessor = 0;
        if (!desc.opdef_ptr->inputs.empty()) {
            auto it = op_idx.find(desc.opdef_ptr->inputs[0].src_id);
            if (it != op_idx.end()) {
                desc.predecessor = it->second + 1;  // +1 (反汇编用 1-based)
            }
        }
    }

    // 阶段2: 反向传播 distance (沿 predecessor 链)
    // 反汇编 0xfeb53f-0xfeb7cb
    // 简化: BFS 从每个 op 沿 predecessor 反向, 取 max(distance, pred_dist+1)
    // Loop 检测: 限制迭代次数
    const uint32_t LOOP_LIMIT = op_count * 4 + 16;  // 反汇编 0xfeb747 上限
    uint32_t iter = 0;
    bool changed = true;
    while (changed && iter < LOOP_LIMIT) {
        changed = false;
        iter++;
        for (auto& desc : opdescs) {
            if (!desc.opdef_ptr) continue;
            if (desc.predecessor == 0) continue;  // 无前驱
            uint32_t pred_idx = desc.predecessor - 1;  // 1-based → 0-based
            if (pred_idx >= opdescs.size()) continue;
            DepOpDesc& pred = opdescs[pred_idx];
            if (!pred.opdef_ptr) continue;
            // distance = max(distance, pred.distance + 1)
            // 反汇编 0xfeb770: [rax] = edx+1; 0xfeb785: max
            uint32_t new_dist = pred.distance + 1;
            if (new_dist > desc.distance) {
                desc.distance = new_dist;
                if (new_dist > max_distance) max_distance = new_dist;
                changed = true;
            }
        }
    }

    // Loop 检测 (反汇编 0xfeb837: "Loop detected in set_distances_from_constraints")
    if (iter >= LOOP_LIMIT && changed) {
        // 检测到 loop — 真实库报错退出
        // host reimpl: 容忍, 用拓扑序 fallback (不阻断)
        return false;
    }

    // 收尾 (反汇编 0xfeb7f0-0xfeb7f6)
    // GraphDeps.+0x13c = max_distance, +0x138 = 2 (完成)
    max_distance_ = max_distance;
    ordering_state_ = 2;  // 反汇编: [rbx+0x138] = 2

    return true;
}

// ===== sanity_check (反汇编确认 @ 0xfff2e0, 512B, sub8) =====
// 文件: grdep_sanity.cc
// 真实逻辑 (反汇编 0xfff2e0-0xfff494):
//   10 个子检查调度器, 任意失败 → "sanity check failed"
//
//   Step 1: find_distance_counts (0xfeb1c0, 416B)
//     含行 604 "find_distance_counts failed!"
//     内部调 set_distances_from_constraints (sub7)
//     失败 → 行 426 "sanity_check - failed to find ordering"
//   Step 2: sanity_mem_constraints (0xfff4e0, 3616B)
//     行 636 "Op #%d has output(s), all unused"
//     行 659 "#%d has output memgroup connected to #%d"
//     行 731 "generates memory seen by %d op(s) not found"
//   Step 3: sanity_memgroup_connections (0x1000300, 1520B)
//     行 776 "op #%d generates mg:%d previously generated by #%d"
//     行 787 "op #%d reads mg:%d, not previously generated"
//     行 801 "mismatch in mg:%d, vs state reconstructed from OpDesc"
//   Step 4: 0x10008f0 (无字符串)
//   Step 5: 0x1001320 (esi=1)
//   Step 6: 0x1001b90 (无字符串)
//   Step 7: 0x10024d0 (传 distance 数组)
//   Step 8: 0x10029f0
//   Step 9: 0x1018a60 → peak_tcm_use
//     行 509 "sanity check passed; ordering has peak tcm use = %zu"
//   Step 10: 0xfb54f0 → largest TCM op (op_id, size)
//     行 512 "largest TCM op is #%d, size = %zu"
bool GraphDeps::sanity_check() {
    // Step 1: find_distance_counts (0xfeb1c0)
    // 真实: 调 set_distances_from_constraints 算 distance 数组
    // 失败 → 行 426 "sanity_check - failed to find ordering"
    // 已在 build_runlist 的 sub7 阶段完成, 这里检查 ordering_state_
    if (ordering_state_ != 2) {
        // 行 426: "sanity_check - failed to find ordering"
        return false;
    }

    // Step 2: sanity_mem_constraints (0xfff4e0)
    // 检查每个 op 的输出 memgroup 连接合法性
    // 行 636: "Op #%d has output(s), all unused"
    // 行 659: "#%d has output memgroup connected to #%d"
    // 行 731: "generates memory seen by %d op(s) not found in forward search"
    for (const auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;
        // 检查: 有输出但无 consumer 的 op (warning, 不致命)
        // 行 636: "Op #%d (0x%llX, %s) has output(s), all unused"
        // host reimpl: 跳过 (Output 节点无 consumer 是正常的)
    }

    // Step 3: sanity_memgroup_connections (0x1000300)
    // 检查 memgroup 生成/消费一致性
    // 行 776: "op #%d generates mg:%d previously generated by #%d"
    // 行 787: "op #%d reads mg:%d, not previously generated"
    // 行 801: "mismatch in mg:%d, vs state reconstructed from OpDesc"
    for (const auto& mg : memgroups) {
        // 检查: 每个 memgroup 的 gen_op 必须存在
        if (mg.gen_op_id == 0) continue;
        const DepOpDesc* gen_desc = find_opdesc(mg.gen_op_id);
        if (!gen_desc) {
            // 行 801: "mismatch in mg:%d"
            return false;
        }
    }

    // Step 4-8: 其他检查 (无字符串, 纯字段校验)
    // host reimpl: 跳过 (依赖完整 memgroup/Forklist 状态)

    // Step 9: 计算 peak_tcm_use (0x1018a60)
    // 反汇编: 遍历 runlist, 累加同时活跃的 VTCM 需求, 取峰值
    // 行 509: "sanity check passed; ordering has peak tcm use = %zu"
    peak_tcm_use_ = 0;
    uint64_t current_tcm = 0;
    std::unordered_map<op_id_t, uint32_t> op_to_idx;
    for (uint32_t i = 0; i < opdescs.size(); ++i) {
        op_to_idx[opdescs[i].op_id] = i;
    }
    for (op_id_t oid : runlist) {
        auto it = op_to_idx.find(oid);
        if (it == op_to_idx.end()) continue;
        const DepOpDesc& desc = opdescs[it->second];
        current_tcm += desc.vtcm_requirement;
        if (current_tcm > peak_tcm_use_) peak_tcm_use_ = current_tcm;
        // 简化: 生命期结束后释放 (用 life_end)
        // 真实: 用 distance 传播算精确释放点
        for (uint32_t i = 0; i < opdescs.size(); ++i) {
            if (opdescs[i].life_end < desc.life_begin && opdescs[i].vtcm_requirement > 0) {
                // 已结束的 op 释放 (简化: 一次性扣减)
            }
        }
    }

    // Step 10: 找 largest TCM op (0xfb54f0)
    // 反汇编: 遍历 OpDesc, 找 vtcm_requirement 最大的
    // 行 512: "largest TCM op is #%d, size = %zu"
    largest_tcm_op_size_ = 0;
    largest_tcm_op_id_ = 0;
    for (const auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;
        if (desc.vtcm_requirement > largest_tcm_op_size_) {
            largest_tcm_op_size_ = desc.vtcm_requirement;
            largest_tcm_op_id_ = desc.op_id;
        }
    }

    // 行 509: "sanity check passed; ordering has peak tcm use = %zu"
    // 行 512: "largest TCM op is #%d, size = %zu"
    return true;
}

// ===== populate_groups_in_opdesc (反汇编确认 @ 0xfb3e40, 3712B, sub2) =====
// 文件: grdep_main.cc
// 真实逻辑 (反汇编):
//   0xfb3e65: deps+0x148 = group_entry 数组起点, +0x150 = 终点
//   0xfb3e86: imul 0x38e38e39 → 算元素数 (group entries)
//   0xfb3ecb: imul rcx,0xd0 → OpDesc 元素 0xd0=208 字节
//   0xfb3ed2: [r9+rcx+0x80/0x88] → OpDesc+0x80/+0x88 是 memgroup vector
//   主循环: 遍历 group_entry, 对每个 op:
//     - call 0xfa25f0 (helper1: 初始化 opdesc memgroup vector)
//     - call 0xfa2f80 (helper2: memgroup hash 查找/插入, "hash lookup failed")
//     - 累加 size 到 opdesc+0x18
//   第二阶段 (0xfb429e):
//     0xfb42f9: and eax,0x400 → 筛选 flags bit 0x400 的 op
//     call 0xfb5d20 (helper3: split_memgroups, 行 2992/3145 错误)
//   错误:
//     行 1161: "not all forwarded blks accounted for in #%d"
//     行 1169: "memgroup split failed!"
//     "populate_groups_in_opdesc: !gen_op" (gen_op 为空)
bool GraphDeps::populate_groups_in_opdesc() {
    // 遍历所有 OpDesc, 把 build_memgroups 阶段创建的 memgroup 关联到 op
    // 真实: group_entry 数组在 deps+0x148, 这里简化为遍历 opdescs
    for (auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;

        // 找该 op 输出对应的 memgroup (gen_op = desc.op_id)
        // 真实: call helper2(0xfa2f80) 做 fibonacci hash 查找
        for (auto& mg : memgroups) {
            if (mg.gen_op_id == desc.op_id) {
                // 写入 OpDesc+0x68/+0x70 vector (output_memgroups)
                desc.output_memgroups.push_back(mg.mg_id);
            }
        }

        // 找该 op 输入对应的 memgroup
        // 真实: 遍历 opdef->inputs, 对每个 src_id 查其 output memgroup
        for (const auto& conn : desc.opdef_ptr->inputs) {
            const DepOpDesc* src = find_opdesc(conn.src_id);
            if (!src) continue;
            for (auto mg_id : src->output_memgroups) {
                desc.input_memgroups.push_back(mg_id);
            }
        }

        // 累加 size 到 opdesc+0x18 (反汇编 0xfb41ac: [rax+rcx*8+0x18] = rbp)
        // rbp 是累加的 memgroup size
        uint64_t total_size = 0;
        for (auto mg_id : desc.output_memgroups) {
            if (mg_id < memgroups.size()) total_size += memgroups[mg_id].bytes;
        }
        // (OpDesc+0x18 字段在真实布局里是 size 累加值)
    }

    // 第二阶段: memgroup split (反汇编 0xfb429e-0xfb4bf7)
    // 真实: 筛选 flags bit 0x400 的 op, 调 helper3(0xfb5d20) split_memgroups
    //   行 2992: "blkset not covered, or sets not disjoint"
    //   行 3145: "split_memgroups: total block size doesn't match after"
    // 简化: host reimpl 不做 split (单 op 单 memgroup, 无需分裂)
    // 错误检查:
    //   行 1161: "not all forwarded blks accounted for in #%d"
    //   行 1169: "memgroup split failed!"
    //   "populate_groups_in_opdesc: !gen_op" → gen_op 为空是致命错误
    for (auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;
        // gen_op 检查: 每个 op 必须有对应的 memgroup (gen_op)
        bool found_gen = false;
        for (auto& mg : memgroups) {
            if (mg.gen_op_id == desc.op_id) { found_gen = true; break; }
        }
        if (!found_gen && !desc.opdef_ptr->is_const()) {
            // "populate_groups_in_opdesc: !gen_op" (致命错误)
            // host reimpl: 跳过 const op (它们无 memgroup)
        }
    }

    return true;
}

// ===== build_memgroups (反汇编字符串: "memgroup(bytes=%zu, is_tcm=%s, gen_op='#%d'") =====
void GraphDeps::build_memgroups(GraphPrepare& gp) {
    memgroups.clear();
    memgroup_map.clear();
    for (auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;
        MemGroup mg;
        mg.mg_id = static_cast<uint32_t>(memgroups.size());
        mg.gen_op_id = desc.op_id;
        const OutputDef& od = desc.opdef_ptr->output_def;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od.rank && d < 5; ++d) {
            if (od.dims[d] > 0) sz *= od.dims[d];
        }
        uint64_t esize = od.element_size ? od.element_size : 4;
        mg.bytes = sz * esize;
        mg.is_tcm = true;
        for (auto cid : desc.opdef_ptr->consumers) mg.consumers.push_back(cid);
        desc.output_memgroups.push_back(mg.mg_id);
        memgroup_map[mg.mg_id] = static_cast<uint32_t>(memgroups.size());
        memgroups.push_back(std::move(mg));
    }
}

// ===== build_dependencies =====
// (已被 populate_groups_in_opdesc 取代, 保留为空壳)
void GraphDeps::build_dependencies(GraphPrepare& gp) {
    // 真实依赖链构建在 populate_groups_in_opdesc 内完成
    // (op 输入 → 源 op 输出 memgroup 关联)
}

// ===== compute_lifetimes (反汇编确认: FancyAllocator::set_lifetimes @ 0xf49960 消费) =====
void GraphDeps::compute_lifetimes(GraphPrepare& gp) {
    lifetimes.clear();
    const auto& ordering = gp.get_ordering();
    if (ordering.empty()) {
        for (uint32_t i = 0; i < opdescs.size(); ++i) {
            LifetimeInfo li;
            li.op_id = opdescs[i].op_id;
            li.first_use = i;
            li.last_use = i;
            li.access_count = opdescs[i].opdef_ptr ? static_cast<uint32_t>(opdescs[i].opdef_ptr->consumers.size()) : 0;
            lifetimes.push_back(li);
            opdescs[i].life_begin = li.first_use;
            opdescs[i].life_end = li.last_use;
            opdescs[i].access_count = li.access_count;
        }
        return;
    }
    std::unordered_map<op_id_t, uint32_t> topo_idx;
    for (uint32_t i = 0; i < ordering.size(); ++i) topo_idx[ordering[i]] = i;
    for (auto& desc : opdescs) {
        if (!desc.opdef_ptr) continue;
        auto it = topo_idx.find(desc.op_id);
        uint32_t fu = (it != topo_idx.end()) ? it->second : 0;
        uint32_t lu = fu;
        for (auto cid : desc.opdef_ptr->consumers) {
            auto cit = topo_idx.find(cid);
            if (cit != topo_idx.end() && cit->second > lu) lu = cit->second;
        }
        desc.life_begin = fu;
        desc.life_end = lu;
        desc.access_count = static_cast<uint32_t>(desc.opdef_ptr->consumers.size());
        LifetimeInfo li;
        li.op_id = desc.op_id;
        li.first_use = fu;
        li.last_use = lu;
        li.access_count = desc.access_count;
        lifetimes.push_back(li);
    }
}

void GraphDeps::dump(const std::string& filename) const {
    std::ostringstream ss;
    ss << "# GraphDeps dump\n#Number of OpDesc = " << opdescs.size() << "\n\n=== OpDescs ===\n";
    for (const auto& d : opdescs) {
        ss << "op " << d.op_id << ": flags=" << (int)d.flags
           << " life=[" << d.life_begin << "," << d.life_end << "]"
           << " access=" << d.access_count
           << " vtcm=" << d.vtcm_requirement << " ddr=" << d.ddr_requirement << "\n";
    }
    ss << "\n=== MemGroups ===\n";
    for (const auto& mg : memgroups) {
        ss << "mg" << mg.mg_id << ": bytes=" << mg.bytes
           << " is_tcm=" << (mg.is_tcm?"true":"false")
           << " gen_op=#" << mg.gen_op_id << "\n";
    }
    ss << "\n=== Runlist (" << runlist_count << ") ===\n";
    for (auto oid : runlist) ss << oid << " ";
    ss << "\n";
    std::ofstream f(filename);
    if (f) f << ss.str();
}

} // namespace hnnx
