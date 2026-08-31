#pragma once
// ============================================================================
// M37 反汇编保真实现：GraphPrepare::phys_alloc_in_runlist @0xf72b00（874B 全指令）
// 证据: audit_verify/reports/M37_phys_alloc_in_runlist_disasm.md
//       audit_verify/asm/f3/M37_phys_alloc_in_runlist.asm
// 规则: 每个逻辑段标注 [0x地址]；未在指令级确认的部分显式标注"未完全理解"
//
// 语义（记录级替身）: 遍历 runlist，每个 op 先做可选的 supertile 成员扇出
// （count() 门 + at() 取成员表），再调本体的分配槽（vtable+0x48 替身为
// alloc_rc 字段）；本体返回非 0 → 记 ERROR 日志（graph_prepare.cc:2173）并短路。
// ============================================================================
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnnx {

// runlist 里的一个 Op（记录级）
struct PhysAllocOp {
    op_id_t id = 0;        // Op::id(Graph) [0xf72b37 PLT 0x6f1940]
    int alloc_rc = 0;      // vtable+0x48 槽的返回值（0=成功）[0xf72ce9]
};

// supertile 成员表的一个条目 —— 真 so 中是 unordered_map 节点:
//   node+0x8=缓存哈希(==id) / +0x10=key(==id) / +0x38..0x40=vector<Op*>
//   （key 到 vector 之间的 node+0x18..0x37 未读，遗留）
struct PhysAllocSupertileEntry {
    op_id_t id = 0;
    // 成员 Op 表：null 项在真代码里被 continue 跳过 [0xf72dc1-0xf72dc4]
    std::vector<op_id_t> member_ids;
    std::vector<bool> member_null;   // 与 member_ids 等长；true = *q == nullptr
};

struct PhysAllocRunlistConfig {
    // this+0x6208 字节门 [0xf72b23 cmpb $0,0x6208(%r14)]：false → 直接走本体分配
    bool supertile_mode = false;
    // this+0x74c8 指针后的 map（记录级：桶数!=0 ⇔ 表非空 [0xf72b47]）
    std::vector<PhysAllocSupertileEntry> supertile_entries;
};

struct PhysAllocRunlistResult {
    int rc = 0;                            // 0 或首个失败 op 的 alloc_rc [0xf72e4d]
    std::vector<op_id_t> called_ids;       // vtable+0x48 调用序列（成员在前、本体在后）
    std::string error_log;                 // rc!=0 时的 graph_prepare.cc:2173 日志
};

// 三段式主体（§1 总控流的逐段移植）。
// op_map: id → Op（成员扇出的 vcall 目标；成员的返回值被丢弃 [0xf72dcf 无 test]）
PhysAllocRunlistResult phys_alloc_in_runlist_disasm(
    const std::vector<PhysAllocOp>& runlist,
    const PhysAllocRunlistConfig& cfg,
    const std::unordered_map<op_id_t, PhysAllocOp>& op_map);

// ERROR 日志文本 [fmt@0x461d95b 逐字节验证; file 'graph_prepare.cc'@0x461dff6]
// %llx 实参 = get_extra_info(this,op)->[0] [0xf72e2c]（字段含义半证，记录级取 op id）
std::string m37_alloc_error_log(op_id_t id);

} // namespace hnnx
