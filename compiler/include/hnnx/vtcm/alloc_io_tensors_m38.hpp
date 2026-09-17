#pragma once
// ============================================================================
// M38 反汇编保真实现：GraphPrepare::allocate_io_tensors @0xf69940（2687B 全指令）
// 证据: audit_verify/reports/M38_allocate_io_tensors_disasm.md
//       audit_verify/asm/f3/M38_allocate_io_tensors.asm
// 规则: 每个逻辑段标注 [0x地址]；未在指令级确认的部分显式标注"遗留/记录级"。
//
// 一句话语义: 为 Input/Output 端点补建张量；Output 节点 op 列表中 OpDef+0x9 bit0
// 置位的每个 op 生成一个 q::QNN_Cast 节点接替（id = Graph::new_id(老id)），最后
// new 一个 q::Output 节点携带（可能更新的）id 数组插入节点表，旧 Output 节点标记可删。
// ============================================================================
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hnnx {

// OutputDef 记录级替身（真结构 0x50B；只有 +4 枚举与 +0x48 调整字被本函数触碰）
struct M38OutputDef {
    int encoding = 0;    // +4 [改写: 1→7 [0xf69f4d], 2→3 [0xf69f6f]]
    int adjust = 0;      // +0x48 [改写: -=0x80 [0xf69f4a] / -=0x8000 [0xf69f6b]]
};

// OpDef 记录级替身（本函数触及的字段）
struct M38OpDef {
    op_id_t id = 0;
    bool needs_cast = false;            // +0x9 bit0 [0xf69df6 testb $1,0x9(%r12)]
    uint64_t field_0x20 = 0;            // +0x20（make_op_node_impl 末参 [0xf69e75]）
    std::vector<op_id_t> ops;           // +0x30..0x38 vector<OpRef>（8B/项）[0xf69ab3/0xf69ac9]
    std::vector<M38OutputDef> outdefs;  // get_outputdefs 结果（0x50B/项）[0xf69df1]
};

// GraphPrepare 记录级替身
struct M38Graph {
    op_id_t input_node_id = 0;          // this+0x5340 [0xf69965]
    op_id_t output_node_id = 0;         // this+0x5348 [0xf69a41]
    std::map<op_id_t, M38OpDef> opdefs; // this+0x6d60 的树 [0xf69975 等]
    uint32_t new_id_seq = 0;            // Graph+0x10 计数器 [0xd2f682-0xd2f688]
    bool config_45dc_eq1 = false;       // this+0x45dc==1（插入第 4 参）[0xf6a1a5]
    bool make_op_node_fails = false;    // 记录级开关：make_op_node_impl 返回 0 [0xf69e84]
    int insert_rc = 0;                  // 记录级开关：0x10baa00 的返回值 [0xf6a1b4]

    // Graph::new_id @0xd2f680（19B 全解）:
    //   (uint64)(seq++) << 32 | (hint & 0xFFFFFFFF) [0xd2f68b-0xd2f68f]
    op_id_t new_id(op_id_t hint) {
        const op_id_t r = (static_cast<op_id_t>(new_id_seq) << 32) | (hint & 0xFFFFFFFFu);
        ++new_id_seq;
        return r;
    }
};

struct M38Result {
    int rc = 0;                          // 返回值 eax [0xf69b9e]
    op_id_t new_output_node_id = 0;      // D 段第二个 new_id（未建则 0）[0xf6a0a9]
    std::vector<op_id_t> final_op_ids;   // 存入新 Output 节点的（新/老）id 数组 [0xf6a0e6/0xf6a130]
    std::vector<op_id_t> tensor_info_ids;// this+0x5370 追加序（每次 0xd129e0 的来源 OpDef）[0xf69c05/0xf69f00]
    std::vector<std::pair<int, int>> enum_rewrites;    // (旧+4, 新+4)，Cast 改写 [0xf69f40-0xf69f74]
    std::vector<int> adjust_after;       // 改写后的 +0x48 终值（与 enum_rewrites 等长）
    bool old_output_marked_deletable = false; // mark_op_deletable 路径 [0xf6a24f]
    bool old_output_flagged_0x3 = false;     // OpDef#1+0x9|=3 路径 [0xf6a236]
    bool collect_deletable_ran = false;      // [0xf6a257]
    std::string warning_log;             // 6676/6740 WARNING（真码有 log-level 门 [0xf69a0e]）
    std::string error_log;               // 6696 致命错 [0xf6a059]
    bool input_node_cleared = false;     // this+0x5340 清零 [0xf69a36]
    bool output_node_cleared = false;    // this+0x5348 清零 [0xf69b79]
};

M38Result allocate_io_tensors_disasm(M38Graph& g);

// 两条 WARNING 文本（fmt@0x4620185 / 0x46201f9 逐字节）
std::string m38_input_removed_warning(op_id_t id);
std::string m38_output_removed_warning(op_id_t id);
// 致命错文本（fmt@0x46201bd 逐字节）
std::string m38_fatal_error_log();

} // namespace hnnx
