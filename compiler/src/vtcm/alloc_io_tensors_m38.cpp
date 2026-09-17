// ============================================================================
// M38 反汇编保真实现 —— 见 include/hnnx/vtcm/alloc_io_tensors_m38.hpp 头注
// 证据基线: audit_verify/reports/M38_allocate_io_tensors_disasm.md (2026-08-28)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// 未逐指令确认处一律注明"遗留"，禁止臆测补齐。
// ============================================================================
#include "hnnx/vtcm/alloc_io_tensors_m38.hpp"
#include <cstdio>

namespace hnnx {

std::string m38_input_removed_warning(op_id_t id) {
    // fmt@0x4620185: '%s:6676:WARNING:graph Input node %llX has been removed\n'
    char buf[128];
    std::snprintf(buf, sizeof buf,
                  "graph_prepare.cc:6676:WARNING:graph Input node %llX has been removed\n",
                  static_cast<unsigned long long>(id));
    return buf;
}

std::string m38_output_removed_warning(op_id_t id) {
    // fmt@0x46201f9: '%s:6740:WARNING:graph Output node %llX has been removed\n'
    char buf[128];
    std::snprintf(buf, sizeof buf,
                  "graph_prepare.cc:6740:WARNING:graph Output node %llX has been removed\n",
                  static_cast<unsigned long long>(id));
    return buf;
}

std::string m38_fatal_error_log() {
    // fmt@0x46201bd: '%s:6696::ERROR:Fatal error in allocate_io_tensors function\n'
    return "graph_prepare.cc:6696::ERROR:Fatal error in allocate_io_tensors function\n";
}

// ---- A 段: 输入节点 [0xf69965-0xf69a41) ------------------------------------
// 返回 false = collect_multi_outputdef 失败（调用方带 rc 返回）
static bool m38_input_part(M38Graph& g, M38Result& res) {
    const op_id_t in_id = g.input_node_id;             // this+0x5340 [0xf69965]
    if (in_id == 0) return true;                       // 0 → 直进 B 段 [0xf6996f]
    auto it = g.opdefs.find(in_id);                    // 树查 this+0x6d60 [0xf69975-0xf699b9]
    if (it == g.opdefs.end()) {
        // WARNING 6676（真码先过 GetLogPriorityLevel>0 [0xf69a0e]）[0xf69a31]
        res.warning_log += m38_input_removed_warning(in_id);
        g.input_node_id = 0;                           // this+0x5340 = 0 [0xf69a36]
        res.input_node_cleared = true;
        return true;                                   // 不 return，继续 B 段
    }
    // 命中: collect_multi_outputdef(this, OpDef, &vec, 0, 0) [0xf69dc]
    //   内部遗留（报告 §8.1）；记录级恒成功 → 建 tensor-info 条目
    res.tensor_info_ids.push_back(it->first);          // 逐项 0xd129e0→this+0x5370 [0xf69c05/0xf69c1a]
    return true;
}

M38Result allocate_io_tensors_disasm(M38Graph& g) {
    M38Result res;

    // ---- A 段 ----
    if (!m38_input_part(g, res)) return res;           // 非零 rc 提前返回 [0xf69e4-0xf6a04]

    // ---- B 段: 输出节点段头 [0xf69a41-0xf69b87) ----
    const op_id_t out_id = g.output_node_id;           // this+0x5348 [0xf69a41]
    if (out_id == 0) { res.rc = 0; return res; }       // 0 → return 0 [0xf69a4e→0xf69b87]
    auto out_it = g.opdefs.find(out_id);               // [0xf69a54-0xf69aa2]
    if (out_it == g.opdefs.end()) {
        res.warning_log += m38_output_removed_warning(out_id); // 6740 [0xf69b74]
        g.output_node_id = 0;                          // this+0x5348 = 0 [0xf69b79]
        res.output_node_cleared = true;
        res.rc = 0;
        return res;                                    // 提前 return 0 [0xf69b87]
    }
    M38OpDef& opdef1 = out_it->second;                 // OpDef#1 [0xf69a9b]

    // new(n*8) 清零表 [0xf69af4-0xf69d4e]；n = OpDef#1+0x30 向量项数 [0xf69ab3-0xf69ad0]
    std::vector<op_id_t> new_ids(opdef1.ops.size(), 0);
    // conditionally_validate_single_quant（内部遗留）[0xf69d55-0xf69d58]
    bool made_new_node = false;                        // 旗 rsp+0x10 = 0 [0xf69d7f-0xf69d81]

    // ---- C 段: 主循环 [0xf69d90-0xf69fdf) ----
    for (size_t i = 0; i < opdef1.ops.size(); ++i) {
        const op_id_t old_id = opdef1.ops[i];          // (rbp+i*8) [0xf69da1]
        auto it2 = g.opdefs.find(old_id);              // 树查 [0xf69db0-0xf69dd2]
        if (it2 == g.opdefs.end()) {
            // E 段致命错: qnndsp_log(0, fmt6696, ...) → return -1 [0xf6a037-0xf6a059]
            res.rc = -1;
            res.error_log = m38_fatal_error_log();
            return res;
        }
        M38OpDef& opdef2 = it2->second;                // OpDef#2 [0xf69ddc]
        op_id_t slot_id = old_id;                      // bit0 清 → 老 id [0xf69dfe]

        if (opdef2.needs_cast) {                       // testb $1,0x9(%r12) [0xf69df6]
            // a. q::QNN_Cast 驻留名惰性初始化（guard@0x6245270）[0xf69e08-0xf6a032] —— 记录级一次性
            // b. 新 id: (seq++<<32)|(old_id&0xFFFFFFFF) [0xf69e16-0xf69e23; new_id@0xd2f680]
            slot_id = g.new_id(old_id);
            // c. OutputDef 改写循环 [0xf69e30-0xf69f74]，0x50 步长 [0xf69f50]
            //   记录级: 真码就地改写 OpDef#2 的 outputdefs；此处同时留痕 (旧枚举, 新枚举)
            for (M38OutputDef& od : opdef2.outdefs) {
                const int old_enc = od.encoding;
                if (od.encoding == 1) {
                    od.adjust -= 0x80;                 // [0xf69f4a addl $-0x80]
                    od.encoding = 7;                   // [0xf69f4d movl $7]
                } else if (od.encoding == 2) {
                    od.adjust -= 0x8000;               // [0xf69f6b]
                    od.encoding = 3;                   // [0xf69f6f]
                }
                res.enum_rewrites.emplace_back(old_enc, od.encoding);
                res.adjust_after.push_back(od.adjust);
            }
            // d. make_op_node_impl(7 参) [0xf69e4d-0xf69e7b] —— 内部遗留；失败开关→return -1
            if (g.make_op_node_fails) {
                res.rc = -1;                           // [0xf69e84→0xf69fe4 r13d=-1]
                return res;
            }
            // f. note_new_node(行号 0x1a38=6712) [0xf69e8d-0xf69e9f]; 旗=1 [0xf69ea4]
            made_new_node = true;
        }
        // 5. 汇合: 新表[i] = 新/老 id [0xf69eab-0xf69eb0]
        new_ids[i] = slot_id;
        // 6/7. 复制首个 OutputDef（0x50B）→ 0xd129e0 → this+0x5370 追加 [0xf69eb4-0xf69f35]
        //   （表空时真码解引用空指针——视为不变式，记录级跳过）
        if (!opdef2.outdefs.empty()) res.tensor_info_ids.push_back(opdef2.id);
    }

    // 循环毕: 旗为 0 → 释放新表 return 0（不进 D）[0xf6a07b-0xf6a08f]
    if (!made_new_node) { res.rc = 0; return res; }

    // ---- D 段: 新 Output 节点 [0xf6a091-0xf6a289) ----
    // q::Output 驻留名惰性初始化（guard@0x6245280, 'Output'@0x4696e72）[0xf6a298-0xf6a2d2]
    res.new_output_node_id = g.new_id(out_id);         // 第二个 new_id [0xf6a09f-0xf6a0ae]
    res.final_op_ids = std::move(new_ids);             // 节点 move 接管新表 [0xf6a102-0xf6a133]
    // 插入节点表 0x10baa00(this+0x6d58, 节点, out_id, this+0x45dc==1) [0xf6a195-0xf6a1b4]
    if (g.insert_rc != 0) {
        res.rc = g.insert_rc;                          // 失败 → 节点析构 → return rc [0xf6a1c3-0xf6a1dc]
        return res;
    }
    // 树查老 Output 节点 id [0xf6a1f2-0xf6a234]
    auto old_it = g.opdefs.find(out_id);
    if (old_it != g.opdefs.end()) {
        res.old_output_flagged_0x3 = true;             // OpDef#1+0x9 |= 3 [0xf6a236]
        // 0xf7f000(this+0x6da0, &OpDef#1+0x20) = push_back [0xf6a238-0xf6a248] —— 记录级留痕
    } else {
        res.old_output_marked_deletable = true;        // mark_op_deletable(out_id) [0xf6a24f-0xf6a252]
    }
    res.collect_deletable_ran = true;                  // collect_deletable_nodes [0xf6a257]
    res.rc = 0;                                        // return 0 [0xf6a278]
    return res;
}

} // namespace hnnx
