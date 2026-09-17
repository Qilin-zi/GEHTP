// ============================================================================
// M37 反汇编保真实现 —— 见 include/hnnx/vtcm/phys_alloc_m37.hpp 头注
// 证据基线: audit_verify/reports/M37_phys_alloc_in_runlist_disasm.md (2026-08-28)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// 未逐指令确认处一律注明"遗留"，禁止臆测补齐。
// ============================================================================
#include "hnnx/vtcm/phys_alloc_m37.hpp"
#include <cstdio>

namespace hnnx {

std::string m37_alloc_error_log(op_id_t id) {
    // fmt@0x461d95b: '%s:2173::ERROR:could not allocate memory for op %llx!!\n\n'
    // file@0x461dff6 = 'graph_prepare.cc' [0xf72e2f/0xf72e36 两个 lea]
    char buf[128];
    std::snprintf(buf, sizeof buf,
                  "graph_prepare.cc:2173::ERROR:could not allocate memory for op %llx!!\n\n",
                  static_cast<unsigned long long>(id));
    return buf;
}

PhysAllocRunlistResult phys_alloc_in_runlist_disasm(
    const std::vector<PhysAllocOp>& runlist,
    const PhysAllocRunlistConfig& cfg,
    const std::unordered_map<op_id_t, PhysAllocOp>& op_map) {
    PhysAllocRunlistResult res;
    if (runlist.empty()) return res;      // count==0 → return 0 [0xf72b0b → 0xf72e1b]

    // this+0x74c8 后的 unordered_map（记录级：id→条目）。
    // 真代码每次查找都是完整内联 find（桶定位+链走，幂/非幂桶两路）——
    // 这里用标准库 unordered_map 等价覆盖；两次独立查找的语义保留（count + at）
    std::unordered_map<op_id_t, const PhysAllocSupertileEntry*> table;
    if (cfg.supertile_mode && !cfg.supertile_entries.empty()) {   // 三重门的前两重
        for (const auto& e : cfg.supertile_entries) table.emplace(e.id, &e);
    }

    for (const auto& op : runlist) {      // [0xf72b20 循环头 / 0xf72cf4 步进]
        // ---- supertile 成员扇出 ----
        // 门 1: this+0x6208 != 0 [0xf72b23]；门 2: 桶数 != 0 [0xf72b47]
        //   （表仅在 supertile_mode 且非空时装载，与两门合取等价）
        if (!cfg.supertile_mode || cfg.supertile_entries.empty()) goto call_self;
        {
            // id = op->id(*this) [0xf72b37]
            // 查找 #1 = count(id): 桶定位（幂: id&(n-1) [0xf72bb5] / 非幂: id%n
            //   [0xf72bee，32 位 div 快路径 0xf72bd6]）+ 链走（hash@8 [0xf72c42]
            //   == id 且 key@0x10 [0xf72c30] == id → 命中；异桶节点止链）
            const auto it1 = table.find(op.id);
            if (it1 == table.end()) goto call_self;      // 未命中 → 不扇出
            // 查找 #2 = at(id): 同一桶/链再走一遍 [0xf72d06-0xf72dfa]；
            //   未命中 → 'unordered_map::at: key not found' @0x398e262
            //   → call 0x8c6970 抛出 [0xf72e5e]。count 后 at 必中（键唯一），
            //   std::unordered_map::at 等价承载该不变式
            const PhysAllocSupertileEntry& e = *table.at(op.id);
            // 成员 vector node+[0x38..0x40) [0xf72d97/0xf72d9b]，步长 8 [0xf72db0]
            for (size_t k = 0; k < e.member_ids.size(); ++k) {
                if (e.member_null.size() > k && e.member_null[k])
                    continue;              // *q == nullptr → continue [0xf72dc1-0xf72dc4]
                // m->vtable[0x48](m, this) [0xf72dcc] —— 返回值被丢弃:
                //   call 后直接 jmp 回循环 [0xf72dcf]，无 test eax（成员失败静默）
                res.called_ids.push_back(e.member_ids[k]);
            }
        }
        // ---- 本体分配（扇出后仍必调）----
    call_self:
        // op->vtable[0x48](op, this) [0xf72ce0-0xf72ce9]
        res.called_ids.push_back(op.id);
        if (op.alloc_rc != 0) {            // [0xf72cec testl; jne]
            // get_extra_info(this, op) [0xf72e27 PLT 0x6eed80]；
            // qnndsp_log(0, fmt, file, extra->[0], "") [0xf72e48]
            res.rc = op.alloc_rc;
            res.error_log = m37_alloc_error_log(op.id);
            return res;                    // 短路返回 [0xf72e4d]
        }
    }
    return res;                            // 全成功 → 0 [0xf72e1b ebp=0]
}

} // namespace hnnx
