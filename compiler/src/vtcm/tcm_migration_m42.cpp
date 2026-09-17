// ============================================================================
// M42 反汇编保真实现 —— 见 include/hnnx/vtcm/tcm_migration_m42.hpp 头注
// 证据基线: audit_verify/reports/M42_tcm_migration_disasm.md (2026-08-28)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// helper 内部为记录级钩子，禁止臆测补齐（M42b 遗留见报告 §5）。
// ============================================================================
#include "hnnx/vtcm/tcm_migration_m42.hpp"
#include <cstdio>

namespace hnnx {

// ---- 日志文本（指令级格式串）-----------------------------------------------
// 3472: "%s:3472::ERROR:Operator named %s (0x%llx) not sufficiently tiled
//        to fit in TCM (0x%u). Requires 0x%zu bytes." @0x55b5ba1 [0x1322f95]
std::string m42_err_3472(const std::string& name, uint64_t opid, uint32_t full,
                         uint64_t demand) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "tcm_migration.cc:3472::ERROR:Operator named %s (0x%llx) "
                  "not sufficiently tiled to fit in TCM (0x%x). Requires 0x%zx bytes.",
                  name.c_str(), (unsigned long long)opid, full,
                  (size_t)demand);
    return buf;
}

// 3459: "%s:3459:WARNING:Operator named %s (0x%llx) not sufficiently tiled
//        to fit well in TCM, some operands spilled to main memory." @0x55b5b25
std::string m42_warn_3459(const std::string& name, uint64_t opid) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "tcm_migration.cc:3459:WARNING:Operator named %s (0x%llx) "
                  "not sufficiently tiled to fit well in TCM, some operands "
                  "spilled to main memory.",
                  name.c_str(), (unsigned long long)opid);
    return buf;
}

std::string m42_err_1819() {   // @0x55b5699 [0x1321bf6]
    return "tcm_migration.cc:1819::ERROR:Fatal error in TCMMigration::run "
           "function due to registration errors.";
}
std::string m42_err_1824() {   // @0x55b56ee [0x1321c80]
    return "tcm_migration.cc:1824::ERROR:Inside TCMMigration::run() function: "
           "scan_input_graph() call failed.";
}
std::string m42_err_1829() {   // @0x55b5742 [0x1323120]
    return "tcm_migration.cc:1829::ERROR:Inside TCMMigration::run() function: "
           "propagate_attributes() call failed.";
}

// ---- 早退尾（P2/P3/P8 共用）-------------------------------------------------
// mode 0x11 配置 [0x1321c32/0x132315c] → 0xf19910 → 公共尾 0x1323191
static void m42_early_exit(M42Trace& tr) {
    tr.early_exit_mode = 0x11;                               // [0x1321c32 movl $0x11]
}

// ---- 主外壳 -----------------------------------------------------------------
void m42_tcm_migration(M42Ctx& ctx, M42BudgetConfig& budget, const M42Hooks& hk,
                       uint32_t vtcm_tile_size, bool cleanup, M42Trace& tr) {
    // P0 前处理 [0x13219c0-0x1321a29]
    if (hk.pre_dce) hk.pre_dce(ctx, false);                  // 0x131fc90 + DCE [0x13219f2-0x13219fa]
    bool dyn = hk.is_dynamic_active ? hk.is_dynamic_active(ctx) : false; // [0x1321a02]
    if (dyn && hk.pre_dce) hk.pre_dce(ctx, true);            // 0xf17d70 + 二次 DCE [0x1321a0b-0x1321a1a]
    tr.time_points.push_back("prepare_before_tcm_migration"); // @0x461de3a [0x1321a29]
    (void)cleanup;                                           // [rsp+0x34] 尾声支，记录级

    // P1 预算 [0x1321bc8-0x1321bda]
    budget.full = vtcm_tile_size;                            // get_vtcm_tile_size [0x1321bc8]
    budget.three_quarters = m42_three_quarters(budget.full); // [0x1321bd4-0x1321bda]

    // P2 TLS 门 [0x1321be1-0x1321c67]
    if (hk.tls_busy && hk.tls_busy()) {
        tr.log += m42_err_1819() + "\n";                     // [0x1321bf6-0x1321c0f]
        m42_early_exit(tr);                                  // → 公共尾 [0x1321c67]
        tr.time_points.push_back("prepare_tcm_migration");   // [0x132319b]
        return;
    }

    // P3 scan_input_graph [0x1321c6c-0x1321cf1]
    if (hk.scan_input_graph) hk.scan_input_graph(ctx);
    tr.scan_ran = true;
    if (ctx.errcount != 0) {                                 // [0x1321c76]
        tr.log += m42_err_1824() + "\n";                     // [0x1321c80-0x1321c99]
        m42_early_exit(tr);
        tr.time_points.push_back("prepare_tcm_migration");
        return;
    }

    // 参考布局码：flag 置 1 → 0x101 [0x1322db1-0x1322dcd / 0x1322c2a-0x1322c4d]
    const uint16_t ref1 = ctx.flag_1c ? 0x101 : ctx.default_code1;
    const uint16_t ref2 = ctx.flag_1e ? 0x101 : ctx.default_code2;

    // P4 第一遍（正向）[0x1322495-0x13224f7]
    for (auto& op : ctx.ops) {
        if (op.kind - 0xa >= 3) continue;                    // 类型门 [0x1322500-0x132250a]
        uint64_t est = hk.estimate_op ? hk.estimate_op(ctx, op) : 0; // 0x13318c0 [0x13227c6]
        ++tr.full_budget_cmp;
        if (est > budget.full) {                             // vs 全量 [0x13227d2]
            // 0x1330a80(rsp+0x170, &op) 旁置记录 [0x13227db-0x13227eb]（记录级：留痕于 estimate）
        } else if (hk.transform) {
            hk.transform(ctx, &op, /*crouton cfg*/ 0);       // 0x1332210 [0x13224b5-0x13224e8]
        }
    }

    // P5 第二遍（逆向）[0x132283d-0x132291c] —— 记录级: 类型分派 0x1331290/0x1334cf0
    for (auto it = ctx.ops.rbegin(); it != ctx.ops.rend(); ++it) {
        M42Op& op = *it;
        if (op.kind == 0xa) {
            // 谓词 0x1331290 扫 op+0x80 端口表 [0x13228c0-0x132290a]（记录级：无操作）
        } else if (op.kind == 0xb && !op.tensors.empty() &&
                   op.tensors[0].tensor_kind != 0xa) {
            // 0x1334cf0(ctx, op) [0x1322896-0x13228af]（记录级：无操作）
        }
    }

    // P6 主判决遍（逆向）[0x1322921-0x1323027]
    for (auto it = ctx.ops.rbegin(); it != ctx.ops.rend(); ++it) {
        if (ctx.stop_flag) continue;                         // ctx+0x14 门 [0x1322960]
        M42Op& op = *it;
        // 头门 +0x5e/+0x5f [0x1322974-0x1322980]（记录级：模型不区分）
        // B 支: +0x5d≠0 且 +0x5c==1 → op 级估计 vs 全量 [0x1322b5f-0x1322ba3]
        if (op.f5d != 0 && op.f5c == 1) {
            uint64_t est = hk.estimate_op ? hk.estimate_op(ctx, op) : 0; // 0x13318c0(…,1)
            ++tr.full_budget_cmp;
            if (est > budget.full) {                         // [0x1322ba0]
                if (hk.transform) hk.transform(ctx, &op, ctx.global_d44); // [0x1322bdb]
            }
            continue;                                        // B 支不再走张量累计 [0x1322be0]
        }
        // A 支: 张量累计 acc += size_after - size_before [0x1322be5-0x1322f41]
        uint64_t acc = 0;
        for (auto& t : op.tensors) {
            if (!m42_layout_code_match(ref1, t.code1)) continue; // 码匹配 [0x1322c89-0x1322cee]
            if ((t.tensor_kind & ~1u) != 0xa) continue;      // [0x1322d35 andl $-2]
            t.size_before = hk.tensor_size ? hk.tensor_size(ctx, t, false) : 0; // [0x1322d4c]
            if (hk.transform) hk.transform(ctx, &t, ctx.global_d40);           // [0x1322d85]
            t.transformed = true;
            t.size_after = hk.tensor_size ? hk.tensor_size(ctx, t, false) : 0;  // [0x1322d9a]
            acc += t.size_after - t.size_before;             // [0x1322d9f-0x1322da7]
        }
        op.demand = acc;
        // 判决 [0x1322f7c-0x1322ff7]
        ++tr.full_budget_cmp;
        if (op.demand > budget.full) {                       // cmpq %r9,%rcx; jbe [0x1322f8b]
            tr.log += m42_err_3472(op.name, op.opid, budget.full, op.demand) + "\n";
            ++ctx.errcount;                                  // [0x1322fbd]
        } else if (hk.log_priority && hk.log_priority() > 0) { // [0x1322fc7-0x1322fce]
            tr.log += m42_warn_3459(op.name, op.opid) + "\n";  // [0x1322fd5]
        }
    }

    // P7 第三遍（正向）[0x132302d-0x13230b7] —— 0xedfea0/0x1335830（记录级：无操作）

    // P8 错误计数边界 [0x1323112]
    if (ctx.errcount != 0) {
        tr.log += m42_err_1829() + "\n";                     // [0x1323120-0x1323139]
        m42_early_exit(tr);
        tr.time_points.push_back("prepare_tcm_migration");
        return;
    }

    // P9 好路径 [0x1323356-0x1323c59]：stop==0 直达管线；否则属性传播（记录级）
    // P10 转换/对齐四连 [0x1323be9-0x1323c0c]
    if (ctx.stop_flag == 0) {
        static const int stages[4] = {0, 1, 2, 3};           // 0x132a8b0/20/0x132bbf0/0x132c1f0
        for (int s : stages) {
            if (hk.convert_pipeline) hk.convert_pipeline(ctx, s);
            tr.convert_stages.push_back(s);
        }
    }

    // P11 第四遍（逆向再判决）[0x1323c63-0x1323e3b]
    for (auto it = ctx.ops.rbegin(); it != ctx.ops.rend(); ++it) {
        M42Op& op = *it;
        if (op.kind != 0xa) continue;                        // [0x1323cb1]
        if (!m42_layout_code_match(ref1, op.f5c | (op.f5d << 8)) ||
            !m42_layout_code_match(ref2, op.f5e | (op.f5f << 8)))
            continue;                                        // [0x1323cc6-0x1323cf8]
        M42TensorRec whole;                                  // op 整体量测 0x132ca20(ctx, op, 0)
        whole.tensor_kind = op.kind;
        uint64_t sz = hk.tensor_size ? hk.tensor_size(ctx, whole, false) : 0;
        ++tr.full_budget_cmp;
        if (sz > budget.full && op.ports != 0) {             // [0x1323d0f/0x1323d25]
            if (hk.transform) hk.transform(ctx, &op, ctx.global_d40); // [0x1323c9b]
        }
    }

    // P12 第五遍（正向）[0x1323e40-0x1323e77] —— 类型 0xb → 0x1334cf0（记录级：无操作）

    // 公共尾 [0x1323191-0x13231a5]
    tr.time_points.push_back("prepare_tcm_migration");       // @0x55b556a [0x13231a5]
}

} // namespace hnnx
