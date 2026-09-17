#pragma once
// ============================================================================
// M42 反汇编保真实现：GraphPrepare::tcm_migration(uint, bool) 外壳
// 证据: audit_verify/reports/M42_tcm_migration_disasm.md
//       audit_verify/asm/f3/M42_tcm_migration.asm
// 规则: 每个逻辑段标注 [0x地址]；helper 内部为记录级（M42b 遗留），以钩子注入。
//
// 一句话语义: TCM 迁移外壳 = "DCE 前处理 → 预算(全量 + (全量/4)*3) → scan →
// 六遍 op 扫描（估计/累计/判决）→ 转换四连 → 计时点收尾"。外壳所有显式
// 比较用全量预算；3/4 值只写入配置块供 helper。判决: 需求>全量 → 3472
// ERROR 且错误计数+1；否则优先级门控下 3459 WARNING 不计数。
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace hnnx {

// ---- 预算规则 --------------------------------------------------------------
// (full/4)*3，uint32 先除后乘 [0x1321bd4 shrl $2; 0x1321bd7 leal (%rax,%rax,2)]
// 注意: 与 full*3/4 在 full 非 4 倍数时结果不同，实现不得交换顺序。
inline uint32_t m42_three_quarters(uint32_t full) {
    return (full / 4u) * 3u;                                // [0x1321bd4-0x1321bda]
}

// 预算配置块（rsp+0x140 同构）：+0x28 全量、+0x2c 3/4   [0x1321bcd/0x1321bda]
struct M42BudgetConfig {
    uint8_t  head[0x28] = {0};      // +0x00..+0x27（32B 状态 + float 1.0 模板）
    uint32_t full = 0;              // +0x28 = get_vtcm_tile_size()
    uint32_t three_quarters = 0;    // +0x2c = (full/4)*3
};

// ---- SWAR popcount 与单比特测试 --------------------------------------------
// 0x5555/0x3333/0x0f0f/0x0101 序列 [0x1323594-0x13235ec]（两处同型）
inline uint64_t m42_popcount(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);            // [0x1323597-0x13235a4]
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull); // [0x13235b0-0x13235c4]
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0full;            // [0x13235c7-0x13235db]
    return (x * 0x0101010101010101ull) >> 56;              // [0x13235de-0x13235ec]
}
// 单比特（0 或恰好 1 位）→ 快路径 [0x13235f0 cmpq $1; ja 多比特支]
inline bool m42_at_most_one_bit(uint64_t x) { return m42_popcount(x) <= 1; }

// ---- crouton 布局码匹配 ------------------------------------------------------
// 码 >= 0x100 表示带标志；两码均 >=0x100 比较低字节；单边 >=0x100 即失配
// [0x1322c89-0x1322cee / 0x1323cc6-0x1323cf8 同型]
inline bool m42_layout_code_match(uint16_t ref, uint16_t code) {
    if (ref >= 0x100 && code >= 0x100) return (uint8_t)ref == (uint8_t)code; // [0x1322c91-0x1322c9c]
    return (ref >= 0x100) == (code >= 0x100) && ref == code;                 // [0x1322ca0-0x1322cb4]
}

// ---- 记录级数据模型 ----------------------------------------------------------
// op+0x68/+0x70 的 0x18B 张量记录 [0x1322c74-0x1322c7f]
struct M42TensorRec {
    uint32_t tensor_kind = 0;   // 张量对象 +0x8 类型标记（计入判据: &-2==0xa [0x1322d35]）
    uint16_t code1 = 0;         // +0x08 布局码
    uint16_t code2 = 0;         // +0x0a 布局码
    uint32_t port = 0;          // +0x0c 端口号
    // 判决口径: 变换前后各测一次尺寸，acc += after - before [0x1322d9f-0x1322da7]
    uint64_t size_before = 0;
    uint64_t size_after = 0;
    bool transformed = false;   // 0x1332210 是否作用于该张量
};

// OpDef 记录级替身（外壳触及的 +0x8/+0x58/+0x5a..+0x5f 字段）
struct M42Op {
    uint32_t kind = 0;                  // +0x8 类型标记（0xa/0xb/0xc 留，其余跳过 [0x1322504]）
    uint32_t id = 0;                    // +0x58（B 支估计器输入）
    uint8_t  f5a = 0, f5b = 0, f5c = 0; // +0x5a/+0x5b（P4 门）、+0x5c（B 支门 ==1）
    uint8_t  f5d = 0, f5e = 0, f5f = 0; // +0x5d（B 支门 !=0）、+0x5e/+0x5f（P6 头门）
    std::vector<M42TensorRec> tensors;  // +0x68/+0x70
    uint32_t ports = 0;                 // +0x80/+0x88 记录数（P2/P11 用）
    uint64_t estimate = 0;              // 0x13318c0 估计量（注入）
    uint64_t demand = 0;                // A 支累计（外壳算出）
    // 日志身份（0x1322f7c-0x1322f88 的 name/id 双链在模型里直接存值）
    std::string name = "op";
    uint64_t opid = 0;
};

// TCMMigration 上下文（rsp+0x78 同构，记录级）
struct M42Ctx {
    uint32_t level = 0;                 // +0x10 参数 esi [0x1321a3b]
    uint8_t  stop_flag = 0;             // +0x14 [0x1322960]
    uint32_t errcount = 0;              // +0x78 [0x1321c76/0x1322fbd/0x1323112]
    std::vector<M42Op> ops;             // +0x30 vector<OpDef*> [0x1321a70]
    // pkg_flag 相邻全局（0x6247d38-0x6247d44）的模型替身
    uint16_t default_code1 = 0;         // @0x6247d38
    uint16_t default_code2 = 0;         // @0x6247d3a
    uint8_t  flag_1c = 0;               // @0x6247d3c（置 1 时参考码 = 0x101 [0x1322db7]）
    uint8_t  flag_1e = 0;               // @0x6247d3e
    uint32_t global_d40 = 0;            // @0x6247d40（0x1332210 第 3 参）
    uint32_t global_d44 = 0;            // @0x6247d44（B 支 0x1332210 第 3 参）
};

// helper 钩子（记录级替身；真码见报告 §5 遗留）
struct M42Hooks {
    // 0x13256f0 scan_input_graph：返回是否置错误计数 [0x1321c71-0x1321c76]
    bool (*scan_input_graph)(M42Ctx&) = nullptr;
    // 0x13318c0(ctx, op, &out, 1) op 级估计器 → 需求量 [0x13227c6/0x1322b94]
    uint64_t (*estimate_op)(M42Ctx&, M42Op&) = nullptr;
    // 0x132ca20(ctx, tensor, flag) 张量尺寸 [0x1322d4c/0x1322d9a/0x1323d03]
    uint64_t (*tensor_size)(M42Ctx&, M42TensorRec&, bool flag) = nullptr;
    // 0x1332210(ctx, op|tensor, &u32) 核心变换（改 size_after 口径由钩子自定）
    void (*transform)(M42Ctx&, void* target, uint32_t arg) = nullptr;
    // 转换/对齐四连 0x132a8b0/0x132ac20/0x132bbf0/0x132c1f0 [0x1323bee-0x1323c0c]
    void (*convert_pipeline)(M42Ctx&, int stage) = nullptr;
    // GetLogPriorityLevel@plt [0x1322fc7]
    int (*log_priority)() = nullptr;
    // TLS 门 [tls@0x623ee88+0x24] [0x1321bed]
    bool (*tls_busy)() = nullptr;
    // 0x131fc90/0xf17d70/dead_code_removal… 前处理（记录级，仅留痕）
    void (*pre_dce)(M42Ctx&, bool dynamic_active) = nullptr;
    bool (*is_dynamic_active)(M42Ctx&) = nullptr;
};

// 外壳执行留痕（供测试断言）
struct M42Trace {
    std::string log;                    // 全部 qnndsp_log 拼接
    std::vector<std::string> time_points;  // mark_time_point 顺序
    std::vector<int> convert_stages;    // P10 四连顺序
    int early_exit_mode = -1;           // -1 无；0 = mode 0x11 早退尾命中
    bool scan_ran = false;
    // 外壳显式使用的预算比较计数（全量口径）
    int full_budget_cmp = 0;
};

// 主入口：tcm_migration 外壳（报告 §2 阶段流水 P0-P12）
void m42_tcm_migration(M42Ctx& ctx, M42BudgetConfig& budget, const M42Hooks& hk,
                       uint32_t vtcm_tile_size, bool cleanup, M42Trace& tr);

// 3472/3459 文本（fmt@0x55b5ba1 / 0x55b5b25，file="tcm_migration.cc"@0x55b5998）
std::string m42_err_3472(const std::string& name, uint64_t opid, uint32_t full, uint64_t demand);
std::string m42_warn_3459(const std::string& name, uint64_t opid);
std::string m42_err_1819();
std::string m42_err_1824();
std::string m42_err_1829();

} // namespace hnnx
