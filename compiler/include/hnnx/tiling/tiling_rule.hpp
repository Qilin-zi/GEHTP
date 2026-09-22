#pragma once
// tiling_rule.hpp — 产品路径 tiling 规则注册表(QNN declare_tiling_rule 的结构移植)
//
// 与 tile_shape_m36c.hpp(记录级 RE 替身)分工:
//   * 本文件是产品路径: op 类型 → shape_fn(op, 预算) → tile 计划
//   * tile_shape_m36c 是 RE 证据, 不进执行路径
//
// shape_fn 契约(QNN tiler DSL 同构):
//   declare_tiling_rule(op, shape_fn)     — 注册(显式调用, 防静态库死条带)
//   shape_fn(op_shape, budget) → plan      — 预算驱动, 不硬编码尺寸
//   budget.vtcm_tile_size = get_vtcm_tile_size() = min(budget,4MB)×3/4
//
// 默认回落: 未注册 op → identity(1 tile 整图), 与 QNN minimize_tiling 等价。
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hnnx {

// tiling 预算(对应 QNN 的 tcm_size_for_tiling 选项)
struct TileBudget {
    uint64_t vtcm_tile_size = 0;   // 可用 VTCM(= get_vtcm_tile_size(), 默认 3MB)
    uint32_t align = 32;           // HMX 硬件 tile 对齐(32)
};

// 单个 op 的形状描述(与 OpDef 解耦, 便于独立单测)
struct OpShape {
    // matmul: M×K × K×N
    uint32_t M = 0, N = 0, K = 0;
    uint32_t weight_bits = 16;     // 权重每元素位宽: 16=f16(2B), 4=Q4_0(0.5B)
    // conv2d: 输出 H/W/C(批量 N 在 shape_fn 外逐图处理)
    uint32_t in_h = 0, in_w = 0, cin = 0;
    uint32_t out_h = 0, out_w = 0, cout = 0;
    uint32_t kh = 1, kw = 1, sh = 1, sw = 1, ph = 0, pw = 0;
};

// 输出分块(与具体 op 解耦的通用视图)
//   matmul: m0/m1=行, n0/n1=列, c0/c1 未用
//   conv  : m0/m1=y,  n0/n1=x,  c0/c1=co 通道区间(c1==c0 表示不切通道)
struct OutputTile {
    uint32_t m0 = 0, m1 = 0;
    uint32_t n0 = 0, n1 = 0;
    uint32_t c0 = 0, c1 = 0;
    uint64_t working_set = 0;      // 该 tile 峰值工作集(act+weight+out, 字节)
};

struct TilingPlan {
    std::string op;
    bool tiled = false;                      // false = 恒等(单 tile 整图)
    std::vector<OutputTile> tiles;           // tiled 时覆盖整个输出, 无洞无重叠
    uint64_t max_working_set = 0;            // 所有 tile 中最大工作集
    bool fits_budget = false;                // max_working_set <= budget
};

// shape_fn: 预算驱动, 返回 tile 计划。op_shape 已由注册表按 op 名匹配。
using ShapeFn = std::function<TilingPlan(const OpShape&, const TileBudget&)>;

// 注册表(单例): op 名 → shape_fn; 未知 op 回落恒等
class TilingRuleRegistry {
public:
    static TilingRuleRegistry& instance();

    void register_rule(const std::string& op, ShapeFn fn);
    bool has(const std::string& op) const;
    TilingPlan plan(const std::string& op, const OpShape& s, const TileBudget& b) const;
    const std::vector<std::string>& registered() const;   // 诊断

private:
    TilingRuleRegistry() = default;
    std::map<std::string, ShapeFn> rules_;
    std::vector<std::string> order_;
};

// 内置 shape_fn
TilingPlan matmul_tile_shape(const OpShape& s, const TileBudget& b);  // 输出分块, K 不切
TilingPlan conv_tile_shape(const OpShape& s, const TileBudget& b);    // 复用 compute_conv_tiles

// 恒等(未注册/最小化 tiling 的回落)
TilingPlan identity_tile_plan(const OpShape& s);

// 在 compiler 初始化时调用一次(镜像 register_builtin_passes 的"一行注册")
void register_builtin_tiling_rules();

} // namespace hnnx
