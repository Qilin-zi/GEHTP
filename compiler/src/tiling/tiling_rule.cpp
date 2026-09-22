#include "hnnx/tiling/tiling_rule.hpp"
#include "hnnx/tiling/conv_tiling.hpp"
#include <algorithm>

namespace hnnx {

// ---- matmul 工作集估算(f16 激活/输出 = 2B; 权重 = weight_bits/8 B) ----
// 输出 tile (Mt × Nt), K 完整:
//   A 切片 = Mt × K × 2
//   B 切片(权重, dequant 实时) = Nt × K × weight_bits / 8
//   C 块   = Mt × Nt × 2
static uint64_t matmul_ws(uint32_t Mt, uint32_t Nt, uint32_t K, uint32_t wbits) {
    uint64_t a = (uint64_t)Mt * K * 2;
    uint64_t b = (uint64_t)Nt * K * wbits / 8;   // 先乘后除(Q4_0=4 → Nt*K/2)
    uint64_t c = (uint64_t)Mt * Nt * 2;
    return a + b + c;
}

TilingPlan matmul_tile_shape(const OpShape& s, const TileBudget& b) {
    TilingPlan p;
    p.op = "matmul";
    uint32_t M = s.M, N = s.N, K = s.K;
    uint32_t align = b.align > 0 ? b.align : 32;
    if (M == 0 || N == 0 || K == 0) { p.tiled = false; return p; }

    // 预算驱动: 从整图起步, 先缩 N 后缩 M, 到工作集 ≤ 预算为止(对齐 align)
    uint32_t mt = M, nt = N;
    if (b.vtcm_tile_size > 0) {
        auto down = [align](uint32_t v) { return (v > align) ? (v - align) : align; };
        while (nt > align && matmul_ws(mt, nt, K, s.weight_bits) > b.vtcm_tile_size) nt = down(nt);
        while (mt > align && matmul_ws(mt, nt, K, s.weight_bits) > b.vtcm_tile_size) mt = down(mt);
    }

    // 生成覆盖 [0,M)×[0,N) 的 tile 网格(无洞无重叠)
    uint64_t maxws = 0;
    for (uint32_t m0 = 0; m0 < M; m0 += mt) {
        uint32_t m1 = std::min(M, m0 + mt);
        for (uint32_t n0 = 0; n0 < N; n0 += nt) {
            uint32_t n1 = std::min(N, n0 + nt);
            OutputTile t;
            t.m0 = m0; t.m1 = m1; t.n0 = n0; t.n1 = n1; t.c0 = 0; t.c1 = 0;
            t.working_set = matmul_ws(m1 - m0, n1 - n0, K, s.weight_bits);
            maxws = std::max(maxws, t.working_set);
            p.tiles.push_back(t);
        }
    }
    p.tiled = (p.tiles.size() > 1);
    p.max_working_set = maxws;
    p.fits_budget = (b.vtcm_tile_size == 0) || (maxws <= b.vtcm_tile_size);
    return p;
}

TilingPlan conv_tile_shape(const OpShape& s, const TileBudget& b) {
    TilingPlan p;
    p.op = "conv2d";
    if (s.out_h == 0 || s.out_w == 0 || s.cout == 0) { p.tiled = false; return p; }

    // 预算驱动: 每输出像素工作集 ≈ kh*kw*cin*2(f16 act), 据此粗估 tile_h。
    // 精确 per-tile 工作集由 compute_conv_tiles 的 halo + 通道共同决定; 原型用此粗估。
    uint32_t tile_h = s.out_h, tile_w = s.out_w;
    if (b.vtcm_tile_size > 0) {
        uint64_t per_pix = (uint64_t)s.kh * s.kw * s.cin * 2;
        if (per_pix > 0) {
            uint64_t max_pix = (b.vtcm_tile_size / 8) / per_pix;   // 留 7/8 给权重+输出+halo
            if (max_pix > 0 && max_pix < (uint64_t)s.out_h * s.out_w) {
                uint32_t th = (uint32_t)std::max<uint64_t>(1, max_pix / s.out_w);
                if (th < s.out_h) { tile_h = th; tile_w = s.out_w; }
            }
        }
    }

    std::vector<ConvTileDesc> ct = compute_conv_tiles(
        s.in_h, s.in_w, s.cin, s.out_h, s.out_w, s.cout,
        s.kh, s.kw, s.sh, s.sw, s.ph, s.pw,
        tile_h, tile_w, /*co_per_tile=*/0);

    uint64_t maxws = 0;
    for (const auto& t : ct) {
        OutputTile o;
        o.m0 = t.out_y0; o.m1 = t.out_y0 + t.out_h;   // y
        o.n0 = t.out_x0; o.n1 = t.out_x0 + t.out_w;   // x
        o.c0 = t.co0;    o.c1 = t.co0 + t.co_n;       // co
        uint64_t inb  = (uint64_t)t.in_h * t.in_w * t.ci * 2;       // 输入切片(halo)
        uint64_t wb   = (uint64_t)t.kh * t.kw * t.ci * t.co_n * 2;  // 权重
        uint64_t outb = (uint64_t)t.out_h * t.out_w * t.co_n * 2;   // 输出块
        o.working_set = inb + wb + outb;
        maxws = std::max(maxws, o.working_set);
        p.tiles.push_back(o);
    }
    p.tiled = (p.tiles.size() > 1);
    p.max_working_set = maxws;
    p.fits_budget = (b.vtcm_tile_size == 0) || (maxws <= b.vtcm_tile_size);
    return p;
}

TilingPlan identity_tile_plan(const OpShape& s) {
    (void)s;
    TilingPlan p;
    p.op = "identity";
    p.tiled = false;
    return p;
}

TilingRuleRegistry& TilingRuleRegistry::instance() {
    static TilingRuleRegistry reg;
    return reg;
}

void TilingRuleRegistry::register_rule(const std::string& op, ShapeFn fn) {
    if (rules_.find(op) == rules_.end()) order_.push_back(op);
    rules_[op] = std::move(fn);
}

bool TilingRuleRegistry::has(const std::string& op) const {
    return rules_.find(op) != rules_.end();
}

TilingPlan TilingRuleRegistry::plan(const std::string& op, const OpShape& s, const TileBudget& b) const {
    auto it = rules_.find(op);
    if (it == rules_.end()) return identity_tile_plan(s);
    return it->second(s, b);
}

const std::vector<std::string>& TilingRuleRegistry::registered() const { return order_; }

void register_builtin_tiling_rules() {
    auto& r = TilingRuleRegistry::instance();
    r.register_rule("matmul", matmul_tile_shape);
    r.register_rule("conv2d", conv_tile_shape);
}

} // namespace hnnx
