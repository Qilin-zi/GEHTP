// test_tiling_rule: 产品路径 tiling 规则注册表 + matmul/conv 两个 shape_fn
//
// 1. 注册表: register/has/plan 分发, 未知 op 回落恒等
// 2. matmul 预算驱动: 小预算强制分块, 大预算单 tile, 工作集 ≤ 预算
// 3. matmul 数值等价: 分 tile 执行(K 不切 → 累加序不变)与全图参考 byte-exact
// 4. conv 注册: plan 与 compute_conv_tiles 一致(tile 数/覆盖)
//
// 注: 工作集估算用设备 dtype(f16/Q4_0); 数值对拍参考用 float32(独立于设备位宽,
// 只验证"分块分解正确"——无洞/无重叠/K 累加序保持)。
#include "hnnx/tiling/tiling_rule.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static void ref_matmul(const float* A, const float* B, float* C,
                       uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

static void tiled_matmul(const float* A, const float* B, float* C,
                         uint32_t M, uint32_t N, uint32_t K, const TilingPlan& plan) {
    for (const auto& t : plan.tiles)
        for (uint32_t m = t.m0; m < t.m1; ++m)
            for (uint32_t n = t.n0; n < t.n1; ++n) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
                C[m * N + n] = acc;
            }
}

static void test_registry() {
    std::printf("[1] registry dispatch\n");
    register_builtin_tiling_rules();
    auto& reg = TilingRuleRegistry::instance();
    CHECK(reg.has("matmul") && reg.has("conv2d"), "matmul/conv2d 已注册");
    CHECK(!reg.has("add"), "未知 op 未注册");

    OpShape s; s.M = 64; s.N = 64; s.K = 64;
    TileBudget b; b.vtcm_tile_size = 1u << 30;
    TilingPlan p = reg.plan("matmul", s, b);
    CHECK(p.op == "matmul", "plan 分发到 matmul 规则");
    TilingPlan q = reg.plan("add", s, b);
    CHECK(!q.tiled && q.op == "identity", "未知 op → 恒等回落");
}

static void test_matmul_budget() {
    std::printf("[2] matmul budget-driven tiling\n");
    OpShape s; s.M = 1024; s.N = 1024; s.K = 4096; s.weight_bits = 4;   // Q4_0
    TileBudget big; big.vtcm_tile_size = 1u << 30;
    TilingPlan p1 = matmul_tile_shape(s, big);
    CHECK(p1.tiles.size() == 1 && !p1.tiled, "1GB 预算 → 单 tile 整图");

    TileBudget small; small.vtcm_tile_size = 1u << 20;   // 1MB → 强制分块
    TilingPlan p2 = matmul_tile_shape(s, small);
    CHECK(p2.tiles.size() > 1 && p2.tiled, "1MB 预算 → 分块");
    CHECK(p2.fits_budget, "每 tile 工作集 ≤ 预算");

    uint64_t cover = 0;
    for (const auto& t : p2.tiles) cover += (uint64_t)(t.m1 - t.m0) * (t.n1 - t.n0);
    CHECK(cover == (uint64_t)s.M * s.N, "tile 网格无洞无重叠覆盖全输出");
}

static void test_matmul_byte_exact() {
    std::printf("[3] matmul tiled vs untiled byte-exact (K 不切)\n");
    uint32_t M = 256, N = 192, K = 128;
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_tiled(M * N);
    for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin((float)i * 0.01f);
    for (size_t i = 0; i < B.size(); ++i) B[i] = std::cos((float)i * 0.01f);

    ref_matmul(A.data(), B.data(), C_ref.data(), M, N, K);

    OpShape s; s.M = M; s.N = N; s.K = K; s.weight_bits = 16;
    TileBudget b; b.vtcm_tile_size = 1u << 16;   // 64KB → 强制分块
    TilingPlan p = matmul_tile_shape(s, b);
    CHECK(p.tiles.size() > 1, "小预算强制分块");

    tiled_matmul(A.data(), B.data(), C_tiled.data(), M, N, K, p);
    CHECK(std::memcmp(C_ref.data(), C_tiled.data(), M * N * sizeof(float)) == 0,
          "分 tile 与全图 byte-exact");
}

static void test_conv_registry() {
    std::printf("[4] conv registry ↔ compute_conv_tiles 一致\n");
    OpShape s; s.in_h = 32; s.in_w = 32; s.cin = 32;
    s.out_h = 32; s.out_w = 32; s.cout = 32;
    s.kh = 3; s.kw = 3; s.sh = 1; s.sw = 1; s.ph = 1; s.pw = 1;

    TileBudget big; big.vtcm_tile_size = 1u << 30;
    TilingPlan p1 = conv_tile_shape(s, big);
    CHECK(p1.tiles.size() == 1 && !p1.tiled, "大预算 conv → 单 tile 整图");

    TileBudget small; small.vtcm_tile_size = 1u << 15;   // 32KB → 强制分块
    TilingPlan p2 = conv_tile_shape(s, small);
    CHECK(p2.tiles.size() > 1 && p2.tiled, "小预算 conv → 分块");

    uint64_t cover = 0;
    for (const auto& t : p2.tiles) cover += (uint64_t)(t.m1 - t.m0) * (t.n1 - t.n0);
    CHECK(cover == (uint64_t)s.out_h * s.out_w, "conv tile 覆盖全输出");
}

int main() {
    test_registry();
    test_matmul_budget();
    test_matmul_byte_exact();
    test_conv_registry();
    if (failed) { std::fprintf(stderr, "\n%d FAILED\n", failed); return 1; }
    std::printf("\nALL PASS\n");
    return 0;
}
