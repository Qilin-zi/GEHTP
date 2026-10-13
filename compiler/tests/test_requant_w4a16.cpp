// test_requant_w4a16 — W4A16 供给侧 atbl/otbl 闭式对拍 (B 段 T004)
//
// 对拍源 = s256/s2560 oracle 反推并逐位验证的公式 (013 memory "w4a16-emit-layout"):
//   atbl = 8*(K/32) 项, atbl[i] = i*(M*8)  (byte offset 进 act 面)
//   otbl = 8*(N/32) 项, otbl[i] = i*(M*8)  (byte offset 进 out 面)
//
// 覆盖:
//   1. s256 (M=256,K=256,N=256)   方形 → atbl/otbl 各 64 项
//   2. s2560 (M=256,K=2560,N=2560) 方形 → atbl/otbl 各 640 项
//   3. K≠N (M=256,K=2560,N=256)    → atbl 640 项 / otbl 64 项 (历史 otbl 误按 K 计长 bug 修正)
//   4. 首项/末项/步长闭式校验 (i*(M*8))
//
// packed_w / folded_bias 不在本测 (oracle 已失, 文档化 gap, 见 requant_w4a16.h)。

#include "requant_w4a16.h"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace requant_w4a16;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while (0)

static bool tbl_matches(const std::vector<uint32_t>& tbl, uint32_t n_exp,
                        uint32_t M, const char* what) {
    if (tbl.size() != n_exp) {
        std::fprintf(stderr, "  %s: size=%zu expect=%u\n", what, tbl.size(), n_exp);
        return false;
    }
    for (uint32_t i = 0; i < n_exp; ++i) {
        uint32_t exp = i * (M * 8u);
        if (tbl[i] != exp) {
            std::fprintf(stderr, "  %s[%u]=%u expect=%u\n", what, i, tbl[i], exp);
            return false;
        }
    }
    return true;
}

static void test_s256_square() {
    const uint32_t M = 256, K = 256, N = 256;
    auto atbl = atbl_for(M, K);
    auto otbl = otbl_for(M, N);
    CHECK(atbl.size() == 64u && tbl_matches(atbl, 64u, M, "atbl s256"),
          "s256 atbl = 64 项, 闭式 i*(M*8)");
    CHECK(otbl.size() == 64u && tbl_matches(otbl, 64u, M, "otbl s256"),
          "s256 otbl = 64 项, 闭式 i*(M*8)");
    CHECK(atbl[0] == 0u && atbl[63] == 63u * 2048u, "s256 atbl 首末项");
    CHECK(otbl[1] == 2048u, "s256 otbl 步长 = M*8 = 2048");
}

static void test_s2560_square() {
    const uint32_t M = 256, K = 2560, N = 2560;
    auto atbl = atbl_for(M, K);
    auto otbl = otbl_for(M, N);
    CHECK(atbl.size() == 640u && tbl_matches(atbl, 640u, M, "atbl s2560"),
          "s2560 atbl = 640 项, 闭式 i*(M*8)");
    CHECK(otbl.size() == 640u && tbl_matches(otbl, 640u, M, "otbl s2560"),
          "s2560 otbl = 640 项, 闭式 i*(M*8)");
    CHECK(atbl[639] == 639u * 2048u, "s2560 atbl 末项");
}

static void test_k_neq_n() {
    // K≠N 非方形: atbl 长随 K(640 项), otbl 长随 N(64 项) —— 历史 bug 修正验证。
    const uint32_t M = 256, K = 2560, N = 256;
    auto atbl = atbl_for(M, K);
    auto otbl = otbl_for(M, N);
    CHECK(atbl.size() == 640u, "K≠N: atbl 长 = 8*(K/32) = 640");
    CHECK(otbl.size() == 64u, "K≠N: otbl 长 = 8*(N/32) = 64 (非 K 基)");
    CHECK(tbl_matches(atbl, 640u, M, "atbl K≠N"), "K≠N atbl 闭式");
    CHECK(tbl_matches(otbl, 64u, M, "otbl K≠N"), "K≠N otbl 闭式");
}

static void test_min_tile() {
    // 最小对齐形状 K=32 → 8 项; M=256 步长不变。
    const uint32_t M = 256, K = 32;
    auto atbl = atbl_for(M, K);
    CHECK(atbl.size() == 8u && tbl_matches(atbl, 8u, M, "atbl K=32"),
          "K=32 → atbl 8 项, 闭式");
}

int main() {
    test_s256_square();
    test_s2560_square();
    test_k_neq_n();
    test_min_tile();
    if (failed) { std::fprintf(stderr, "\n%d FAILED\n", failed); return 1; }
    std::printf("\nALL PASS\n");
    return 0;
}
