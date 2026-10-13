// requant_w4a16.cc — W4A16 供给侧生成器 (B 段 T004, 013-executor-hoist)
//
// 只实现可位级验证的 atbl/otbl 闭式 (已对 s256/s2560 oracle bit-exact 命中);
// packed_w / folded_bias 因 oracle 已失降级为文档化 gap (见 requant_w4a16.h 尾注)。

#include "requant_w4a16.h"

namespace requant_w4a16 {

std::vector<uint32_t> atbl_for(uint32_t M, uint32_t K) {
    // 表长 = 8*(K/32) = K/4; 每项 = i*(M*8), byte offset 进 act 面 (u32 LE)。
    // 布局定论源: 013 memory "w4a16-emit-layout" — atbl[i] = i*(M*8),
    // s256(64 项)/s2560(640 项) 均 bit-exact 命中 oracle。
    const uint32_t n = (K / 32u) * 8u;
    std::vector<uint32_t> tbl(n);
    const uint32_t stride = M * 8u;
    for (uint32_t i = 0; i < n; ++i) tbl[i] = i * stride;
    return tbl;
}

std::vector<uint32_t> otbl_for(uint32_t M, uint32_t N) {
    // 表长 = 8*(N/32) = N/4; 每项 = i*(M*8), byte offset 进 out 面 (u32 LE)。
    // 注意: 表长随 N (out 列) 走, 非 K —— 历史 otbl 误按 8*(K/32) 计长
    // (K≠N 时永不命中/越界) 的 bug 修正版 (contract-w4a16-op.md 供给侧节)。
    const uint32_t n = (N / 32u) * 8u;
    std::vector<uint32_t> tbl(n);
    const uint32_t stride = M * 8u;
    for (uint32_t i = 0; i < n; ++i) tbl[i] = i * stride;
    return tbl;
}

} // namespace requant_w4a16
