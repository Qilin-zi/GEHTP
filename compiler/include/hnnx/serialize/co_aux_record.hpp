#pragma once
// ============================================================================
// co_aux_record.hpp — CO_AUX 记录 (pickle body +0x0c)
//
// RE 依据: golden_identity.bin 实际字节 (M22 §2.1) + A 的 serialize.cc。
// CO_AUX 布局 (全 u32 小端):
//   [tag]     = aux_tag_word('Co') = 0x6f4390bc
//   [nquads]  = 8 + strtab_words   (tag 之后的 word 总数, 含 nquads 自身)
//   [n_names] = 类名数量
//   [n_op]    = op 数
//   [strtab]  = 各名 NUL 分隔, 4B 对齐
//   [trailer] = 5 word 元数据
//
// 注意: 这是 CO_AUX (op/类名 + 计数), 与 class-index 记录 (class_index_record.hpp)
// 是不同记录。A 的 emit_co_aux_record 字节级验证正确。
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace hnnx {

// CO_AUX trailer 默认 5 word (identity golden 实测)。部分字段图相关(如 n_names/n_op 编码),
// 待 RE (见 M22 §2.1 trailer 值)。
inline constexpr uint32_t kCoAuxTrailer[5] = {
    0x80010000u, 0x00010006u, 0x80000010u, 0x80000010u, 0x00000004u
};

// 纯函数: 生成 CO_AUX 记录的 u32 字序列 (不含 tag 字; 调用方自行套 aux_tag_word)。
// nquads = 1(n_names) + 1(n_op) + strtab_words + 5(trailer)。
// 注意 nquads 统计 tag 之后的 word 数, 不含 tag 字本身。
std::vector<uint32_t> make_co_aux_record(
    const std::vector<std::string>& names, uint32_t n_op,
    const uint32_t trailer[5] = kCoAuxTrailer);

} // namespace hnnx
