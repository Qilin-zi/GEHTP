#pragma once
// ============================================================================
// class_index_record.hpp — class-index 记录 (make_class_index_aux_record 的输出)
//
// RE 依据: audit_verify/reports/M22_serialize_class_index.md
//   hnnx::Serializer::make_class_index_aux_record @0x12eb840 (1230B) 逐指令解码。
//
// 注意: 这是 class-index 记录 (类名→索引映射), 不是 CO_AUX 记录。
//   CO_AUX = [tag][nquads][n_names][n_op][strtab][trailer×5] (见 M22 §2.1)。
//
// 记录布局 (全 u32 小端):
//   word[0] = n             (类名数量)
//   word[1] = strtab_words  (= (n + Σlen + 3) >> 2)
//   words[2 .. 2+strtab_words)   = strtab (各名 NUL 分隔, 补齐到 4B)
//   words[2+strtab_words .. end) = counter 表 (每类 1 个 u32 索引)
//
// tag: 'Co'=0x6f43 (flag=false) / 'Ct'=0x7443 (flag=true)。
// 上 wire 时经 aux_tag_word 编码: ((tag<<16)|tag)^0xffff → 0x6f4390bc。
//
// 这是"计算不照抄"的核心: 记录完全从 (类名, 索引) 列表算出。
// ============================================================================
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace hnnx {

constexpr uint32_t kClassIndexTagCo = 0x6f43u;  // 'Co' (LE)
constexpr uint32_t kClassIndexTagCt = 0x7443u;  // 'Ct' (LE)

// 纯函数: 从 (类名, 索引) 列表生成 class-index 记录的 u32 字序列。
// 输出长度 = 2 + strtab_words + n 字。
std::vector<uint32_t> make_class_index_record(
    const std::vector<std::pair<std::string_view, uint32_t>>& classes);

// aux tag 编码 (与 A 的 aux_tag_word 一致, RE 验证):
//   tag_word = ((tag<<16) | tag) ^ 0xffff
inline uint32_t aux_tag_word(uint32_t tag) {
    return ((tag << 16) | (tag & 0xffffu)) ^ 0xffffu;
}

} // namespace hnnx
