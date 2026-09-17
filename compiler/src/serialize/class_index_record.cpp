#include "hnnx/serialize/class_index_record.hpp"

#include <cstring>

namespace hnnx {

// 逐指令复刻 make_class_index_aux_record @0x12eb840 (见 M22):
//   n = classes.size()
//   Σlen = Σ string_view.size()
//   strtab_words = (n + Σlen + 3) >> 2
//   分配 (n + strtab_words + 2) 字, 清零
//   word[0]=n, word[1]=strtab_words
//   然后逐元素: 写 name 字节 + NUL; 写 u32 索引到 counter 表
std::vector<uint32_t> make_class_index_record(
    const std::vector<std::pair<std::string_view, uint32_t>>& classes)
{
    const size_t n = classes.size();

    // Σlen (M22: 12eb890 add 0x8(%rax),%ebp 累加 string_view.size)
    size_t total_len = 0;
    for (const auto& c : classes) total_len += c.first.size();

    // strtab_words = (n + Σlen + 3) >> 2  (M22: 12eb89c add $0x3; 12eb89f shr $0x2)
    const uint32_t strtab_words = static_cast<uint32_t>((n + total_len + 3) >> 2);

    // 总字数 = n + strtab_words + 2  (M22: 12eb8a2 lea (%r8,%rbp,1); add $0x2)
    const size_t total_words = n + strtab_words + 2;

    std::vector<uint32_t> rec(total_words, 0u);

    // word[0] = n, word[1] = strtab_words (M22: 12eb924/12eb927)
    rec[0] = static_cast<uint32_t>(n);
    rec[1] = strtab_words;

    // strtab 区从 word[2] 起; counter 表从 word[2+strtab_words] 起
    // (M22: 12eba3c add $0x2,%ebp → r13=r15+ebp*4; 12eba45 rbp=r15+8)
    uint8_t* strtab = reinterpret_cast<uint8_t*>(rec.data() + 2);
    uint32_t* counters = rec.data() + 2 + strtab_words;

    for (const auto& c : classes) {
        const std::string_view name = c.first;
        if (!name.empty()) {
            std::memcpy(strtab, name.data(), name.size());  // M22: 12eba80 memcpy
            strtab += name.size();
        }
        *strtab++ = 0x00;              // NUL 终止 (M22: 12eba50 movb $0x0)
        *counters++ = c.second;        // 索引 (M22: 12eba58/12eba5b)
    }

    return rec;
}

} // namespace hnnx
