#include "hnnx/serialize/co_aux_record.hpp"

#include <cstring>

namespace hnnx {

std::vector<uint32_t> make_co_aux_record(
    const std::vector<std::string>& names, uint32_t n_op,
    const uint32_t trailer[5])
{
    const uint32_t n_names = static_cast<uint32_t>(names.size());

    // strtab: 各名 NUL 分隔, 补到 4B
    std::string strtab;
    for (const auto& nm : names) {
        strtab += nm;
        strtab.push_back('\0');
    }
    while (strtab.size() % 4 != 0) strtab.push_back('\0');
    const uint32_t strtab_words = static_cast<uint32_t>(strtab.size() / 4);

    // nquads = 1(自身) + 1(n_names) + 1(n_op) + strtab_words + 5(trailer)
    // 统计 tag 之后的 word 总数, 含 nquads 自身 (A: "nquads 不含 tag 字" = 8 + strtab_words)。
    const uint32_t nquads = 1u + 1u + 1u + strtab_words + 5u;

    std::vector<uint32_t> rec;
    rec.reserve(1 + nquads);  // nquads + (n_names/n_op/strtab/trailer)
    rec.push_back(nquads);
    rec.push_back(n_names);
    rec.push_back(n_op);

    const uint32_t* sw = reinterpret_cast<const uint32_t*>(strtab.data());
    for (uint32_t i = 0; i < strtab_words; ++i) rec.push_back(sw[i]);

    for (uint32_t i = 0; i < 5; ++i) rec.push_back(trailer[i]);

    return rec;
}

} // namespace hnnx
