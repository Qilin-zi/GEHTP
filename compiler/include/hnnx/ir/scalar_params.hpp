#pragma once
// scalar_params 的打包/解析(loader 写入 OpDef::op_data; 构造/提取器/发射器消费)
//
// 布局:
//   [u32 count]
//   per param:
//     [u32 name_len][name bytes][u8 kind: 0=num 1=str]
//     num → [f64 value];  str → [u32 len][bytes]
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hnnx {

struct ScalarParam {
    std::string name;
    bool is_numeric = false;
    double value_num = 0.0;
    std::string value_str;
    int64_t as_int() const { return static_cast<int64_t>(value_num); }
};

inline void scalar_pack_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
    b.push_back(static_cast<uint8_t>(v >> 16));
    b.push_back(static_cast<uint8_t>(v >> 24));
}

inline uint32_t scalar_unpack_u32(const std::vector<uint8_t>& b, size_t& off) {
    uint32_t v = static_cast<uint32_t>(b[off]) | (static_cast<uint32_t>(b[off + 1]) << 8) |
                 (static_cast<uint32_t>(b[off + 2]) << 16) | (static_cast<uint32_t>(b[off + 3]) << 24);
    off += 4;
    return v;
}

inline double scalar_unpack_f64(const std::vector<uint8_t>& b, size_t& off) {
    double v;
    std::memcpy(&v, b.data() + off, 8);
    off += 8;
    return v;
}

inline std::vector<uint8_t> pack_scalar_params(const std::vector<ScalarParam>& sp) {
    std::vector<uint8_t> b;
    scalar_pack_u32(b, static_cast<uint32_t>(sp.size()));
    for (const auto& p : sp) {
        scalar_pack_u32(b, static_cast<uint32_t>(p.name.size()));
        b.insert(b.end(), p.name.begin(), p.name.end());
        if (p.is_numeric) {
            b.push_back(0);
            const uint8_t* pv = reinterpret_cast<const uint8_t*>(&p.value_num);
            b.insert(b.end(), pv, pv + 8);
        } else {
            b.push_back(1);
            scalar_pack_u32(b, static_cast<uint32_t>(p.value_str.size()));
            b.insert(b.end(), p.value_str.begin(), p.value_str.end());
        }
    }
    return b;
}

inline std::vector<ScalarParam> unpack_scalar_params(const std::vector<uint8_t>& blob) {
    std::vector<ScalarParam> out;
    if (blob.size() < 4) return out;
    size_t off = 0;
    uint32_t n = scalar_unpack_u32(blob, off);
    for (uint32_t i = 0; i < n; i++) {
        if (off + 4 > blob.size()) break;
        uint32_t nlen = scalar_unpack_u32(blob, off);
        if (off + nlen + 1 > blob.size()) break;
        ScalarParam p;
        p.name.assign(reinterpret_cast<const char*>(blob.data() + off), nlen);
        off += nlen;
        uint8_t kind = blob[off++];
        if (kind == 0) {
            if (off + 8 > blob.size()) break;
            p.is_numeric = true;
            p.value_num = scalar_unpack_f64(blob, off);
        } else {
            if (off + 4 > blob.size()) break;
            uint32_t slen = scalar_unpack_u32(blob, off);
            if (off + slen > blob.size()) break;
            p.value_str.assign(reinterpret_cast<const char*>(blob.data() + off), slen);
            off += slen;
        }
        out.push_back(std::move(p));
    }
    return out;
}

// 便捷: 按名取 scalar; 缺省返回 nullptr
inline const ScalarParam* scalar_get(const std::vector<ScalarParam>& sp, const std::string& name) {
    for (const auto& p : sp)
        if (p.name == name) return &p;
    return nullptr;
}

} // namespace hnnx
