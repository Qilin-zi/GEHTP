#pragma once
// ============================================================================
// barrel.hpp — .serialized.bin 容器 (barrel + pickle TLV 骨架)
//
// RE 依据: A 的 serialize.cc (wire 常量经 M09 字节级验证为真) + M24 §5.3
//   serialize_file_barrel_core @0xfdae20 (796B)
//
// 结构 (这是"内层", 外层 system header 另属 B 的 context_binary_writer):
//   [barrel 头]  magic 0x3790FA5C + hwords|ver_lo + TLV 链 + ENDHDR
//   [零填充到 pickle_start]
//   [pickle 头]  magic 0x7309F72B + hwords|ver + TLV 链 + ENDHDR
//   [pickle body] MODES_AUX + CO_AUX + Section3 + tail
//   [CRC footer]
//
// barrel tag 编码: 两 ASCII 字符的 16 位 (低字节=第一字符, 高字节=第二字符)。
//   recdesc = (tag << 16) | rlen, rlen = 含 recdesc 的 word 数。
// ============================================================================
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

namespace hnnx {

// barrel magic 与版本 (M09 验证为真)
constexpr uint32_t kBarrelMagic  = 0x3790FA5Cu;  // HDR_MAGIC_MULTI
constexpr uint32_t kPickleMagic  = 0x7309F72Bu;  // PICKLE_MAGIC
constexpr uint32_t kBarrelVerLo  = 0x8001u;      // multi-nsp + version 1
constexpr uint32_t kPickleVerLo  = 0x0001u;      // version 1

namespace barrel_tags {
    constexpr uint32_t IDENT    = 0x6449;  // 'I' + 256*'d'
    constexpr uint32_t SIZE     = 0x7a53;  // 'S' + 256*'z'
    constexpr uint32_t VERSION  = 0x7256;  // 'V' + 256*'r'
    constexpr uint32_t OPTIONS  = 0x704f;  // 'O' + 256*'p'
    constexpr uint32_t MEMORY   = 0x6d4d;  // 'M' + 256*'m'
    constexpr uint32_t CONTENTS = 0x6354;  // 'T' + 256*'c'
    constexpr uint32_t MULTI    = 0x754d;  // 'M' + 256*'u'
    constexpr uint32_t ENDHDR   = 0x7a5a;  // 'Z' + 256*'z'
}

constexpr uint32_t kMultiSerAlign = 4096;  // MULTI_SER_ALIGN
constexpr uint32_t kSize0Align    = 64;    // barrel 头对齐

// 简单字节流写入器 (LE u32)
class ByteWriter {
public:
    ByteWriter() = default;
    void u32(uint32_t v) {
        buf_.push_back(static_cast<uint8_t>(v));
        buf_.push_back(static_cast<uint8_t>(v >> 8));
        buf_.push_back(static_cast<uint8_t>(v >> 16));
        buf_.push_back(static_cast<uint8_t>(v >> 24));
    }
    void bytes(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }
    void zeros(size_t n) { buf_.insert(buf_.end(), n, 0); }
    size_t size() const { return buf_.size(); }
    size_t tell() const { return buf_.size(); }
    void patch_u32(size_t pos, uint32_t v) {
        if (pos + 4 <= buf_.size()) {
            buf_[pos] = static_cast<uint8_t>(v);
            buf_[pos+1] = static_cast<uint8_t>(v >> 8);
            buf_[pos+2] = static_cast<uint8_t>(v >> 16);
            buf_[pos+3] = static_cast<uint8_t>(v >> 24);
        }
    }
    const std::vector<uint8_t>& data() const { return buf_; }

    // TLV record: recdesc = (tag<<16)|rlen, rlen = 1 + payload words
    void record(uint32_t tag, const std::vector<uint32_t>& words) {
        u32((tag << 16) | static_cast<uint32_t>(words.size() + 1));
        for (uint32_t w : words) u32(w);
    }

private:
    std::vector<uint8_t> buf_;
};

// 写 barrel 头 (IDENT/SIZE/VERSION/OPTIONS/MEMORY/CONTENTS/MULTI/ENDHDR)。
// options_words: OPTIONS 记录 payload (8808B 配置 → GOLDEN_OPTIONS_U32)。
// 返回 [SIZE[0] 回填位置, SIZE[1] 回填位置]。
struct BarrelHeaderPatch { size_t size0_pos; size_t size1_pos; };
BarrelHeaderPatch write_barrel_header(ByteWriter& w, const std::vector<uint32_t>& options_words,
                                      uint32_t num_pickles);

// 写 pickle 头 (IDENT/SIZE/VERSION/OPTIONS/MEMORY/ENDHDR)。返回 SIZE[1] 回填位置。
size_t write_pickle_header(ByteWriter& w, const std::vector<uint32_t>& options_words);

} // namespace hnnx
