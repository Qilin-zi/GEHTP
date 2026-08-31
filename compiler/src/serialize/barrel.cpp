#include "hnnx/serialize/barrel.hpp"

#include <cstring>

namespace hnnx {

BarrelHeaderPatch write_barrel_header(ByteWriter& w, const std::vector<uint32_t>& options_words,
                                      uint32_t num_pickles)
{
    // 1. magic + hwords|ver_lo 占位
    w.u32(kBarrelMagic);
    size_t hwords_pos = w.tell();
    w.u32(0);  // 回填 hwords|ver_lo

    // 2. IDENT: 50 u32 (200B) 图名 (identity 图全 0)
    w.record(barrel_tags::IDENT, std::vector<uint32_t>(50, 0));

    // 3. SIZE: recdesc + SIZE[0] 占位 + SIZE[1] 占位 + SIZE[2]=0 + SIZE[3]=0x44
    size_t size0_pos = w.tell() + 4;   // 跳过 recdesc
    w.record(barrel_tags::SIZE, {0, 0, 0, 0x44});
    size_t size1_pos = size0_pos + 4;  // SIZE[1] 紧随 SIZE[0]

    // 4. VERSION: "v2.48.40.260702151143." (24B)
    {
        const char* ver = "v2.48.40.260702151143.";
        std::vector<uint32_t> vw(6, 0);
        std::memcpy(vw.data(), ver, std::min<size_t>(strlen(ver), 22));
        w.record(barrel_tags::VERSION, vw);
    }

    // 5. OPTIONS: 8808B 配置
    w.record(barrel_tags::OPTIONS, options_words);

    // 6. MEMORY: 13 u32 全 0 (identity)
    w.record(barrel_tags::MEMORY, std::vector<uint32_t>(13, 0));

    // 7. CONTENTS: 6 u32
    w.record(barrel_tags::CONTENTS, {0x1, 0x300, 0x10001, 0x500, 0x00fffff0, 0x100});

    // 8. MULTI: num_pickles + 0
    w.record(barrel_tags::MULTI, {num_pickles, 0});

    // 9. ENDHDR
    w.record(barrel_tags::ENDHDR, {});

    // 10. 回填 hwords|ver_lo + SIZE[0]
    size_t end_pos = w.tell();
    uint32_t hwords = static_cast<uint32_t>(end_pos / 4);
    w.patch_u32(hwords_pos, (kBarrelVerLo << 16) | (hwords & 0xffff));
    uint32_t size0 = static_cast<uint32_t>((end_pos + kSize0Align - 1) & ~(kSize0Align - 1));
    w.patch_u32(size0_pos, size0);

    return BarrelHeaderPatch{size0_pos, size1_pos};
}

size_t write_pickle_header(ByteWriter& w, const std::vector<uint32_t>& options_words)
{
    // 1. magic + hwords|ver 占位
    w.u32(kPickleMagic);
    size_t hwords_pos = w.tell();
    w.u32(0);

    // 2. IDENT: 50 u32
    w.record(barrel_tags::IDENT, std::vector<uint32_t>(50, 0));

    // 3. SIZE: 6 u32 (SIZE[0]=0x0c00, SIZE[1] 回填, SIZE[2..5]=0/0x44/0x80/0x11)
    size_t size0_pos = w.tell() + 4;
    w.record(barrel_tags::SIZE, {0x00000c00, 0, 0, 0x44, 0x80, 0x11});
    size_t size1_pos = size0_pos + 4;

    // 4. VERSION
    {
        const char* ver = "v2.48.40.260702151143.";
        std::vector<uint32_t> vw(6, 0);
        std::memcpy(vw.data(), ver, std::min<size_t>(strlen(ver), 22));
        w.record(barrel_tags::VERSION, vw);
    }

    // 5. OPTIONS
    w.record(barrel_tags::OPTIONS, options_words);

    // 6. MEMORY: 13 u32 (identity: 0x0a,0,0,0x20,0,0x45,0x40,0,...)
    w.record(barrel_tags::MEMORY, {0x0a, 0, 0, 0x20, 0, 0x45, 0x40, 0, 0, 0, 0, 0, 0});

    // 7. ENDHDR
    w.record(barrel_tags::ENDHDR, {});

    // 8. 回填 hwords|ver
    size_t end_pos = w.tell();
    size_t bytes_from_magic = end_pos - (hwords_pos - 4);
    uint32_t hwords = static_cast<uint32_t>(bytes_from_magic / 4);
    w.patch_u32(hwords_pos, (kPickleVerLo << 16) | (hwords & 0xffff));

    return size1_pos;
}

} // namespace hnnx
