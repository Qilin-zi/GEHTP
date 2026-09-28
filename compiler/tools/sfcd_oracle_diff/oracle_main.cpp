// oracle_main.cpp — 跑 REQNNFRAME 本地 oracle 写侧，打印组装词流（供对拍）
#include "hnnx/vtcm/spillfill_g4.hpp"
#include <cstdio>

int main() {
    using namespace hnnx;
    std::vector<G4SfcdCopyRecord> recs;
    {
        std::vector<G4SfcdCopyEntry> es = {{0x1000, 0x800, 0},
                                           {0x1800, 0x400, 0},
                                           {0xABC0, 0x1000, 0}};
        recs.push_back(g4_sfcd_finalize_copies(2, 0x400, es));
    }
    {
        std::vector<G4SfcdCopyEntry> es = {{0x40, 0x80, 0}};
        recs.push_back(g4_sfcd_finalize_copies(3, 0x8000, es));
    }
    {
        std::vector<G4SfcdCopyEntry> es = {{0x1230, 0x10, 0}, {0x1240, 0x10, 0}};
        recs.push_back(g4_sfcd_finalize_copies(0x2F, 0, es));
    }
    std::vector<G4SfcdWaitEntry> set_events = {{9, 0x1234}, {0xABCDEF, 0xFFFFFFFF}};

    // 变体 1: 窄式 waits + ckpt=7
    {
        std::vector<G4SfcdWaitEntry> waits = {{3, 7}, {0xABC, 0xFF}, {0xFFFFF, 0}};
        auto v = g4_sfcd_assemble(7, recs, set_events, waits);
        std::printf("%zu\n", v.size());
        for (auto w : v) std::printf("%08x ", w);
        std::printf("\n");
    }
    // 变体 2: 宽式 waits + ckpt=7
    {
        std::vector<G4SfcdWaitEntry> waits = {{5, 0x100}, {0x20000, 0xFFFF}, {7, 0x1000}};
        auto v = g4_sfcd_assemble(7, recs, set_events, waits);
        std::printf("%zu\n", v.size());
        for (auto w : v) std::printf("%08x ", w);
        std::printf("\n");
    }
    // 变体 3: 无 waits + ckpt=0
    {
        auto v = g4_sfcd_assemble(0, recs, set_events, {});
        std::printf("%zu\n", v.size());
        for (auto w : v) std::printf("%08x ", w);
        std::printf("\n");
    }
    return 0;
}
