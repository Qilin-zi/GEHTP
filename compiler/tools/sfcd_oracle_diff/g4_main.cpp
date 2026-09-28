// g4_main.cpp — 跑 104 树写侧，打印组装词流（同一输入集，供对拍 oracle_main）
#include "hnnx/vtcm/spillfill_g4.hpp"
#include <cstdio>

using namespace hnnx;

static void run(int32_t ckpt,
                const std::vector<G4SfcdWaitEntry>& waits) {
    G4SlcArea area;
    area.area_id = 0x1234;
    area.is_fill = false;
    area.dma_checkpoint = ckpt;

    G4SlcRecord r0;
    r0.rec_type = G4_REC_SPILLFILL;
    r0.ddr_pool = 2;
    r0.sf_offset = 0x400;
    r0.copies = {{0x1000, 0x800, 0}, {0x1800, 0x400, 0}, {0xABC0, 0x1000, 0}};
    area.records.push_back(r0);

    G4SlcRecord r1;
    r1.rec_type = G4_REC_SPILLFILL;
    r1.ddr_pool = 3;
    r1.sf_offset = 0x8000;
    r1.copies = {{0x40, 0x80, 0}};
    area.records.push_back(r1);

    G4SlcRecord r2;
    r2.rec_type = G4_REC_SPILLFILL;
    r2.ddr_pool = 0x2F;
    r2.sf_offset = 0;
    r2.copies = {{0x1230, 0x10, 0}, {0x1240, 0x10, 0}};
    area.records.push_back(r2);

    G4SlcRecord r3;
    r3.rec_type = G4_REC_SETPROGRESS;
    r3.mb_idx = 9;
    r3.value_to_set = 0x1234;
    area.records.push_back(r3);

    G4SlcRecord r4;
    r4.rec_type = G4_REC_SETPROGRESS;
    r4.mb_idx = 0xABCDEF;
    r4.value_to_set = 0xFFFFFFFF;
    area.records.push_back(r4);

    if (!waits.empty()) {
        G4SlcRecord r5;
        r5.rec_type = G4_REC_WAITFOR;
        r5.pairs = waits;
        area.records.push_back(r5);
    }

    const auto v = g4_construct_sfcd(area);
    std::printf("%zu\n", v.size());
    for (auto w : v) std::printf("%08x ", w);
    std::printf("\n");
}

int main() {
    run(7, {{3, 7}, {0xABC, 0xFF}, {0xFFFFF, 0}});              // 窄式
    run(7, {{5, 0x100}, {0x20000, 0xFFFF}, {7, 0x1000}});        // 宽式
    run(0, {});                                                  // 无 waits
    return 0;
}
