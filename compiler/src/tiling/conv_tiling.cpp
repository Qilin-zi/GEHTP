#include "hnnx/tiling/conv_tiling.hpp"

#include <algorithm>

namespace hnnx {

std::vector<ConvTileDesc> compute_conv_tiles(
    uint32_t in_h, uint32_t in_w, uint32_t cin,
    uint32_t out_h, uint32_t out_w, uint32_t cout,
    uint32_t kh, uint32_t kw,
    uint32_t sh, uint32_t sw,
    uint32_t ph_begin, uint32_t pw_begin,
    uint32_t tile_h, uint32_t tile_w,
    uint32_t co_per_tile) {
    std::vector<ConvTileDesc> tiles;

    if (out_h == 0 || out_w == 0) return tiles;
    uint32_t th = (tile_h > 0) ? tile_h : out_h;
    uint32_t tw = (tile_w > 0) ? tile_w : out_w;
    if (sh == 0) sh = 1;
    if (sw == 0) sw = 1;
    uint32_t co_s = (co_per_tile > 0 && co_per_tile < cout) ? co_per_tile : cout;

    for (uint32_t co0 = 0; co0 < cout; co0 += co_s) {
        uint32_t co_n = std::min(co_s, cout - co0);
        for (uint32_t y0 = 0; y0 < out_h; y0 += th) {
            uint32_t oh = std::min(th, out_h - y0);
            for (uint32_t x0 = 0; x0 < out_w; x0 += tw) {
                uint32_t ow = std::min(tw, out_w - x0);
                ConvTileDesc t{};
                t.out_y0 = y0; t.out_x0 = x0; t.out_h = oh; t.out_w = ow;
                // halo 推导: [y0*sh - ph_begin, ... + (oh-1)*sh + kh), 钳位 [0, in_h)
                long long r0 = (long long)y0 * sh - ph_begin;
                long long r1 = r0 + (long long)(oh - 1) * sh + kh;
                long long c0 = (long long)x0 * sw - pw_begin;
                long long c1 = c0 + (long long)(ow - 1) * sw + kw;
                r0 = std::max(r0, 0LL); c0 = std::max(c0, 0LL);
                r1 = std::min(r1, (long long)in_h); c1 = std::min(c1, (long long)in_w);
                t.in_y0 = (uint32_t)r0; t.in_x0 = (uint32_t)c0;
                t.in_h = (uint32_t)std::max(r1 - r0, 0LL);
                t.in_w = (uint32_t)std::max(c1 - c0, 0LL);
                t.kh = kh; t.kw = kw; t.sh = sh; t.sw = sw;
                t.ph_begin = ph_begin; t.pw_begin = pw_begin;
                t.ci = cin; t.co = cout;
                t.co0 = co0; t.co_n = co_n;
                tiles.push_back(t);
            }
        }
    }
    return tiles;
}

} // namespace hnnx
