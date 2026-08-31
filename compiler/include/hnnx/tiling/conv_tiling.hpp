#pragma once
// conv_tiling.hpp — 通用单算子 conv 空间分块(含 halo 推导 + 边界钳位 + C-split)
//
// 纯函数模块(不依赖 Op/Graph), 供:
//   * 编译器: Conv2d extra_info 的 tiling 段(serialize_opdef extractor)
//   * 设备侧: wtop_emit / 引擎按 tile 描述做 im2col+GEMM 切片
// halo 公式(通用, 任意 kh/kw/sh/sw/pad/dilation=1):
//   输出 tile 起于 (y0,x0) 尺寸 (th,tw), 输入切片(含 halo)为
//     rows: [y0*sh - ph_begin, y0*sh - ph_begin + (th-1)*sh + kh)  钳位 [0, in_h)
//     cols: 同理
// C-split: co_per_tile>0 时输出通道切成 [co0, co0+co_n) 段, 与空间分块正交。
#include <cstdint>
#include <vector>

namespace hnnx {

// 单 tile 描述(固定二进制布局: 19×u32 = 76B, 设备侧同构解析)
struct ConvTileDesc {
    uint32_t out_y0 = 0, out_x0 = 0, out_h = 0, out_w = 0;  // 输出 tile 位置/尺寸
    uint32_t in_y0 = 0, in_x0 = 0, in_h = 0, in_w = 0;      // 输入切片(含 halo, 已钳位)
    uint32_t kh = 1, kw = 1, sh = 1, sw = 1;                // 卷积几何
    uint32_t ph_begin = 0, pw_begin = 0;                    // pad(单侧; same-pad 取 begin)
    uint32_t ci = 0, co = 0;                                // 全图通道数
    uint32_t co0 = 0, co_n = 0;                             // C-split 切片(co_n==co 表示不切)
    uint32_t block_ref = 0;                                 // VTCM 块引用(阶段7 分配后回填)
};

// 通用 conv 空间分块计算。
// tile_h/tile_w: 期望输出 tile 尺寸(0 表示整图 = 单 tile);
// co_per_tile: 输出通道切片(0 表示不切)。
// 返回: 空间 tile × 通道 tile 的笛卡尔积(行主序: 通道段为最外层)。
std::vector<ConvTileDesc> compute_conv_tiles(
    uint32_t in_h, uint32_t in_w, uint32_t cin,
    uint32_t out_h, uint32_t out_w, uint32_t cout,
    uint32_t kh, uint32_t kw,
    uint32_t sh, uint32_t sw,
    uint32_t ph_begin, uint32_t pw_begin,
    uint32_t tile_h, uint32_t tile_w,
    uint32_t co_per_tile);

} // namespace hnnx
