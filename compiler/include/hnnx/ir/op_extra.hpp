#pragma once
// per-op 参数 schema 契约(extractor 写 → serialize_opdef 尾随 → wtop_emit 消费)
//
// 每个 op 型一个固定字节布局(小端, 无 padding 歧义——显式 u32/u64 字段序)。
// 提取器从 op_data(scalar_params 打包)+ tensor_param const + output_def 形状
// 推导参数; emit 端按同名 struct 解析。新算子 = 新 struct + 注册 extractor。
#include <cstdint>

namespace hnnx {

// Eltwise_Binary / Eltwise_Unary / ElementWiseNeuron / Eltwise_Ternary:
// operation 值(QNN 约定, 实测 2.48): binary 0=ADD 1=SUB 2=MUL 3=DIV
struct ExtraEltwise {
    uint32_t operation;      // +0
    uint32_t eltwise_type;   // +4 (Eltwise_Ternary 用: 0=SELECT)
};

// Softmax / Gather / Concat / Reduce / CumulativeSum: axis 系
struct ExtraAxis {
    int32_t axis;            // +0
    uint32_t reduce_type;    // +4 (Reduce: 0=SUM 1=AVG ...)
    uint32_t keep_dims;      // +8
    uint32_t exclusive;      // +12 (CumulativeSum)
    uint32_t reverse;        // +16 (CumulativeSum)
};

// RmsNorm
struct ExtraRmsNorm {
    double epsilon;          // +0
    uint32_t axes;           // +8 (通常 1 个轴)
    uint32_t pad;            // +12
};

// FullyConnected: 纯投影 M×K @ K×N(+bias)。M 由输入形状得, K/N 来自权重 const
struct ExtraFc {
    uint32_t m;              // +0  行(输出激活行)
    uint32_t k;              // +4  内维
    uint32_t n;              // +8  列(输出通道)
    uint32_t has_bias;       // +12
};

// MatMul
struct ExtraMatMul {
    uint32_t m;              // +0
    uint32_t k;              // +4
    uint32_t n;              // +8
    uint32_t transpose_in0;  // +12
    uint32_t transpose_in1;  // +16
    uint32_t batched;        // +20  batch-BMM(q·kᵀ per head: 批维不折叠)
};

// StridedSlice: ranges 从 tensor_param const 读出(begin/end/strides 各 4 字节/维)
struct ExtraStridedSlice {
    uint32_t rank;           // +0
    uint32_t begin_mask;     // +4
    uint32_t end_mask;       // +8
    uint32_t new_axes_mask;  // +12
    uint32_t shrink_axes;    // +16
    uint32_t ranges_offset;  // +20 (const 池偏移, 指向 rank×3×u32 数组)
};

// Reshape: 目标形状(dims 数组直接进 extra, rank ≤ 5)
struct ExtraReshape {
    uint32_t rank;           // +0
    int64_t dims[5];         // +4..+43
};

// Transpose: perm 数组(rank ≤ 5)
struct ExtraTranspose {
    uint32_t rank;           // +0
    int64_t perm[5];         // +4..+43
};

// Conv2d: 60B 固定头(全小端 u32/u64, 设备侧同构解析), 后随 tiling 段
// [u32 tile_h][u32 tile_w][u32 co_per_tile][u32 num_tiles][ConvTileDesc × num_tiles]
struct ExtraConv {
    uint32_t sh = 1, sw = 1;
    uint32_t ph_begin = 0, ph_end = 0, pw_begin = 0, pw_end = 0;
    uint32_t dh = 1, dw = 1, group = 1, kh = 1, kw = 1;
    uint64_t weight_src = 0, bias_src = 0;
};

// DepthWiseConv2d(SSM conv 等): 几何字段(ExtraConv 的缩减版)
struct ExtraDepthwiseConv {
    uint32_t kh;             // +0
    uint32_t kw;             // +4
    uint32_t sh;             // +8
    uint32_t sw;             // +12
    uint32_t pad_l;          // +16
    uint32_t pad_r;          // +20
    uint32_t pad_t;          // +24
    uint32_t pad_b;          // +28
    uint32_t group;          // +32
};

// Pad
struct ExtraPad {
    uint32_t scheme;         // +0
    double constant_value;   // +8
};

// ScatterNd
struct ExtraScatterNd {
    uint32_t reduction;      // +0
};

// Split: 段数 + 每段大小(最多 8 段, 常用 2-3)
struct ExtraSplit {
    uint32_t axis;           // +0
    uint32_t num_splits;     // +4
    uint32_t sizes[8];       // +8..+39
    uint32_t split_index;    // +40  本副本在 Split 的段序号(loader 注入)
};

} // namespace hnnx
