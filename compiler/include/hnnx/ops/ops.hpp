#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>

namespace hnnx {

// Forward declare
struct Graph;

// Base op class for all HTP operations
// Source: typical_op_prepare.cc, op_def.cc, match_op.cc

// 指令选择: 根据 op 类型 + dtype 选择 HVX/HMX wrapper 变体。
// 返回 wrapper 标识符 (供序列化记录运行期用哪个内核)。
// Source: hvx_wrapper_select.cc, hmx_op_select.cc
enum class HwWrapper : uint8_t {
    HVX_Vector     = 0,  // 128B 向量内核
    HVX_Scalar      = 1,  // 标量后备
    HMX_Matrix      = 2,  // HMX 矩阵内核 (Conv/MatMul)
    HMX_MatrixInt4  = 3,  // HMX 4-bit 量化内核
    Ref_Host        = 0xFE,// host reference (无 DSP)
};
HwWrapper select_wrapper(const std::string& op_type, DType dtype, uint32_t soc_type);

class TypicalOp : public Op {
public:
    TypicalOp() { graph = nullptr; }
    float cost(const Graph* g) const override;
    void serialize_internal(class Serializer&, int) const override;

    // Op type name (set by constructor)
    std::string op_type_name;
    // 解析后的 op 参数 (从 OpDef::op_data 提取)
    std::vector<uint8_t> params;
    // 缓存的 output_def (从 OpDef 拷贝，供 tiler/cost 直接读取)
    OutputDef cached_out_def{};
    // 选中的硬件 wrapper (指令选择结果)
    HwWrapper wrapper = HwWrapper::Ref_Host;
    // op_id and input connections (set by factory, used by execute_host after DCE)
    op_id_t op_id = 0;
    std::vector<InputConn> exec_inputs;

    // 执行接口: host-side reference 实现 (D 阶段, 定义在 ops.cpp)
    // inputs: 每个输入缓冲的指针; outputs[0]: 输出缓冲; out_def: 输出形状
    // in_defs: 各输入的 OutputDef (供 MatMul 等需要输入形状的 op 使用)
    virtual void execute(const std::vector<const uint8_t*>& inputs,
                         uint8_t* output,
                         const struct OutputDef& out_def) const;
    virtual void execute(const std::vector<const uint8_t*>& inputs,
                         uint8_t* output,
                         const struct OutputDef& out_def,
                         const std::vector<OutputDef>& in_defs) const;
};

// Op registration macros - each op registers its constructor
// Source: op_package_registry.cc, op_package_ops_opts_registration.cc

#define REGISTER_HTP_OP(name) \
    static bool _reg_##name = hnnx::OpRegistry::instance().register_op_fn(#name, name::construct)

// All 200+ operations, organized by category
// Source files: _conv.cc, _matmul.cc, _relu.cc, etc.

// === Convolution ops ===
struct ConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv.cc, conv_opts.cc
    // Parameters: stride, padding, dilation, group, weight, bias
};

struct DepthWiseConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: dwconv.cc, dwconv_opts.cc
};

struct DilatedConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: dilated_conv.cc
};

struct GroupedConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: grouped_conv.cc
};

struct SparseConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: sparse_conv.cc
};

struct IndiceConvOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: indice_conv.cc
};

struct ConvResidualScalingOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv_residual_scaling.cc
};

struct ConvFusedBiasScaleOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv_fused_biasscale.cc
};

struct ConvActivationsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv_activations.cc
};

struct ConvConvertTransposeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv_convert_transpose.cc
};

struct ConvSpecialEncodingsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: conv_special_encodings.cc
};

// === MatMul ops ===
struct MatMulOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: matmul.cc, matmul_opts.cc
};

struct DenseOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: dense.cc
};

// === Elementwise ops ===
struct AddOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _add.cc
};

struct SubOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _sub.cc (implicit in elementwise.cc)
};

struct MulOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _mul.cc
};

struct DivOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _div.cc
};

struct MinMaxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _minmax.cc
};

struct SquaredDiffOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _squared_difference.cc
};

struct ElementWiseFloorDivOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _elementwise_floordiv.cc
};

struct ElementWiseSignOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _elementwise_sign.cc
};

// === Activation ops ===
struct ReluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _relu.cc, relu.cc
};

struct SigmoidOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _sigmoid.cc, sigmoid.cc
};

struct TanhOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _tanh.cc
};

struct EluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _elu.cc
};

struct GeluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _gelu.cc, gelu.cc
};

struct SwishOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _swish.cc
};

struct HardSigmoidOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _hardsigmoid.cc
};

struct HardSwishOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _hardswish.cc
};

struct SoftplusOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _softplus.cc
};

struct LeakyReluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _leaky_relu_opts.cc
};

struct PReluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _prelu.cc
};

struct ClampOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _clamp.cc
};

struct LinearClipOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _linearclip.cc
};

// === Math ops ===
struct ExpOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _exp.cc
};

struct LogOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _log.cc
};

struct SqrtRsqrtOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _sqrt_rsqrt.cc
};

struct PowerOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _power.cc
};

struct AbsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _abs.cc
};

struct SinOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _cos_sin.cc
};

struct AtanOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _atan.cc
};

struct AsinOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _asin.cc
};

struct LogitOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _logit.cc
};

// === Rounding ops ===
struct CeilingOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _ceiling.cc
};

struct FloorOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _floor.cc
};

struct RoundOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _round.cc
};

struct NegOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _elementwiseneg.cc
};

// === Pooling ops ===
struct AvgPoolOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _avgpool.cc
};

struct MaxPoolOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _maxpool.cc
};

// === Normalization ops ===
struct BatchNormOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _batchnorm.cc
};

struct LayerNormOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _layernorm.cc
};

struct InstanceNormOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _instance_norm.cc
};

struct GroupNormOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _groupnorm.cc
};

struct RmsNormOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _rms_norm.cc
};

struct LrnOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _lrn.cc
};

// === Softmax ops ===
struct SoftmaxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _softmax.cc
};

struct LogSoftmaxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _logsoftmax.cc
};

struct MaskedSoftmaxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _masked_softmax.cc
};

// === RNN ops ===
struct LstmOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _lstm.cc, _monolithic_lstm.cc
};

struct GruOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _gru.cc
};

// === Shape manipulation ops ===
struct ReshapeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _reshape.cc
};

struct TransposeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _transpose.cc
};

struct ConcatOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _concat.cc (split.cc)
};

struct SplitOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _split.cc
};

struct FlattenOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _flatten.cc
};

struct ChannelShuffleOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _channelshuffle.cc
};

// === Slice/Pad ops ===
struct SliceOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _slice.cc
};

struct StridedSliceOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _stridedslice.cc
};

struct PadOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _pad.cc
};

struct MirrorPadOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _mirrorpad.cc
};

// === Gather/Scatter ops ===
struct GatherOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _gather.cc
};

struct GatherElementsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _gather_elements.cc
};

struct GatherNdOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _gather_nd.cc
};

struct ScatterElementsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _scatter_elements.cc
};

struct ScatterNdOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _scatternd.cc
};

// === Reduction ops ===
struct ReduceOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _reduce.cc
};

struct ArgMinMaxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _argminmax.cc
};

struct CumSumOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _cumsum.cc
};

// === Cast/Quantize ops ===
struct CastOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _cast_op.cc
};

struct FpCastOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _fp_cast_op.cc
};

struct QuantizeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _quantize.cc
};

struct DequantizeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _dequantize.cc
};

struct RequantizeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _requantize.cc
};

// === Compare ops ===
struct CompareOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _compare.cc
};

struct CompareEqualOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _compare_equal.cc
};

struct LogicalOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _logical.cc
};

struct LogicalCompareOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _logical_compare.cc
};

struct SelectOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _select.cc
};

struct IsInfOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _isinf.cc
};

struct IsNanOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _isnan_opts.cc
};

// === Resize ops ===
struct ResizeBilinearOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _resize_bilinear.cc
};

struct ResizeNearestOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _resize_nearest.cc
};

struct ResizeTrilinearOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _resize_trilinear.cc
};

struct ResizeCubicOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _resizecubic.cc
};

// === Other ops ===
struct TopKOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _topk.cc, _topk_bitonic.cc
};

struct NmsOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _nms.cc, _nms_bitonic.cc
};

struct OneHotOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _onehot.cc
};

struct GridSampleOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _grid_sample.cc
};

struct RoiAlignOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _roialign.cc
};

struct ExtractGlimpseOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _extract_glimpse.cc
};

struct EinsumOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _einsum.cc
};

struct HadamardOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _hadamard.cc
};

struct SwiGluOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _swiglu.cc
};

struct ToRgbOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _to_rgb.cc
};

struct StftOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _stft.cc
};

// === Space/Depth transform ops ===
struct SpaceToDepthOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _space_to_depth.cc
};

struct DepthToSpaceOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _depth_to_space.cc
};

struct FrameToDepthOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _frame_to_depth.cc
};

struct DepthToFrameOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _depth_to_frame.cc
};

struct SpaceToBatchOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _space_to_batch.cc
};

struct BatchToSpaceOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _batch_to_space.cc
};

// === Attention/RoPE ops ===
struct RotaryPosEmbdOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _rotary_pos_embd.cc
};

struct RopeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _rope.cc
};

struct RopeCoeffPackOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _rope_coeff_pack.cc
};

// === Misc ops ===
struct BiasAddOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _biasadd.cc
};

struct HashtableLookupOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _hashtable_lookup.cc
};

struct DecodeBboxOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _decode_bbox.cc
};

struct ConvertCenterToCornerOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _convert_center_to_corner.cc
};

struct RandomNormalLikeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _random_normal_like.cc
};

struct RandomUniformLikeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _random_uniform_like.cc
};

struct BitrearrangeOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _bitrearrange.cc
};

struct NonZeroOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _nonzero.cc
};

struct AllTrueOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _all_true.cc
};

struct PartialSumOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _partial_sum.cc
};

struct DepthAccumulateOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _depth_accumulate.cc
};

struct WidthToHeightOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _width_to_height.cc
};

struct UnpackOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _unpack.cc
};

struct TileOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _tile.cc
};

// === Conditional/Switch ops ===
struct ConditionalOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _conditional.cc
};

struct ConditionalAddOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _conditional_add.cc
};

struct SwitchedOp : TypicalOp {
    static std::unique_ptr<Op> construct(const OpIoPtrs& io, op_id_t id);
    // Source: _switched_op.cc
};

} // namespace hnnx

namespace hnnx { void register_all_ops(); }
