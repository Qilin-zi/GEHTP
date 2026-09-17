#pragma once
#include "hnnx/ops/ops.hpp"

namespace hnnx {

// HLX (Hexagon Lightweight eXtensions) operators
// Source: hlx_abs.cc, hlx_ceiling.cc, hlx_conv_weights.cc, hlx_elementwise.cc, etc.

struct HlxAbsOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxCeilingOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxConvWeightsOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxElementwiseOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxElementwiseSignOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxExpOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxFloorOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxGeluOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxHadamardOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxLinearClipOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxPreluOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxQElementwiseOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxResizeBilinearGeneralOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxRotaryPosEmbdOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxRoundOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };
struct HlxSoftmaxOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };

// HMX (Hexagon Matrix eXtensions) - for matrix operations
// Source: hmx.cc
struct HmxOp : TypicalOp { static std::unique_ptr<Op> construct(const OpIoPtrs&, op_id_t); };

void register_hlx_hmx_ops();

} // namespace hnnx
