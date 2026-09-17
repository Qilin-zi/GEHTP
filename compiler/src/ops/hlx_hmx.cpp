#include "hnnx/ops/hlx_hmx.hpp"
#include "hnnx/ir/op_registry.hpp"

namespace hnnx {

static std::unique_ptr<Op> hlx_construct(const OpIoPtrs& io, op_id_t id) { return nullptr; }

std::unique_ptr<Op> HlxAbsOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxCeilingOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxConvWeightsOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxElementwiseOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxElementwiseSignOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxExpOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxFloorOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxGeluOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxHadamardOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxLinearClipOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxPreluOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxQElementwiseOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxResizeBilinearGeneralOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxRotaryPosEmbdOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxRoundOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HlxSoftmaxOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }
std::unique_ptr<Op> HmxOp::construct(const OpIoPtrs& io, op_id_t id) { return hlx_construct(io, id); }

void register_hlx_hmx_ops() {
    auto& reg = OpRegistry::instance();
    reg.register_op_fn("HlxAbs", HlxAbsOp::construct);
    reg.register_op_fn("HlxCeiling", HlxCeilingOp::construct);
    reg.register_op_fn("HlxConvWeights", HlxConvWeightsOp::construct);
    reg.register_op_fn("HlxElementwise", HlxElementwiseOp::construct);
    reg.register_op_fn("HlxElementwiseSign", HlxElementwiseSignOp::construct);
    reg.register_op_fn("HlxExp", HlxExpOp::construct);
    reg.register_op_fn("HlxFloor", HlxFloorOp::construct);
    reg.register_op_fn("HlxGelu", HlxGeluOp::construct);
    reg.register_op_fn("HlxHadamard", HlxHadamardOp::construct);
    reg.register_op_fn("HlxLinearClip", HlxLinearClipOp::construct);
    reg.register_op_fn("HlxPrelu", HlxPreluOp::construct);
    reg.register_op_fn("HlxQElementwise", HlxQElementwiseOp::construct);
    reg.register_op_fn("HlxResizeBilinearGeneral", HlxResizeBilinearGeneralOp::construct);
    reg.register_op_fn("HlxRotaryPosEmbd", HlxRotaryPosEmbdOp::construct);
    reg.register_op_fn("HlxRound", HlxRoundOp::construct);
    reg.register_op_fn("HlxSoftmax", HlxSoftmaxOp::construct);
    reg.register_op_fn("Hmx", HmxOp::construct);
}

} // namespace hnnx
