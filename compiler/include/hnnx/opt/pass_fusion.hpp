#pragma once

namespace hnnx {

// pass_fusion —— 激活融合(注册制, PHASE_3, 默认关)。
// 8 条 kLegacyFusionRules 逐条注册为一个 GraphPass(name="fusion_<producer>
// _<consumer>", 小写), matcher=fusion_match、rewrite=fuse_pair。
// 默认 enabled=false: 产出 ConvActivations / 带激活的 Add/MatMul 无
// wtop_emit 翻译、设备 kernel(exec_conv2d/add/binary) 无激活参数 →
// 主路径不得开启; 等设备/emit 补齐后用 --enable-pass fusion_* 打开。
void register_pass_fusion_rules();

} // namespace hnnx
