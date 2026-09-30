#pragma once
// pass_fc_bias —— FullyConnected bias 折叠(PHASE_3, 默认开)
// =====================================================================
// 模式: FullyConnected(2 输入, 权重 const) → [Reshape] → Eltwise_Binary(ADD,
//        另一输入为 N 元素 const bias)  ⇒  bias 并入 FC inputs[2], 删 Add。
//
// 与 8 条 legacy 融合规则的本质区别: 本规则**设备准入** ——
//   emit:  op_matmul 读 ExtraFc.has_bias + inputs[2] → OP_MATMUL_F16_BIAS
//          (opcode 36, arity 8, 追加 bias 槽引用);
//   设备:  exec_matmul_f16 标量路径在 f32 累加器上加 bias[n](单次舍入,
//          比"matmul 存 f16 再 Add"少一次 f16 往返);
//   host:  ops.cpp FC 非 batched 分支同款 bias 处理, host/device 口径一致。
// 数值: 折叠前 matmul 出 f16 → Add 读回 f32 加 bias → f16; 折叠后 f32 累加器
//       直接加 bias → f16。单次舍入差 ≤1 ulp f16, 精度门"不回归"内。
//
// 保守门(任一不满足即不折, 图不动):
//   - FC 仅 2 输入(尚无 bias), 权重 inputs[1] 为 const;
//   - 中间 Reshape 可选, 存在时须 1:1(仅 Add 一个消费者);
//   - Add 为 Eltwise_Binary 且 operation==ADD(0), 另一输入是 N 元素 const
//     (N = FC 输出末维; 严格 [N], 不猜广播语义);
//   - **HMX 门**: M/K/N 全 32 倍数(设备会走 HMX crouton)时不折 ——
//     内核 v1 仅在标量路径应用 bias(HMX bias 页逐列装载未上板验证),
//     挡住可避免"HMX 快路径静默退化标量"的性能悬崖。0.8B 实测 18 处
//     in_proj_a (M=32,K=1024,N=16) 全部 N%32!=0 → 正常折叠。
//
// 可信标记: pass 在 fc->op_data 写 scalar "has_bias"=1; extract_fc_extra
// (→ .bin ExtraFc → emit) 与 host execute 只认这个 scalar —— 注入形态下
// inputs[2] 常是 N 元素 wscale const, 形状启发不可靠(已废弃)。
namespace hnnx {

void register_pass_fc_bias();

} // namespace hnnx
