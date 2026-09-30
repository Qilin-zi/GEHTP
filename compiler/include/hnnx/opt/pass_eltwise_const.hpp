#pragma once
// pass_eltwise_const_fold —— 同型 const 二元链合并(PHASE_1, 默认开)
// =====================================================================
// 规则(数值恒等容差内, 非位级):
//   - 同型链: EB(c1)∘EB(c2) → EB(c')(c' = f64 运算后 cast 回原 const dtype);
//     已确证操作码: 0=ADD, 13=MULTIPLY(数字枚举直读, 同型只信枚举相等);
//   - Div→Mul: EB(DIV)(x, c) → EB(MUL)(x, 1/c), 仅当 c const 且非零
//     非 denormal; 1/c 用 f32 先算再入池(与设备除法路径容差一致)。
//
// 保守门: 中间张量无旁路消费者(单消费者门, 同 fc_bias 纪律); 操作数须
// 编译期 const(const_pool 可读)。
//
// 数值: f64 计算 + 单次 cast ≤1 ULP f16(SC-002 口径)。
// 幂等: 合并后链消失; 逃生: --no-pass eltwise_const_fold。
namespace hnnx {

void register_pass_eltwise_const();

// Div→Mul 独立注册(默认关): 设备 EB MULTIPLY 溢出口径曾致末行 NaN,
// 见 pass_eltwise_const.cpp 头注; --enable-pass eltwise_div_to_mul 开启。
void register_pass_eltwise_div_to_mul();

} // namespace hnnx
