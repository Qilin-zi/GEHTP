#pragma once

namespace hnnx {

// pass_const_fold —— 常量折叠(求值版, PHASE_0)。
// 对应真实 const_prop/const_prop_extract_outputs(graph_prepare.cc), 但真实
// 库的求值语义未 RE; 这里接我方 TypicalOp::execute host 参考核求值。
// 铁律(照抄 const_prop 壳注释): 折叠必须算出值 —— 只标 OP_CONST 不存值
// 会让 emit 跳过 op → 消费方读空槽 → 数值错(M3c probe 实锤)。
// 求值失败/不在白名单 → 不折(保守, 数值永不错)。
void register_pass_const_fold();

} // namespace hnnx
