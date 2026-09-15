#pragma once

namespace hnnx {

// pass_shape_fold —— 形状归一化(PHASE_1)。
// 结构匹配模式(均为纯 IR 重写, 零设备改动):
//   A. 恒等 Reshape/Flatten(输出 dims == 输入 dims)→ 删除 + 消费者改线;
//   B. Reshape→Reshape 链(1:1)→ 链首并入链尾(目标 dims 不变);
//   C. 恒等 Transpose(全部 tensor_param 为恒等序列)→ 删除 + 消费者改线。
// 安全门: A/C 恒等折叠后消费者看到的形状与原来完全相同(producer 的
// output_def ≡ 被删 op 的 output_def), 对任意消费者都安全; B 只要求
// 链尾仍是 Reshape(按元素数读平铺数组)且链首仅被链尾消费。
// 跨布局模式(Transpose∘Transpose perm 合成)暂不实现 —— perm 常数可能被
// 多 op 共享, 合成需替换共享 const, 待有非共享性判定后补。
void register_pass_shape_fold();

} // namespace hnnx
