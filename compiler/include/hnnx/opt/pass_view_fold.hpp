#pragma once
// pass_view_fold —— 视图链合并/吸收(PHASE_1, 默认开)
// =====================================================================
// 三条规则(全部位级恒等: 视图变换是纯索引重排, 无浮点参与):
//   R1  SS∘SS   相邻 StridedSlice 区间代数合成单 op; 合成结果全区间
//               恒等 → 改线删除(消费 producer);
//   R2  Reshape 吸收: SS 的输入是单消费者"纯 1 维增删" Reshape → SS 直接
//               索引底层张量, Reshape 标 dead;
//   R3  T∘T     两 Transpose 置换复合为恒等 → 双删; 非恒等不动
//               (无通用合成发射优化, 留后续)。
//
// 语义口径(关键锚定, 2026-09-30 核实): host(src/ops/ops.cpp StridedSlice)
// 与设备(kernels exec_slice)均**只认 ranges const**——每轴三元组交错
// [b,e,s]×rank(loader 拍平 [[b,e,s]...]); begin/end/new_axes/shrink 四掩码
// 是惰性元数据, 发射/执行两侧都不读。故本 pass 的合并语义 = ranges 区间
// 代数, 掩码不参与(合成后顺手清零, 防陈旧元数据误导阅读)。
//
// 保守门(任一不满足不动图):
//   - 单消费者门: 中间张量被旁路消费时不合(同 fc_bias_fold 纪律);
//   - ranges const 可解析(const_pool 内, rank*3 个 i32, rank≤5);
//   - 输入 rank 与 ranges rank 右对齐后逐轴一致(秩不配不猜);
//   - 合成后外层选择区间不越界(b + (cnt-1)*s < dim)。
//
// 幂等: 合并后链消失, 二跑零改写。逃生: --no-pass view_fold。
// stats: view_fold=<matched>/<rewritten>(既有 stats_line 格式)。
#include <cstdio>

namespace hnnx {

void register_pass_view_fold();

// T004 普查: prepare 视角的视图链基数 dump(graph_prepare.cpp 的
// GEHTP_DUMP_VIEWS 环境变量钩子调用; 与 pass 同谓词, 避免口径分叉)。
void dump_view_census(class GraphPrepare* gp, std::FILE* out);

} // namespace hnnx
