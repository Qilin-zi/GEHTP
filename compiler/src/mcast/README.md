# src/mcast/ — 多播优化

对应真实 `grdep_mcast_optimizer.cc`。多 NSP 间的多播通信优化。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `mcast_optimizer.cpp` | `grdep_mcast_optimizer.cc` @ 0x1075FF0 等 | McastOptimizer::optimize: supercast 合并 (重叠 MCID 合并), build_lp_input, ILP 模型 (CLP/HiGHS simplex) |

## supercast

当多个 McSend 共享 MCID 且 receiver 集合重叠时, 合并成单个 supercast 减少通信次数。
test_e2e 验证: 2 个重叠 MCID=10 的 send 合并成 1 个。

## ILP

真实库用线性规划求最优多播调度; 框架引用 `clp_simplex.cc` / `optimization_solver_highs.h`, 当前为占位。
