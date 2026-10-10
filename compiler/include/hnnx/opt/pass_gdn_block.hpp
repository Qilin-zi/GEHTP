#pragma once
// 009 US3: linear_attn 整块折叠 pass (pass_gdn_block.cpp)
#include <string>

namespace hnnx {

// 注册进 PassManager (name="gdn_block", default-off, PHASE_0 注册首位)
void register_pass_gdn_block();

// C1 覆盖率统计: recognized/lowered/fallback/coverage (+逐层 fallback 原因)。
// hnnx_compile --opt-stats 在 stats_line 后追加输出。
std::string gdn_block_opt_stats();

} // namespace hnnx
