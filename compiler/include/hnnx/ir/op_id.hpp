#pragma once
// hnnx::op_id_t —— 唯一定义点 (types.hpp 与 fancy_allocator 精确族共用, 避免双头重复别名)
#include <cstdint>

namespace hnnx {
using op_id_t = uint64_t;
} // namespace hnnx
