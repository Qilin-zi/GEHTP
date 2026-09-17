// ============================================================================
// serializer_bridge.cpp —— 双世界链接桥 TU
//
// 背景 (M33-5): 导出符号由定义 TU 的视角决定。serializer.cpp/deserializer.cpp
// 以 -DHNNX_SER_PRECISE 编译 → ctor 符号与 .so 逐字节一致:
//   _ZN4hnnx10SerializerC2ERK12GraphPreparePNS_9AllocatorEPcm  (@0x12f1320)
//   _ZN4hnnx12DeserializerC2EPKcmP5Graph                      (@0xcfcf20)
// 但旧族调用点 (graph_prepare.cpp / 各测试, types.hpp 的 hnnx::GraphPrepare /
// hnnx::Graph 近似名) 引用的是 hnnx 修饰版本 → 链接缺口。
//
// 本 TU 以 -DHNNX_SER_BRIDGE 含 serializer.hpp: 两名并见 (全局 GraphPrepare/
// Graph 前置仅本 TU 可见, 不外泄 → 不产生 using-directive 歧义), 定义旧族签名
// ctor, 委托转发到精确族本体。布局/虚表不变 —— 纯转发行。
// M36 (GraphPrepare 反演为全局名) 后旧族调用点消失, 本文件整体删除。
// ============================================================================
#define HNNX_SER_BRIDGE 1
#include "hnnx/serialize/serializer.hpp"

namespace hnnx {

Serializer::Serializer(HNNX_GP_T const &gp, Allocator *alloc, char *buf, size_t buflen)
    : Serializer(reinterpret_cast<::GraphPrepare const &>(gp), alloc, buf, buflen)
{
}

Deserializer::Deserializer(char const *data, size_t size, HNNX_GRAPH_T *graph)
    : Deserializer(data, size, reinterpret_cast<::Graph *>(graph))
{
}

} // namespace hnnx
