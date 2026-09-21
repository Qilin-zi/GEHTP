#pragma once
// ============================================================================
// fa::RuntimeAllocator —— 分配器中间层 (libHtpPrepare.so 2.48.40.260702, x86_64)
//
// 规范来源: docs/GEHTP_BACKPORT_PLAN.md §1 (P3 行为保持重构, 2026-09-21)
//
// 层级 (§1.1): .so 真三层 hnnx::Allocator → fa::RuntimeAllocator → fa::FancyAllocator
//   证据: FancyAllocator ctor @0xf3f2c0 首调
//   _ZN2fa16RuntimeAllocatorC2EN4hnnx9Allocator4ModeER5Graph@plt (基类 ctor);
//   RuntimeAllocator C2/C1 @0xd8a880, 虚表 _ZTVN2fa16RuntimeAllocatorE @0x5ec1ec0。
//   本树恢复中间层: fa::RuntimeAllocator ← fa::FancyAllocator。
//   RuntimeAllocator **暂不继承 hnnx::Allocator** (M35 两世界统一前不动), 原因:
//     ① tensor_base.hpp 内联成员直调 map_block_reference/deserialize_blocks,
//        需 fa::RuntimeAllocator 完整类型; 本头若再经 hnnx::Allocator 回环
//        包含 tensor_base.hpp 则循环不可解;
//     ② serializer.cpp DefaultSerAllocator 已 : hnnx::Allocator + FancyAllocator
//        双基 (Serializer ctor @0x12f14ab dynamic_cast 跨投影); RuntimeAllocator
//        再携带 hnnx::Allocator 会成菱形, dynamic_cast 歧义;
//     ③ hnnx::Allocator ctor 需 ::Graph&, FancyAllocator 缺省构造链
//        (VtcmCacheInstance 成员 / 各测试栈对象) 无 Graph 供给路径。
//   graph_ 退化为可空指针 (默认 nullptr), 虚面仅保留析构 (槽位工厂经基指针
//   删除旧者所需, 对应 .so "vptr+0x08 deleting dtor")。
//
// 字段布局 (§1.3 定稿, sizeof == 0xb8; P4 接口承诺, 定稿后不再改):
//   +0x00 vptr            (虚析构)
//   +0x08 graph_          (.so: 继承自 Allocator 的 Graph&; 本树可空 Graph*)
//   +0x10 mode_           (u32; ctor `movl %esi,0x10(%rdi)` @0xd8a8a2)
//   +0x14 f_14_           (bool @0xd8a8b3)
//   +0x15..+0x3f 未解码预留
//   +0x40 tcm_pool_base_  (static_assert 证据)
//   +0x48 tcm_pool_size_  (static_assert 证据)
//   +0x50 largest_memory_alloc_size_ (static_assert 证据)
//   +0x58..+0x87 未解码预留
//   +0x88 shared_tensors_ (§1.4 记录面; G4 视角 = PoolDesc 数组, 步长 0x30)
//   +0xa0 shared_spillfill_base_ (static_assert 证据; G4 视角 = spillfill 池句柄)
//   +0xa8/+0xac/+0xb0 slot_sizes_[3] (§2.5 三槽尺寸, set_spillfill_size 回存面)
//   +0xb4 对齐填充
// 成员偏移由 tests/test_runtime_alloc.cpp 运行时钉死 (非标准布局类 offsetof
// 告警规避), sizeof 由下方 static_assert 钉死。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <vector>

// Graph 双世界桥 (同 deserz.hpp/ser_ops_interface.hpp): 精确族 TU
// (-DHNNX_SER_PRECISE) 用全局 ::Graph (mangling 与 .so 一致); 旧族 TU 用
// types.hpp 的 hnnx::Graph —— 不引入全局声明, 免 `using namespace hnnx` 歧义。
#if defined(HNNX_SER_PRECISE)
class Graph;
#if !defined(HNNX_GRAPH_T)
#define HNNX_GRAPH_T ::Graph
#endif
#else
namespace hnnx {
class Graph;
} // namespace hnnx
#if !defined(HNNX_GRAPH_T)
#define HNNX_GRAPH_T hnnx::Graph
#endif
#endif

namespace hnnx {
class Deserz; // serialize/deserz.hpp (class-key: class)
} // namespace hnnx

namespace fa {

// ---- §1.4 记录面: 48B (0x30) 步长记录 --------------------------------------
// .so: shared_tensors 记录数 = ((end-begin) >> 4) × 0xAAAAAAAAAAAAAAAB
//      (0xAAAA...AB 是 3 的模逆: (字节数/16)/3 的精确除, 向量字节数恒为
//      48 的倍数 —— "÷3 语义")。本树 std::vector 的 size() 直接给出记录数。
// G4 视角 (include/hnnx/vtcm/spillfill_g4.hpp G4PoolDesc, 同一 0x30 步长面):
//   size@+0x10, u16 flags@+0x1e (bit0=shared [0xf4ce98 orl $1],
//                               bit4=far   [0xf4cec2 orl $0x10])。
// 其余字段未解码, 以预留保持步长; 解码属 P4/P5 写侧。
struct SharedTensorRecord {
    uint64_t f_00 = 0;  // +0x00 未解码
    uint64_t f_08 = 0;  // +0x08 未解码
    uint32_t size = 0;  // +0x10 G4PoolDesc.size
    uint32_t f_14 = 0;  // +0x14 未解码
    uint16_t f_18 = 0;  // +0x18 未解码
    uint16_t f_1a = 0;  // +0x1a 未解码
    uint16_t f_1c = 0;  // +0x1c 未解码
    uint16_t flags = 0; // +0x1e G4PoolDesc.flags (bit0=shared, bit4=far)
    uint64_t f_20 = 0;  // +0x20 未解码
    uint64_t f_28 = 0;  // +0x28 未解码
};
static_assert(sizeof(SharedTensorRecord) == 0x30, "§1.4 记录步长 48B (0x30)");

class RuntimeAllocator {
  public:
    // Mode (.so hnnx::Allocator::Mode; make_allocator 两分支均传 0, §1.2)
    static constexpr uint32_t MODE_ALLOC_VIRTUAL = 0;

    RuntimeAllocator(); // graph=nullptr, mode=AllocVirtual (FancyAllocator 缺省构造链用)
    RuntimeAllocator(uint32_t mode, HNNX_GRAPH_T &graph); // .so ctor 同形 (Mode, Graph&)
    virtual ~RuntimeAllocator(); // 槽位工厂经基指针 deleting dtor (vptr+0x08 面)

    RuntimeAllocator(const RuntimeAllocator &) = delete;
    RuntimeAllocator &operator=(const RuntimeAllocator &) = delete;

    // ---- §1.2 make_allocator 槽位工厂 (@0xf4e3b0) --------------------------
    // .so: fancy ? new(0x3a0)+FancyAllocator : new(0xb8)+RuntimeAllocator;
    //      两分支 Mode 均传 esi=0 (AllocVirtual); 写入 *(Allocator**)this = p
    //      并返回 this (调用方 Graph ctor @0xd23509 以栈槽为 this, 随后写
    //      this+0x1d8, 旧者调 vptr+0x08 deleting dtor)。
    // 0x3a0 是 .so FancyAllocator 真分配尺寸 (本树对象布局不同, 不作 P3 契约)。
    static RuntimeAllocator *make_allocator(HNNX_GRAPH_T &graph, bool fancy);
    // 槽位变体: 先经虚析构删除旧占用者, 再 *slot = 新实例; 返回 slot。
    // slot == nullptr 或创建失败语义: slot 为 nullptr 时直接返回 nullptr。
    static RuntimeAllocator **make_allocator_at(RuntimeAllocator **slot, HNNX_GRAPH_T &graph, bool fancy);

    // ---- §1.4 记录面访问函数 (@0xd8d640 / @0xd8d6e0) ------------------------
    // 与 tensor_base.hpp 旧 stub 期一致: 仅声明不定义 (全树无调用点,
    // 链接不引用; Tensor 内联调用点原样编译)。真实记录面算法 (48B 记录
    // 读写 + §1.5 错误两型) 属 P5+ 跨组复用升级。
    void const *map_block_reference(unsigned off, unsigned size) const noexcept;
    void deserialize_blocks(hnnx::Deserz &dctx, void const **table, size_t nblocks);

    // ---- §1.3 字段访问面 (P4 接口承诺) --------------------------------------
    HNNX_GRAPH_T *graph() const noexcept { return graph_; }
    void set_graph(HNNX_GRAPH_T *g) noexcept { graph_ = g; }

    uint32_t mode() const noexcept { return mode_; }
    void set_mode(uint32_t m) noexcept { mode_ = m; } // FancyAllocator::set_mode(int) 同名字面

    bool f_14() const noexcept { return f_14_; }
    void set_f_14(bool v) noexcept { f_14_ = v; }

    // set_tcm_pool / set_largest_memory_alloc_size: .so 为 hnnx::Allocator
    // 虚槽 (+0x28/+0x30), RuntimeAllocator 实现体; 本树虚面属 M35, 先给直写面。
    void set_tcm_pool(void *base, size_t size) noexcept
    {
        tcm_pool_base_ = base;
        tcm_pool_size_ = size;
    }
    void *tcm_pool_base() const noexcept { return tcm_pool_base_; }
    size_t tcm_pool_size() const noexcept { return tcm_pool_size_; }

    void set_largest_memory_alloc_size(size_t s) noexcept { largest_memory_alloc_size_ = s; }
    size_t largest_memory_alloc_size() const noexcept { return largest_memory_alloc_size_; }

    // §1.4 记录面 (G4 PoolDesc 数组同一面)
    std::vector<SharedTensorRecord> &shared_tensors() noexcept { return shared_tensors_; }
    const std::vector<SharedTensorRecord> &shared_tensors() const noexcept { return shared_tensors_; }

    uint64_t shared_spillfill_base() const noexcept { return shared_spillfill_base_; }
    void set_shared_spillfill_base(uint64_t v) noexcept { shared_spillfill_base_ = v; }

    // §2.5 三槽 (+0xa8/+0xac/+0xb0); 越界按 §1.5② 抛 std::out_of_range("vector")
    uint32_t slot_size(unsigned i) const;
    void set_slot_size(unsigned i, uint32_t v);
    static constexpr unsigned SLOT_COUNT = 3;

  protected:
    HNNX_GRAPH_T *graph_ = nullptr;      // +0x08
    uint32_t mode_ = 0;             // +0x10
    bool f_14_ = false;             // +0x14
    unsigned char reserved_15_[0x40 - 0x15]{}; // +0x15..+0x3f 未解码预留
    void *tcm_pool_base_ = nullptr; // +0x40
    uint64_t tcm_pool_size_ = 0;    // +0x48
    uint64_t largest_memory_alloc_size_ = 0; // +0x50
    unsigned char reserved_58_[0x88 - 0x58]{}; // +0x58..+0x87 未解码预留
    std::vector<SharedTensorRecord> shared_tensors_; // +0x88 (24B → +0x88..+0x9f)
    uint64_t shared_spillfill_base_ = 0; // +0xa0
    uint32_t slot_sizes_[3] = {0, 0, 0};   // +0xa8/+0xac/+0xb0
    uint32_t reserved_b4_ = 0;             // +0xb4 对齐填充
};

static_assert(sizeof(RuntimeAllocator) == 0xb8, "§1.3 RuntimeAllocator sizeof 定稿 0xb8");

} // namespace fa
