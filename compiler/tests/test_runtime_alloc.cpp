// Test: fa::RuntimeAllocator 中间层 (P3 行为保持重构验收)
// 规范: docs/GEHTP_BACKPORT_PLAN.md §1
//   §1.1 层级恢复 (FancyAllocator : public RuntimeAllocator)
//   §1.2 make_allocator 槽位工厂 (@0xf4e3b0: fancy 分派, Mode 恒 0)
//   §1.3 字段布局定稿 (sizeof 0xb8; 成员偏移此处运行时钉死 —— P4 接口承诺)
//   §1.4 记录面 48B 步长 ÷3 语义
//   §1.5② 越界抛 std::out_of_range("vector")
#include "hnnx/vtcm/runtime_alloc.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/ir/types.hpp" // hnnx::Graph (旧族桥: HNNX_GRAPH_T=hnnx::Graph)
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using fa::FancyAllocator;
using fa::RuntimeAllocator;
using fa::SharedTensorRecord;

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char *msg)
{
    if (cond) {
        ++tests_passed;
    } else {
        ++tests_failed;
        std::cout << "  FAIL: " << msg << "\n";
    }
}

// 布局探针: protected 字段经派生类暴露, 指针差钉死 §1.3 偏移
// (非标准布局类规避 offsetof; 成员函数不影响对象布局)。
struct LayoutProbe : RuntimeAllocator {
    static size_t off_graph() { LayoutProbe p; return po(p, &p.graph_); }
    static size_t off_mode() { LayoutProbe p; return po(p, &p.mode_); }
    static size_t off_f14() { LayoutProbe p; return po(p, &p.f_14_); }
    static size_t off_tcm_base() { LayoutProbe p; return po(p, &p.tcm_pool_base_); }
    static size_t off_tcm_size() { LayoutProbe p; return po(p, &p.tcm_pool_size_); }
    static size_t off_largest() { LayoutProbe p; return po(p, &p.largest_memory_alloc_size_); }
    static size_t off_shared_tensors() { LayoutProbe p; return po(p, &p.shared_tensors_); }
    static size_t off_spillfill_base() { LayoutProbe p; return po(p, &p.shared_spillfill_base_); }
    static size_t off_slot(unsigned i) { LayoutProbe p; return po(p, &p.slot_sizes_[i]); }

  private:
    template <typename M>
    static size_t po(LayoutProbe &p, const M *m)
    {
        return static_cast<size_t>(reinterpret_cast<const char *>(m) -
                                   reinterpret_cast<const char *>(&p));
    }
};

// 删除观察器: 验证 make_allocator_at 经基指针虚析构删除旧占用者
// (§1.2 "旧者调 vptr+0x08 deleting dtor" 面对应)
struct TrackingAlloc : RuntimeAllocator {
    static int dtor_count;
    ~TrackingAlloc() override { ++dtor_count; }
};
int TrackingAlloc::dtor_count = 0;

int main()
{
    std::cout << "== RuntimeAllocator P3 中间层验收 ==\n";

    // ---- §1.3 字段布局定稿 (P4 接口承诺) ----
    check(sizeof(RuntimeAllocator) == 0xb8, "sizeof(RuntimeAllocator) == 0xb8");
    check(sizeof(LayoutProbe) == 0xb8, "sizeof(派生无新成员) == 0xb8");
    check(LayoutProbe::off_graph() == 0x08, "graph_ @ +0x08");
    check(LayoutProbe::off_mode() == 0x10, "mode_ @ +0x10");
    check(LayoutProbe::off_f14() == 0x14, "f_14_ @ +0x14");
    check(LayoutProbe::off_tcm_base() == 0x40, "tcm_pool_base_ @ +0x40");
    check(LayoutProbe::off_tcm_size() == 0x48, "tcm_pool_size_ @ +0x48");
    check(LayoutProbe::off_largest() == 0x50, "largest_memory_alloc_size_ @ +0x50");
    check(LayoutProbe::off_shared_tensors() == 0x88, "shared_tensors_ @ +0x88");
    check(LayoutProbe::off_spillfill_base() == 0xa0, "shared_spillfill_base_ @ +0xa0");
    check(LayoutProbe::off_slot(0) == 0xa8, "slot_sizes_[0] @ +0xa8");
    check(LayoutProbe::off_slot(1) == 0xac, "slot_sizes_[1] @ +0xac");
    check(LayoutProbe::off_slot(2) == 0xb0, "slot_sizes_[2] @ +0xb0");

    // ---- §1.4 记录面: 48B 步长 ÷3 语义 ----
    check(sizeof(SharedTensorRecord) == 0x30, "SharedTensorRecord 步长 48B (0x30)");
    {
        RuntimeAllocator ra;
        check(ra.shared_tensors().empty(), "shared_tensors 初始空");
        ra.shared_tensors().push_back(SharedTensorRecord{});
        ra.shared_tensors().push_back(SharedTensorRecord{});
        ra.shared_tensors().back().size = 0x1234;
        ra.shared_tensors().back().flags = 0x11; // bit0=shared | bit4=far
        const auto &v = ra.shared_tensors();
        // ÷3 语义: 向量字节数恒为 48 的倍数, 记录数 = (字节数/16)/3
        check((v.size() * sizeof(SharedTensorRecord)) % 48 == 0, "记录字节数 48 对齐");
        check(v.size() == (v.size() * sizeof(SharedTensorRecord) / 16) / 3, "÷3 记录数语义");
        check(v[1].size == 0x1234 && v[1].flags == 0x11, "记录字段 (G4PoolDesc 面) 读写");
    }

    // ---- 字段访问面 ----
    {
        RuntimeAllocator ra;
        check(ra.graph() == nullptr && ra.mode() == 0 && !ra.f_14(), "缺省: graph=null mode=0 f_14=0");
        check(ra.tcm_pool_base() == nullptr && ra.tcm_pool_size() == 0, "缺省: tcm pool 空");
        check(ra.largest_memory_alloc_size() == 0, "缺省: largest=0");
        check(ra.shared_spillfill_base() == 0, "缺省: spillfill_base=0");
        int dummy = 0;
        ra.set_tcm_pool(&dummy, 0x2000);
        check(ra.tcm_pool_base() == &dummy && ra.tcm_pool_size() == 0x2000, "set_tcm_pool 往返");
        ra.set_largest_memory_alloc_size(0x100000);
        check(ra.largest_memory_alloc_size() == 0x100000, "largest 往返");
        ra.set_shared_spillfill_base(0xdead0000);
        check(ra.shared_spillfill_base() == 0xdead0000, "spillfill_base 往返");
        ra.set_f_14(true);
        check(ra.f_14(), "f_14 往返");
        ra.set_slot_size(0, 0x10000);
        ra.set_slot_size(1, 0x20000);
        ra.set_slot_size(2, 0x30000);
        check(ra.slot_size(0) == 0x10000 && ra.slot_size(1) == 0x20000 && ra.slot_size(2) == 0x30000,
              "三槽读写 (+0xa8/+0xac/+0xb0)");
    }

    // ---- §1.5② 错误模型: 越界抛 std::out_of_range("vector") ----
    {
        RuntimeAllocator ra;
        bool threw_get = false, threw_set = false;
        try {
            (void)ra.slot_size(3);
        } catch (const std::out_of_range &e) {
            threw_get = true;
            check(std::string(e.what()) == "vector", "out_of_range what() == \"vector\"");
        }
        try {
            ra.set_slot_size(99, 1);
        } catch (const std::out_of_range &e) {
            threw_set = true;
        }
        check(threw_get, "slot_size 越界抛 out_of_range (§1.5②)");
        check(threw_set, "set_slot_size 越界抛 out_of_range (§1.5②)");
    }

    // ---- §1.2 make_allocator 分派路径 ----
    {
        hnnx::Graph g{};

        // fancy=false → RuntimeAllocator 本体
        RuntimeAllocator *p0 = RuntimeAllocator::make_allocator(g, false);
        check(p0 != nullptr, "make_allocator(fancy=false) 非空");
        check(typeid(*p0) == typeid(RuntimeAllocator), "fancy=false 动态类型 = RuntimeAllocator");
        check(dynamic_cast<FancyAllocator *>(p0) == nullptr, "fancy=false 非 FancyAllocator");
        check(p0->mode() == RuntimeAllocator::MODE_ALLOC_VIRTUAL, "Mode 恒 0 (AllocVirtual, §1.2)");
        check(p0->graph() == &g, "graph 指针写入");
        delete p0; // 经基指针虚析构删除

        // fancy=true → FancyAllocator
        RuntimeAllocator *p1 = RuntimeAllocator::make_allocator(g, true);
        check(p1 != nullptr, "make_allocator(fancy=true) 非空");
        check(dynamic_cast<FancyAllocator *>(p1) != nullptr, "fancy=true 动态类型 = FancyAllocator");
        check(typeid(*p1) == typeid(FancyAllocator), "fancy=true typeid = FancyAllocator");
        check(p1->mode() == RuntimeAllocator::MODE_ALLOC_VIRTUAL, "fancy 分支 Mode 亦 0 (§1.2)");
        check(p1->graph() == &g, "fancy 分支 graph 指针写入");
        delete p1;
    }

    // ---- §1.2 make_allocator_at 槽位语义 ----
    {
        hnnx::Graph g{};
        RuntimeAllocator *slot = nullptr;

        // 空槽 → 写入新实例, 返回槽本身 ("写入 *(Allocator**)this = p 并返回 this")
        RuntimeAllocator **r = RuntimeAllocator::make_allocator_at(&slot, g, false);
        check(r == &slot, "make_allocator_at 返回槽指针");
        check(slot != nullptr && typeid(*slot) == typeid(RuntimeAllocator), "槽写入 RuntimeAllocator");

        // 旧占用者经虚析构删除 (派生 dtor 计数观察 deleting dtor 路径)
        delete slot;
        slot = new TrackingAlloc();
        int before = TrackingAlloc::dtor_count;
        RuntimeAllocator::make_allocator_at(&slot, g, true);
        check(TrackingAlloc::dtor_count == before + 1, "旧占用者经虚析构删除 (deleting dtor)");
        check(dynamic_cast<FancyAllocator *>(slot) != nullptr, "槽位替换为 FancyAllocator");
        delete slot;
        slot = nullptr;

        // nullptr 槽 → 安全返回 nullptr
        check(RuntimeAllocator::make_allocator_at(nullptr, g, false) == nullptr,
              "make_allocator_at(nullptr) 返回 nullptr");
    }

    // ---- §1.1 层级: FancyAllocator is-a RuntimeAllocator + 行为保持锚点 ----
    {
        FancyAllocator fa; // 缺省构造链不变 (VtcmCacheInstance 成员/旧测试同路径)
        check(dynamic_cast<RuntimeAllocator *>(&fa) != nullptr, "FancyAllocator is-a RuntimeAllocator");
        check(fa.graph() == nullptr && fa.mode() == 0, "FancyAllocator 缺省基类状态");
        fa.set_mode(2); // 旧 API (int) — 写基类 mode_ 字段
        check(fa.mode() == 2, "FancyAllocator::set_mode(int) 写基类 mode_");
        fa.set_mode(0);

        // allocate_with_lifetime 行为不变 (与 test_vtcm_reuse 同语义锚点):
        // 两组不同生命周期 → 顺序 bump, 跨组无复用
        std::vector<FancyAllocator::AllocRequest> reqs = {
            {1, 1024, 0, 2}, // op_id, size, life_begin, life_end
            {2, 1024, 1, 1},
        };
        auto res = fa.allocate_with_lifetime(reqs, 0x100000);
        check(res.size() == 2, "行为锚点: allocate_with_lifetime 结果数");
        // life_end 降序: op1(le=2) 先放 → offset 0; op2(le=1) 次组 → offset 1024
        check(res[1].offset == 0 && res[2].offset == 1024, "行为锚点: 顺序 bump 偏移不变");
        check(fa.total_vtcm_used() == 2048 && fa.vtcm_saved_by_reuse() == 0,
              "行为锚点: 跨组无复用 (RuntimeAllocator 算法属 P5+)");
    }

    // VtcmCacheInstance 包装面不变 (含内嵌 FancyAllocator 新基类子对象)
    {
        hnnx::VtcmCacheInstance vtcm(0, 0x400000);
        check(vtcm.usable_size() == 3145728, "VtcmCacheInstance::usable_size 不变 (4MB×0.75)");
        check(dynamic_cast<RuntimeAllocator *>(&vtcm.allocator()) != nullptr,
              "VtcmCacheInstance::allocator() is-a RuntimeAllocator");
    }

    std::cout << "passed=" << tests_passed << " failed=" << tests_failed << "\n";
    return tests_failed == 0 ? 0 : 1;
}
