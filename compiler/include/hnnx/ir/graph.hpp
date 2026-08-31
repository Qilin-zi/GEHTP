#pragma once
// ============================================================================
// Graph 类字节级重实现 —— 第一阶段（接口 + 布局骨架 + 已验证方法体）
// libHtpPrepare.so 2.48.40.260702, x86_64-linux-clang, stripped
//
// 证据链（全部字节级，无推断；未知区段以精确边界的 opaque 数组标注）:
//   * _ZTV5Graph @0x5ebdc58 (0x1E8 字节, 终点 0x5ebde40 = dcrate 异常 typeinfo):
//     59 个虚槽（D1/D0 + 57 个虚函数），逐槽经 .rela.dyn 解析:
//     36 个 GLOB_DAT(导出符号, 名字/签名全部可读) + 23 个 RELATIVE(内部函数)。
//   * _ZTI5Graph @0x5ebdf60: __si_class_type_info → 单继承;
//     父类 typeinfo @0x5ebe6a8 = "14HexagonNNGraph"（无命名空间, 抽象接口,
//     任何头文件中均无声明 —— 接口仅存在于 vtable 槽序中）。
//   * sizeof(Graph) == 0x6C00, align 128 —— hexagon_nn_deserialize_graph
//     @0xd3d409: `mov $0x6c00,%edi; mov $0x80,%esi; call operator new(align_val_t)`
//     后紧接 Graph 反序列化构造调用 (0xd3d478)。
//   * 源文件名（日志字符串证据）: hexagon_nn_graph.cc @0x39ab41d。
//   * 主构造 C2(HexagonNNEnv&,unsigned,alloc_sel,optionpair const*,unsigned)
//     @0xd22c60 (3668B) 与析构 D1 @0xd263d0 (2830B) 逐指令解码 → 字段图。
//   * C API 交叉验证: hexagon_nn_get_id_from_graph @0xcdafdf `call *0x10(%rax)`
//     → 槽 +0x10 = 图 ID getter (函数体 @0xd352c0 读 +0x45d8)。
//   * 构造后状态检查: 0xd3d4af `call *0x60(%rax)` = get_status, 结果与 0x6b 比较。
//
// ---- vptr 相对槽位表 (59 槽) --------------------------------------------
//  +0x00 ~Graph D1                    +0x08 ~Graph D0
//  +0x10 get_graph_id        [内部@0xd352c0] = (u32)*(this+0x45d8)
//  +0x18 append_node(string const&,unsigned,InputDef const*,size_t,
//                    OutputDef const*,size_t,unsigned long const*)
//  +0x20 append_const_node(unsigned,OutputDef const&,unsigned char const*,size_t)
//  +0x28 set_node_ids(unsigned,unsigned,unsigned)
//  +0x30 prepare(HexagonNNEnv&)   @0xd34ff0: log"prepare() is called in execute
//        mode"@0x39a928b → 返回 -1 (本 .so 为 execute 模式构建, prepare 被移除)
//  +0x38 execute(HexagonNNEnv&)   @0xd2c010 (527B)
//  +0x40 get_hextimate_output(HextimateOutput*) @0xd34e90 = 0
//  +0x48 set_file_io(shared_ptr<basic_iostream>,HexagonNNFileType)
//        @0xd34ea0: 两种 HexagonNNFileType 均为 log→-1 (execute 模式不支持)
//  +0x50 trigger_abort()          @0xd34e30: lock orl $4,0x30(*(this+0x6810))
//  +0x58 trigger_cached_acquire_cancel() @0xd34e80 → (this+0x6810) 尾调用 0xddca90
//  +0x60 get_status()             @0xd2f5b0 (88B)
//  +0x68 [内部@0xd2f990] ((*(this+0x5370))-(*(this+0x5368)))>>4   16B 元素计数
//  +0x70 [内部@0xd2f2b0] ((*(this+0x5358))-(*(this+0x5350)))>>4 - (u32)*(this+0x6508)
//  +0x78 get_defs_for_inputs(vector<OutputDef>&)   @0xd2f6a0
//  +0x80 get_defs_for_outputs(vector<OutputDef>&)  @0xd2f9b0
//  +0x88 is_opname_registered(string const&)
//  +0x90 fixup_axis_const_node(unsigned,unsigned,unsigned)
//  +0x98 [内部@0xd45af0] log"%s:27::ERROR:method removed\n" → 返回 7
//  +0xa0 fixup_node_shape(InputDef,unsigned char const*,unsigned)
//  +0xa8 [内部@0xd352d0] 返回 *(this+0x6560)
//  +0xb0 [内部@0xd352e0] 返回 *(this+0x6568)
//  +0xb8 [内部@0xd45ad0] 返回 0xfffffff9 (-7, 已移除方法)
//  +0xc0 [内部@0xd45ae0] 返回 0xfffffff9 (-7, 已移除方法)
//  +0xc8 set_option_item(HexagonNNEnv&,char const*,string const&)
//  +0xd0 get_option_item(char const*,string&) const
//  +0xd8 set_shared_spillfill(far_vm_ptr_tmpl_t<unsigned long>,unsigned long)
//  +0xe0 set_mc_shared_buffer(far_vm_ptr_tmpl_t<ulong> const&,unsigned long)
//  +0xe8 set_shared_doorbells(far_vm_ptr_tmpl_t<ulong> const&,unsigned)
//  +0xf0 set_shared_tensors(far_vm_ptr_tmpl_t<ulong> const&,unsigned)
//  +0xf8 set_graph_name(string const&)   @0xd28110: addq $0x45e0,%rdi;
//          jmp basic_string::operator=(string const&)@plt —— 按引用赋值
//  +0x100 get_graph_name()               @0xd28120: leaq 0x45e0(%rdi),%rax; ret
//          —— 返回 +0x45e0 字符串的地址 (std::string const*)
//  +0x108 set_hmx_implicit_pwr_ctrl(bool) @0xd34de0 (返回旧值)
//  +0x110 [内部@0xd352f0] if((p=*(this+0x6810))) *(u8*)(p+0x3c)=(u8)b
//  +0x118 [内部@0xd35310] return (p=*(this+0x6810)) && *(u8*)(p+0x3c)!=0
//  +0x120 explicit_cached_release(HexagonNNEnv&)
//  +0x128 check_yield_cached_release(HexagonNNEnv&)
//  +0x130 [内部@0xd35330] (this+0x6b00) 上尾调用 0xd3c620
//  +0x138 [内部@0xd35340] 返回 (u32)*(this+0x6b3c)
//  +0x140 [内部@0xd35350] (this+0x6b00) 上尾调用 0xd3c710
//  +0x148 [内部@0xd35360] call 0xd3c7f0(this+0x6b00); 结果==0 时再调
//          Graph::check_valid_pmu_sampler_event —— pmu 系
//  +0x150 [内部@0xd35390] (this+0x6b00) 上尾调用 0xd3ccb0
//  +0x158 optrace_size() const
//  +0x160 optrace_get(traceevent*,unsigned)
//  +0x168 get_exec_tid() const
//  +0x170 get_op_id_and_addr(void const*)
//  +0x178 get_num_opinfos() const
//  +0x180 get_opinfo_data(opinfo*,unsigned)
//  +0x188 [内部@0xd353a0] 返回 this+0x140 (arena1)
//  +0x190 [内部@0xd353b0] 返回 this+0x168 (arena2)
//  +0x198 pmu_sampler() const
//  +0x1a0 [内部@0xd353c0] 返回 this+0x190 (arena3)
//  +0x1a8 [内部@0xd353d0] p=*(this+0x6b90);
//          返回 {rax=p, rdx= p && *(void**)p==this} (子图回指检测)
//  +0x1b0 get_input_output_tensor(int,bool)
//  +0x1b8 [内部@0xd34d10] xor eax,eax;ret → 0
//  +0x1c0 get_full_allocator() const
//  +0x1c8 [内部@0xd353f0] no-op ret
//  +0x1d0 [内部@0xd35400] no-op ret
//
// ---- 字段布局（构造 C2/析构 D1 钉死; 其余为精确边界 opaque 区段）---------
//  +0x0000 vptr
//  +0x0008 "GSTART" 魔数 (movl 'GSTA'@+0x8; movl 'RT\0'@+0xb) + 1 字节
//  +0x0010 u32 = 0x1010
//  +0x0014..0xc7  opaque(0xb4) —— 构造清零
//  +0x00c8 void* —— new(0x1E0)+memset0+本地初始化(0xd360b0)
//  +0x00d0 内嵌 hnnx::Crate (D1: Crate::clear(Graph*)+~Crate;
//          crate+0x40 == Graph+0x110 计数器 ←→ M28 add_record_slot 结论互证)
//  +0x0118 void* = *(0xc8)+0x198
//  +0x0140 Arena1 {new(0x4800) base; cursor; end; -1; bool} 0x28B —— 槽+0x188
//  +0x0168 Arena2 {new(0x3400) ...} —— 槽+0x190
//  +0x0190 Arena3 {new(0x10800) ...} —— 槽+0x1a0
//  +0x01d8 hnnx::Allocator* —— D1 中 tracked_free 一律经此 (0xd2646d 起)
//  +0x0200 内嵌 hnnx::DMA_Manager (0x4380B; D1: wait_all + reset)
//  +0x4580 u32 = 1
//  +0x45d8 u32 graph_id ← 主构造第 2 参 (edx)
//  +0x45dc u32 = 1
//  +0x45e0 24B 字符串: D1 的 c_str 模式 = testb $1,@+0x45e0;
//          长串数据指针 @+0x45f0, 短串数据 @+0x45e1 (构造自空串 0x4628a0e)
//  +0x45f8 u32 = 0
//  +0x4600..546f opaque —— 含 +0x4e00/-1、+0x4e10/-1、+0x4e14/(u16)-1、
//          0x5350/58 与 0x5368/70 两对 16B 元素 vector(槽+0x68/+0x70 计数源)、
//          0x5398..54d0 内侵入链表数组(0x18 步长自引用)
//  +0x5cc8/+0x5ccc/+0x6338 vtcm 选项 int (get_vtcm_option_size @0xd34fb0)
//  +0x6508 u32 —— 槽 +0x70 的减数
//  +0x6560/+0x6568 void* —— 槽 +0xa8/+0xb0
//  +0x6570..0x6807 不透明 (无锚点)
//  +0x6808 u32 vtcm_size (get_vtcm_size @0xd34fe0)
//  +0x6810 void* 状态对象 —— +0x30 原子标志(request_yield|=2, abort|=4), +0x3c bool
//  +0x68b0 void* 后台工作线程对象 (vptr; +0x8 int —— request_bkgrnd_yield 读)
//  +0x68c8 bool hmx_implicit_pwr_ctrl
//  +0x68d0..6bff 尾区 —— 0x6948 子对象(D1 本地析构 0xcf37c0)、0x6b00 pmu
//          子对象、0x6b3c u32、0x6b88 互斥体、0x6b90 子图指针、0x6ba8 止
//  总计 0x6C00。
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "op_def.hpp" // OutputDef, DType

class Graph;
class HexagonNNEnv;
class MultiVxuInfo;
struct traceevent;
struct opinfo;
struct HextimateOutput;
struct optionpair;
struct hexagon_nn_wide_iovec_t;
enum HexagonNNFileType : int;

namespace fa {
struct FancyAllocator; // M32: get_full_allocator (+0x1c0 槽) 返回引用
} // namespace fa

namespace hnnx {
class Crate;
// M32: graph_crate(Graph&) @plt 0x6f2f50 —— 返回 Graph 内嵌 crate (Graph+0xd0,
//   0x48 字节; graph.hpp 布局注 + GraphOrdering 互证)。调用点
//   VariadicOpBase::assign_input_pointers @0x139fc81 等。
Crate *graph_crate(Graph &); // @0x6f2f50
} // namespace hnnx

namespace hnnx {
template <typename T> struct far_vm_ptr_tmpl_t; // 17far_vm_ptr_tmpl_t (set_shared_* 参数)
class Deserializer;
class Deserz;
class Allocator;
class DMA_Manager;
class Crate;
class Event;
class Executable;
class OpIoPtrs; // 定义于 op_io_ptrs.hpp (Graph::allocator +0x1d8 直读的友元)
class TypicalOpUtil; // 定义于 typical_op_io.hpp (do_allocate 直读 +0x1d8)
template <unsigned, unsigned> class TypicalOpIoBase; // allocate 内联 do_allocate
class VariadicOpBase; // 定义于 variadic_op_base.hpp (allocate @0x139fa60 直读)
struct OpExtraInfo; // 定义于 op.hpp (SDK op_extra_info.h: sizeof 24)
struct OpExtraAttrib; // OpExtraInfo 的 prepare 期扩展子类 (位域首单元 @+0x18)
// 注: ListType 是 enum (hnnx::ListType {MainList,VecList,MtxList,EltList}),
//     由 executable.hpp 给出定义, 不可在此前置为 class。
} // namespace hnnx

// ============================================================================
// HexagonNNGraph —— 抽象接口基类 (typeinfo @0x5ebe6a8 "14HexagonNNGraph",
// 无基类; 仅以 59 槽 vtable 槽序存在 —— 槽内名字不可知的 23 个虚函数以
// slot 前缀命名, 注释给出已验证语义。)
// ============================================================================
class HexagonNNGraph {
  public:
    virtual ~HexagonNNGraph() {} // 槽 +0x00(D1) / +0x08(D0)

    virtual unsigned get_graph_id() const = 0;                                    // +0x10
    virtual int append_node(std::string const &, unsigned, InputDef const *, size_t,
                            OutputDef const *, size_t, unsigned long const *) = 0; // +0x18
    virtual int append_const_node(unsigned, OutputDef const &, unsigned char const *,
                                  size_t) = 0;                                    // +0x20
    virtual int set_node_ids(unsigned, unsigned, unsigned) = 0;                   // +0x28
    virtual int prepare(HexagonNNEnv &) = 0;                                      // +0x30
    virtual int execute(HexagonNNEnv &) = 0;                                      // +0x38
    virtual int get_hextimate_output(HextimateOutput &) = 0;                      // +0x40
    virtual int set_file_io(std::shared_ptr<std::basic_iostream<char>>,
                            HexagonNNFileType) = 0;                               // +0x48
    virtual void trigger_abort() = 0;                                             // +0x50
    virtual void trigger_cached_acquire_cancel() = 0;                             // +0x58
    virtual int get_status() = 0;                                                 // +0x60
    virtual size_t slot68_num_inputs() const = 0;                                 // +0x68
    virtual size_t slot70_num_outputs() const = 0;                                // +0x70
    virtual void get_defs_for_inputs(std::vector<OutputDef> &) = 0;               // +0x78
    virtual void get_defs_for_outputs(std::vector<OutputDef> &) = 0;              // +0x80
    virtual bool is_opname_registered(std::string const &) = 0;                   // +0x88
    virtual int fixup_axis_const_node(unsigned, unsigned, unsigned) = 0;          // +0x90
    virtual int slot98_removed() = 0;                                             // +0x98
    virtual int fixup_node_shape(InputDef, unsigned char const *, unsigned) = 0; // +0xa0
    virtual void *slot_a8_get_6560() = 0;                                         // +0xa8
    virtual void *slot_b0_get_6568() = 0;                                         // +0xb0
    virtual int slot_b8_removed() = 0;                                            // +0xb8
    virtual int slot_c0_removed() = 0;                                            // +0xc0
    virtual int set_option_item(HexagonNNEnv &, char const *, std::string const &) = 0; // +0xc8
    virtual int get_option_item(char const *, std::string &) const = 0;           // +0xd0
    virtual int set_shared_spillfill(hnnx::far_vm_ptr_tmpl_t<unsigned long>, unsigned long) = 0; // +0xd8
    virtual int set_mc_shared_buffer(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &,
                                   unsigned long) = 0;                                // +0xe0
    virtual int set_shared_doorbells(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &, unsigned) = 0; // +0xe8
    virtual int set_shared_tensors(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &, unsigned) = 0; // +0xf0
    virtual void set_graph_name(std::string const &) = 0;                         // +0xf8
    virtual std::string const *get_graph_name() = 0;                              // +0x100
    virtual bool set_hmx_implicit_pwr_ctrl(bool) = 0;                             // +0x108
    virtual void slot110_set_6810_3c(bool) = 0;                                   // +0x110
    virtual bool slot118_get_6810_3c() const = 0;                                 // +0x118
    virtual int explicit_cached_release(HexagonNNEnv &) = 0;                      // +0x120
    virtual int check_yield_cached_release(HexagonNNEnv &) = 0;                   // +0x128
    virtual int slot130_pmu_forward() = 0;                                        // +0x130
    virtual unsigned slot138_get_6b3c() const = 0;                                // +0x138
    virtual int slot140_pmu_forward() = 0;                                        // +0x140
    virtual int slot148_pmu_check() = 0;                                          // +0x148
    virtual int slot150_pmu_forward() = 0;                                        // +0x150
    virtual size_t optrace_size() const = 0;                                      // +0x158
    virtual void optrace_get(traceevent *, unsigned) = 0;                         // +0x160
    virtual unsigned get_exec_tid() const = 0;                                    // +0x168
    virtual unsigned long get_op_id_and_addr(void const *) = 0;                   // +0x170
    virtual size_t get_num_opinfos() const = 0;                                   // +0x178
    virtual void get_opinfo_data(opinfo *, unsigned) = 0;                         // +0x180
    virtual void *slot188_arena1() = 0;                                           // +0x188
    virtual void *slot190_arena2() = 0;                                           // +0x190
    virtual void const *pmu_sampler() const = 0;                                  // +0x198
    virtual void *slot1a0_arena3() = 0;                                           // +0x1a0
    virtual std::pair<void *, bool> slot1a8_child_check() const = 0;              // +0x1a8
    virtual void *get_input_output_tensor(int, bool) = 0;                         // +0x1b0
    virtual void *slot1b8_zero() = 0;                                             // +0x1b8
    // M32 修正返回类型: GraphPrepare::get_full_allocator @0xf849e0 返回
    // dynamic_cast<fa::FancyAllocator&>(…) (bad_cast 失败), Graph 基类同槽
    // @0xd35140 恒抛 runtime_error("wide crouton not supported") —— 均为引用返回。
    virtual fa::FancyAllocator &get_full_allocator() const = 0;                   // +0x1c0
    virtual void slot1c8_noop() = 0;                                              // +0x1c8
    virtual void slot1d0_noop() = 0;                                              // +0x1d0
};
static_assert(sizeof(HexagonNNGraph) == 8);

// ============================================================================
// Graph —— 具体类 (全局作用域, _ZTS5Graph = "5Graph")
// ============================================================================
class Graph : public HexagonNNGraph {
  public:
    // 嵌套枚举 (mangling: NS_9alloc_selE / NS_18deser_runlist_modeE)。
    // alloc_sel 为 32 位: 主构造第 3 参经 ecx 传递 (SysV: 枚举 → 32 位寄存器)。
    // deser_runlist_mode 的值 2 被模板实例 deserialize_runlist<2> 引用。
    enum alloc_sel : int { /* 枚举项未在 .so 中露出 */ };
    enum deser_runlist_mode : int { /* 值 2 存在; 其余未露出 */ };

    // 三段 arena: new(size)+memset0 → {base, cursor=base, end=base+size, -1, false}
    // (构造 @0xd22d78-0xd22e83 三次分配: 0x4800 / 0x3400 / 0x10800)
    struct arena_hdr {
        void *base;      // +0x00
        void *cursor;    // +0x08
        void *end;       // +0x10
        int64_t counter; // +0x18, 初值 -1
        bool flag;       // +0x20
    };
    static_assert(sizeof(arena_hdr) == 0x28);

    // ---- 已验证方法体（逐指令对应 .so; 大方法为声明, 实现于库侧）----

    // 槽 +0x10 ↔ @0xd352c0: mov 0x45d8(%rdi),%eax; ret
    unsigned get_graph_id() const override { return graph_id; }

    // 槽 +0x40 ↔ @0xd34e90: xor eax,eax; ret (Hextimate 未启用)
    int get_hextimate_output(HextimateOutput &) override { return 0; }

    // 槽 +0x50 ↔ @0xd34e30: lock orl $0x4, 0x30(p=*(this+0x6810));
    //   随后 vcall+0x10(get_status) 仅为日志 (GetLogPriorityLevel>=3) —— 省略。
    void trigger_abort() override
    {
        __sync_or_and_fetch(reinterpret_cast<unsigned *>(static_cast<unsigned char *>(status_obj) + 0x30), 4u);
    }

    // 槽 +0x68 ↔ @0xd2f990: ((0x5370)-(0x5368)) 算术右移 4 —— 16B 元素 vector 计数
    size_t slot68_num_inputs() const override
    {
        return (reinterpret_cast<uintptr_t>(v5370) - reinterpret_cast<uintptr_t>(v5368)) >> 4;
    }
    // 槽 +0x70 ↔ @0xd2f2b0: ((0x5358)-(0x5350))>>4 再减 (u32)0x6508 (带符号)
    size_t slot70_num_outputs() const override
    {
        return ((reinterpret_cast<uintptr_t>(v5358) - reinterpret_cast<uintptr_t>(v5350)) >> 4) - v6508;
    }

    // 槽 +0x98 ↔ @0xd45af0: qnndsp_log(0,"%s:27::ERROR:method removed\n",
    //   "hexagon_nn_graph.cc") → 7 (日志走 qnndsp, 此处以 stderr 表达同一事实)
    int slot98_removed() override
    {
        fprintf(stderr, "%s:27::ERROR:method removed\n", "hexagon_nn_graph.cc");
        return 7;
    }

    // 槽 +0xa8/+0xb0 ↔ @0xd352d0/@0xd352e0
    void *slot_a8_get_6560() override { return v6560; }
    void *slot_b0_get_6568() override { return v6568; }
    // 槽 +0xb8/+0xc0 ↔ @0xd45ad0/@0xd45ae0: mov $0xfffffff9,%eax; ret
    int slot_b8_removed() override { return -7; }
    int slot_c0_removed() override { return -7; }

    // 槽 +0x108 ↔ @0xd34de0: 读旧值(bl) → 写新值 → 返回旧值
    bool set_hmx_implicit_pwr_ctrl(bool b) override
    {
        const bool old = hmx_implicit_pwr_ctrl;
        hmx_implicit_pwr_ctrl = b;
        return old;
    }

    // 槽 +0x110 ↔ @0xd352f0: p=*(this+0x6810); p 非空时 *(u8*)(p+0x3c)=(u8)b
    void slot110_set_6810_3c(bool b) override
    {
        if (unsigned char *const p = static_cast<unsigned char *>(status_obj)) p[0x3c] = (unsigned char)b;
    }
    // 槽 +0x118 ↔ @0xd35310: p=*(this+0x6810); return p && p[0x3c]!=0
    bool slot118_get_6810_3c() const override
    {
        const unsigned char *const p = static_cast<const unsigned char *>(status_obj);
        return p && p[0x3c] != 0;
    }

    // 槽 +0x138 ↔ @0xd35340: mov 0x6b3c(%rdi),%eax; ret
    unsigned slot138_get_6b3c() const override { return v6b3c_ref(); }

    // 槽 +0x188/+0x190/+0x1a0 ↔ @0xd353a0/@0xd353b0/@0xd353c0
    void *slot188_arena1() override { return &arena1; }
    void *slot190_arena2() override { return &arena2; }
    void *slot1a0_arena3() override { return &arena3; }

    // 槽 +0x1a8 ↔ @0xd353d0: p=*(this+0x6b90);
    //   rax=p(空则 0); rdx=(p && *(void**)p==this)
    std::pair<void *, bool> slot1a8_child_check() const override
    {
        void *const p = child_graph_ref();
        return {p, p && *static_cast<void *const *>(p) == static_cast<void *>(const_cast<Graph *>(this))};
    }

    // 槽 +0x1b8 ↔ @0xd34d10: xor eax,eax; ret
    void *slot1b8_zero() override { return nullptr; }
    // 槽 +0x1c8/+0x1d0 ↔ @0xd353f0/@0xd35400: 单条 ret
    void slot1c8_noop() override {}
    void slot1d0_noop() override {}

    // ---- 非虚小方法（已验证）----

    // @0xd34d20: lock orl $0x2, 0x30(*(this+0x6810)); (日志省略)
    void request_yield()
    {
        __sync_or_and_fetch(reinterpret_cast<unsigned *>(static_cast<unsigned char *>(status_obj) + 0x30), 2u);
    }

    // @0xd34fe0: mov 0x6808(%rdi),%eax; ret
    unsigned get_vtcm_size() const { return vtcm_size; }

    // @0xd34fb0: a=*(int*)0x6338; a!=-1→a; b=*(int*)0x5cc8; b!=-1→b<<10;
    //            否则 *(int*)0x5ccc<<20
    unsigned get_vtcm_option_size() const
    {
        int v = vtcm_opt_6338;
        if (v != -1) return (unsigned)v;
        v = vtcm_opt_5cc8;
        if (v != -1) return (unsigned)v << 10;
        return (unsigned)vtcm_opt_5ccc << 20;
    }

    // 槽 +0xf8 ↔ @0xd28110: addq $0x45e0,%rdi; jmp operator=(string const&)@plt
    void set_graph_name(std::string const &s) override { graph_name_str = s; }
    // 槽 +0x100 ↔ @0xd28120: leaq 0x45e0(%rdi),%rax; ret
    std::string const *get_graph_name() override { return &graph_name_str; }

    // 析构 (2830B) / 主构造 (3668B) / 反序列化构造: 实现于库侧 (声明)
    ~Graph() override;
    Graph(HexagonNNEnv &, unsigned graph_id_in, alloc_sel, optionpair const *, unsigned);

    // ---- 其余导出方法: 签名逐字取自符号 mangling (audit_verify/graph_syms.txt
    //     共 143 条), 大函数体属后续阶段; 此处声明接口主线 ----
    int prepare(HexagonNNEnv &) override;        // @0xd34ff0: log → -1
    int execute(HexagonNNEnv &) override;        // @0xd2c010 (527B)
    int get_status() override;                   // @0xd2f5b0
    void get_defs_for_inputs(std::vector<OutputDef> &) override;  // @0xd2f6a0
    void get_defs_for_outputs(std::vector<OutputDef> &) override; // @0xd2f9b0
    int append_node(std::string const &, unsigned, InputDef const *, size_t, OutputDef const *, size_t,
                    unsigned long const *) override;               // @0xd35020
    int append_const_node(unsigned, OutputDef const &, unsigned char const *, size_t) override;
    int set_node_ids(unsigned, unsigned, unsigned) override;
    bool is_opname_registered(std::string const &) override;
    int fixup_axis_const_node(unsigned, unsigned, unsigned) override; // @0xd350b0
    int fixup_node_shape(InputDef, unsigned char const *, unsigned) override;
    int set_option_item(HexagonNNEnv &, char const *, std::string const &) override;
    int get_option_item(char const *, std::string &) const override;
    int set_shared_spillfill(hnnx::far_vm_ptr_tmpl_t<unsigned long>, unsigned long) override;
    int set_mc_shared_buffer(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &, unsigned long) override;
    int set_shared_doorbells(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &, unsigned) override; // @0xd28130
    int set_shared_tensors(hnnx::far_vm_ptr_tmpl_t<unsigned long> const &, unsigned) override;
    void trigger_cached_acquire_cancel() override;                      // @0xd34e80
    int explicit_cached_release(HexagonNNEnv &) override;
    int check_yield_cached_release(HexagonNNEnv &) override;
    size_t optrace_size() const override;
    void optrace_get(traceevent *, unsigned) override;
    unsigned get_exec_tid() const override;
    unsigned long get_op_id_and_addr(void const *) override; // @0xd2a0f0
    size_t get_num_opinfos() const override;                 // @0xd2a240
    void get_opinfo_data(opinfo *, unsigned) override;
    void const *pmu_sampler() const override;                // @0xd2bf70
    void *get_input_output_tensor(int, bool) override;       // @0xd2f210
    // @0xd35140: 恒抛 std::runtime_error("wide crouton not supported") (16B 异常, 串 @0x39a942a)
    fa::FancyAllocator &get_full_allocator() const override;
    int slot130_pmu_forward() override;
    int slot140_pmu_forward() override;
    int slot148_pmu_check() override;
    int slot150_pmu_forward() override;
    int set_file_io(std::shared_ptr<std::basic_iostream<char>>, HexagonNNFileType) override;

    // ---- 布局: 已验证字段 + 精确边界 opaque 区段 (总计 0x6C00) ----
  private:
    // M32: OpIoPtrs::allocator()/get_output_for_cloned_op 直读 +0x1d8 字段
    //     (@0xf85540: mov 0x1d8(%rax); @0xf8544b 同) —— 与 Tensor 侧
    //     `friend class ::Op` (受保护虚槽直调) 同源的访问权证据。
    //     TypicalOpUtil::do_allocate @0x1399170 与
    //     TypicalOpIoBase<N,N>::allocate @0x1399bc5 同样直读 (mov 0x1d8(%rsi))。
    friend class hnnx::OpIoPtrs;
    friend class hnnx::TypicalOpUtil;
    template <unsigned, unsigned> friend class hnnx::TypicalOpIoBase; // allocate 内联 do_allocate
    friend class hnnx::VariadicOpBase; // allocate @0x139fa60 同样直读 +0x1d8

    char gstart_magic[8];        // +0x008 "GSTART\0"+pad (movl×2 立即数)
    unsigned ver_1010;           // +0x010 = 0x1010
    unsigned char pad_14[0xb4];  // +0x014..0xc7 (构造清零)
    void *obj_0x1E0;             // +0x0c8
    unsigned char crate_raw[0x48]; // +0x0d0 内嵌 hnnx::Crate (0xd0..0x118)
    void *obj_0x1E0_at_198;      // +0x118
    unsigned char pad_120[0x20]; // +0x120..0x13f
    arena_hdr arena1;            // +0x140 (0x28)
    arena_hdr arena2;            // +0x168 (0x28)
    arena_hdr arena3;            // +0x190 (0x28)
    unsigned v1b8;               // +0x1b8 = 0
    unsigned char pad_1bc[0x1c]; // +0x1bc..0x1d7
    hnnx::Allocator *allocator;  // +0x1d8
    unsigned char pad_1e0[0x20]; // +0x1e0..0x1ff
    unsigned char dma_manager[0x4380]; // +0x200 内嵌 hnnx::DMA_Manager
    unsigned v4580;              // +0x4580 = 1
    unsigned char pad_4584[0x54]; // +0x4584..0x45d7
    unsigned graph_id;           // +0x45d8
    unsigned v45dc;              // +0x45dc = 1
    std::string graph_name_str;  // +0x45e0 (24B; set/get_graph_name 证据见上)
    unsigned v45f8;              // +0x45f8 = 0
    unsigned char pad_45fc[4];   // +0x45fc..0x45ff
    // +0x4600..0x534f: 内嵌子对象(0xd22ff1 构造)+杂项+runlist 区。
    //   已知锚点: 0x4e00/-1, 0x4e08/0, 0x4e10/-1, 0x4e14/(u16)-1, 0x4e18 子对象,
    //   0x52a0/-1 (均在下方 pad 内, 未单列)
    unsigned char pad_4600[0x5350 - 0x4600]; // +0x4600..0x534f
    void *v5350;                 // +0x5350 16B 元素 vector begin (槽 +0x70 减数源)
    void *v5358;                 // +0x5358 同上 end
    unsigned char pad_5360[8];   // +0x5360..0x5367
    void *v5368;                 // +0x5368 16B 元素 vector begin (槽 +0x68 计数源)
    void *v5370;                 // +0x5370 同上 end
    // +0x5378..0x5cc7: 0x5398..0x54d0 侵入链表数组(0x18 步长自引用),
    //   0x5da8 (D1 触及) —— 边界内不透明
    unsigned char pad_5378[0x5cc8 - 0x5378]; // +0x5378..0x5cc7
    int vtcm_opt_5cc8;           // +0x5cc8 (get_vtcm_option_size 第 2 优先级, KB)
    int vtcm_opt_5ccc;           // +0x5ccc (第 3 优先级, MB)
    unsigned char pad_5cd0[0x6338 - 0x5cd0];
    int vtcm_opt_6338;           // +0x6338
    unsigned char pad_633c[0x6508 - 0x633c];
    unsigned v6508;              // +0x6508
    unsigned char pad_650c[0x6560 - 0x650c];
    void *v6560;                 // +0x6560
    void *v6568;                 // +0x6568
    // +0x6570..0x6807: 不透明 (无已钉死锚点)
    unsigned char pad_6570[0x6808 - 0x6570];
    unsigned vtcm_size;          // +0x6808
    unsigned char pad_680c[4];   // +0x680c
    void *status_obj;            // +0x6810
    unsigned char pad_6818[0x68b0 - 0x6818];
    void *bkgrnd_worker;         // +0x68b0
    unsigned char pad_68b8[0x10];
    bool hmx_implicit_pwr_ctrl;  // +0x68c8
    // +0x68c9..0x6bff 尾区: 0x6948 子对象(本地析构 0xcf37c0), 0x6b00 pmu 子对象,
    //   0x6b3c u32, 0x6b88 互斥体(nn_mutex), 0x6b90 子图指针, 0x6ba8 止
    unsigned char tail_68c9[0x6c00 - 0x68c9];

    // pad 区内锚点偏移常量（供后续阶段/调试引用）
    static constexpr unsigned off_v4e00 = 0x4e00, off_v4e18 = 0x4e18;

    unsigned &v6b3c_ref() const
    {
        return *reinterpret_cast<unsigned *>(const_cast<unsigned char *>(tail_68c9) + (0x6b3c - 0x68c9));
    }
    void *const &child_graph_ref() const
    {
        return *reinterpret_cast<void *const *>(tail_68c9 + (0x6b90 - 0x68c9));
    }

    // ---- Op 侧经 PLT 调用的 Graph 非虚接口 (符号可读, 函数体属 Graph 未反演区段;
    //      Op 代码 (如构造/ id/ set_chkpts) 直接调用 → 公有成员) ----
  public:
    // _ZN5Graph14set_extra_infoEPK2OpRKN4hnnx11OpExtraInfoE @plt 0x6eb6a0
    //   (Op 构造 @0xd4e010: 写 vptr 后以栈上 {my_id, -1} 调用)
    void set_extra_info(Op const *, hnnx::OpExtraInfo const &);
    // _ZN5Graph14get_extra_infoEPK2Op @plt 0x6f3240 (非 const, set_chkpts 经此写入 [8])
    hnnx::OpExtraAttrib &get_extra_info(Op const *);
    // _ZNK5Graph14get_extra_infoEPK2Op @plt 0x6eed80 (id/get_*_store_type/get_serialize_flags/
    //   serialize_internal 均经此读取)
    hnnx::OpExtraAttrib const &get_extra_info(Op const *) const;
    // _ZNK5Graph15is_hmx_threadedEv @plt 0x6ebaa0 (get_op_store_type/get_serialize_flags 尾部调用)
    bool is_hmx_threaded() const;

  private:
    friend struct GraphLayoutProbe;

  public:
    static constexpr size_t total_size = 0x6C00;
};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
struct GraphLayoutProbe { // 经 friend 访问 Graph 私有布局做编译期校验
    static constexpr bool ok =
        sizeof(Graph) == 0x6C00 && offsetof(Graph, graph_id) == 0x45d8 &&
        offsetof(Graph, graph_name_str) == 0x45e0 && offsetof(Graph, arena1) == 0x140 &&
        offsetof(Graph, arena2) == 0x168 && offsetof(Graph, arena3) == 0x190 &&
        offsetof(Graph, allocator) == 0x1d8 && offsetof(Graph, v5350) == 0x5350 &&
        offsetof(Graph, v5358) == 0x5358 && offsetof(Graph, v5368) == 0x5368 &&
        offsetof(Graph, v5370) == 0x5370 && offsetof(Graph, vtcm_opt_5cc8) == 0x5cc8 &&
        offsetof(Graph, vtcm_opt_5ccc) == 0x5ccc && offsetof(Graph, vtcm_opt_6338) == 0x6338 &&
        offsetof(Graph, v6508) == 0x6508 && offsetof(Graph, v6560) == 0x6560 &&
        offsetof(Graph, v6568) == 0x6568 && offsetof(Graph, vtcm_size) == 0x6808 &&
        offsetof(Graph, status_obj) == 0x6810 && offsetof(Graph, bkgrnd_worker) == 0x68b0 &&
        offsetof(Graph, hmx_implicit_pwr_ctrl) == 0x68c8;
};
#pragma clang diagnostic pop
static_assert(GraphLayoutProbe::ok, "Graph 布局: sizeof/全部锚点偏移 (M29)");
