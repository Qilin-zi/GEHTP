#pragma once
// ============================================================================
// deserz.hpp — hnnx::DeSerError / DCrate / Deserz (M33 字节级)
//
// Deserz ctor @0xcfcda0 逐指令解码 + SDK deserializer.h / crate.h /
// deser_concurrent.h / dcrate_inlines.h 三角互证。sizeof(Deserz) = 0xd8。
//
//   ctor: Deserz(Deserializer *full_deser, char const *p, size_t n, Graph *g)
//   写入序 (即成员声明序; 基类 DeSerError 非多态 → 让位于 vptr, 落 +0x08):
//     +0x00 vptr                     ← GOT 0x623f388 (_ZTVN4hnnx6DeserzE) + 0x10
//     +0x08 errstr = 0               ← DeSerError 基
//     +0x10 current_op_deser_fn_p = 0
//     +0x18 allocator = 0            (fa::RuntimeAllocator*)
//     +0x20 d_crate{0,0,0}           (DCrate 0x18: nextp/limitp/cratep)
//     +0x38 op_deserializer_map      ← call 0xcfb120 初始化的 .data 静态表
//                                      (@0x62431d8) 之地址
//     +0x40 tensor_deserializer_map  ← call 0xcfb4a0 静态表 (@0x6243218)
//     +0x48 graph_ptr = arg4
//     +0x50 full_deser = arg1
//     +0x58 bufstart = arg2 p        +0x60 bufend = p+n
//     +0x68 bufp = p                 +0x70 buf_limit = p
//     +0x78 bytes_filled = 0
//     +0x80 op_extra_info: id=0, chkpts={-1,-1}, op_tag 不写 (委托 ctor 不触及)
//     +0x98 next_tensordef_index = 1 (u32); +0x9c format_version = 0 (int)
//     +0xa0 seg_fixup_state{}        (runlist_fixup_state 0x38, 全零)
//
// M32 镜像的调用点偏移全部对上: +0x18 allocator / +0x20+0x28 bump 游标
// (= DCrate nextp/limitp) / +0x30 crate (= cratep) / +0x50 共享上下文
// (= full_deser) / +0x68 读游标 (= bufp) / +0x70 读末尾 (= buf_limit) /
// +0x9c 压缩格式 (= format_version) / +0xa0 段修正子对象 (= seg_fixup_state)。
//
// Deserializer ctor @0xcfcf20 内联同体基类构造, 差异仅两处:
//   +0x50 存 this (自引用); 基类段完成后 vptr 换 _ZTVN4hnnx12DeserializerE
//   (GOT 0x623f328)。+0x70 buf_limit 基类段不写 (0xcfcf92 直跳 +0x78),
//   Deserializer 扩展段 0xcfd0b8 补写 p —— 两路径终值相同 (滑动窗口起点)。
// ============================================================================
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include "hnnx/ir/op_extra_info.hpp"

namespace fa {
struct RuntimeAllocator; // 全局 fa:: (mangling N2fa16RuntimeAllocator)
                          // 注: 本树 tensor_base.hpp 内另有 hnnx::fa 同名定义 (M35 统一)
} // namespace fa

// Tensor/Graph 为全局类 (mangling 证据: _ZN4hnnx27deserialize_tensor_register…
// PFNS_8uptr_DWDI6TensorEE; _ZN4hnnx12DeserializerC2EPKcmP5Graph)。双世界桥同
// ser_ops_interface.hpp: 精确族 TU (-DHNNX_SER_PRECISE) 用全局名 → 导出符号与
// .so 一致; 旧族 TU 用 types.hpp 的 hnnx:: 近似名, 不引入全局声明 —— 免
// `using namespace hnnx` 歧义。
#if defined(HNNX_SER_PRECISE)
class Tensor;
class Graph;
#define HNNX_TENSOR_T ::Tensor
#define HNNX_GRAPH_T ::Graph
#else
namespace hnnx {
class Tensor;
class Graph;
} // namespace hnnx
#define HNNX_TENSOR_T hnnx::Tensor
#define HNNX_GRAPH_T hnnx::Graph
#if defined(HNNX_SER_BRIDGE)
class Graph; // 精确族真身 —— 仅桥 TU 可见 (不外泄 → 无 using-directive 歧义)
#endif
#endif

namespace hnnx {

class Deserz;
class Deserializer;
class DeserSegDescs;
class Crate;
struct uptr_Tensor;

// ---------------------------------------------------------------------------
// DeSerError (SDK serialize_defs.h): 非多态单成员。Deserz 与 Serializer 的公共
// 错误通道 —— Serializer vmi typeinfo (@0x6057708) base0 = 本类 @偏移 8
// (本地 _ZTI @0x5eb9c38; __base_class_type_info 字序 {指针先, offset_flags 后},
// base0 {0x5eb9c38, 0x802}=偏移 8 public / base1 {0x6057530, 0x2}=偏移 0
// public = SerOpsInterface 主基)。
// ---------------------------------------------------------------------------
struct DeSerError {
    char const *errstr = nullptr; // +0x00
};

// ---------------------------------------------------------------------------
// deser_segment_span (SDK deser_concurrent.h:35): {base, limit}
// ---------------------------------------------------------------------------
struct deser_segment_span {
    void *base;
    void *limit;
};

// ---------------------------------------------------------------------------
// runlist_fixup_state (SDK deser_concurrent.h:43) — sizeof 0x38
// ---------------------------------------------------------------------------
struct runlist_seg_descriptor;
struct fixup_supplemental_recs;

struct runlist_fixup_state {
    unsigned segno = 0;                          // +0x00
    size_t *crate_begin = nullptr;               // +0x08 段数据在 crate 中的起点
    runlist_seg_descriptor *seg_desc = nullptr;  // +0x10 对应段描述符
    uint32_t base_tensor_index = 0;              // +0x18 本段首个张量索引
    uint32_t base_blocktable_index = 0;          // +0x1c 本段首个块表索引
    uint32_t base_sharedobj_index = 0;           // +0x20 本段首个共享对象索引
    size_t *fixup_list_head = nullptr;           // +0x28 修正链头 (无则 null)
    fixup_supplemental_recs *fixup_supplemental; // +0x30 附加修正链
};
static_assert(sizeof(runlist_fixup_state) == 0x38);

// ---------------------------------------------------------------------------
// DCrate (SDK deser_concurrent.h:229) — Crate 的反序列化代理。sizeof 0x18。
// nextp 非空时在 [nextp, limitp) 内 bump 分配 (do_alloc: 对齐 >4 时圆整,
// 溢出抛 dcrate_seg_overflow_error); 否则落到 cratep (Crate 慢径)。
// Deserz::setup_dcrate_out @0xcfcf10 写 nextp/limitp (= Deserz+0x20/+0x28)。
// ---------------------------------------------------------------------------
class DCrate {
    friend class Deserz; // Deserz 游标访问面直读 nextp/limitp (M32 偏移证据)

    // 三者要么全空, 要么全非空且 4 对齐
    void *nextp = nullptr;   // +0x00
    void *limitp = nullptr;  // +0x08
    Crate *cratep = nullptr; // +0x10

  public:
    DCrate() {}
    ~DCrate() {}
    DCrate(DCrate const &) = default;
    DCrate(DCrate &&) = default;
    DCrate &operator=(DCrate const &) = default;
    DCrate &operator=(DCrate &&) = default;
    explicit DCrate(Crate &c) : cratep(&c) {}
    void set_crate(Crate &c) { cratep = &c; }
    Crate *crate() { return cratep; }
    bool is_active() const { return nextp != nullptr; }
    constexpr size_t bytes_remaining() const { return (char *)limitp - (char *)nextp; }
    char *next_loc() { return (char *)nextp; }
    std::pair<char *, char *> range_remain() { return {(char *)nextp, (char *)limitp}; }
    void set_memory_range(void *base, unsigned len)
    {
        nextp = base;
        limitp = (void *)((char *)base + len);
    }
    void remove_memory_range()
    {
        nextp = nullptr;
        limitp = nullptr;
    }

    template <typename T, typename... Args> T *emplace(Args &&...args);
    template <typename T> T *emplace0(Deserz &dctx);
    template <typename FI, typename FD, typename SA> void *emplace_explicit(Deserz &dctx, FI, FD, SA);
    template <typename T, bool DTOR_OK = false> T *alloc_array(size_t n);

  private:
    void *do_alloc(size_t align, size_t amount); // dcrate_inlines.h 语义, 库侧定义
};

// 段溢出异常 (SDK deser_concurrent.h:213); 0xcf5240 抛出的 8 字节对象
class dcrate_seg_overflow_error : public std::exception {
  public:
    dcrate_seg_overflow_error() noexcept {}
    ~dcrate_seg_overflow_error() {}
    dcrate_seg_overflow_error(dcrate_seg_overflow_error const &) = default;
    dcrate_seg_overflow_error(dcrate_seg_overflow_error &&) = default;
    dcrate_seg_overflow_error &operator=(dcrate_seg_overflow_error const &) = default;
    dcrate_seg_overflow_error &operator=(dcrate_seg_overflow_error &&) = default;

    char const *what() const noexcept override;
};

// 多字补充记录尾词高 3 位类别编码 (均须 4..7; 0 = 短记录)
constexpr unsigned SUPPFIXUP_CAT_tensor = 4;
constexpr unsigned SUPPFIXUP_CAT_sharedobj = 5;
constexpr unsigned SUPPFIXUP_CAT_blocktable = 6; // 索引打包进一个词
constexpr unsigned SUPPFIXUP_CAT_blocktable_full = 7; // .. 打包进两个词
constexpr unsigned SUPPFIXUP_CAT_SHIFT = 29u;

bool fixup_encode_for_blocktable(runlist_fixup_state &seginfo, uint32_t idx, uint32_t table_offs, void **ptrloc);

[[noreturn]] void throw_dcrate_seg_overflow(); // @0xcf5240

// ---------------------------------------------------------------------------
// 反序列化函数指针与注册表 (SDK deserializer.h)
// ---------------------------------------------------------------------------
class Deserz;

using op_deserializer_fn = uptr_Tensor (*)(Deserz &);
using tensor_deserializer_fn = uptr_Tensor (*)(Deserz &);

// trick_stringview_lt —— 中点字节先比的 string_view 比较器
struct trick_stringview_lt {
    bool operator()(std::string_view a, std::string_view b) const
    {
        unsigned const na = unsigned(a.size());
        unsigned const nb = unsigned(b.size());
        if (na != nb) return na < nb;
        char const *const pa = a.data();
        char const *const pb = b.data();
        if (pa == pb || na == 0) return false; // pa==pb 是常见情形
        unsigned const char_a = pa[na >> 1];
        unsigned const char_b = pb[na >> 1];
        if (char_a != char_b) return char_a < char_b;
        return ::memcmp(pa, pb, na) < 0;
    }
};

using op_deserializer_map_t =
    std::map<std::string_view, std::pair<op_deserializer_fn, bool>, trick_stringview_lt>;
using tensor_deserializer_map_t = std::map<std::string_view, tensor_deserializer_fn, trick_stringview_lt>;

// ---------------------------------------------------------------------------
// Deserz —— 低层读游标 + 反序列化上下文。sizeof 0xd8。
// 虚面 (_ZTVN4hnnx6DeserzE @0x5eb9bd8): D1@+0x00 (0xcfce70) / D0@+0x08
// (0xcfce80) / fill_buffer@+0x10 (0xcfde30) / deserialize_fread@+0x18 (0xcfd930)。
// ---------------------------------------------------------------------------
class Deserz : public DeSerError {
    friend class Deserializer;
    friend class DeserTensorConn;

  protected:
    Deserz(Deserializer *full_deser, char const *p, size_t n, HNNX_GRAPH_T *g = nullptr);

  public:
    Deserz(Deserz const &) = default;
    virtual ~Deserz(); // 首个虚函数 (SDK 约定); D1/D0 见上

    // 段处理三件套 (仅用于启动一个段)
    void setup_source_span(deser_segment_span const &); // @0xcfcea0
    void setup_dcrate_out(void *base, size_t len);      // @0xcfcf10
    void setup_next_tensor_index(unsigned const idx) { next_tensordef_index = idx; }
    void initial_l2fetch();                             // @0xcfcee0

    typedef uint32_t object_identity_type;

    // true 若本 Deserz 实为 Deserializer 自身的基类子对象。
    // (SDK 内联定义于 deserializer.h —— 需 Deserializer 完整类型,
    //  故体在本库 serializer.hpp 中 Deserializer 定义之后给出。)
    bool is_base_deser() const;

    using op_deserialize_info_ptr_t = op_deserializer_map_t::const_iterator;
    using op_deserialize_fn_list_t = std::vector<op_deserialize_info_ptr_t>;
    using tensor_deserialize_fn_list_t = std::vector<tensor_deserializer_fn>;

    // ---- 表访问 (转 full_deser; 内联定义见 SDK deserializer.h:623+) ----
    op_deserialize_fn_list_t &get_op_deserialize_fn_list();
    tensor_deserialize_fn_list_t &get_tensor_deserialize_fn_list();
    std::vector<void *const *> &get_blocktable_link_table();

    void deserialize_tensor_def(HNNX_TENSOR_T const *tensor_ptr);
    void deserialize_tensor_ref(HNNX_TENSOR_T const *&where);
    void deserialize_tensor_refs(HNNX_TENSOR_T const **ptrs, unsigned n);
    object_identity_type deserialize_object_identity();
    void need_tensor_fixup(object_identity_type oid, HNNX_TENSOR_T const **where);

    HNNX_GRAPH_T &graph() const { return *graph_ptr; }
    Crate *crate() { return d_crate.crate(); }
    DCrate *dcrate() { return &d_crate; }
    DeserSegDescs const &get_segments() const; // → full_deser->segments
    op_deserializer_map_t const &get_op_deser_map() const { return *op_deserializer_map; }

    bool is_aligned_const_format() const; // → full_deser->aligned_const_format_flag
    bool has_pending_tensor_updates();

    // ---- 导出方法 (nm 逐一核对; 返回类型不入 mangling 者依 SDK) ----
    // 返回值为 16B 平凡结构 → SysV rax:rdx 双寄存器 (0xcfd516 xor rdx /
    // 0xcfd5a3-0xcfd5c4 返回表项地址直证; Shape<4>::deserialize @0xd94134
    // 以 rdx 为主判 —— rdx==0 即既有对象, rax 即其指针; rdx≠0 为新对象
    // 应写入的共享表项槽位)。
    struct shared_obj_lookup {
        void const *existing; // rax
        void **new_where;     // rdx (共享表项地址 —— 表为 vector<void*>)
    };
    static_assert(sizeof(shared_obj_lookup) == 16);
    shared_obj_lookup deserialize_shared_obj_func(void const **where); // @0xcfd490
    void deserialize_uint32_arr(unsigned int *arr, unsigned long n);        // @0xcfd8b0
    void deserialize_uint32_arr_sizet(unsigned long *arr, unsigned long n); // @0xcfd8d0
    void deserialize_skip_words(unsigned long n);                           // @0xcfde10
    uint64_t deser_u64_slowpath();                                          // @0xcfdee0
    void deserialize_buf_withlen(unsigned long len, void *p);               // @0xcfdf40
    void deserialize_buf(unsigned long len, void *p);                       // @0xcfdfe0
    // @0xcfe000/x3/x4 — 8/12/16 字节平凡结构 (两寄存器 sret 调用约定)
    struct uint32_x2_t {
        unsigned a, b;
    };
    struct uint32_x3_t {
        unsigned a, b, c;
    };
    struct uint32_x4_t {
        unsigned a, b, c, d;
    };
    uint32_x2_t deserialize_uint32_x2(); // @0xcfe000
    uint32_x3_t deserialize_uint32_x3(); // @0xcfe060
    uint32_x4_t deserialize_uint32_x4(); // @0xcfe0e0
    // @0xcfc6f0: u32 长度前缀 + 4 对齐字节串。name_buf 在 Deserializer (+0x10a),
    // 故仅完整 Deserializer 上可调 (SDK 明注)。
    std::string_view deserialize_str();

    // apply_segment_fixups @0xcf5810 (const — 修正落在本对象外)
    void apply_segment_fixups(runlist_fixup_state &) const;

    // ---- 游标内联 (SDK buffer_offset/buffer_remain) ----
    size_t buffer_offset() const { return bufp - bufstart; }
    size_t buffer_remain() const { return bufend - bufp; }
    char const *cur() const { return bufp; }
    char const *base() const { return bufstart; }

    // ---- M32 访问面 (调用点证据偏移; 布局成员化后直读) ----
    bool is_compressed() const noexcept { return format_version != 0; }              // +0x9c
    // classic_format — LayoutTensor 反序列化构造 @db5456 (`cmpl $0,0x9c(dctx); jne`)
    // 逐字节: 非 classic 时 nblocks=1 (延迟指针解析下读不到 shape 对象)。
    bool classic_format() const noexcept { return format_version == 0; }             // +0x9c
    unsigned char *read_cursor() const noexcept { return (unsigned char *)bufp; }    // +0x68
    void set_read_cursor(unsigned char *p) noexcept { bufp = (char const *)p; }
    unsigned char *read_end() const noexcept { return (unsigned char *)buf_limit; }  // +0x70
    unsigned char *scratch_ptr() const noexcept { return (unsigned char *)d_crate.nextp; } // +0x20
    void set_scratch_ptr(unsigned char *p) noexcept { d_crate.nextp = p; }
    unsigned char *scratch_end() const noexcept { return (unsigned char *)d_crate.limitp; } // +0x28
    unsigned char *bump_cursor() const noexcept { return scratch_ptr(); }
    unsigned char *bump_limit() const noexcept { return scratch_end(); }
    void set_bump_cursor(unsigned char *p) noexcept { set_scratch_ptr(p); }
    Crate *scratch_crate() const noexcept { return d_crate.cratep; } // +0x30
    void *shared_ctx() const noexcept { return full_deser; }         // +0x50
    void *shared_context() const noexcept { return full_deser; }
    unsigned char *shared_subobject() noexcept // +0xa0 = &seg_fixup_state
    {
        return reinterpret_cast<unsigned char *>(&seg_fixup_state);
    }
    fa::RuntimeAllocator *runtime_allocator() const noexcept { return allocator; } // +0x18
    // 虚槽 +0x10 (第 3 槽 = fill_buffer) 的手工分发 (M32 假 vtable 实验同位)
    unsigned char *refill() noexcept
    {
        char const *(*fn)(Deserz *);
        void *vt;
        memcpy(&vt, this, sizeof vt);
        memcpy(&fn, static_cast<unsigned char *>(vt) + 0x10, sizeof fn);
        return (unsigned char *)fn(this);
    }
    void fetch_more() noexcept // 旧镜像名: 同槽 void 形
    {
        refill();
    }

    // ---- 数据成员 (声明序 = ctor 写入序 = 布局序) ----
    op_deserializer_fn const *current_op_deser_fn_p{}; // +0x10
    fa::RuntimeAllocator *allocator;                   // +0x18 (无 {} — ctor 置 0)
    DCrate d_crate;                                    // +0x20 (0x18, 含 crate 指针)

  protected:
    // 提升指针: 免每用一次静态锁 (ctor: 两 call 初始化 .data 静态表后存址)
    op_deserializer_map_t const *op_deserializer_map;         // +0x38
    tensor_deserializer_map_t const *tensor_deserializer_map; // +0x40
    HNNX_GRAPH_T *graph_ptr{};                                // +0x48
    Deserializer *full_deser;                                 // +0x50

    char const *bufstart;  // +0x58 当前缓冲起点
    char const *bufend;    // +0x60 首个不可读字节
    char const *bufp;      // +0x68 下一读位置
    char const *buf_limit; // +0x70 ≤ bufend; 触发 fill_buffer 之处
    size_t bytes_filled;   // +0x78 此前已填充字节

    OpExtraInfo op_extra_info; // +0x80 (0x18: id/chkpts/op_tag)

    unsigned next_tensordef_index = 1; // +0x98 (属 tensorconn 但须置于 Deserz)
    int format_version = 0;            // +0x9c 0=classic, 1=July/2023

    // 多线程解码用; 仅段级 Deserz 实例使用 (完整 Deserializer 内恒零构造)
    runlist_fixup_state seg_fixup_state{}; // +0xa0

    // ---- 虚下溢钩子 ----
    virtual char const *fill_buffer();                              // @0xcfde30
    virtual void deserialize_fread(void *p, size_t len, bool align); // @0xcfd930
};

static_assert(sizeof(Deserz) == 0xd8, "Deserz 布局");

// 自由函数 (nm): @0xcfc4c0 慢径 op 索引 / @0xcfca90 类型注册 / @0xcfcb60 张量
unsigned deserialize_op_idx_slow(Deserz &, unsigned int idx);
void deserialize_tensor_register(std::type_info const &, char const *, op_deserializer_fn);
uptr_Tensor deserialize_tensor(Deserz &dctx);

} // namespace hnnx
