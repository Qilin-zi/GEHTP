#define HNNX_SER_PRECISE 1 // 本 TU 为精确族: Op/Tensor/GraphPrepare/Graph 用全局名 (导出符号与 .so 一致)
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/ir/tensor_base.hpp" // hnnx::Allocator 完整类型 (tensor_base:202) + ::OutputDef 先行
#include "hnnx/ir/op.hpp"          // hnnx::uptr_Tensor 完整类型 (op.hpp:273)
#include "hnnx/vtcm/fancy_allocator.hpp" // ::fa::FancyAllocator 完整类型 (ctor dynamic_cast)
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace {
// 兜底分配器 (仅重实现侧): 同时具有 hnnx::Allocator (dynamic_cast 源, 主基) 与
// fa::FancyAllocator (目标) 两基 —— Serializer ctor @0x12f14ab 的
// dynamic_cast<FancyAllocator&>(*alloc) 跨投影视成立。虚面全空: 序列化路径只
// 存指针不解引用 (fancy_alloc 在本移植中无使用点)。
// 真身层级 (M35 全量反演): FancyAllocator ctor @0xf3f2c0 首调
//   fa::RuntimeAllocator::RuntimeAllocator(hnnx::Allocator::Mode, Graph&)
// ⇒ hnnx::Allocator ← fa::RuntimeAllocator ← fa::FancyAllocator。
// Allocator::graph 成员仅存储: 空引用占位不被读取。
class DefaultSerAllocator final : public hnnx::Allocator, public ::fa::FancyAllocator {
  public:
    DefaultSerAllocator() : hnnx::Allocator(hnnx::AllocatorMode::AllocVirtual, *(::Graph *)nullptr) {}
    ~DefaultSerAllocator() override {}
    void allocate_n(void **, size_t, size_t, size_t, hnnx::MemoryClass, unsigned, DType) override {}
    void allocate_persistent_blocks(void **, size_t, size_t, size_t, unsigned) override {}
    void set_mode(hnnx::AllocatorMode) override {}
    void set_tcm_pool(void *, size_t) override {}
    void set_largest_memory_alloc_size(size_t) override {}
};
DefaultSerAllocator g_default_ser_allocator;
} // namespace

namespace hnnx {
Allocator *default_serializer_allocator() { return &g_default_ser_allocator; }
} // namespace hnnx

namespace hnnx {

// Tagged record format:
// [uint32 encoded_tag] = (tag & 0xFFFF | tag << 16) ^ 0xFFFF
// [uint32 word_count]   = ceil(data_size / 4)
// [uint32 third_field]
// [bytes data]          (4-byte aligned)
//
// Separators:
// 0xFA0000FA = normal segment separator
// 0xFA0000FE = aux-data start
// 0xBEEFF00D = end-of-segment marker

// encode/decode_bin_tag 唯一实现于 src/serialize/bin_format.cpp (本文件曾重复定义 → 链接冲突)

// ---------------------------------------------------------------------------
// 组容器 helper @0x12ef9e0: arr = new void*[n](0) (旧 arr 先 delete[]),
// n 落 +0x08, freelist/y 清零, +0x20 写 1.0f。
// ---------------------------------------------------------------------------
static ser_group ser_group_alloc(ser_group &g, size_t n)
{
    delete[] reinterpret_cast<unsigned char **>(g.arr);
    g.arr = new void *[n]();
    g.n = n;
    g.freelist = nullptr;
    g.y = 0;
    g.ratio = 1.0f;
    return ser_group(); // unused (按引用操作)
}

// ---------------------------------------------------------------------------
// Serializer ctor @0x12f1320 —— 逐指令复刻:
//   grp_a = helper(this+0x10, 0x200) (0x1000B); grp_b = helper(this+0x38, 0x20)
//   (0x100B); f_60=0/f_68=0/f_70=-1/f_78=0; str1..6 默认构造;
//   gprep_p=arg1; bufstart=bufp=buf; bufend=buf+len; bytes_filled=0;
//   f_c8=0 (movw); f_d0=0; f_d8=-1; f_e0=0; f_108=0/f_10c=0/f_110=1
//   (movq 合写); f_178=0; fancy_alloc = dynamic_cast<fa::FancyAllocator*>(arg2)
//   (失败 __cxa_bad_cast —— C++ 引用形自动); f_188=0; f_190=0 (movw);
//   0x1b8 q0; 0x1c0..0x1ef 清零; f_1f0=0; str7..9 默认; f_2c8=0; f_2d0=0;
//   f_2d4=0x8000; f_2d8=0 (movw); f_2da=0; 0x2dc..0x353 清零; f_358=0。
// ---------------------------------------------------------------------------
Serializer::Serializer(::GraphPrepare const &gp, Allocator *alloc, char *buf, size_t buflen)
    : grp_a(), grp_b(), f_70(0xffffffffu), f_d8(0xffffffffu), f_110(1), f_2d4(0x8000u)
{
    (void)ser_group_alloc(grp_a, 0x200);
    (void)ser_group_alloc(grp_b, 0x20);

    gprep_p = &gp;
    bufstart = buf;
    bufend = buf + buflen;
    bufp = buf;

    // @0x12f1440: movq %rax,0x10c —— 覆盖 f_10c=0/f_110=1 两 u32 (in-class 已置)
    // @0x12f14ab: __dynamic_cast(alloc, Allocator→fa::FancyAllocator); 空即 bad_cast
    fancy_alloc = &dynamic_cast<::fa::FancyAllocator &>(*alloc);
}

// D1 @0x12f1730 逆序: f_358 pimpl 虚删除 → 9 strings → grp_b → grp_a。
// C++ 生成序 = 成员声明的逆序, 与此一致。
Serializer::~Serializer()
{
    delete f_358;
    f_358 = nullptr;
    delete[] grp_b.arr;
    delete[] grp_a.arr;
    grp_b.arr = grp_a.arr = nullptr;
}

// ---- 行为面 (旧 .cpp 移植; 真实方法体 0x12f1a60 等未逐指令, 编码不变) ----

void Serializer::serialize_fwrite(void const *data, size_t size, bool align4)
{
    if (mode() == Mode::Prescan) {
        int pad = 0;
        if (align4) {
            uint32_t misalign = (static_cast<uint32_t>(current_position()) + static_cast<uint32_t>(size)) & 3;
            if (misalign != 0) pad = 4 - misalign;
        }
        bytes_filled += size + pad;
        return;
    }

    if (align4) {
        uint32_t misalign = static_cast<uint32_t>(current_position()) & 3;
        if (misalign != 0) {
            int pad = 4 - misalign;
            std::memset(bufp, 0, pad);
            bufp += pad;
        }
    }

    if (size > 0 && data) {
        std::memcpy(bufp, data, size);
        bufp += size;
    }
}

void Serializer::flush_buffer()
{
    // @0x12f19f0 未逐指令; 满缓冲语义: 已写字节转记 bytes_filled, 游标回卷
    bytes_filled += size_t(bufp - bufstart);
    bufp = bufstart;
}

void Serializer::serialize_uint32(uint32_t a, uint32_t b, uint32_t c)
{
    serialize_fwrite(&a, 4, false);
    serialize_fwrite(&b, 4, false);
    serialize_fwrite(&c, 4, false);
}

void Serializer::write_uint32(uint32_t val)
{
    if (mode() == Mode::Write)
        serialize_fwrite(&val, 4, false);
    else
        bytes_filled += 4;
}

void Serializer::write_tagged_record(uint32_t tag, void const *data, int data_size)
{
    // 状态机 (重实现侧, f_188): 0=idle 1=writing 2=aux_pending 3=aux_writing
    if (f_188 == 2) {
        write_uint32(SEPARATOR_AUX);
        f_188 = 3;
    }

    uint32_t encoded = encode_bin_tag(tag);
    uint32_t word_count = (uint32_t(data_size) + 3) >> 2;
    serialize_uint32(encoded, word_count, word_count);

    if (data && data_size > 0) serialize_fwrite(data, size_t(data_size), true);

    if (f_188 == 0) f_188 = 1;
}

// @0x12f3f80: bytes_filled + bufp − bufstart
unsigned long Serializer::bytes_written() { return current_position(); }

// @0x12f1be0 未逐指令 —— 预扫描总字节数 (prescan 完成后 = current_position)
unsigned long Serializer::measure_bytes() { return current_position(); }

// @0x12f24e0 未逐指令 —— aux 数据回写 (重实现侧: 直接缓冲内改写)
void Serializer::rewrite_auxdata(unsigned long offset, unsigned n_words, void const *data, unsigned words,
                                 bool a, bool b)
{
    (void)n_words;
    (void)a;
    (void)b;
    if (mode() != Mode::Write || data == nullptr || words == 0) return;
    char *dst = bufstart + offset;
    size_t space = size_t(bufend - dst);
    size_t n = std::min(size_t(words) * 4u, space);
    std::memcpy(dst, data, n);
}

// 旧 do_insert_preload_op (serialize_oplist.cc:418 注释恢复): 位置/计数三字
// 落 +0x340/+0x348/+0x350 (方法访问区, 见头注)
void Serializer::do_insert_preload_op()
{
    uint64_t position = current_position();
    f_348 = f_340;
    f_340 = unsigned(position);
    f_350++;
}

// ---------------------------------------------------------------------------
// SerOpsInterface 18 槽 (M23/M32 行为移植; .so 体 0x12ed280..0x12edb60 待逐指令)
// ---------------------------------------------------------------------------
void Serializer::op_serialize_func(Op const *op, unsigned n_in, Tensor const *const *in_tens, unsigned n_out,
                                    uptr_Tensor const *out_tens, unsigned variadic_flag, unsigned extra)
{
    (void)variadic_flag;
    (void)extra;
    if (!op) return;
    uint32_t hdr[3] = {n_in, n_out, 0};
    serialize_fwrite(hdr, 12, true);
    for (unsigned i = 0; i < n_in && in_tens; ++i)
        if (in_tens[i]) tensor_serialize(in_tens[i]);
    for (unsigned i = 0; i < n_out && out_tens; ++i)
        if (out_tens[i].ptr) tensor_serialize(out_tens[i].ptr);
}

void Serializer::op_for_tensor_func(Op const *op, unsigned n_out, uptr_Tensor const *out_tens)
{
    op_serialize_func(op, 0, nullptr, n_out, out_tens, opMODE_typical, 0);
}

void Serializer::prescan_ops_func(Op *const *seq_of_ops, unsigned n_ops, bool last)
{
    (void)seq_of_ops;
    (void)last;
    uint32_t n = n_ops;
    serialize_fwrite(&n, 4, true);
}

void Serializer::graph_io_tensors(unsigned n_in, uptr_Tensor const *in_tensors, unsigned n_out,
                                   uptr_Tensor const *out_tensors, bool input_only)
{
    uint32_t hdr[2] = {n_in, input_only ? 0u : n_out};
    serialize_fwrite(hdr, 8, true);
    for (unsigned i = 0; i < n_in && in_tensors; ++i)
        if (in_tensors[i].ptr) tensor_serialize(in_tensors[i].ptr);
    if (!input_only)
        for (unsigned i = 0; i < n_out && out_tensors; ++i)
            if (out_tensors[i].ptr) tensor_serialize(out_tensors[i].ptr);
}

void Serializer::checkpoints_table(Checkpoints const &c) { (void)c; }

void Serializer::before_runlists(unsigned nops_norun, unsigned nops_main, unsigned nops_vector, unsigned nops_mtx,
                                  unsigned n_runlist_seg_descs)
{
    uint32_t hdr[5] = {nops_norun, nops_main, nops_vector, nops_mtx, n_runlist_seg_descs};
    serialize_fwrite(hdr, 20, true);
}

void Serializer::after_non_runlist() { write_uint32(SEPARATOR_NORMAL); }
void Serializer::after_runlist() { write_uint32(SEPARATOR_NORMAL); }

void Serializer::serialize_op(Op const &op, unsigned op_seqno)
{
    (void)&op;
    write_uint32(op_seqno);
}

void Serializer::tensor_serialize(Tensor const *tens)
{
    (void)tens;
    write_uint32(0); // 占位: 张量定义记录 (真实 @0x12edb30)
}

void Serializer::shape_serialize(ShapeFlags const *basep, unsigned rank)
{
    if (basep && rank) serialize_fwrite(basep, sizeof(ShapeFlags) * rank, true);
}

OpSerHandle Serializer::op_special(Op const *op)
{
    (void)op;
    return make_opser_handle(0);
}

void Serializer::spcl_done(OpSerHandle &) {}
void Serializer::spcl_add_u32(OpSerHandle &, uint32_t const *p, unsigned n)
{
    if (p && n) serialize_fwrite(p, size_t(n) * 4, true);
}
void Serializer::spcl_add_sized_vec(OpSerHandle &, uint32_t const *data, bool extra)
{
    (void)extra;
    if (data) serialize_fwrite(data, 4, true);
}
void Serializer::spcl_fill_nullptr(OpSerHandle &, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) write_uint32(0);
}
void Serializer::spcl_add_in_tensor(OpSerHandle &, Tensor const *t) { tensor_serialize(t); }
void Serializer::spcl_add_out_tensor(OpSerHandle &, uptr_Tensor const &t)
{
    if (t.ptr) tensor_serialize(t.ptr);
}

SerializerPimpl::~SerializerPimpl() = default;

// ---------------------------------------------------------------------------
// FileSerializer —— 扩展区按 D1@0x12f2850 与临时构造片段 0x12f2660 复刻
// ---------------------------------------------------------------------------
FileSerializer::FileSerializer(::GraphPrepare const &gp, Allocator *alloc, char *buf, size_t buflen)
    : Serializer(gp, alloc, buf, buflen)
{
    fs_buf = nullptr;
    fs_368 = 0;
    fs_370 = 0;
    fs_378 = 0;
    fs_380 = 0;
    fs_388 = 0;
    fs_390 = 0;
}

void FileSerializer::flush_buffer()
{
    // @0x12f28e0 未逐指令; 读 fs_370/fs_368 —— 落盘暂存
    bytes_filled += size_t(bufp - bufstart);
    bufp = bufstart;
}

void FileSerializer::serialize_fwrite(void const *data, size_t size, bool align4)
{
    Serializer::serialize_fwrite(data, size, align4);
}

void FileSerializer::rewrite_auxdata(unsigned long offset, unsigned n_words, void const *data, unsigned words,
                                     bool a, bool b)
{
    Serializer::rewrite_auxdata(offset, n_words, data, words, a, b);
}

unsigned long FileSerializer::bytes_written()
{
    flush_buffer();
    return Serializer::bytes_written();
}

// ---------------------------------------------------------------------------
// Deserz —— ctor @0xcfcda0 内联展开式 (成员 in-class 初始化后仅补差异):
//   两个静态表指针 (call 0xcfb120/0xcfb4a0 的 .data 提升结果), buf_limit 仅在
//   独立构造路径写 p (Deserializer 路径 0xcfcf92 直跳 +0x78 不写)。
// ---------------------------------------------------------------------------
static op_deserializer_map_t const g_op_deser_map_init;     // @0x62431d8 静态表替身
static tensor_deserializer_map_t const g_tens_deser_map_init; // @0x6243218 静态表替身

Deserz::Deserz(Deserializer *full_deser_in, char const *p, size_t n, ::Graph *g)
    : DeSerError(), current_op_deser_fn_p(nullptr), allocator(nullptr), d_crate(),
      op_deserializer_map(&g_op_deser_map_init), tensor_deserializer_map(&g_tens_deser_map_init),
      graph_ptr(g), full_deser(full_deser_in), bufstart(p), bufend(p + n), bufp(p),
      buf_limit(full_deser_in == nullptr ? p : nullptr), bytes_filled(0), op_extra_info(),
      next_tensordef_index(1), format_version(0), seg_fixup_state()
{
    // +0x80 OpExtraInfo: id=0/chkpts={-1,-1} (in-class 已写); op_tag 不触及 ✓
}

Deserz::~Deserz() = default;

// @0xcfcf10 逐指令 (两条存储, 不触 cratep):
//   movq %rsi,0x20(%rdi)  — d_crate.nextp = base
//   addq %rsi,%rax(len) → movq %rax,0x28(%rdi) — limitp = base + len
void Deserz::setup_dcrate_out(void *base, size_t len)
{
    d_crate.nextp = base;
    d_crate.limitp = static_cast<char *>(base) + len;
}

// @0xcfde30 逐指令:
//   buf_limit >= bufend → qnndsp_log(0, "%s:462::ERROR:over-read of serialized
//   data\n", "deserializer.cc") (deserializer.cc:462; 本侧略日志直通) 后
//   throw std::length_error("deserialize underflow")。
//   否则 buf_limit = (bufend − buf_limit > 0x10000) ? buf_limit + 0x8000 : bufend
//   (双 cmov 组合的精确边界), 返回 bufp (不变)。
char const *Deserz::fill_buffer()
{
    if (buf_limit >= bufend) throw std::length_error("deserialize underflow");
    buf_limit = (size_t(bufend - buf_limit) > 0x10000) ? buf_limit + 0x8000 : bufend;
    return bufp;
}

// @0xcfd930 逐指令:
//   pad 由 (bufp + size − bufstart) & 3 决定 (读"后"对齐, 32 位算术), 非 0 时
//   = 4 − mis; 边界一律 buf_limit (+0x70) 而非 bufend; 窗口耗尽走虚槽 +0x10
//   (fill_buffer, 真尽抛 length_error), memcpy 按 min(size, limit−bufp) 分块。
//   dst == null: 跳过 size + pad (同样经 buf_limit 钳制/补窗)。
void Deserz::deserialize_fread(void *dst, size_t size, bool align4)
{
    uint32_t pad = 0;
    if (align4) {
        uint32_t mis = static_cast<uint32_t>(uint64_t(bufp + size - bufstart)) & 3;
        if (mis != 0) pad = 4 - mis;
    }

    size_t total = size + pad; // dst==null 路径的合计跳过量
    char *cdst = static_cast<char *>(dst);

    while (total > 0) {
        size_t avail = size_t(buf_limit - bufp);
        size_t chunk = total < avail ? total : avail;
        if (chunk == 0) {
            fill_buffer(); // 虚分发 (+0x10)
            continue;
        }
        if (cdst != nullptr && size > 0) {
            std::memcpy(cdst, bufp, chunk);
            cdst += chunk;
            size -= chunk;
        }
        bufp += chunk;
        total -= chunk;
    }
}

void Deserz::deserialize_uint32_arr(unsigned int *arr, unsigned long n) { deserialize_fread(arr, n * 4, false); }
void Deserz::deserialize_uint32_arr_sizet(unsigned long *arr, unsigned long n)
{
    deserialize_fread(arr, n * 8, false);
}

void Deserz::deserialize_skip_words(unsigned long n) { bufp += n * 4; }

uint64_t Deserz::deser_u64_slowpath()
{
    uint64_t lo, hi;
    deserialize_fread(&lo, 4, false);
    deserialize_fread(&hi, 4, false);
    return lo | (hi << 32);
}

void Deserz::deserialize_buf_withlen(unsigned long len, void *p) { deserialize_fread(p, len, true); }
void Deserz::deserialize_buf(unsigned long len, void *p) { deserialize_fread(p, len, false); }

// @0xcfe000 族逐指令: 逐字直读 —— 每字前 bufp >= buf_limit 则虚调 fill_buffer
// (+0x10), 无 fread 的分块/对齐路径。x3 @0xcfe060 / x4 @0xcfe0e0 同构。
Deserz::uint32_x2_t Deserz::deserialize_uint32_x2()
{
    if (bufp >= buf_limit) fill_buffer();
    uint32_x2_t r;
    std::memcpy(&r.a, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.b, bufp, 4);
    bufp += 4;
    return r;
}

Deserz::uint32_x3_t Deserz::deserialize_uint32_x3()
{
    uint32_x3_t r;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.a, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.b, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.c, bufp, 4);
    bufp += 4;
    return r;
}

Deserz::uint32_x4_t Deserz::deserialize_uint32_x4()
{
    uint32_x4_t r;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.a, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.b, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.c, bufp, 4);
    bufp += 4;
    if (bufp >= buf_limit) fill_buffer();
    std::memcpy(&r.d, bufp, 4);
    bufp += 4;
    return r;
}

// @0xcfc6f0 逐指令:
//   full_deser != this → throw runtime_error("bad deserialize_str");
//   窗口检查后读 u32 len; len >= 0x1001 (name_buf 0x1000) → throw
//   runtime_error("Deserializing too many bytes");
//   虚调 deserialize_fread (+0x18) 将 len 字节 (align=1) 读入
//   full_deser + 0x10a 的 name_buf, 返回 {name_buf, len}。
std::string_view Deserz::deserialize_str()
{
    if (full_deser != this) throw std::runtime_error("bad deserialize_str");
    if (bufp >= buf_limit) fill_buffer();
    uint32_t len;
    std::memcpy(&len, bufp, 4);
    bufp += 4;
    if (len >= 0x1001) throw std::runtime_error("Deserializing too many bytes");
    char *namep = reinterpret_cast<char *>(full_deser) + 0x10a; // name_buf
    deserialize_fread(namep, len, true);                        // 虚分发 (+0x18)
    return std::string_view(namep, len);
}

} // namespace hnnx
