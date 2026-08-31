#pragma once
// ============================================================================
// serializer.hpp — hnnx::Serializer / FileSerializer / Deserializer (M33 字节级)
//
// 证据基: libHtpPrepare.so x86_64 (QNN 2.48.40.260702)
//   * Serializer ctor @0x12f1320 (GraphPrepare const&, Allocator*, char*, size_t)
//     —— 逐指令解码, sizeof = 0x360; D1 @0x12f1730 / D0 @0x12f18c0。
//   * Serializer vtable _ZTVN4hnnx10SerializerE @0x6057558 (vptr 值
//     0x6057568+0x10; GOT 0x623f070): **25 槽** = SerOpsInterface 18 槽
//     (+0x00..+0x88) + D1/D0 + flush_buffer@0x12f19f0 + serialize_fwrite@
//     0x12f1a60 + rewrite_auxdata@0x12f24e0 + measure_bytes@0x12f1be0 +
//     bytes_written@0x12f3f80 (= bytes_filled + bufp − bufstart)。
//   * typeinfo _ZTIN4hnnx10SerializerE @0x6057708 = __vmi_class_type_info,
//     flags=0, base_count=2 —— __base_class_type_info 字序 {type_info* 先,
//     offset_flags 后}: base0 = DeSerError @偏移 8 (ofl 0x802) / base1 =
//     SerOpsInterface @偏移 0 (ofl 0x2, 主基)。声明序
//     `: public SerOpsInterface, public DeSerError` 同序复现。
//   * FileSerializer: 匿名命名空间本地类 (typeinfo 名 N4hnnx14FileSerializerE
//     @0x55b3833, __si_class_type_info 基→Serializer); 本地 vtable @0x6057630
//     (vptr 值 0x6057640, D1 @0x12f2850 首指令写入值即此)。D1/D0 证明扩展首成员
//     +0x360 = new[]/delete[] 所有权缓冲; FileSerializer::flush_buffer@
//     0x12f28e0 / serialize_fwrite @0x12f2a00 / rewrite_auxdata @0x12f2e60 /
//     bytes_written @0x12f29e0 / measure_bytes 共享基类 @0x12f1be0。
//     Serializer::rewrite_auxdata 尾部在 bufstart+8 处**就地内联构造一个临时
//     FileSerializer** (异常清理调基类 D2 @0x12f27c4) —— 扩展成员 0x360..0x390
//     的写入序即由此片段钉死。
//   * Deserializer ctor @0xcfcf20 `Deserializer(char const*, size_t, Graph*)`:
//     Deserz 基内联同体 (+0x50 存 this; +0x70 buf_limit 不写), sizeof ≈0x11e8。
//     24 导出方法 0xcf5400–0xcff3c0。
// 布局外的一切 (Mode/write_tagged_record 等) 为重实现侧行为面, 注释标明。
// ============================================================================
// 注: 不含 types.hpp —— 本头须同时服务精确族 (op_def/tensor_base, 与 types.hpp
// 的 hnnx::OpDef/OutputDef 近似定义互斥) 与旧族 (调用方自会先含 types.hpp)。
#include "hnnx/serialize/ser_ops_interface.hpp"
#include "hnnx/serialize/deserz.hpp"
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

// Forward declarations
// GraphPrepare 双世界桥 (同 ser_ops_interface.hpp 的 Op/Tensor 桥): 真身为全局
// 类 (mangling _ZN4hnnx10SerializerC2ERK12GraphPrepare…)。Graph 桥 HNNX_GRAPH_T
// 由 deserz.hpp 提供 —— 本头 35 行已先含之, 勿重复定义。
#if defined(HNNX_SER_PRECISE)
class GraphPrepare;
#define HNNX_GP_T ::GraphPrepare
#else
namespace hnnx {
class GraphPrepare;
} // namespace hnnx
#define HNNX_GP_T hnnx::GraphPrepare
#if defined(HNNX_SER_BRIDGE)
class GraphPrepare; // 精确族真身 —— 仅桥 TU 可见 (不外泄 → 无 using-directive 歧义)
#endif
#endif

namespace hnnx {

class Serializer;
class FileSerializer;
class Deserializer;
class Allocator; // PNS_9AllocatorE —— hnnx::Allocator 两世界同名

} // namespace hnnx

namespace fa {
class FancyAllocator; // 全局 fa:: (mangling N2fa14FancyAllocator; hnnx::fa 不存在)
} // namespace fa

namespace hnnx {

// .bin format: tagged record stream
// Each record: [uint32 encoded_tag][uint32 word_count][uint32 third_field][data...]
// Tag encoding: (tag & 0xFFFF | tag << 16) ^ 0xFFFF
// Separators: 0xFA0000FA (normal), 0xFA0000FE (aux-data), 0xBEEFF00D (end-of-segment)

// 重实现侧兜底: 近似 GraphPrepare 的 allocator_ 无供给路径 (恒空), 而真实
// Serializer ctor @0x12f14ab 无条件 dynamic_cast<FancyAllocator&>(*alloc)。
// 返回一个可跨投射到 fa::FancyAllocator 的空实现分配器 (定义见 serializer.cpp;
// 真身 GraphPrepare 恒有 Allocator&, 不经此路)。
Allocator *default_serializer_allocator();

// Tag encoding
uint32_t encode_bin_tag(uint32_t tag);
uint32_t decode_bin_tag(uint32_t encoded);

// Blocktable serialization
void serialize_blocktable(Serializer& ser, const void* blocktable, size_t size);
void serialize_qp_record(Serializer& ser, const std::vector<uint32_t>& qp_ids);
void serialize_internal(Serializer& ser, const void* data, size_t size);
void get_serialization_indices(const void** objects, size_t count, uint32_t* indices);
void* deserialize_block_pointer(Deserializer& deser);
void gpe_serialize_to_mem(Serializer& ser, const void* gpe_data, size_t size);
void make_plan_for_deser_by_segments(Serializer* ser, int num_segments, uint32_t ops_per_segment);

// Known tag IDs (from do_serialize decompilation @ 0xf64fa0, verified):
// Header/config tags (written in order):
//   0xEF4D = io_dma_bypass, 0xE347 = io_tensors_config, 0xD352 = extra_config
//   0xD349 = multicast_config, 0x5647 = runlist_segment_desc (8 bytes)
//   0xC953 = self_slicing flags, 0x4650 = MC cache config
//   0xC955 = pass registry size (variable), 0xCF55 = another registry (variable)
//   0x5350 = IO counts (input_count, output_count, 8 bytes)
enum BinFormatTag : uint32_t {
    TAG_IO_DMA_BYPASS        = 0xEF4D,
    TAG_SPILL_FILL_INSTEAD   = 0x4453,
    TAG_EXTENDED_UDMA        = 0xD446,
    TAG_IO_TENSORS_CONFIG    = 0xE347,
    TAG_EXTRA_CONFIG         = 0xD352,
    TAG_MULTICAST_CONFIG     = 0xD349,
    TAG_NUM_SEGMENTS         = 0x5248,
    TAG_VEC_RUNLIST_COUNT    = 0x524C,
    TAG_RUNLIST_SEGMENT_DESC = 0x5647,
    TAG_SELF_SLICING         = 0xC953,
    TAG_SLICING_CONFIG       = 18000,  // 0x4650
    TAG_PASS_REGISTRY        = 0xC955, // pass registry size (variable)
    TAG_ANOTHER_REGISTRY     = 0xCF55, // another registry (variable)
    TAG_IO_COUNTS            = 0x5350, // input_count, output_count (8 bytes)
    TAG_RUNLIST_AUX          = 0x5453,
    TAG_INTERFACE_BASELINE   = 0xE358,
    // Reimplementation-specific records (graph structure round-trip)
    // These use custom tags not in the real .bin to avoid conflicts
    TAG_OP_RECORD            = 0x4F50, // 'OP' runlist op descriptor
    TAG_IO_TENSOR_DESC       = 0x494F, // 'IO' input/output tensor descriptor
    TAG_GRAPH_HEADER         = 0x4748, // 'GH' graph header (op+io counts)
    TAG_CONST_EXTENT         = 0x4345, // 'CE' const extent descriptor table
    TAG_PLAN_ORDER           = 0x504C, // 'PL' Scheduler(ST-Cut)计划执行序: [u32 count][u32 ids...]
    TAG_SPILL_FILL_OP        = 0x5346, // 'SF' 单张量 spill/fill 记录: [u64 op_id][u32 block_id][u64 ddr_offset][u64 size]
};

constexpr uint32_t SEPARATOR_NORMAL  = 0xFA0000FA;
constexpr uint32_t SEPARATOR_AUX     = 0xFA0000FE;
constexpr uint32_t SEPARATOR_END     = 0xBEEFF00D;

// .bin header (大端! 从真实 test_minimal.serialized.bin 头部确认)
// 无文件级 BEEF/FA00 magic — 那些是 DSP 内存分隔符, 不在 .bin 文件里。
// 真实格式: 计数 + 偏移表 (大端 uint32)
struct BinHeader {
    uint32_t num_graphs;       // [0x00] 大端, 真实样本 = 2
    uint32_t num_records;      // [0x04] 大端, 真实样本 = 3
    uint32_t reserved_08;      // [0x08] = 0
    uint32_t flags;            // [0x0C] = 1
    uint64_t offset_table;     // [0x10] 大端偏移表基址
    uint64_t block_size;       // [0x18] 大端块大小 (样本 = 1MB)
    uint64_t base_address;     // [0x20] 大端基址 (样本 = 0x2E5E9000)
};

// (文件级 magic 校验不存在; "Magic check failed" 指类型注册表的 magic)
// Deserializer::get_name 里的类型 magic (反汇编确认):
constexpr uint32_t TYPE_MAGIC_1 = 0x71A6009B; // get_name cmp.eq 常量 1
constexpr uint32_t TYPE_MAGIC_2 = 0xEBC0FEFE; // get_name cmp.eq 常量 2

// class index: 类型注册表, 反序列化时找到正确的 tensor/op 构造函数
// Source: Deserializer::auxdata_class_index(tag, bool)
struct ClassIndexEntry {
    uint32_t class_id;  // 类型 ID
    uint32_t name_len;  // 类型名长度
    // 后跟 name_len 字节类型名 (4 字节对齐)
};

// segment span: 段在 .bin 中的字节范围
// Source: Deserz::apply_segment_fixups, skip_to_after_span
struct DeserSegmentSpan {
    uint32_t segment_index;
    uint32_t op_count;
    uint64_t byte_offset; // 段起始偏移
    uint64_t byte_size;   // 段大小
};

// const extent descriptor: 常量数据在 const pool 中的位置
// Source: Deserializer::extract_const_extent_table / auxdata_read_const_extent_descriptor
struct ConstExtentDesc {
    uint64_t op_id;        // 所属 op
    uint64_t offset;       // 在 const pool 中的偏移
    uint64_t size;         // 字节数
    uint32_t tensor_type;  // tensor 类型 ID (见 ClassIndexEntry)
    uint32_t reserved;
};

// ---------------------------------------------------------------------------
// SerializerPimpl —— +0x358 所指对象的多态面 (D1@0x12f1730: 非空时调其
// vptr+0x08 = deleting-dtor ⇒ 基类含虚析构)。成员未反演。
// ---------------------------------------------------------------------------
class SerializerPimpl {
  public:
    virtual ~SerializerPimpl();
};

// ---------------------------------------------------------------------------
// ser_group —— 组容器 0x28 (ctor @0x12f1320 调 helper @0x12ef9e0:
//   helper(this, n): arr = new void*[n](0) (旧 arr 先 delete[]), n 落 +0x08,
//   freelist/y 清零, +0x20 写 1.0f。D1 逆序: freelist 链遍历 (node->next@
//   node+0) 逐 delete → delete[] arr。语义标签 M33 未全解, 偏移/尺寸已证。)
// ---------------------------------------------------------------------------
struct ser_group {
    void **arr = nullptr;        // +0x00 元素数组 (ctor: n*8 字节清零分配)
    size_t n = 0;                // +0x08 容量 (组 A 0x200 / 组 B 0x20)
    void **freelist = nullptr;   // +0x10 空闲链 (结点首字为 next)
    unsigned long y = 0;         // +0x18 (M33 未解码)
    float ratio = 1.0f;          // +0x20 ctor 写 1.0f
};
static_assert(sizeof(ser_group) == 0x28);

// ---------------------------------------------------------------------------
// Serializer —— sizeof 0x360。成员序 = ctor 写入序 = 布局序。
// 未解码语义的字段命名 f_<off>, 字节区间以注释钉死; 大片 ctor 清零区按
// 实际清零边界声明为 pad 数组 (清零≠无成员 —— 仅证明 ctor 写 0)。
// ---------------------------------------------------------------------------
class Serializer : public SerOpsInterface, public DeSerError {
  public:
    // @0x12f1320: +0x180 经 __dynamic_cast(alloc, Allocator→fa::FancyAllocator)
    // 失败即 __cxa_bad_cast —— allocator 实参必须已是 FancyAllocator。
    Serializer(HNNX_GP_T const &gp, Allocator *alloc, char *buf, size_t buflen);
#if defined(HNNX_SER_BRIDGE)
    // 桥 TU (serializer_bridge.cpp) 双名并见: 补精确族签名声明 (定义在
    // serializer.cpp) 供下方旧族 ctor 委托转发; 旧族 ctor 本体亦在桥 TU 定义。
    Serializer(::GraphPrepare const &gp, Allocator *alloc, char *buf, size_t buflen);
#endif

    // ---- 重实现侧行为面 (不入 .so 布局; 消费方 graph_prepare.cpp 等) ----
    enum class Mode : int {
        Prescan = 0, // 仅计数
        Write = 1,   // 实写
    };
    void set_mode(Mode m) { f_108 = (m == Mode::Prescan) ? 1 : 0; } // 借 +0x108 (ctor-0 字节)
    Mode mode() const { return f_108 ? Mode::Prescan : Mode::Write; }
    void set_error(char const *msg) { errstr = msg; }               // DeSerError @+0x08
    bool has_error() const { return errstr != nullptr; }
    // bytes_written 虚槽同式 (@0x12f3f80): bytes_filled + bufp − bufstart
    size_t current_position() const
    {
        return bytes_filled + size_t(bufp - bufstart);
    }
    void write_uint32(uint32_t val);            // 行为面 (见 serializer.cpp)
    void write_tagged_record(uint32_t tag, const void *data, int data_size); // 行为面
    void serialize_uint32(uint32_t a, uint32_t b, uint32_t c);                // 行为面
    void do_insert_preload_op();                                              // 行为面 (+0x340 区)

    size_t buffer_remain() const { return size_t(bufend - bufp); }
    char *cursor() const { return bufp; }

    virtual ~Serializer(); // D1 @0x12f1730 / D0 @0x12f18c0

  protected:
    // ---- 虚面 (vtable @0x6057558, 25 槽; 序 = 槽序) ----
    // SerOpsInterface 18 槽 (+0x00..+0x88) —— 全覆盖, 体见 serializer.cpp
    void op_serialize_func(HNNX_OP_T const *op, unsigned n_in, HNNX_TENSOR_T const *const *in_tens, unsigned n_out,
                           uptr_Tensor const *out_tens, unsigned variadic_flag, unsigned extra) override;
    void op_for_tensor_func(HNNX_OP_T const *op, unsigned n_out, uptr_Tensor const *out_tens) override;
    void prescan_ops_func(HNNX_OP_T *const *seq_of_ops, unsigned n_ops, bool last) override;
    void graph_io_tensors(unsigned n_in, uptr_Tensor const *in_tensors, unsigned n_out,
                          uptr_Tensor const *out_tensors, bool input_only) override;
    void checkpoints_table(Checkpoints const &) override;
    void before_runlists(unsigned nops_norun, unsigned nops_main, unsigned nops_vector, unsigned nops_mtx,
                         unsigned n_runlist_seg_descs) override;
    void after_non_runlist() override;
    void after_runlist() override;
    void serialize_op(HNNX_OP_T const &, unsigned op_seqno) override;
    void tensor_serialize(HNNX_TENSOR_T const *tens) override;
    void shape_serialize(ShapeFlags const *basep, unsigned rank) override;
    OpSerHandle op_special(HNNX_OP_T const *op) override;
    void spcl_done(OpSerHandle &) override;
    void spcl_add_u32(OpSerHandle &, uint32_t const *p, unsigned n) override;
    void spcl_add_sized_vec(OpSerHandle &, uint32_t const *data, bool extra) override;
    void spcl_fill_nullptr(OpSerHandle &, unsigned n) override;
    void spcl_add_in_tensor(OpSerHandle &, HNNX_TENSOR_T const *) override;
    void spcl_add_out_tensor(OpSerHandle &, uptr_Tensor const &) override;

  public:
    // vtable 后 5 槽 (+0x98..+0xb0)。访问级别不入二进制; 供落盘路径直调。
    virtual void flush_buffer();                                             // +0x98 @0x12f19f0
    virtual void serialize_fwrite(void const *data, size_t size, bool align4); // +0xa0 @0x12f1a60
    virtual void rewrite_auxdata(unsigned long offset, unsigned n_words, void const *data,
                                 unsigned words, bool a, bool b);             // +0xa8 @0x12f24e0
    virtual unsigned long measure_bytes();                    // +0xb0? @0x12f1be0
    virtual unsigned long bytes_written();                    // 末槽 @0x12f3f80

    // ---- 数据成员 (0x360; ctor @0x12f1320 写入序) ----
    ser_group grp_a;               // +0x10 (helper n=0x200 → 0x1000B)
    ser_group grp_b;               // +0x38 (helper n=0x20 → 0x100B)
    unsigned f_60 = 0;             // +0x60 u32 0
    uint64_t f_68 = 0;             // +0x68
    unsigned f_70 = 0xffffffffu;   // +0x70 u32 -1
    uint64_t f_78 = 0;             // +0x78
    unsigned char f_80[8];         // +0x80..0x87 ctor 未触及 (str1 前间隙)
    std::string str1;              // +0x88
    HNNX_GP_T const *gprep_p{}; // +0xa0 (真身 RK12GraphPrepare 全局)
    char *bufstart{};              // +0xa8 当前缓冲首
    char *bufend{};                // +0xb0 = buf+buflen
    char *bufp{};                  // +0xb8 = buf
    size_t bytes_filled = 0;       // +0xc0
    unsigned f_c8;                 // +0xc8 word 0 (movw! 0xca-0xcb 不触及)
    uint64_t f_d0 = 0;             // +0xd0
    unsigned f_d8 = 0xffffffffu;   // +0xd8 u32 -1
    uint64_t f_e0 = 0;             // +0xe0
    unsigned char f_e8[8];         // +0xe8..0xef ctor 未触及 (str2 前间隙)
    std::string str2;              // +0xf0 (尾字 0x100-0x107 即其 +0x10)
    unsigned char f_108 = 0;       // +0x108 字节 0 (重实现借作 Prescan 门控)
    unsigned f_10c = 0;            // +0x10c u32 0 (ctor movq %rax,0x10c 合写)
    unsigned f_110 = 1;            // +0x110 u32 1 (同上 qword 高半)
    std::string str3;              // +0x118
    std::string str4;              // +0x130 (尾字 0x140-0x147 即其 +0x10)
    std::string str5;              // +0x148
    std::string str6;              // +0x160 (ctor: movups 0x158 跨 str5 尾/str6 头)
    unsigned f_178 = 0;            // +0x178 u32 0 (str6 尾 0x170-0x177 即其 +0x10)
    ::fa::FancyAllocator *fancy_alloc{}; // +0x180 (dynamic_cast 结果; 失败 bad_cast)
    unsigned f_188 = 0;            // +0x188 u32 0
    unsigned short f_190 = 0;      // +0x190 word 0
    unsigned char pad_192[0x198 - 0x192] = {}; // +0x192..0x197 (ctor 未触及)
    unsigned char pad_198[0x1b8 - 0x198] = {}; // +0x198..0x1b7 ctor 清零区
    unsigned char pad_1b8[0x1c0 - 0x1b8] = {}; // +0x1b8..0x1bf (0x1b8 q0 后 4 字节不触及)
    unsigned char pad_1c0[0x1f0 - 0x1c0] = {}; // +0x1c0..0x1ef ctor 清零区 (48B)
    unsigned f_1f0 = 0;                       // +0x1f0 u32 0
    unsigned char pad_1f4[4];                 // +0x1f4..0x1f7 对齐填充
    std::string str7;              // +0x1f8
    unsigned char f_210[8];        // +0x210..0x217 ctor 未触及 (str8 前间隙)
    std::string str8;              // +0x218
    unsigned char f_230[0x28];     // +0x230..0x257 ctor 未触及 (0x28 —— 疑似第三组容器)
    std::string str9;              // +0x258
    unsigned char pad_270[0x2c8 - 0x270] = {}; // +0x270..0x2c7 (ctor 未触及)
    unsigned f_2c8 = 0;            // +0x2c8 u32 0
    unsigned char f_2d0 = 0;       // +0x2d0 字节 0
    unsigned f_2d4 = 0x8000u;      // +0x2d4 u32 0x8000
    unsigned short f_2d8 = 0;      // +0x2d8 word 0
    unsigned char f_2da = 0;       // +0x2da 字节 0
    unsigned char pad_2dc[0x330 - 0x2dc] = {}; // +0x2dc..0x32f ctor 清零区 (movups×5)
    unsigned f_330 = 0, f_334 = 0; // +0x330/+0x334 (方法访问区, 语义未解)
    unsigned f_338 = 0, f_33c = 0; // +0x338/+0x33c
    unsigned f_340 = 0;            // +0x340 当前位置 (旧注释; 40 处方法访问)
    unsigned f_344 = 0;            // +0x344
    unsigned f_348 = 0;            // +0x348 前一位置
    unsigned f_34c = 0;            // +0x34c (ctor movq 0x34c 合写首)
    unsigned f_350 = 0;            // +0x350 preload 计数
    unsigned char pad_354[4];      // +0x354..0x357 对齐填充
    SerializerPimpl *f_358 = nullptr; // +0x358 pimpl (D1: delete → 其 vptr+0x08 虚删除槽)
};
// RE 布局保真断言按 MSVC ABI 标定(真身 .so 在 Windows 工具链);Linux(g++/Itanium
// ABI)下成员布局不同,此断言不适用——序列化行为走显式成员访问,与 sizeof 无关。
#if defined(_MSC_VER)
static_assert(sizeof(Serializer) == 0x360, "Serializer 布局");
#endif

// ---------------------------------------------------------------------------
// FileSerializer —— Serializer 的文件落盘派生 (本地类, sizeof ≥0x398)。
// 扩展成员 0x360..0x390 由 D1@0x12f2850 (free +0x360) 与 flush_buffer/
// serialize_fwrite/rewrite_auxdata 的访问序钉死; ctor 在 .so 中内联于调用点
// (Serializer::rewrite_auxdata@0x12f2646 起的临时构造片段 + graph_prepare
// serialize_file @0xf830d0), 未有独立符号。
// ---------------------------------------------------------------------------
class FileSerializer : public Serializer {
  public:
    // 重实现侧: 基类 ctor + 扩展清零 (对齐内联片段 0x12f2660-0x12f26a0)
    explicit FileSerializer(HNNX_GP_T const &gp, Allocator *alloc, char *buf, size_t buflen);

  protected:
    void flush_buffer() override;        // @0x12f28e0
    void serialize_fwrite(void const *data, size_t size, bool align4) override; // @0x12f2a00
    void rewrite_auxdata(unsigned long offset, unsigned n_words, void const *data, unsigned words, bool a,
                         bool b) override;                                          // @0x12f2e60
    unsigned long bytes_written() override; // @0x12f29e0: 先 flush_buffer 再同式

    // ---- 扩展成员 (基类 0x360 之后) ----
    char *fs_buf = nullptr;   // +0x360 暂存缓冲 (new[]/delete[] 所有权; D1 释放)
    unsigned fs_368 = 0;      // +0x368 u32 (flush_buffer 读)
    uint64_t fs_370 = 0;      // +0x370 (flush_buffer 读)
    unsigned fs_378 = 0;      // +0x378 u32
    uint64_t fs_380 = 0;      // +0x380
    uint64_t fs_388 = 0;      // +0x388
    uint64_t fs_390 = 0;      // +0x390 (rewrite_auxdata 临时构造: 缓冲基址)
};
#if defined(_MSC_VER)
static_assert(sizeof(FileSerializer) >= 0x398);
#endif

// ---------------------------------------------------------------------------
// Deserializer —— sizeof ≈0x11e8。ctor @0xcfcf20 内联 Deserz 基 (唯一差异:
// +0x50 存 this 自引用; +0x70 buf_limit 不写), 随后 vptr 换 _ZTVN4hnnx12DeserializerE
// (GOT 0x623f328, 表 @0x5eb9c08) 并初始化扩展区。
// 扩展区 ctor 写入: +0xd8 vector 清零; +0xf0 0x18 清零 (tensorconn);
// +0x108 word 0; name_buf[4096]@+0x10a 不初始化; +0x1110 u32 0; +0x1118 0;
// +0x1120 u32 0; +0x1128..0x1138 清零 (segments/pickle_len_words);
// +0x1160/+0x1178 std::map 自引用空表 (端结点在表+0x08);
// +0x11a0..0x11e8 = 三个 vector (fn 列表 ×2 + blocktable 链表)。
// ---------------------------------------------------------------------------
class Deserializer : public Deserz {
  public:
    Deserializer(char const *data, size_t size, HNNX_GRAPH_T *graph); // @0xcfcf20
#if defined(HNNX_SER_BRIDGE)
    Deserializer(char const *data, size_t size, ::Graph *graph); // 精确族 (定义在 deserializer.cpp)
#endif
    virtual ~Deserializer();

    // Deserz::is_base_deser 之内联体 (需本类完整):
    // "本 Deserz 是否 Deserializer 自身的基类子对象"
    friend inline bool Deserz::is_base_deser() const;

    // ---- 重实现侧行为面 (deserializer.cpp; 地址注释 = .so 对应方法) ----
    void load_header(size_t offset);
    void auxdata_deserialize_segments(uint32_t param);
    void auxdata_class_index(uint32_t tag, bool flag);
    void auxdata_temparr_sizes(uint32_t tag);
    void auxdata_read_const_extent_descriptor(uint32_t tag);
    void extract_const_extent_table(uint32_t tag);
    void extract_const_extent_data(uint64_t offset, uint32_t size, void *dst, uint64_t a, uint64_t b);
    void resize_object_tables(const struct runlist_auxdata_seg_desc &);
    void segmentjob_deserialize_ops(uint32_t seg_idx, uint32_t job_idx);
    void handle_auxdata_deser(uint32_t tag, uint32_t size);
    uint32_t crate_size_according_to_segments();
    void get_forward_span(uint32_t seg, uint32_t a, uint32_t b);
    void skip_to_after_span(const DeserSegmentSpan &span);
    void set_graph(HNNX_GRAPH_T *g) { graph_ptr = g; }
    const char *get_name(uint64_t a, uint64_t b);

    // ---- 扩展数据成员 (Deserz 0xd8 之后) ----
    std::vector<void *> objindex;            // +0xd8 对象索引表
    unsigned char tensorconn[0x18] = {};     // +0xf0 (SDK DeserTensorConn, M33 未逐字段)
    bool tc_flag1 = false, tc_flag2 = false; // +0x108 word 0
    char name_buf[0x1000];                   // +0x10a 名字缓冲 (ctor 不初始化)
    unsigned char pad_110a[0x1110 - 0x110a]; // +0x110a..0x110f 对齐填充
    unsigned x_1110 = 0;                     // +0x1110
    uint64_t x_1118 = 0;                     // +0x1118
    unsigned x_1120 = 0;                     // +0x1120
    unsigned char segs_1128[0x1140 - 0x1128] = {}; // +0x1128..0x113f 清零 (segments 头部)
    unsigned char pad_1140[0x1160 - 0x1140] = {}; // +0x1140..0x115f (segments 续, 未逐字段)
    std::map<std::string_view, uint64_t> named_cexdescs;   // +0x1160 (键值类型 M33 未证)
    std::map<std::string_view, uint64_t> named_weight_bufs; // +0x1178 (同上)
    unsigned char pad_1190[0x11a0 - 0x1190] = {}; // +0x1190..0x119f
    op_deserialize_fn_list_t op_deserialize_fn_list;      // +0x11a0
    tensor_deserialize_fn_list_t tensor_deserialize_fn_list; // +0x11b8
    std::vector<void *const *> blocktable_link_table;     // +0x11d0
};

#if defined(_MSC_VER)
static_assert(sizeof(Deserializer) == 0x11e8, "Deserializer 布局 (0xd8 Deserz 基 + 扩展区)");
#endif

// is_base_deser (SDK deserializer.h:624 原式) —— Deserializer 完整后给出
inline bool Deserz::is_base_deser() const { return static_cast<Deserz const *>(full_deser) == this; }

} // namespace hnnx
