#pragma once
// ============================================================================
// G4 反汇编保真实现：spill/fill（SFCD 运行时格式 + slc 序列化 + 分配面 + 桩）
// 证据: audit_verify/reports/G4_spillfill_disasm.md (2026-08-31)
//       audit_verify/asm/f3/G4_grdep_spillfill.asm (0x100b000-0x1016000)
//       audit_verify/asm/f3/G4_slc_allocator.asm   (0x1293000-0x129a000)
//       audit_verify/asm/f3/G4_slc_post_pass.asm   (0x129a000-0x12a4000)
// 规则: 每个逻辑段标注 [0x地址]；grdep 重写簇 / 0x129f0b0 设计主体 /
//       SFCD 写侧为记录级（G4b 遗留），不臆测补齐。
//
// 一句话语义: spill/fill 在 .so 里分四层 —— ① insert_spill_fill 是永久桩
// [0x106d7d0]；② 运行时 SFCD 二进制 = 3 词头 + 四类记录（tcm 块 / 0x80 wait /
// 0x81 set / 非法:2095），子项 64B 对齐步进 [0x1013c58]；③ slc 序列化侧
// rec_type 三值枚举 0/1/2 [0x1294cd7]；④ 分配面 = 池 2 三槽 + shared/far 位
// (PoolDesc+0x1e bit0/bit4) [0xf4ce98/0xf4cec2]。
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace hnnx {

// ---- 对齐 --------------------------------------------------------------------
// SFCD 子项游标步进: cur += (len + 0x3f) & ~0x3f        [0x1013c58-0x1013c5e]
inline uint32_t g4_round_up_64(uint32_t n) {
    return (n + 0x3fu) & ~0x3fu;                              // [0x1013c58]
}
// FancyAllocator 64KB 取整: (n + 0xFFFF) & 0xFFFF0000   [0xf4cd14-0xf4cd1e]
inline uint32_t g4_round_up_64k(uint32_t n) {
    return (n + 0xffffu) & 0xffff0000u;                       // [0xf4cd14]
}

// ============================================================================
// §A SFCD（SpillFill Control Data）运行时格式 —— dump 真身 0x1013a90 全解
// ============================================================================

// 记录四类判据 = 首词最高字节 [0x1013b8a js / 0x1013cd6 cmpb $0x81 / 0x1013cde cmpb $0x80]
enum class G4SfcdKind {
    TcmBlock,          // w0 < 0x80000000: (pool << 16) | nblocks，w1 = blob 偏移
    WaitForProgress,   // top byte 0x80，bit16 选单词/对形态
    SetProgress,       // top byte 0x81: mb_idx = w0 & 0xFFFFFF, val = w1
    BadHeader          // 其它 → :2095 ERROR（不推进游标，记错继续）
};

// 头部（3 词）: +0 u32 total_bytes（尾校验用它+4）；+4 i32 checkpoint_index；
//               +8 u32 低 24 位 = 记录数；记录自 +0xc 起 [0x1013ab0/0x1013af7/0x1013b12/0x1013b33]
struct G4SfcdHeader {
    uint32_t total_bytes = 0;      // [+0]
    int32_t  checkpoint_index = 0; // [+4]
    uint32_t record_count = 0;     // [+8] & 0xFFFFFF
};

// tcm 块子项（8 字节 {u32 w, u32 len}）
// tcm_off = w & ~0xF，type_nibble = w & 0xF  [0x1013bfe/0x1013c0b]
// type nibble 的取值语义未在任何打印中命名（G4b 遗留）
struct G4SfcdBlockEntry {
    uint32_t tcm_offset = 0;
    uint32_t type_nibble = 0;
    uint32_t len = 0;
    uint32_t cursor_before = 0;    // 元组行第 2 参（打印时的 mempool 游标）
};

struct G4SfcdWaitEntry { uint32_t mb_idx = 0, val = 0; };

// 解析结果（一次走完的副作用汇总）
struct G4SfcdWalk {
    G4SfcdHeader hdr;
    bool zero_records = false;         // [+8]&0xFFFFFF == 0 → "?? zero records??" 后返回
    uint64_t cursor_bytes = 0;         // 走完所有记录后的 (游标 − 基址) 字节数
    bool bad_length = false;           // cursor_bytes != [+0] + 4  [0x1013e93-0x1013ea3]
    uint64_t bad_length_words = 0;     // 差值 >> 2
    uint32_t blocks_total = 0;         // Σ 子项 len（不对齐）@ [rsp+0x14]
    std::vector<std::string> errors;   // :2095 非法头（每条一处，游标停驻）
};

// dump 真身模型: dump_sfcd(f, sfcd, is_fill(edx), verbose(ecx)) @0x1013a90
// 返回打印文本；walk 非空时回填解析结果。
// 注意 fill/spill 由调用方参数 is_fill 决定（cmovne 选 fmt/箭头），与记录内容无关
// [0x1013b37-0x1013b5f]。
std::string g4_dump_sfcd(const uint32_t* sfcd, bool is_fill, bool verbose,
                         G4SfcdWalk* walk = nullptr);

// :2095 错误文本（qnndsp_log 侧 + 文件侧双写）
// "%s:2095::ERROR:Bad SFCD record header %08X\n" @0x4626e9e [0x1013eff]
std::string g4_err_2095(uint32_t bad_hdr);
// "%s   !!!!  Bad SFCD record header %08X" @0x4626eca [0x1013f20]
std::string g4_err_2095_file(uint32_t bad_hdr);

// ============================================================================
// §B slc 序列化侧（slc_allocator.cc，序列化器 0x1294570）
// ============================================================================

// rec_type 三值枚举 [0x1294cdb: cmpl $0x2 / 0x1294cec: cmpl $0x1 / 0x1294d14: testl]
// 其它值直接跳过 [0x129533e]
enum G4RecType {
    G4_REC_SPILLFILL   = 0,  // 拷贝记录（值序列化为字符串 "spillfill"）
    G4_REC_WAITFOR     = 1,  // {mb_idx, wait_for_val} 对 @ +0x8..+0x10
    G4_REC_SETPROGRESS = 2   // mb_idx @ +0x8, value_to_set @ +0xc
};

// tagged-kv 机制: 键串 OR 0x4050000000000000（string_tag）[0x1294605 orq]
constexpr uint64_t G4_STRING_TAG = 0x4050000000000000ull;
// 已核值标签: is_fill/is_multi_nsp 的 bool 词 = 10(true)/9(false)
//   `cmpb $1; movw $0xa; sbbw $0` [0x1294b6b-0x1294ba9 / 0x1294bb0-0x1294bee]
// dma_checkpoint 的 i32 标签 = 0xe [0x1294bf5-0x1294c35]
// 其余值标签 6/7/8/0xb/0xc/0xd 在键表区出现但逐键归属未核（G4b 遗留）
constexpr uint64_t G4_TAG_BOOL_TRUE = 0xa;
constexpr uint64_t G4_TAG_BOOL_FALSE = 0x9;
constexpr uint64_t G4_TAG_I32_DMA_CHECKPOINT = 0xe;

// spillfill 记录的 copies 元素（12 字节: tcm_offset, copy_len, cache_hints）
struct G4SlcCopy { uint32_t tcm_offset = 0, copy_len = 0, cache_hints = 0; };

// 记录结构（0x40 字节步长 [0x1294cba]），三类共用体按 rec_type 取舍:
//   0: ddr_pool@+0xc, sf_offset@+0x10, total_copy@+0x14, copies 向量@+0x20/+0x28
//   1: pairs 向量@+0x8/+0x10（8B 元素 {mb_idx, wait_for_val}）
//   2: mb_idx@+0x8, value_to_set@+0xc
//   +0x38: 与 rec_type 冗余的类别镜象（分派后各支再校验，不等 → 错误路径）
struct G4SlcRecord {
    uint32_t rec_type = 0;                    // +0x00
    uint32_t ddr_pool = 0;                    // +0x0c (type 0)
    uint32_t sf_offset = 0;                   // +0x10 (type 0)
    uint32_t total_copy = 0;                  // +0x14 (type 0)
    std::vector<G4SlcCopy> copies;            // +0x20 (type 0)
    std::vector<G4SfcdWaitEntry> pairs;       // +0x08 (type 1)
    uint32_t mb_idx = 0;                      // +0x08 (type 2；type 1 元素内)
    uint32_t value_to_set = 0;                // +0x0c (type 2)
};

// 区域级字段: 名字下标@+0x30, i32@+0x38/+0x3c, is_fill 字节@+0x40,
//             is_multi_nsp@+0x41, dma_checkpoint i32@+0x44, 记录向量@+0x60/+0x68
struct G4SlcArea {
    uint64_t area_id = 0;        // 名字 "SLC_spillfill_area_0x%llx" 的十六进制主体
    int32_t  runlist_idx = 0;    // +0x38 侧
    int32_t  nsp_id = 0;         // +0x3c 侧
    bool     is_fill = false;    // +0x40
    bool     is_multi_nsp = false; // +0x41
    int32_t  dma_checkpoint = 0; // +0x44
    std::vector<G4SlcRecord> records;
};

// 序列化产物（顺序即写出顺序）——键表实证于 [0x1294622-0x12947a4]
struct G4TaggedField {
    std::string key;             // 写出时 OR string_tag
    int value_tag = 0;           // 0 = 未核（G4b）；bool=9/0xa；dma_checkpoint=0xe
    bool is_string = false;
    std::string sval;
    uint64_t uval = 0;
};

// 序列化一个区域（区域级键 + 逐记录键）。返回 tagged 字段流；
// 键顺序: SLC_spillfill_area_0x<id>, runlist_idx, memgroup_tags, is_fill,
//         is_multi_nsp, dma_checkpoint, records, nsp_id（memgroup_tags 值编码未核 → 占位 tag=0）
std::vector<G4TaggedField> g4_serialize_slc_area(const G4SlcArea& area);

// ============================================================================
// §C 运行时分配面（fa::FancyAllocator / RuntimeAllocator）
// ============================================================================

constexpr uint32_t G4_SPILLFILL_POOL_ID = 2;   // [0xf4cd91]/[0xf4ce66] ecx=2;
                                              // is_shared_spillfill 亦要求 pool==2 [0xd8b552]
constexpr uint32_t G4_COPYLESS_WEIGHTS_POOL = 0xf; // is_copyless_weights [0xd8b570-0xd8b579]

// PoolDesc: 步长 0x30，size@+0x10，u16 flags@+0x1e（bit0=shared, bit4=far）
constexpr uint16_t G4_POOL_SHARED = 0x1;       // [0xf4ce98 orl $1]
constexpr uint16_t G4_POOL_FAR = 0x10;         // [0xf4cec2 orl $0x10]

struct G4PoolDesc {
    uint32_t size = 0;          // +0x10
    uint16_t flags = 0;         // +0x1e
};

// FancyAllocator 模型（只建 G4 触及的字段）:
//   +0x88 池表（PoolDesc 数组）、+0xa0 spillfill 池句柄、
//   +0xa8/+0xac/+0xb0 三槽尺寸、+0x290 shared 尺寸
struct G4FancyAllocator {
    std::vector<G4PoolDesc> pools;   // +0x88 起，步长 0x30
    uint64_t spillfill_pool = 0;     // +0xa0（池句柄，模型里 = 池表下标+1）
    uint32_t slot_sizes[3] = {0, 0, 0}; // +0xa8/+0xac/+0xb0
    uint64_t shared_size_290 = 0;    // +0x290
};

// set_spillfill_size(u32 sizes[3]) @0xf4ccf0 —— 三槽全展开
// [0xf4cd09/+0, 0xf4cd2d/+4, 0xf4cd53/+8]:
//   sizes[i]==0 跳过；否则原地 64K 取整且 total += 取整值 + 0x10000（每槽多加 64KB）
//   total > 0xFFFFFF00 → throw runtime_error("oversize mem pool")  [0xf4cd75]
//   allocate_new_pool(align=0x10000, total, pool_id=2)；三槽回存 +0xa8/+0xac/+0xb0
// 返回 total。
uint64_t g4_set_spillfill_size(G4FancyAllocator& fa, uint32_t sizes[3]);

// set_spillfill_shared_size(u64) @0xf4ce30:
//   size==0 → 返回 0 [0xf4ce34]；+0x290 = size；sz = 64K 取整；
//   建池 2 描述符，pd.size = sz，flags |= SHARED；
//   env 开关(byte@+0x6143) 且 sz >= env u32(@0x6148) << 20 → flags |= FAR
//   [0xf4cea0-0xf4cec5]。返回 sz。
uint64_t g4_set_spillfill_shared_size(G4FancyAllocator& fa, uint64_t size,
                                      bool env_far_enabled, uint32_t env_far_mb);

// RuntimeAllocator::is_shared_spillfill(pool, flags) @0xd8b550:
//   pool==2 && !(flags & 1) && env_shared_set(+0x5c9c)  [0xd8b552-0xd8b567]
bool g4_is_shared_spillfill(uint32_t pool, uint16_t flags, bool env_shared_set);

// FancyAllocator::can_mempool_be_far(PoolDesc&) @0xf4cee0 —— far 判据谓词版
//   [0xf4cee0-0xf4cefe]
bool g4_can_mempool_be_far(const G4PoolDesc& pd, bool env_far_enabled,
                           uint32_t env_far_mb);

// ---- grdep 调用点的三槽填法（set_spillfill_size@plt @0x1010cc5）-------------
// 多域形态 [0x1010c02 cmpb/je !=0]: 三域各取峰值(+0x28)逐槽 64K 取整
void g4_fill_slots_multi(const uint32_t peaks[3], uint32_t out[3]);
// 单域形态: 三峰值 argmax 选槽——峰值2 严格大于 max(峰值0,峰值1) → 槽 2；
//   否则 (峰值1 > 峰值0) ? 1 : 0 [0x1010c6f-0x1010c8d]；
//   三峰值之和为 0 → 不分配（全 0 返回）[0x1010c64 testl/je]
void g4_fill_slots_single(const uint32_t peaks[3], uint32_t out[3]);

// ============================================================================
// §D 桩与检查点 op
// ============================================================================

// GraphPrepare::insert_spill_fill(vector<tagged>&) @0x106d7d0 —— 永久桩
//   qnndsp_log(0, fmt@0x462adf1, "insert_spillfill.cc", "") 后 return -1
int g4_insert_spill_fill(std::string& log);

// make_dma_checkpoint_op(graph, opid, mb_idx, is_set) @0xd95ac0 —— 记录级模型
//   new 0x18B op；is_set ? vtable 0x5ec2488(set) : 0x5ec2568(wait) [0xd95af2]
//   +0x8 = mb_idx, +0x10 = 0；insert_op(opid, op, owned=false) [0xd95b57]
//   （wait/set-progress 就是图上的检查点 op，与 SFCD 0x80/0x81 类一一对应）
struct G4CheckpointOp {
    uint64_t vtable = 0;      // 0x5ec2488(set) / 0x5ec2568(wait)
    uint32_t mb_idx = 0;      // +0x8
    uint32_t value = 0;       // +0x10（构造时恒 0）
    uint64_t opid = 0;
    bool owned = false;       // insert_op 第 3 参 = false
};
G4CheckpointOp g4_make_dma_checkpoint_op(uint64_t graph, uint64_t opid,
                                         uint32_t mb_idx, bool is_set);

} // namespace hnnx
