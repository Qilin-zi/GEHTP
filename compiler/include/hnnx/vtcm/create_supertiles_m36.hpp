#pragma once
// ============================================================================
// M36 反汇编保真实现：create_supertiles @0x1313ac0 + P2 小函数簇
// 证据: audit_verify/reports/M36_create_supertiles_disasm.md
//       audit_verify/asm/_ZN12GraphPrepare17create_supertilesEv_0000000001313ac0.asm
//       audit_verify/asm/f3/M36_helper_*.asm
// 规则: 每个逻辑段标注 [0x地址]；未在指令级确认的部分显式标注"未完全理解"
//
// 与 src/vtcm/supertile.cpp 的关系：那边是早期 DP 重构（推测性），
// 本文件按 M36 逐指令证据重写，两者并存，冒烟测试只测本文件。
// ============================================================================
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <map>
#include <string>

namespace hnnx {

class GraphPrepare;

// ============================================================================
// Op-id 间接表 —— vector<u32>，真 so 中挂在 GraphPrepare+0x7440
// 每项 4B 覆盖 1024 个 id（id>>10 = 表项下标）
//
// tag u32 位布局 [0x12face0 / 0x12fad60 / 0x12faf20 三函数联合反推]:
//   bit31      溢出/远跳标记
//   bit30      supertile-chunk 标记（bit30+bit31 同置=负 tag → 读下一表项）
//   bits27-29  3-bit 类型
//   bits22-26  5-bit 字段号
//   bits 0-21  22-bit payload
//   (tag & 0xF8000000)==0x78000000 → 相对跳转: 真表项 = idx + (tag & 0xFFFFF)
// ============================================================================
class OpIdTable {
public:
    std::vector<uint32_t> entries;   // 真表 [GraphPrepare+0x7440]

    // resolve_table_index(id) [0x12face0，全指令]
    //   id==0 → 0；idx = id>>10；tag = entries[idx-1]（1 基下标，-0x4(%r8,%rsi,4)）
    //   远跳 tag: (tag & 0xF8000000)==0x78000000 → idx += tag & 0xFFFFF，重读
    //   负 tag (bit30 且 bit31): 越界保护后返回 idx 本身
    //   越界 → 错误路径 0xcf27a0（本实现返回 0 并置 error）
    uint64_t resolve_table_index(uint64_t id, bool* error = nullptr) const;

    // resolve_full_id(id) [0x12fad60，全指令]
    //   同上，但返回 (新表项下标 << 10) | (id & 0x3ff)
    //   ——即把低 10 位 chunk 序号拼到目标表项的空间里 [0x12fad84-0x12fadb9]
    uint64_t resolve_full_id(uint64_t id, bool* error = nullptr) const;

    // extract_field(id) [0x12faf20，全指令]
    //   定位表项后返回 tag & 0x3FFFFF（22 位 payload）
    //   正常 tag 有 bit30 → 返回 payload；负 tag → 读下一表项原值 [0x12faf8b]
    uint32_t extract_field(uint64_t id, bool* error = nullptr) const;

    // compare_ids(a,b) [0x12fadf0，仅前 12B 指令级]
    //   a==b → 0；a!=0 && a!=b → 1；a==0 的分支未 dump（遗留）
    //   本实现只保留已证部分
    static int compare_ids(uint64_t a, uint64_t b);

    // ---- 表构建（供测试/装配）----
    // 真表以 0x2000 字节步长增长（每项 4B）；set_plain 写普通项
    void grow_for(uint64_t max_id);
    void set_plain(uint64_t id, uint32_t tag);
    void set_far(uint64_t id, uint64_t target_id);       // 写 0x78000000|(delta)
    void set_negative(uint64_t id);                      // 写 bit30|bit31 负 tag

    static constexpr uint32_t FAR_MARKER   = 0x78000000; // [0x12fad0e/0x12fad97]
    static constexpr uint32_t NEG_BIT30    = 0x40000000; // [0x12fad31/0x12fadc1]
    static constexpr uint32_t NEG_BIT31    = 0x80000000; // (testl %eax,%eax; js)
    static constexpr uint64_t IDS_PER_ENTRY = 1024;      // id>>10
    static constexpr uint64_t CHUNK_MASK    = 0x3ff;     // [0x12fad84]
};

// ============================================================================
// autothread_size(OpDef*, idx) [0x10c3690，全指令]
//   T = *(u32*)(OpDef->[0x18] + 0x6304)   // 线程数
//   S = *(u32*)(OpDef + 0x50 + idx*8)     // 尺寸字段
//   T >= 2: q=(S+T-1)/T; result=min(round_up(q,M),S)  M=tbl[idx]
//   tbl @0x39b7650 = {1, 8, 8, 32}  （4 项 u32，文件字节实测）
//   T <  2: result = S
//   conditionally_validate_single_quant 的返回值未被使用（已证，不实现副作用）
// ============================================================================
uint32_t autothread_size(uint32_t num_threads, uint32_t size_field, unsigned idx);

// ============================================================================
// insert_spill_fill [0x106d7d0，38B 存根，全指令]
//   本 so 构建里永远 qnndsp_log('%s:17::ERROR:insert_spill_fill not supported\n')
//   并返回 -1。真实现不在此 so 中 —— REQNN 侧只能走参考文档路径。
// ============================================================================
int insert_spill_fill_stub(std::string* log_out = nullptr);

// ============================================================================
// make_dma_checkpoint_op(y, j, b) [0xd95ac0，全指令] —— 描述符级
//   b != 0 → class DmaCheckpointWait, vptr=0x5ec2488
//   b == 0 → class DmaCheckpointSet,  vptr=0x5ec2568
//   （M36 修正：旧文档 Set/Wait 映射写反；typeinfo 经 [vptr-8]→[+8] 追出）
//   序列化槽位: Wait = 0xd96b30/0xd96b60/0xd96b70, Set = 0xd96c70/0xd96ca0/0xd96cb0
// ============================================================================
enum class DmaCheckpointClass : uint32_t {
    Wait = 0x5ec2488u,   // b != 0
    Set  = 0x5ec2568u,   // b == 0
};
struct DmaCheckpointDesc {
    DmaCheckpointClass vtable_class;  // 真值即 vptr，保留 u32 便于比对
    uint32_t param_j;                 // 写入对象 [8] 的 u32
};
inline DmaCheckpointDesc make_dma_checkpoint_op(uint64_t /*y*/, uint32_t j, bool b) {
    // [0xd95ac0]: test %ebp,%ebp; je → b!=0 走 Wait 分支
    return {b ? DmaCheckpointClass::Wait : DmaCheckpointClass::Set, j};
}

// ============================================================================
// make_SyncOp(y) [0xdac440，全指令] —— 描述符级
//   new_id → Op::Op(graph,y) → vptr=0x5ec31f8 (typeinfo=SyncOp)
//   → insert_op(y, {op,false}, take_flag=true) → 返回 (op, id)
//   静态注册 thunk 在 0xdac530 之后（register_op_info + deserialize_op_register,
//   float cost 256.0@0x399db94）——非本函数逻辑
// ============================================================================
struct SyncOpDesc {
    static constexpr uint32_t VTABLE = 0x5ec31f8u;  // [0xdac4a5 附近 mov]
    op_id_t new_id;
};
inline SyncOpDesc make_sync_op_desc(op_id_t y) { return {y}; }

// ============================================================================
// create_supertiles 三阶段 [0x1313ac0，5244B]
//
// Phase 1 [0x1313c58 起] 分组（遍历 grdeps->[0x80] 记录，0xd0 步长，上限
//   grdeps->[0x110]，uVar44=1..count 反编译实证）:
//   记录门: rec[0]!=0 && rec[8]!=0 && rec[0x98]!=0
//   名字黑名单: rec+0x28 (tag 指针) 与 6 个 interned name-tag 比对
//     [0x1313b95-0x1313bc2 六连 cmp]，写者=静态初始化 thunk
//     [0x710806-0x710865 六次 string_tag_t::map_str(literal)]:
//       q::*InputSlicePad / q::*OutputSlice / q::Concat /
//       q::ConvLayer.opt.activations_to_vtcm / q::SlicePad_shape_inplace /
//       q::Slice_contig.tcm   （字面量@0x4617c40/0x462a7b6/0x39b97da/
//                              0x461df6d/0x39bac58/0x55b452d 逐字节验证）
//   资格: (count(gp+0x5fe8) > 1 && rec+0x20 & 0x8) ||
//         (count(gp+0x5fe0) > 1 && (rec+0x20 & 0x110004) == 4)
//     —— 0x8=HMX 位、0x4=HVX 位；与 supertile.cc:138 日志后缀选择一致
//       [0x131375a test $0x4 -> ' (HVX)'; 0x1313768 test $0x8 -> ' (HMX)';
//        否则 ' (error)'; 另有独立非空标志加 ' (autothreaded)']
//   键: pair<resolved_id(经 0x12fad60), tag 指针> -> custom_vec<u32>(run 位置)
//   分组异常 -> ERROR 406: '%s:406::ERROR:unexpected value size for grouping
//                          <%zu, %s>\n' @0x55b4541（supertile.cc:406，逐字节验证）
//   日志: '%s:138:Combine %zu tiles into supertile 0x%llx %s%s%s:\n'
//         @0x55b44b0，DEBUG(4)，GetLogPriorityLevel()>=4 才发 [0x131379a]
//
// Phase 2 [0x1313efc-0x13141a0] 组内整理:
//   空组 -> ERROR；单元素组 -> erase；多元素 -> 按 resolved id stable_sort
//   （>=129 元素走 nothrow-halving scratch —— 分配策略，不影响序）
//
// Phase 3 [0x131415e-0x13143ce] 建 SuperTileOp + chunk 切分:
//   预算: miss -> budget = gp[0x5fd8] << 10; hit(哈希表 gp+0x6e40/0x6e48,
//         命中条件 entry[8]==hash && entry[0x10]==hash) -> gp[0x74c0]
//   张量尺寸累加: vtable +0x60/+0xa0 与 tensor+0xd0
//   维度/层级取法: 0x12fa730 —— tag->{type,value,seq} 三元组流构建器
//     （每层 12B 槽 3x u32 [0x12fa901 lea (%rax,%rax,2)]，见
//      SupertileTagTriple/decode_tag_triples；追加原语 0x12fb8a0 [0x12fa8de]）
//   除数搜索: cnt 从候选向下递减直到 (c*b) % cnt == 0
//   chunk id: (entry_index << 10) | chunk_seq，循环内 id+1 且 & 0x3ff 回绕、
//             跳过 0x400 倍数（skip 0x7ff 位置）
//   grdep 登记: fc5910/fc5af0（本文件 SupertileDepReg）
// ============================================================================

// 0x12fa730 尾部实证 [0x12fa880-0x12fab07]：沿 id 链逐层产出的三元组。
//   根三元组（本 id 表项 tag，不看符号位）:
//     type  = (tag >> 27) & 7;  value = (tag & 0x7FFFFFF) + 1  [0x12fa89d/0x12fa8f2/0x12fa8f6]
//     链种子 next = entries[idx] 原值 [0x12fa896]（半证: 加载点在 dump 外）
//   链循环每步: seq = id & 0x3ff（远跳补 seq |= delta<<10 [0x12fa9ae]）;
//     type = (tag >> 27) & 7           [0x12fa93a-0x12fa93d]
//   value 两种形态:
//     正 tag (bit31 清): (tag>>22 & 0x1f) + 1            [0x12fa9ec->0x12fa937]
//     负 tag (bit31 置): (tag & 0x7FFFFFF) + 1           [0x12fa930]
//   链式续追: bit30 置且正 -> next = tag & 0x3FFFFF (22 位) [0x12fa9e4-0x12fa9f4]
//   bit30+bit31 同置 -> 越界查后 next = entries[idx] 原值  [0x12fa9fc-0x12faa09]
//   停链:   next == 0 或终态 tag（bit30 清）
//   遗留: 追加原语 0x12fb8a0 内部；真代码先数链长再倒序写槽
//         [0x12faa68-0x12fab12 negl ebp / 0x12fa8fe]，物理次序与本实现访问序
//         相反（已声明差异）；尾部结果记录 {终值, 终表项下标} [0x12faa0e-0x12faa40] 未建模
struct SupertileTagTriple {
    uint32_t type;   // 3-bit
    uint32_t value;  // 字段号+1 或 payload+1（见上）
    uint32_t seq;    // 链上 id 低 10 位
};
// 沿链产出三元组（追加到 out），返回条数；*error 置位表示链越界中断
size_t decode_tag_triples(const OpIdTable& t, uint64_t id,
                          std::vector<SupertileTagTriple>& out, bool* error = nullptr);
struct SupertileChunk {
    uint64_t chunk_id;       // (entry_index<<10)|seq
    std::vector<op_id_t> member_ops;
    uint64_t tensor_bytes = 0;
};

struct SupertileGroupResult {
    uint64_t resolved_id = 0;
    uint32_t tag = 0;
    std::vector<SupertileChunk> chunks;
    uint64_t budget = 0;
};

// grdep 侧两个 static（物理上落在 build_graph_deps 区间，objdump 标签错觉）:
//
// fc5910(this, rec) 依赖记录有效切片数 [全指令]:
//   flags&0x200000 (supertiled): v=get_supertiled_ops_info; bytes>0x1f 时
//     rec[0x2]&0x40 → 逐元素累加 self_slicing_num_slices [0xfc5990]
//     否则 → bytes/16 [0xfc5975]；空/短 → throw runtime_error(
//     "num_internal_threads") [0xfc59a9, 串@0x4623207]
//   flags&0x400000 → 尾调 self_slicing_num_slices [0xfc5945]
//   否则 1
struct SupertileDepRecord {
    uint32_t flags = 0;          // bit21(0x200000)=supertiled, bit22(0x400000)=self-slicing
    uint8_t  extra_bits_2 = 0;   // rec+2 字节，bit6(0x40) → 逐元素累加
    op_id_t  op = 0;
};
uint32_t dep_effective_num_slices(const SupertileDepRecord& rec,
                                  const std::vector<uint32_t>& chunk_slice_counts,
                                  uint32_t self_slicing_count);

// fc5af0(grdeps, rec, count, id, flag) supertile per-chunk 依赖注册
//   [0xfc5b1a] count<=1 早退
//   [0xfc5b42-0xfc5b5a] oprec(rec->idx-1)*0xd0 + 0x20 处 flags & 0xc 必须
//     是 4 或 8（幂次校验），否则 qnndsp_log('%s:3457::ERROR:#%d does not
//     have valid resource flags for supertile\n', 'grdep_main.cc'@0x46229f9)
//   [0xfc5ba6] operator new(count*8) 建 u64 数组
//   尾部（M36b 补证）: 数组先三段 memset-0 清零 [0xfc5c50 主循环 32 项/迭代 /
//     0xfc5ce0 余段 / 0xfc5d00 标量尾]，再按成员 id 列表逐项走解析链
//     0xf9dda0（返回向量大小必须 == i+1 的不变式 [0xfc5e03-0xfc5e1e]），
//     过 grdeps 记录 flags 门（&0x200 [0xfc5e2a] / &0xf==5 [0xfc5e3a] /
//     &0x200000 [0xfc5e43] / &0x100400 [0xfc5e5b]）。
//     精确槽值公式依赖 0xf9dda0 —— 归入 build_graph_deps 里程碑；
//     本实现按"chunk_id 起连续 count 个"填充（已声明差异）。
struct GrdepOpRecord {           // 真布局: grdeps->[0x80]，0xd0 步长
    uint64_t resource_flags = 0; // 记录 +0x20 字段
};
enum class SupertileDepRegStatus { Ok, EarlyExit, BadResourceFlags };
SupertileDepRegStatus register_supertiled_dep(
    std::vector<GrdepOpRecord>& op_records, uint32_t rec_op_index,
    uint32_t count, uint64_t chunk_id,
    std::vector<std::vector<uint64_t>>& chunk_id_store);

// ---- 主入口（记录级；装配方把 REQNN OpDef 图映射进来）----
struct SupertileCandidate {
    op_id_t op_id;
    uint32_t tag;             // 分组 tag（Phase1 pair 的第二元）
    uint64_t tensor_bytes;    // Phase3 尺寸累加的该项
    // Phase1 记录门与资格（M36b 修正）:
    const char* op_name = "";         // rec+0x28 的 name-tag（黑名单比对用）
    uint32_t resource_flags = 0;      // rec+0x20（0x8=HMX 位，0x4=HVX 位）
    bool record_valid = true;         // rec[0]!=0 && rec[8]!=0 && rec[0x98]!=0
};
// 6 个永不入 supertile 的名字（静态初始化 thunk [0x710806-0x710865] 的
// map_str 字面量，逐字节验证）—— 装配方默认注入
inline const char* const k_supertile_name_blacklist[6] = {
    "q::*InputSlicePad",
    "q::*OutputSlice",
    "q::Concat",
    "q::ConvLayer.opt.activations_to_vtcm",
    "q::SlicePad_shape_inplace",
    "q::Slice_contig.tcm",
};
struct CreateSupertilesConfig {
    uint64_t budget_default = 0;    // gp[0x5fd8]<<10
    uint64_t budget_hash_hit = 0;   // gp[0x74c0]
    // Phase1 名字黑名单（默认 = k_supertile_name_blacklist；传 {""} 禁用）
    std::vector<const char*> blacklist_names =
        {k_supertile_name_blacklist, k_supertile_name_blacklist + 6};
    // Phase1 资格 [反编译 lVar39 循环实证]:
    //   (hmx_count > 1 && resource_flags & 0x8) ||
    //   (hvx_count > 1 && (resource_flags & 0x110004) == 4)
    uint64_t hmx_count = 0;         // gp+0x5fe8
    uint64_t hvx_count = 0;         // gp+0x5fe0
    // Phase3 哈希命中表（gp+0x6e40，命中条件 entry[8]==hash && entry[0x10]==hash，
    // 哈希键 = id>>10）。此处以命中 id 集合注入；命中 → budget_hash_hit
    std::vector<uint64_t> hash_hit_ids;
};
struct CreateSupertilesStats {
    size_t groups_in = 0, groups_kept = 0, chunks_out = 0;
    std::vector<std::string> errors;   // ERROR 406 等
};

// 三阶段主体。op 表只读；输出分组结果与 chunk 划分。
std::vector<SupertileGroupResult> create_supertiles_disasm(
    const OpIdTable& id_table,
    const std::vector<SupertileCandidate>& ops,
    const CreateSupertilesConfig& cfg,
    CreateSupertilesStats* stats = nullptr);

// Phase3 除数搜索 [0x131415e-0x13143ce 区间]: cnt 从候选向下递减直到
// (c*b) % cnt == 0。c=每 chunk 张量字节上限基准, b=成员数。
uint32_t supertile_find_divisor(uint64_t total_bytes, uint32_t member_count,
                                uint32_t cnt_start);

// Phase3 chunk id 步进 [0x12fa220 建组 + create 循环]: id+1 & 0x3ff 回绕，
// 跳过 0x400 倍数（0x7ff 之后回 0）。返回推进后的 chunk seq。
uint32_t supertile_next_chunk_seq(uint32_t seq);

} // namespace hnnx
