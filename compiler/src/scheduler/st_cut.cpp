// ============================================================================
// st_cut.cpp — libHtpPrepare.so「stcut 调度/切分」管线重实现（M17–M30 反汇编结论）
//
// 源对象: real_so/libHtpPrepare.so（GraphPrepare::schedule_for_alloc @0x1302900，
//         源文件 st_cut.cc——由日志格式串 "%s:3101:…"/"%s:3331:…" 与计时名
//         "stcut_full_schedule"/"stcut_measure_peak"/"stcut_delay_dma_again" 自证）
//
// 调用全景（重实现入口 = 文件尾 stcut_full_schedule）：
//   schedule_for_alloc @0x1302900
//     建图段（早期阶段，本文件不重放其内部，见 stcut_build_initial_state 注）:
//       read_nodes@0x12fc820 → collate_sibs@0x12fd0e0 → relate_by_tensor
//       → node_hash@0x1300210 → quick_early_sort@0x1300600 → block_relate@0x1300a40
//       → stcut_connect_nodes@0x1300cc0 → add_dependencies@0x1305130
//       → strong_relevel@0x12fd600 → arrange_sibs@0x12fffd0 → clean_sibs@0x13056f0
//       → delay_dma@0x13065f0（=本轮评分/峰值）
//     full_schedule 重试循环 @1303ee0–13044c5（M27 §1/§3，逐指令实证）
//     → 恢复最优序@13044cb → delay_dma_again@0x12fc740 → convert_to_ids（内联未划出）
//
// 实现约定：
//   * 每个函数头注释 = 反汇编地址 + 里程碑报告；每段逻辑行内 [0x地址] 对位；
//   * .so 用 rdtsc/16 计时（M23 §3.4 双采样净效果 /16）→ StCutContext::tick()；
//   * 「反汇编未完全理解」段：按已证契约实现，未证细节显式标注，不臆断。
// ============================================================================

#include "hnnx/scheduler/st_cut.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <set>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace hnnx {

// ============================================================================
// 计时 / RNG 原语
// ============================================================================

// .so: 1309860 rdtsc → 130986d >>4；1309871 rdtsc → 1309882 (TSC>>3)&~1（双采样，净效果 /16）
uint64_t StCutContext::tick()
{
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc() >> 4;                 // [1309862-130986d TSC>>4]
#else
    // 非 x86 宿主的等价替身（原对象仅 x86-64）
    return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) >> 4;
#endif
}

// .so: 130df50 门内 0xd79830 rand@plt，状态 = ctx+0x2a8（M29 §0 表）
// 反汇编未完全理解：0xd79830 的 PRNG 算法未拆；此处以同一状态字的 LCG 等价实现，
// 只保证「取模消费」这一控制流形状。
uint64_t StCutContext::rand_next(StCutContext &ctx)
{
    ctx.rng_state = ctx.rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (ctx.rng_state >> 33) & 0x7fffffff;   // 31bit，与 rand() 值域一致
}

// ============================================================================
// M31-3 基础层
// ============================================================================

// ---- 0x1307780 — ctx 阶段快照：五表拷贝 + [ctx+0xc]=0（M24 §0 表）----
// 五表对应：+0x190←+0x68（节点槽）、+0x1a8←+0x80（组标志）、+0x1c0←+0xb0（权重表）、
//           +0x1f0←+0xe0（add 集）、+0x208←+0xf8（subtract 集）
// guard [ctx+0x220]：仅首轮拍（M27 §1 注 1306782 调用点）
void stcut_snapshot_five(StCutContext &ctx)
{
    ctx.snap_nodes = ctx.node_slots;       // → +0x190
    ctx.snap_flags = ctx.group_flags;      // → +0x1a8
    ctx.snap_weights = ctx.weight_table;   // → +0x1c0
    ctx.snap_add = ctx.add_set;            // → +0x1f0
    ctx.snap_subtract = ctx.subtract_set;  // → +0x208
    ctx.mode = 0;                          // [1307812 movl $0] 复用期（0x1300cc0/0x1301040 消费）
    ctx.snapshot_taken = true;             // [ctx+0x220 guard 置位]
}

// ---- 0x1307170 — 五表还原（工作表 ← 快照表）（M28）----
void stcut_restore_five(StCutContext &ctx)
{
    ctx.node_slots = ctx.snap_nodes;       // +0x68 ← +0x190
    ctx.group_flags = ctx.snap_flags;      // +0x80 ← +0x1a8
    ctx.weight_table = ctx.snap_weights;   // +0xb0 ← +0x1c0
    ctx.add_set = ctx.snap_add;            // +0xe0 ← +0x1f0
    ctx.subtract_set = ctx.snap_subtract;  // +0xf8 ← +0x208
}

// ---- 0x130ccd0 — 权重树懒初始化 = grain count 写入方（M29 §5）----
// guard [ctx+0x18] 置 1（一次性）；遍历 [ctx+0x68] 节点表建树（key=节点号），
// count = Σ_{rel ∈ ctx+0xe0[id] add 集} ctx+0xb0[rel].@8（grain 量纲权重）。
// 写一次后只读（0x130cea0/0x13065f0 只消费）。树本体 ctx+0x20（根@0x28/大小@0x30，
// 48B 节点 {left,right,POD}，0x868c50 再平衡）→ 以 std::map 等价。
void stcut_grain_tree_build(StCutContext &ctx)
{
    if (ctx.tree_built) return;            // [guard ctx+0x18]
    ctx.tree_built = true;
    for (uint32_t id = 0; id < ctx.node_slots.size(); ++id) {
        uint64_t count = 0;
        for (uint32_t rel : ctx.add_set[id])
            count += ctx.weight_table[rel].weight;   // ctx+0xb0[rel].field@+8
        ctx.grain_tree[id] = count;
    }
}

// ---- 权重树查询（0x130cea0 主循环的 find-or-insert 防御路径：未命中 count=0）----
uint64_t stcut_grain_of(StCutContext &ctx, uint32_t id)
{
    stcut_grain_tree_build(ctx);
    auto it = ctx.grain_tree.find(id);
    if (it == ctx.grain_tree.end()) {      // [_Znwm(0x30) 防御插入 count=0，M29 §4]
        it = ctx.grain_tree.emplace(id, 0).first;
    }
    return it->second;
}

// ============================================================================
// M31-4 建图层
// ============================================================================

// ---- 0x1307820 — add 集追加（M24 §3.1）----
// .so 为手写 push_back：快路径 130784f；扩容 1307868 max(size+1, 2x)。
// std::vector::push_back 与之等价（size≥1 时同 2x）。
void stcut_add_push(StCutContext &ctx, uint32_t node, uint32_t wslot)
{
    if (node >= ctx.add_set.size()) ctx.add_set.resize(node + 1);
    ctx.add_set[node].push_back(wslot);    // [130784f 快路径 / 1307868 扩容]
}

// ---- 0x1307960 — subtract 集追加（M24 §3.2，与 0x1307820 逐指令同构）----
void stcut_sub_push(StCutContext &ctx, uint32_t node, uint32_t wslot)
{
    if (node >= ctx.subtract_set.size()) ctx.subtract_set.resize(node + 1);
    ctx.subtract_set[node].push_back(wslot);   // [130798f 快路径 / 13079a8 扩容]
}

// ---- 0x1307f60 — 24B 账本记录 push_back，返回新记录指针（M24 §3.3）----
// 扩容 1307f9b：旧容量 = 字节×0xAAAAAAAAAAAAAAAB(/3 魔数)，新容量 max(旧+1,2x)，
// 上限 0x555555555555555；130807a 返回 end−0x18（调用方可就地改写）。
// M34 破译：rdi = 账本对象指针（connect 的 cb 即此物），非 ctx 全局账本——
// 脚手架记账入调用者局部账本（0x1309a60 rsp+0x90），一次试切内整体撤销。
StCutReplayRec *stcut_replay_push(std::vector<StCutReplayRec> &ledger,
                                  const StCutReplayRec &r)
{
    ledger.push_back(r);
    return &ledger.back();                 // [130807a-130807e]
}

// ---- 0x1301040 — 节点注册（M24 §3.5，[ctx+0xc] 消费者）----
// [1301068 cmpl $0,0xc(%rdi)]：mode≠0 → 追加路（节点指针 push ctx+0x68、
//   flags|=mode 后 push ctx+0x88 字节账本 [13011af-13011c6]）；
// mode==0 → 复用路：pop [ctx+0x160] 节点槽空闲栈 [13010ad]，
//   [ctx+0x68][slot]=节点 [13010c4]、[ctx+0x80][slot]=flags [13010cb]、
//   [ctx+0x128][slot]=0 三态复位 [13010d6]。返回槽索引。
// ★asm 复核（13010a4-13010ab）：复用路入口先 cmp begin/end——空闲栈为空时
//   je 0x130106e 落回追加路。空栈无条件 pop 是 UB（实现期冒烟测试挂点）。
uint32_t stcut_register_node(StCutContext &ctx, uint32_t node, uint32_t flags)
{
    uint32_t slot;
    if (ctx.mode != 0 || ctx.free_node_slots.empty()) {  // [1301068 / 13010a4-13010ab]
        ctx.node_slots.push_back(node);               // [130106e push ctx+0x68]
        ctx.group_flags.push_back((uint8_t)(flags | ctx.mode)); // [13011af orl 0xc → 13011c6]
        slot = (uint32_t)ctx.node_slots.size() - 1;
        // [13012a9 r12=旧节点数+1] 级联 resize 到新节点数：u16 标记表 +0x128
        // [13012ad-13012dd]、两张 24B 元素表 +0xe0/+0xf8 [13012e4-130133c]、
        // cand_tables +0x240 [13013bc-1301416（8B 槽，grow→0x13117f0）]。
        ctx.add_set.resize(ctx.node_slots.size());
        ctx.subtract_set.resize(ctx.node_slots.size());
        ctx.mark_table.resize(ctx.node_slots.size(), 0);
        ctx.cand_tables.resize(ctx.node_slots.size()); // [1301416 movq %rax,0x248(%r15)]
    } else {                                          // [130109d 复用路（栈非空）]
        slot = ctx.free_node_slots.back();            // [13010ad pop ctx+0x160]
        ctx.free_node_slots.pop_back();
        ctx.node_slots[slot] = node;                  // [13010c4]
        ctx.group_flags[slot] = (uint8_t)flags;       // [13010cb]
        ctx.mark_table[slot] = 0;                     // [13010d6 movw $0]
    }
    return slot;
}

// ---- 0x1307aa0 — 节点槽退役（M24 §3.4）----
void stcut_retire_node(StCutContext &ctx, uint32_t slot)
{
    ctx.add_set[slot].clear();             // [1307aca end=begin 清空不释放]
    ctx.subtract_set[slot].clear();        // [1307ada]
    ctx.node_slots[slot] = 0;              // [1307adf 节点指针清空]
    ctx.group_flags[slot] = 0x83;          // [1307af2 movb $0x83]
    ctx.free_node_slots.push_back(slot);   // [1307b06 push → ctx+0x160]
}

// ---- 0x1307c90 / 0x1307d90 — 替换对重指向 A/B 侧（M24 §3.6；唯一调用方 0x130bc70）----
// A 侧 [1307d7d]：weight_table[slot].canon = new_partner
void stcut_repoint_a(StCutContext &ctx, uint32_t wslot, uint32_t new_partner)
{
    ctx.weight_table[wslot].canon = new_partner;   // [1307d7d field0=newPartner]
}

// B 侧 [1307e98]：对侧（idx 链）槽的 canon = new_partner（镜像）
void stcut_repoint_b(StCutContext &ctx, uint32_t wslot, uint32_t new_partner)
{
    uint32_t far = ctx.weight_table[wslot].idx;    // [@4 链]
    ctx.weight_table[far].canon = new_partner;     // [1307e98 对侧 field0=newPartner]
}

// ---- 0x1300cc0 — stcut_connect_nodes：替换对记录核心（M23 §2 全解）----
// 签名: u32(ctx, arg2=src, arg3=dst, threshold, callback)；返回映射到 arg3 的槽 A。
// M34 破译：cb = 调用者账本（vector 指针）——命中路推 {A, 旧权重, type0}（重放
// 恢复权重），建对路推 {A, 0, type2}（重放从四条 per-node 链表整体撤除建对）。
uint32_t stcut_connect_nodes(StCutContext &ctx, uint32_t src, uint32_t dst,
                             uint64_t threshold, std::vector<StCutReplayRec> *cb)
{
    // [1300cec-1300d29] src==dst 时打日志（优先级1）——实现略（纯日志）
    // [1300d31-1300d77] 在 add集[src] 里找 weight_table[槽].canon == dst
    for (uint32_t ebx : ctx.add_set[src]) {
        if (ctx.weight_table[ebx].canon == dst) {
            // ---- 命中路径（已存在替换对 → 累加权重）[1300fc9-1301022] ----
            if (cb != nullptr && threshold != 0) {         // [1300fc9/1300fce 两门]
                StCutReplayRec r;
                r.op = ebx;                                 // [1300fd6 record.op = ebx]
                r.value = ctx.weight_table[ebx].weight;     // [1300fda record.value = 旧weight]
                r.type = 0;                                 // [1300fe4 type=0 置权重]
                stcut_replay_push(*cb, r);                  // [1300ff9 call 0x1307f60(cb, &rec)]
            }
            uint64_t w = threshold + ctx.weight_table[ebx].weight;   // [1301008 add]
            if (w > STCUT_MAX_WEIGHT) w = STCUT_MAX_WEIGHT;          // [130100d-1301019 min 0x5f5e0ff]
            ctx.weight_table[ebx].weight = w;                         // [130101d]
            return ebx;                                     // [1300fb8 返回 ebx]
        }
    }

    // ---- 未命中 → 建立新替换对，两路径由 [ctx+0xc] 门控 [1300d79] ----
    uint32_t a, b;
    if (ctx.mode == 0) {
        // 快路径 [1300d85-1300e26]：从 [ctx+0x178] 权重槽空闲栈弹出 2 个复用
        // [1300d96-1300d9d] 栈内 <2 元素 → 转慢路径
        if (ctx.free_weight_slots.size() < 2) {
            // 落入慢路径（见下）
            goto slow_path;
        }
        b = *(ctx.free_weight_slots.end() - 2);   // [1300da7 ebp = output[count-2]（B）]
        a = *(ctx.free_weight_slots.end() - 1);   // [1300dab ebx = output[count-1]（A）]
        ctx.free_weight_slots.resize(ctx.free_weight_slots.size() - 2); // [1300daf-1300db3 弹 2]
        ctx.weight_table[a] = { dst, b, threshold, 0 };            // [1300dc5-1300dd2 A{canon=dst,idx=B,w,flags=0}]
        ctx.threshold_ledger[a] = 0;                               // [1300dda-1300de1]
        ctx.weight_table[b] = { src, a, STCUT_MAX_WEIGHT, ctx.mode | 4u }; // [1300e02-1300e13 B{canon=src,idx=A,w=0x5f5e0ff,flags=mode|4}]
        ctx.threshold_ledger[b] = 0;                               // [1300e17-1300e1e]
    } else {
slow_path:
        // 慢路径 [1300e2b-1300ed3]：追加 2 槽（0x1308b10 24B push ×2 + 0x12fc610 u64 push ×2）
        a = (uint32_t)ctx.weight_table.size();     // [1300e43-1300e4d 新索引 = 槽数]
        b = a + 1;                                 // [1300e4d lea 0x1(%rbx)]
        ctx.weight_table.push_back({ dst, b, threshold, ctx.mode });          // [1300e6f slot A]
        ctx.threshold_ledger.push_back(0);                                    // [1300e7b-1300e8c]
        ctx.weight_table.push_back({ src, a, STCUT_MAX_WEIGHT, ctx.mode | 4u }); // [1300eb8 slot B]
        ctx.threshold_ledger.push_back(0);                                    // [1300ebd-1300ed3]
    }

    // ---- 公共尾 [1300ed8-1300f07]：四向建链 ----
    stcut_add_push(ctx, src, a);          // [1300ee0 0x1307820(ctx, arg2, A)]
    stcut_add_push(ctx, dst, b);          // [1300ee5 0x1307820(ctx, arg3, B)]
    stcut_sub_push(ctx, src, b);          // [1300ef2 0x1307960(ctx, arg2, B)]
    stcut_sub_push(ctx, dst, a);          // [1300eff 0x1307960(ctx, arg3, A)]

    // ---- [ctx+0x110] 位图同步到槽数 [1300f0c-1300f75] ----
    if (ctx.bitmap.size() * 32 < ctx.weight_table.size())
        ctx.bitmap.resize((ctx.weight_table.size() + 31) / 32, 0);
    else if (ctx.bitmap.size() > (ctx.weight_table.size() + 31) / 32)
        ctx.bitmap.resize((ctx.weight_table.size() + 31) / 32);

    // ---- 回调 [1300f7c-1300fa3]：记录 {A, 0, type=2}（撤销建对：四链表撤除）----
    if (cb != nullptr) {
        StCutReplayRec r;
        r.op = a; r.value = 0; r.type = 2;         // [1300f84-1300f91]
        stcut_replay_push(*cb, r);                 // [1300fa3 call 0x1307f60(cb, &rec)]
    }
    return a;                                      // [1300fb8 返回 slot A]
}

// ---- 0x13080a0 — 6-case 账本重放（M22 §3 + M34 逐指令复核；倒序弹出）----
// 24B 记录 {op@0, value@8, type@0x10}；分发表 0x55b3d04（movslq 相对跳转，type≤5）：
//   type0 [130811f]：weight_table[op].weight = value（恢复累加前旧值）；
//   type1 [1308136]：日志（GetLogPriorityLevel≥100 才打印，字符串 0x55b42b0）；
//   type2 [1308151]：撤销建对——A、B 双槽从四条 per-node 链表 swap-remove：
//     A 从 add_set[B.canon]（扫描1 1308178-1308232）与 subtract_set[A.canon]
//       （扫描4 13082f0-1308335）移除；
//     B 从 add_set[A.canon]（扫描2 1308252-130829d）与 subtract_set[B.canon]
//       （扫描3 13082a0-13082ed）移除。权重表槽位保留（不弹）。
//   type3 [13080d4]：弹计数（[ctx+0x70]-=8 节点表弹 1 + [ctx+0x88]-=1 标志弹 1）；
//   type4 [13081a8]：重建建对——type2 的精确逆：A→add_set[B.canon]（push1 13081ef）、
//     B→add_set[A.canon]（push2 130842a）、B→subtract_set[B.canon]（push3 1308561）、
//     A→subtract_set[A.canon]（push4 130868f）；
//   type5 [13081fe]：日志（优先级≥100，字符串 0x55b4288）。
// 槽位归属不变式（由 type2/type4 的四向操作反推）：A{canon=后继} ∈ add_set[前驱]+
// subtract_set[后继]；B{canon=前驱,&4} ∈ add_set[后继]+subtract_set[前驱]。
void stcut_replay_apply(StCutContext &ctx, std::vector<StCutReplayRec> &log)
{
    auto list_remove = [](std::vector<uint32_t> &v, uint32_t slot) {   // swap-with-last
        for (size_t j = 0; j < v.size(); ++j)
            if (v[j] == slot) { v[j] = v.back(); v.pop_back(); return; }
    };
    for (size_t i = log.size(); i-- > 0; ) {       // 反序弹（重放 = 撤销）
        StCutReplayRec &r = log[i];
        if (r.op >= ctx.weight_table.size()) continue;   // 防御：槽越界（.so 无此门）
        const uint32_t a = r.op;
        const uint32_t b = ctx.weight_table[a].idx;
        const uint32_t host_b = ctx.weight_table[a].canon;   // A.canon（B 的对端宿主）
        const uint32_t host_a = ctx.weight_table[b].canon;   // B.canon（A 的对端宿主）
        switch (r.type) {
        case 0:
            ctx.weight_table[r.op].weight = r.value;
            break;
        case 1:
        case 5:
            break;                                 // 纯日志（优先级≥100/常规）
        case 2:
            if (host_a < ctx.add_set.size())  list_remove(ctx.add_set[host_a], a);      // 扫描1
            if (host_b < ctx.add_set.size())  list_remove(ctx.add_set[host_b], b);      // 扫描2
            if (host_b < ctx.subtract_set.size()) list_remove(ctx.subtract_set[host_b], b); // 扫描3
            if (host_a < ctx.subtract_set.size()) list_remove(ctx.subtract_set[host_a], a); // 扫描4
            break;
        case 3:
            if (!ctx.node_slots.empty()) ctx.node_slots.pop_back();      // [ctx+0x70]-=8
            if (!ctx.group_flags.empty()) ctx.group_flags.pop_back();    // [ctx+0x88]-=1
            break;
        case 4:
            if (host_a < ctx.add_set.size())  stcut_add_push(ctx, host_a, a);           // push1
            if (host_b < ctx.add_set.size())  stcut_add_push(ctx, host_b, b);           // push2
            if (host_b < ctx.subtract_set.size()) stcut_sub_push(ctx, host_b, b);       // push3
            if (host_a < ctx.subtract_set.size()) stcut_sub_push(ctx, host_a, a);       // push4
            break;
        default:
            break;                                 // 反汇编未完全理解：超出 0-5 的 type 未见过
        }
    }
    log.clear();
}

// ---- 0x1309230 — 组可达洪泛收集（M24 §1 全解）----
// 从 seed 沿 add 集逐代洪泛，收集「权重>阈值账本且组标志允许」的槽；返回收集计数。
// 0x1309810 循环 B 以返回值≠0 判断「还有可搬运对象」。
uint64_t stcut_flood_collect(StCutContext &ctx, uint32_t seed, uint32_t target)
{
    // [130926d/1309285] 两表清零
    std::fill(ctx.mark_table.begin(), ctx.mark_table.end(), (uint16_t)0);
    std::fill(ctx.bitmap.begin(), ctx.bitmap.end(), (uint32_t)0);

    std::vector<uint32_t> frontier = { seed };     // [13092b4-13092c0 _Znwm(4) 单元素]
    std::vector<uint32_t> result;
    ctx.mark_table[seed] = 1;                      // [13092e9 三态表[seed]=1]
    uint32_t generation = 2;                       // [13092fc 代计数=2]
    uint64_t count = 0;                            // r8
    bool touched = false;                          // r11d / [rsp+0x38]

    while (!frontier.empty()) {                    // [1309330 逐代循环]
        result.clear();
        for (uint32_t ecx : frontier) {
            if (ctx.mark_table[ecx] != 1) continue;    // [1309352 cmpw $1 → ja 跳过]
            ctx.mark_table[ecx] = (uint16_t)generation; // [130935d 改写代号防重入]
            for (uint32_t ebp : ctx.add_set[ecx]) {     // [130936c add集[ecx] 逐成员]
                ctx.inner_loop_count++;                 // [13093f1 addq $1,0x2a0]
                uint32_t r15 = ctx.weight_table[ebp].canon;  // [1309404 field0]
                if (ctx.mark_table[r15] != 0) continue;      // [1309408 tag==0 才进]
                if (ctx.threshold_ledger[ebp] >= ctx.weight_table[ebp].weight)
                    continue;                                // [130941b 阈值>=权重 → 无可取]
                if ((ctx.group_flags[r15] & 0x2) && r15 != target)
                    continue;                                // [1309429/1309430 bit0x2 置位须 field0==target]
                result.push_back(r15);                       // [1309388-13093b8 结果 push（扩容内联 1309443-130958e）]
                ctx.mark_table[r15] = 1;                     // [13093bb 下一代 frontier]
                ctx.bitmap[ebp >> 5] |= 1u << (ebp & 31);    // [1309433 位图置位]
                ++count;                                     // r8++
                touched = touched || (r15 == target);        // r11d |= (field0==target)
            }
        }
        if (touched) break;                          // [13095a0 testb $0x1 → 触达 target 即停]
        ++generation;                                // [13095c3 代号++]
        frontier.swap(result);                       // [13095c8-13095db 两 vector 互换]
    }
    return count;                                    // [130962f mov %r8,%rax]
}

// ---- 0x13096b0 — 递归容量受限权重搬运（M24 §2 全解）----
// 自 seed 递归向下游抽取至多 cap 的权重，记入阈值账本（成员 +，其 idx 对槽 −）；
// 单步 <10000 提前收敛 [13097d0]。
// ★asm 更正（M24 §2 笔误，实现期直接复核 13096ca-13096d3 确证）：
//   seed==target 时 [13096ce movq %r8,0x10(%rsp)] = 写入第 4 参 cap 后直接跳返回
//   [13097ed]——返回 cap（终点吸收全部请求量），非 0。空 add 集 [13097e6] 才返 0
//   （xorl %eax,%eax → [rsp+0x10]）。若按返 0 实现，0x1309810 的循环 A/循环 B
//   将互喂死循环（搬运永不落账 → 洪泛门永开）。
uint64_t stcut_weight_pull(StCutContext &ctx, uint32_t seed, uint32_t target,
                           uint64_t cap, uint32_t depth)
{
    if (seed == target) return cap;                  // [13096ca cmp / 13096ce 吸收]
    if (seed >= ctx.add_set.size() || ctx.add_set[seed].empty()) return 0; // [13096e8-13096f9]
    ++depth;                                         // [13096ff 递归深度计数]
    uint64_t acc = 0;                                // [130970d 累计器=0]
    for (uint32_t ebx : ctx.add_set[seed]) {         // [1309742-13097e1 循环]
        if (ebx / 32 >= ctx.bitmap.size() ||
            !((ctx.bitmap[ebx >> 5] >> (ebx & 31)) & 1u))
            continue;                                // [130975b bt 位图=0 → 跳过]
        uint64_t avail = ctx.weight_table[ebx].weight - ctx.threshold_ledger[ebx]; // [1309772-1309777]
        uint64_t step = (avail >= cap) ? cap : avail;  // [130977b-1309783 cmovge → min]
        uint32_t canon = ctx.weight_table[ebx].canon;  // [130978c field0]
        uint64_t got = stcut_weight_pull(ctx, canon, target, step, depth); // [130979e 递归]
        if (got == 0) {                              // [13097a6 返回0 → 清位继续]
            ctx.bitmap[ebx >> 5] &= ~(1u << (ebx & 31));
            continue;
        }
        ctx.threshold_ledger[ebx] += got;            // [13097b3 账本[ebx] += 搬运量]
        acc += got;                                  // [13097b7 累计器 +=]
        uint32_t idx = ctx.weight_table[ebx].idx;    // [13097c8 @4]
        ctx.threshold_ledger[idx] -= got;            // [13097cc 账本[idx] −=]
        if (step < STCUT_SMALL_STEP) return acc;     // [13097d0 <10000 提前返回]
        cap -= got;                                  // [13097d9 cap 用尽额度]
        if (cap == 0) break;
    }
    return acc;
}

// ---- 0x1309810 — 调度迭代驱动（M23 §3 全解）----
uint64_t stcut_iterate(StCutContext &ctx, uint32_t seed, uint32_t target)
{
    ++ctx.call_count;                                       // [130981e ctx+0x10 += 1]
    std::fill(ctx.threshold_ledger.begin(), ctx.threshold_ledger.end(), (uint64_t)0); // [130983f memset]
    if (stcut_flood_collect(ctx, seed, target) == 0)        // [130984d 首次探测]
        return 0;                                           // [1309855 返 0]
    uint64_t total = 0;                                     // [130985b r12=0]
    // 收敛性（M35 收口，全区间复核）：1309810-1309930 内仅三处出口——初探归零
    //   [1309852 testq]、循环 A 搬运归零 [13098c5 testq]、循环 B 探测归零
    //   [1309924 testq]；无任何上限比较/计数器（cmp/dec 全区仅 memset 与计时
    //   减法）⇒ .so 无迭代上限，终止纯靠不变式：
    //   ①洪泛在「与 target 同代收集的叶子」上提前停（[13095a0] touched 即停），
    //     该叶子的出边槽拿不到位图标记 → 搬运无法跨越 → 账本永不落满 → 洪泛门
    //     永开（A/B 互喂）；
    //   ②seed==target 时 weight_pull 纯吸收返 cap 且无账本副作用 → 循环 A 永不
    //     归零（.so 真实输入由上游保证 pivot≠S 锚，不触发）。
    // 重实现保留防御上限防宿主挂死（.so 无此门，属忠实分歧，已注明）。
    size_t guard = 0;
    const size_t guard_cap = 4 * ctx.weight_table.size() + 64;
    for (;;) {
        // 循环 A [1309860-13098c8]：批量搬运 0x13096b0(cap=0x5f5e0ff) 至返回 0
        for (;;) {
            if (++guard > guard_cap) break;                 // 防御上限（.so 无此门）
            uint64_t t0 = StCutContext::tick();             // [1309860-130986d TSC1>>4]
            uint64_t n = stcut_weight_pull(ctx, seed, target, STCUT_MAX_WEIGHT, 0); // [1309897]
            uint64_t t1 = StCutContext::tick();             // [130989f-13098aa TSC3>>4]
            ctx.batch_time += t1 - t0;                      // [13098ae-13098bb → ctx+0x288]
            total += n;                                     // [13098c2]
            if (n == 0) break;                              // [13098c5-13098c8]
        }
        if (guard > guard_cap) break;                       // 防御上限出口
        // 循环 B [13098ca-1309927]：探测 0x1309230，非 0 回循环 A
        uint64_t t0 = StCutContext::tick();
        uint64_t r = stcut_flood_collect(ctx, seed, target);  // [13098f9]
        uint64_t t1 = StCutContext::tick();
        ctx.probe_time += t1 - t0;                          // [1309910-130991d → ctx+0x290]
        if (r == 0) break;                                  // [1309924-1309927]
    }
    return total;                                           // [130992d]
}

// ---- 0x1309940 — 替换对批量记账 wrapper（M22/M23 + M34 账本贯通；0x1309a60 调 2 次）----
// [1309969 ecx=0x5f5e0ff 哨兵阈值] 4 次 0x1300cc0（cb = 调用者局部账本，M34 破译：
// 0x1309a60 栈参传 rsp+0x90——脚手架对全部记账，试切末 [1309f37] 整体重放撤销）：
//   [1309974] connect(arg2, arg4) 直连；
//   [13099d3] 循环1 遍历 add集[arg4]，跳 filter&4 或 canon==arg3 → connect(canon, arg3)；
//   [1309a00] 循环2 遍历 arg5 向量 → connect(arg2, elem)；
//   [1309a30] 循环3 遍历 arg6 向量 → connect(elem, arg3)。
// [1309a55] 尾跳 0x1309810(ctx, arg2, arg3)，返回其累计处理数。
uint64_t stcut_batch_accounting(StCutContext &ctx, uint32_t arg2, uint32_t arg3, uint32_t arg4,
                                std::vector<uint32_t> *arg5, std::vector<uint32_t> *arg6,
                                std::vector<StCutReplayRec> *cb)
{
    static const std::vector<uint32_t> kEmpty;
    const std::vector<uint32_t> &v5 = arg5 ? *arg5 : kEmpty;
    const std::vector<uint32_t> &v6 = arg6 ? *arg6 : kEmpty;

    stcut_connect_nodes(ctx, arg2, arg4, STCUT_MAX_WEIGHT, cb);   // [1309974]
    // 循环1 [1309990-13099d8]：.so 在循环头预载 begin/end——迭代期间 connect 向
    // add_set[arg4] 的追加不在本趟内重访。C++ range-for 同语义，但追加触发重分配
    // 会使捕获指针悬空（UB）——先快照即忠实且安全。
    std::vector<uint32_t> add4_snapshot = ctx.add_set[arg4];
    for (uint32_t rel : add4_snapshot) {
        if (ctx.weight_table[rel].flags & 4) continue;
        if (ctx.weight_table[rel].canon == arg3) continue;
        stcut_connect_nodes(ctx, ctx.weight_table[rel].canon, arg3, STCUT_MAX_WEIGHT, cb); // [13099d3]
    }
    for (uint32_t e : v5)                                               // [13099da-1309a0c 循环2]
        stcut_connect_nodes(ctx, arg2, e, STCUT_MAX_WEIGHT, cb);   // [1309a00]
    for (uint32_t e : v6)                                               // [1309a0e-1309a3c 循环3]
        stcut_connect_nodes(ctx, e, arg3, STCUT_MAX_WEIGHT, cb);   // [1309a30]

    return stcut_iterate(ctx, arg2, arg3);                              // [1309a55 尾调用]
}

// ---- 0x1305130 — stcut_add_dependencies 层标记原语（M34 逐指令全解）----
// 签名：f(ctx, esi=seed)。软边正向分层，结果写 ctx+0x128 u16 标记表。
// 算法（地址注释对应反汇编）：
//   memset 标记表                                     [1305157-130516a]
//   A 队 = {seed}（seed 不预置标记）                   [1305191-13051bb]
//   B 队 = {}（就绪前沿索引暂存）                      [1305183]
//   level = 2                                         [13051c3]
//   while (A 非空)                                    [13051f6]
//     count = A.size()                                [1305203]（本层快照长度）
//     B 复位                                           [130521c/1305583 写指针=begin]
//     for i in [0, count):                            [130523a-1305243]
//       node = A[i]                                   [1305249]
//       循环1 软前驱就绪检查（subtract 集 ctx+0xf8）：
//         对端 = wt[wt[slot].idx].canon                [130528f-1305298 配对槽跳转]
//         mark[对端] > 1 → 本条已安置                  [130529b ja]
//         slot.flags & 4 → 本条不算依赖                [13052a2]（B 半边）
//         存在未安置软前驱 → 节点本层不就绪             [13052a9 → 130523a]
//       就绪：B.push(i)                               [13052bf]
//       循环2 软后继扩散（add 集 ctx+0xe0）：
//         slot.flags & 4 → 跳过                        [130544b]
//         child = wt[slot].canon                       [1305452 直取]
//         mark[child] != 0 → 跳过                      [130545d]
//         A.push(child); mark[child] = 1               [1305413/1305427]
//     层末（倒序 ×2 展开）：
//       mark[A[B[j]]] = level                          [13055cd/13055fa]
//       A[B[j]] = A.back(); A.pop_back()               [13055d1-13055da swap-remove]
//     level++                                          [13051ee]（首层不加：13051da 直跳 13051ff）
//   释放 A/B                                           [1305633-130564a]
// 标记语义：0=未达，1=在队，≥2=已定层（首层 2）。层号边界：软前驱全定层才就绪，
// 就绪节点的软后继下层入队。软关系图须为 DAG（.so 同样无环检测，环上节点将永
// 不就绪——A 不空即死循环，真实输入为数据流导出的 DAG）。
void stcut_mark_levels(StCutContext &ctx, uint32_t seed, uint32_t max_levels)
{
    std::fill(ctx.mark_table.begin(), ctx.mark_table.end(), (uint16_t)0); // [130516a memset]
    if (seed >= ctx.add_set.size()) return;
    std::vector<uint32_t> queue_a = { seed };                   // [1305191-13051bb] A 队
    std::vector<uint32_t> queue_b;                              // [1305183] B 队（前沿索引）
    uint32_t level = 2;                                         // [13051c3]
    while (!queue_a.empty()) {                                  // [13051f6]
        if (level - 2 >= max_levels) break;   // 概要级调用方护栏（.so 无界；见头注）
        const size_t count = queue_a.size();                    // [1305203]
        queue_b.clear();                                        // [130521c/1305583]
        for (size_t i = 0; i < count; ++i) {                    // [130523a-1305243]
            const uint32_t node = queue_a[i];                   // [1305249]
            bool ready = true;                                  // 循环1：软前驱就绪检查
            for (uint32_t slot : ctx.subtract_set[node]) {      // [1305254-1305287] ctx+0xf8
                uint32_t peer = ctx.weight_table[ctx.weight_table[slot].idx].canon; // [130528f-1305298]
                if (peer >= ctx.mark_table.size()) continue;
                if (ctx.mark_table[peer] > 1) continue;         // [130529b] 对端已定层
                if (ctx.weight_table[slot].flags & 4) continue; // [13052a2] B 半边/硬边
                ready = false;                                  // [13052a9 → 130523a]
                break;
            }
            if (!ready) continue;
            queue_b.push_back((uint32_t)i);                     // [13052bf] 登记前沿索引
            for (uint32_t slot2 : ctx.add_set[node]) {          // [13053ef-1305440] ctx+0xe0
                if (ctx.weight_table[slot2].flags & 4) continue; // [130544b]
                uint32_t child = ctx.weight_table[slot2].canon; // [1305452]
                if (child >= ctx.mark_table.size()) continue;
                if (ctx.mark_table[child] != 0) continue;       // [130545d]
                queue_a.push_back(child);                       // [1305413/13054e3]
                ctx.mark_table[child] = 1;                      // [1305427] 在队占位
            }
        }
        for (size_t k = queue_b.size(); k-- > 0; ) {            // [1305580-130562c] 倒序 ×2 展开
            uint32_t node = queue_a[queue_b[k]];                // [13055c2-13055c6]
            ctx.mark_table[node] = (uint16_t)level;             // [13055cd/13055fa] 正式层号
            queue_a[queue_b[k]] = queue_a.back();               // [13055da]
            queue_a.pop_back();                                 // [13055d6]
        }
        ++level;                                                // [13051ee]
    }
}

// ---- 0x1308c50 — stcut_add_dependencies 的 &4 孪生（M34 逐指令全解）----
// 签名：f(ctx, esi=死参数, edx=seed, rcx=&外部标记表)。骨架与 0x1305130 逐指令
// 同构（memset/A{seed}/B/level=2/循环1 就绪门/循环2 扩散/层末 swap-remove 写层号），
// 仅两处 &4 极性相反——本函数只走 flags&4 槽（= B 半边，canon=前驱）：
//   memset 的是 arg4 指向的外部 vector<u16>                 [1308c7a-1308c86]
//   种子 = arg3（esi 从未被消费——死参数）                    [1308ccb]
//   level = 2                                               [1308cdf]
//   循环1 就绪门（subtract 集 ctx+0xf8，配对槽跳转取对端）：
//     mark[对端] > 1 → 已安置                                [1308dbb ja]
//     slot.flags 非 &4 → 不算依赖（★与 0x1305130 相反）       [1308dc2 je 跳过非 &4]
//     存在未安置的 &4 对端 → 不就绪                           [1308dc9 → 1308d50]
//   就绪：B.push(前沿索引)                                   [1308ddf]
//   循环2 扩散（add 集 ctx+0xe0）：
//     slot.flags 非 &4 → 跳过（★相反）                       [1308f6b je]
//     child = wt[slot].canon                                 [1308f72]
//     mark[child] != 0 → 跳过                                [1308f79]
//     A.push(child); mark[child] = 1                         [1308f2d/1308f3d]
//   层末：mark[A[B[j]]] = level + swap-remove（×2 展开）      [13090e0-130914b]
//   level++（首层不加 [1308cfb 直跳 1308d1f]）                [1308d0e]
// 语义：沿 B 半边（canon=前驱）反向分层；就绪要求 &4 对端（= 后继，经
// subtract 集的 B 槽伙伴链）全定层——反向波自洽（靠种子侧先定层）。
// 调用点仅提案器 [130a391]：外部表 = add_dependencies 结果的 assign 副本
// （长度继承，值随即被 memset 清掉）[130a36a-130a382]。
void stcut_level_mark_bfs(StCutContext &ctx, uint32_t seed, std::vector<uint16_t> &marks,
                          uint32_t max_levels)
{
    std::fill(marks.begin(), marks.end(), (uint16_t)0);         // [1308c7d memset 外部表]
    if (seed >= ctx.add_set.size()) return;
    std::vector<uint32_t> queue_a = { seed };                   // [1308cad-1308cd7] A 队
    std::vector<uint32_t> queue_b;                              // [1308c9f] B 队（前沿索引）
    uint32_t level = 2;                                         // [1308cdf]
    while (!queue_a.empty()) {                                  // [1308d16]
        if (level - 2 >= max_levels) break;   // 概要级调用方护栏（.so 无界；见头注）
        const size_t count = queue_a.size();                    // [1308d23-1308d2f]
        queue_b.clear();                                        // [1308d3c 复位]
        for (size_t i = 0; i < count; ++i) {                    // [1308d50-1308d5e]
            const uint32_t node = queue_a[i];                   // [1308d69]
            bool ready = true;                                  // 循环1：&4 对端就绪检查
            for (uint32_t slot : ctx.subtract_set[node]) {      // [1308d6d-1308da7] ctx+0xf8
                if (!(ctx.weight_table[slot].flags & 4)) continue; // [1308dc2 ★非 &4 跳过]
                uint32_t peer = ctx.weight_table[ctx.weight_table[slot].idx].canon; // [1308daf-1308db8]
                if (peer >= marks.size()) continue;
                if (marks[peer] > 1) continue;                  // [1308dbb] 对端已定层
                ready = false;                                  // [1308dc9 → 1308d50]
                break;
            }
            if (!ready) continue;
            queue_b.push_back((uint32_t)i);                     // [1308ddf] 登记前沿索引
            for (uint32_t slot2 : ctx.add_set[node]) {          // [1308f0a-1308f57] ctx+0xe0
                if (!(ctx.weight_table[slot2].flags & 4)) continue; // [1308f6b ★非 &4 跳过]
                uint32_t child = ctx.weight_table[slot2].canon; // [1308f72]
                if (child >= marks.size()) continue;
                if (marks[child] != 0) continue;                // [1308f79]
                queue_a.push_back(child);                       // [1308f2d]
                marks[child] = 1;                               // [1308f3d] 在队占位
            }
        }
        for (size_t k = queue_b.size(); k-- > 0; ) {            // [13090a0-130914b] 倒序 ×2 展开
            uint32_t node = queue_a[queue_b[k]];                // [13090e2-13090e6]
            marks[node] = (uint16_t)level;                      // [13090e9/1309119] 正式层号
            queue_a[queue_b[k]] = queue_a.back();               // [13090f7/1309123]
            queue_a.pop_back();                                 // [13090f3/130913a]
        }
        ++level;                                                // [1308d0e]
    }
}

// ---- 0x13087f0 — remove_edge（全解）：从四条 per-node 链表撤除一对槽 ----
// p1=canon(slot) [130882a]、pslot=idx(slot) [130882e]、p2=canon(pslot) [1308837]；
// 四处 swap-with-last 删除：add_set[p2]←slot [130883f-130887a] / add_set[p1]←pslot
// [1308880-13088ca] / subtract_set[p2]←pslot [13088d0-130891c] / subtract_set[p1]←slot
// [130891f-130895a]；replay≠0 时追加 {op=slot, 0, type=4} [130895d-130897f]。
// 不置任何 flag、无输出参——M30 旧 cleanup_node 的 |=0x80/追加输出为概要级误读。
// 分裂器调用点 [130b7b6] 传 rsp+0x30 一次性汇，随后 [130b7bb-130b7ce] 直接
// delete 且从不读取 → 等价 nullptr。
void stcut_remove_edge(StCutContext &ctx, uint32_t slot, std::vector<StCutReplayRec> *replay)
{
    auto rm = [](std::vector<uint32_t> &v, uint32_t x) {
        for (size_t j = 0; j < v.size(); ++j)
            if (v[j] == x) { v[j] = v.back(); v.pop_back(); return; }
    };
    uint32_t p1 = ctx.weight_table[slot].canon;          // [130882a]
    uint32_t pslot = ctx.weight_table[slot].idx;         // [130882e]
    uint32_t p2 = ctx.weight_table[pslot].canon;         // [1308837]
    rm(ctx.add_set[p2], slot);                           // [130883f-130887a]
    rm(ctx.add_set[p1], pslot);                          // [1308880-13088ca]
    rm(ctx.subtract_set[p2], pslot);                     // [13088d0-130891c]
    rm(ctx.subtract_set[p1], slot);                      // [130891f-130895a]
    if (replay) {                                        // [130895d testq %rdx]
        StCutReplayRec r;
        r.op = slot; r.value = 0; r.type = 4;            // [1308962-130896f]
        stcut_replay_push(*replay, r);                   // [130897f 0x1307f60]
    }
}

// ---- 建图层输入落地（早期阶段产物 → ctx 表）----
// 对应 schedule_for_alloc 建图段（M25 §1.3 阶段表：read_nodes@13030e0 → collate_sibs
// → relate_by_tensor → node_hash → quick_early_sort → block_relate → connect_nodes
// → add_dependencies → strong_relevel → arrange_sibs → clean_sibs → delay_dma）。
// 各早期阶段内部未全解——本函数只落地其对 ctx 的已证最终效果：
//   节点槽注册（0x1301040 追加路，mode=0x40 建表期 [12fba54]）与关系对建槽。
// ★替换对不变式（asm 复核）：运行期 0x1300cc0 建对恒为 A.flags=mode /
//   B.flags=mode|4（[1300e13]）。B 侧的 &4 是结构性标记：B 槽（canon=src）在
//   add_set[dst] 里带 &4 = 验证器跳过的反向边 + Kahn 终段的真依赖（生产者先于
//   消费者）。若两侧对称不置 4，验证器沿 canon=src 反向泛洪，闭包退化为无向
//   连通闭包 → 任何序都判非法（冒烟测试实证）。
// 反汇编未完全理解：初始表 B 侧权重（.so 建对 B 恒 0x5f5e0ff，但 grain 树 =
//   Σ add集权重若含 0x5f5e0ff 会使 flow 量纲失衡，故初始表疑为实权重——两难
//   未决，此处取实权重）。hard 输入标志不落到槽 flags（M34 修正，见下方
//   fl_a 处死锁论证）；硬依赖的序约束由 Kahn 终段经 B 半边（恒 &4）消费。
// ---- 0x12fc820 — stcut_read_nodes（M33 逐指令解码，见 hpp 头注）----
// .so 侧 cand_tables 为 8B 指针数组：每槽指向 map A 某组值区（组成员表
// vector<u32>），同组节点共享同一指针 [12fcc79 movq %rbp,(%rsi,%rdx,8)]。
// 本实现存表值——提案器/修正器对 cand_tables 只读，观测等价。
void stcut_read_nodes(StCutContext &ctx)
{
    // ①扁平索引（0x12fc862-0x12fc8c2）：reserve(n)+按 id 顺 push {payload,id}，
    //   槽位 i = 节点 i；.so 用它把组员 id → 槽指针（排序比较器 0x12fd0d0
    //   被优化成同基址槽指针比较），此处直接以 id 排序，语义相同。
    const size_t n = ctx.node_slots.size();
    if (n == 0) return;                              // [12fc8d7 空图直落清理]
    for (auto &kv : ctx.relation_groups) {
        auto &members = kv.second;
        // ②成员经扁平索引收集→排序→写回（12fc9a0-12fcc51；越界者告警并跳过
        //   [12fca99-12fcadb "Node index %u is larger than size %zu", st_cut.cc]）
        std::vector<uint32_t> sorted;
        sorted.reserve(members.size());
        for (uint32_t id : members) {
            if ((size_t)id >= n) {                   // [12fcaa7 cmpq %rbp,%rbx / jb 告警]
                fprintf(stderr, "st_cut.cc: Node index %u is larger than size %zu\n", id, n);
                continue;
            }
            sorted.push_back(id);
        }
        std::sort(sorted.begin(), sorted.end());     // [12fcb8d std::sort + 比较器 0x12fd0d0]
        members.swap(sorted);                        // [12fcbf0-12fcc51 写回成员表]
        // ③cand_tables[id] = 本组成员表（共享指针的按值等价）
        for (uint32_t id : members)
            ctx.cand_tables[id] = members;           // [12fcc79]
        // ④组登记（.so：&成员表 push 进 ctx+0x258 [12fc916→0x12fcfa0]）
        ctx.group_list_keys.push_back(kv.first);
    }
}

void stcut_build_initial_state(StCutContext &ctx, const StCutGraphInput &in)
{
    ctx = StCutContext{};                            // 0x12fba20 构造（[ctx+0xc]=0x40 建表期）
    ctx.mode = 0x40;                                 // [12fba54 movl $0x40]
    for (uint32_t i = 0; i < in.node_count; ++i)
        stcut_register_node(ctx, i, 0);              // 0x1301040 追加路
    for (const auto &r : in.relations) {
        uint32_t a = (uint32_t)ctx.weight_table.size();
        uint32_t b = a + 1;
        // A.flags = mode（connect_nodes 本体 [1300dc5-1300e13] 从不给 A 置 4）。
        // M34 修正：撤销「hard→A&4」旧猜测——若 A 侧可置 4，0x1308c50 的就绪门
        // （&4 槽对端）会在硬前驱上自锁：前驱入队唯一途径 = 本节点就绪后经 B 半边
        // 扩散，而本节点就绪又等该前驱定层 → 环死锁；.so 在含硬依赖的真实图上
        // 不挂 → &4 恒为 B 半边结构标记（Kahn 终段消费 B 槽 canon=前驱）。
        uint32_t fl_a = (uint32_t)0x40;                                // A.flags = mode
        uint32_t fl_b = (uint32_t)0x40 | 4u;                           // B.flags = mode|4 [1300e13 不变式]
        ctx.weight_table.push_back({ r.dst, b, r.weight, fl_a });     // slot A {canon=dst,idx=B,w,flags}
        ctx.threshold_ledger.push_back(0);
        ctx.weight_table.push_back({ r.src, a, r.weight, fl_b });     // slot B {canon=src,idx=A,mode|4}
        ctx.threshold_ledger.push_back(0);
        stcut_add_push(ctx, r.src, a);               // [0x1307820 建链（与 0x1300cc0 公共尾同构）]
        stcut_add_push(ctx, r.dst, b);
        stcut_sub_push(ctx, r.src, b);               // [0x1307960]
        stcut_sub_push(ctx, r.dst, a);
    }
    ctx.bitmap.resize((ctx.weight_table.size() + 31) / 32, 0);        // [ctx+0x110 同步]
    ctx.working_order = in.initial_order;
    ctx.relation_groups = in.groups;                 // map A（+0x228）装载期内容
    stcut_read_nodes(ctx);                           // 0x12fc820 建图链第 1 段 [13030e0]
    ctx.mode = 0;                                    // 进入复用期（快照前等价态）
}

// ============================================================================
// M31-5 切分层（M30 六件套 + 辅助）
// ============================================================================

// ---- 0x130ab30 — 切点提案器（M32 逐指令解码，替换 M31 概要级替身）----
// .so 签名（实测）：f(ctx, StCutRec *best /*rsi*/, journal向量的指针 /*rdx*/,
//                   arg4 /*rcx, 全函数未用*/, 排除集 /*r8, 节点键@0x1c/32B 树*/,
//                   out_before /*r9*/, out_after /*栈参 0xc0(%rsp)*/) → u32 切点
// 流程：
//   ①邻对选择 [130ab55-130ab7d]：fwd = journal[best.@0]（链前向索引；
//     初始记录的 @0=2 哨兵恰指向 3 号初始记录）；back = journal[fwd.@8]；
//     filter_id = journal[back].id（0xc(%rsp)，链一致时 = best.id）；
//     picked = {fwd, back} 中 id 较大者（正常链 = 前向后继）。
//   ②外部候选路 [130ab9d-130ad96]（verdict=journal[picked].@0x20 ≠ 0 时）：
//     表 = ctx+0x240[journal[picked].@0x24]；门：表号<表数、非空、字节长≤0xef；
//     过滤 ownership[node]==filter_id [130ac69 cmpw]；命中→切点=hits[count/2]
//     [130ad68 (end−begin)>>3 即元素数/2 中位]；切点==0 或空命中→落通用路。
//   ③通用路 [130ad97-130b001]：0x130a330(ctx, best.@0x14, journal[best.@0].@0x10, &树)
//     ——第二入口参=链前向邻居的 sub_entry；扫 [range_start, +span_len)（经
//     ctx+0x158 双指针取工作序），跳 ctx+0x80[id]&4 [130ae3a]；
//     w = tree[node]（operator[] 缺失插 0 [130ae41-130af24]）× cand_tables[node]
//     元素数 [130af3b-130af5e]（表字节≥0xf1 → w=0 [130af62]）；
//     jitter 门 [[ctx+0x270]+0x15c]≠0：w += (rand%(cfg+1)×w)>>6 [130af7e-130afaf]；
//     w>best 且不在排除集（arg5 树 walk [130afbd-130afea]）→ 更新 [130adf9]。
//   ④随机兜底 [130b01a-130b058]（切点==0 或空区间 [130aff0]）：
//     pivot = order[range_start + rand % span_len]（32/64 位除法自适应 [130b033]）。
//   ⑤统一切集构造 [130b05c-130b33b]：表 = cand_tables[pivot]，门同②；
//     过滤同②；verdict≠0 → 切点改取「距两端最远且不在排除集」的命中元素
//     （距离严格递增扫描 [130b21d-130b268]，r9 从 0 起且比较含 +1），
//     verdict==0 → 中位 [130b2c8]；随后按切点劈半：切点前的命中→out_before，
//     切点后→out_after（0x853eb0 逐元素 push [130b2cc-130b33b]）。
//   ⑥终验+登记 [130b34a-130b495]：切点==0 或已在排除集 → 二次随机兜底（不再
//     复查 [130b386]）；最终不在排除集 → 插入（32B 节点、再平衡 0x868c50、
//     size++ [130b456-130b495]）后返回。
// 候选表 ctx+0x240 的填充方在上游（未解码）——空表时本函数退化为
// 「随机切点 + 空切集」，与 .so 行为一致。
uint32_t stcut_cut_propose(StCutContext &ctx, size_t best,
                           std::set<uint32_t> &exclusion,
                           std::vector<uint32_t> &out_before,
                           std::vector<uint32_t> &out_after)
{
    out_before.clear();
    out_after.clear();
    const StCutRec &rec = ctx.journal[best];

    // ① 邻对选择 [130ab55-130ab7d]
    uint64_t fwd = rec.link_fwd;                          // rax = *(rsi) = best.@0
    uint32_t filter_id = 0;                               // 0xc(%rsp)
    uint32_t picked = (uint32_t)best;
    if (fwd < ctx.journal.size()) {
        uint64_t back = ctx.journal[fwd].link_back;       // journal[fwd].@0x8
        uint32_t back_id = (back < ctx.journal.size()) ? ctx.journal[back].id : 0;
        filter_id = back_id;
        picked = (back_id > ctx.journal[fwd].id) ? (uint32_t)back : (uint32_t)fwd;
    }
    uint8_t verdict_byte = 0;
    if (picked < ctx.journal.size()) verdict_byte = (uint8_t)ctx.journal[picked].field20; // [130ab95 → 0xf(%rsp)]

    auto table_gate = [&](uint32_t idx, std::vector<uint32_t> &hits) -> bool {
        hits.clear();
        if (idx >= ctx.cand_tables.size()) return false;  // [130abc8 表号<表数]
        auto &tbl = ctx.cand_tables[idx];
        if (tbl.empty() || tbl.size() * sizeof(uint32_t) > 0xef) return false; // [130abe5 字节≤0xef]
        for (uint32_t node : tbl)
            if (node < ctx.ownership.size() && ctx.ownership[node] == filter_id)
                hits.push_back(node);                     // [130ac69 过滤]
        return !hits.empty();
    };

    // ② 外部候选路
    int64_t pivot = -1;
    if (verdict_byte != 0) {                              // [130ab9d testb]
        std::vector<uint32_t> hits;
        if (table_gate(ctx.journal[picked].field24, hits)) // [130abaf @0x24]
            pivot = (int64_t)hits[hits.size() / 2];        // [130ad68 中位]
    }

    // ③ 通用路（pivot 未定或 ==0 时；.so 经 je 直落 [130ad86/130ad97]）
    if (pivot <= 0) {
        std::map<uint32_t, uint64_t> tree;                // 0x60-0x78(%rsp) std::map
        uint32_t far_sub = (fwd < ctx.journal.size()) ? ctx.journal[fwd].sub_entry
                                                      : rec.sub_entry; // [130adb0 journal[best.@0].@0x10]
        stcut_build_local_tree(ctx, rec.add_entry, far_sub, tree);      // [130adc3]
        uint64_t best_w = 0;                              // 0x10(%rsp)
        uint32_t best_node = 0;                           // 0x18(%rsp)
        for (uint32_t pos = rec.range_start;
             pos < rec.range_start + rec.span_len; ++pos) {              // [130ae0f 边界]
            if (pos >= ctx.working_order.size()) break;   // 防御（.so 信任区间几何）
            uint32_t node = ctx.working_order[pos];       // [130ae25 经 ctx+0x158]
            if (ctx.group_flags[node] & 4) continue;      // [130ae3a]
            uint64_t tv = tree[node];                     // [130af24 operator[]]
            uint64_t w = 0;                               // [130af62 默认 0]
            if (node < ctx.cand_tables.size() &&
                ctx.cand_tables[node].size() * sizeof(uint32_t) < 0xf1)  // [130af62 <0xf1]
                w = (uint64_t)ctx.cand_tables[node].size() * tv;         // [130af3e-130af5e]
            if (ctx.config.jitter != 0) {                 // [130af7e cfg+0x15c]
                uint64_t r = StCutContext::rand_next(ctx) % (ctx.config.jitter + 1);
                w += (r * w) >> 6;                        // [130afa7-130afaf]
            }
            if (w > best_w && exclusion.count(node) == 0) {              // [130afb2/130afbd]
                best_w = w;
                best_node = node;                         // [130adf9 接受]
            }
        }
        // 树析构 [130b00b 0x8f0720]
        pivot = (int64_t)best_node;
    }

    // ④ 随机兜底 [130b01a-130b058]（含空区间出口 [130aff0] 直落）
    auto random_fallback = [&]() -> int64_t {
        if (rec.span_len == 0) return 0;                  // 除零防御（.so 依区间非空）
        uint64_t r = StCutContext::rand_next(ctx) % rec.span_len;        // [130b021-130b045]
        uint64_t pos = rec.range_start + r;               // [130b04e +0x28]
        if (pos >= ctx.working_order.size()) return 0;    // 防御
        return (int64_t)ctx.working_order[pos];           // [130b055 经 ctx+0x158]
    };
    if (pivot <= 0) pivot = random_fallback();

    // ⑤ 统一切集构造 [130b05c-130b33b]
    if (pivot > 0 && (uint32_t)pivot < ctx.cand_tables.size()) {
        std::vector<uint32_t> hits;
        if (table_gate((uint32_t)pivot, hits)) {
            uint32_t cand = (uint32_t)pivot;              // [130b0bb 0x18(%rsp) 预置]
            if (verdict_byte != 0) {
                // 距两端最远且不在排除集（r9 从 0 起、门 d≥r9、更新 r9=d+1 → 严格递增）
                size_t best_d = 0;                        // r9
                for (size_t i = 0; i < hits.size(); ++i) {
                    size_t d = hits.size() - 1 - i;
                    if (i > d) d = i;                     // [130b21d-130b229 max(余量,已过)]
                    if (d >= best_d && exclusion.count(hits[i]) == 0) { // [130b22d/130b23f]
                        best_d = d + 1;                   // [130b23b rax+1]
                        cand = hits[i];                   // [130b268]
                    }
                }
            } else {
                cand = hits[hits.size() / 2];             // [130b2c8 中位]
            }
            pivot = (int64_t)cand;
            // 劈半 [130b2cc-130b33b]：切点前→out_before(arg6)，切点后→out_after(arg7)
            bool past = false;                            // bpl，遇切点清零
            for (uint32_t h : hits) {
                if (h == (uint32_t)pivot) { past = true; continue; }
                (past ? out_after : out_before).push_back(h); // [130b320/130b30b 0x853eb0]
            }
        }
    }

    // ⑥ 终验 + 登记 [130b34a-130b495]
    if (pivot <= 0 || exclusion.count((uint32_t)pivot) != 0)
        pivot = random_fallback();                        // [130b386 二次兜底，不复查]
    if (pivot > 0) exclusion.insert((uint32_t)pivot);     // [130b456-130b495]
    return (uint32_t)pivot;                               // [130b42f eax=r15d]
}

// ---- 0x1309a60 — 迁移执行器（M30 §1 + M35 参数语义修正）----
// 参数（develop [130db34-130db76] 逐寄存器）：
//   esi = journal[best].add_entry；edx = journal[journal[best].link_fwd].sub_entry
//   （= 本函数 seed [rsp+0x14]，BFS 种子与导出扫描基）；
//   ecx = develop 游标 [rsp+0x18]（记账 connect 对端 / add集 扫描下标）；
//   r8/r9 = develop 局部两向量（提案器 Path A 填充的切集两半；Path B 为空）。
uint64_t stcut_cut_execute(StCutContext &ctx, const StCutRec &rec, uint32_t seed,
                           uint32_t cursor, uint16_t new_rec_id,
                           std::vector<uint32_t> &cands_before,   // 提案 arg6：切点前半
                           std::vector<uint32_t> &cands_after,    // 提案 arg7（栈参）：切点后半
                           std::vector<uint32_t> &out,
                           std::vector<uint32_t> &exported)
{
    // [1309ab1] 估量（M22/M23 批量记账 wrapper）——两半切集 = 提案器 arg6/arg7：
    //   循环2 连 arg2→e（前半，S 侧），循环3 连 e→arg3（后半，seed 侧）
    // M34 账本破译：脚手架对记入本试切局部账本（.so rsp+0x90 vector，xmm0 清零建
    // [1309a8a-1309aa1]），饱和/收尾两次重放都指向它——脚手架不跨试切存活。
    std::vector<StCutReplayRec> trial_ledger;        // [1309a8d rsp+0x90 局部账本]
    std::vector<uint32_t> empty;
    uint64_t est = stcut_batch_accounting(ctx, rec.add_entry, seed, cursor,
                                          &cands_before, &cands_after, &trial_ledger);
    if (est >= STCUT_BATCH_CAP) {                    // [1309ab6 cmpq $0x5f5e0ff 冷路径]
        stcut_replay_apply(ctx, trial_ledger);       // [1309acc 0x13080a0：撤销全部脚手架]
        est = stcut_batch_accounting(ctx, rec.add_entry, seed, cursor, &empty, &empty,
                                     &trial_ledger); // [1309afd 空切集重试（继续记账）]
    }

    out.clear();                                     // [1309b07]
    std::fill(ctx.mark_table.begin(), ctx.mark_table.end(), (uint16_t)0); // [1309b38 memset ctx+0x128]
    std::vector<uint32_t> worklist = { seed };       // [1309b6c-1309b70 工作列表={rsp+0x14=seed}]

    while (!worklist.empty()) {                      // [1309bc4 换缓冲 BFS 主循环]
        std::vector<uint32_t> next;
        for (uint32_t ebp : worklist) {
            if (ebp != seed) {                       // [1309bcb cmpl 种子跳过归属写]
                // [1309bdf-1309c00] ★归属改写无围栏：越界门 + 无条件 ownership[ebp]=新号。
                //   M30 旧实现的自造「ownership[ebp]==rec.id」门已删（.so 无此比较）；
                //   闭包范围由记账 connect 的阈值门/flags&4 账本门/mark 表/group 门
                //   共同限制（[1309d30-1309ef7]），非归属门。
                if (ebp < ctx.ownership.size()) {
                    ctx.ownership[ebp] = new_rec_id; // [1309bf8 ★归属改写：节点→新记录号]
                    out.push_back(ebp);              // [1309c04]
                }
            }
            // 邻接展开 [1309d30-1309ef7]（seed 同样展开）
            for (uint32_t rel : ctx.add_set[ebp]) {
                uint32_t canon = ctx.weight_table[rel].canon;   // 表[rel].@0
                uint32_t flags = ctx.weight_table[rel].flags;   // 表[rel].@0x10
                if (flags & 4) {                                 // flags&4 边的阈值账本门
                    uint32_t partner = ctx.weight_table[rel].idx;             // rel.@4
                    if (!(ctx.weight_table[partner].weight > ctx.threshold_ledger[partner]))
                        continue;                                // [1309d95-1309dad]
                }
                if ((ctx.group_flags[canon] & 3) != 0) continue; // [1309d60 通用门：未冻结]
                if (ctx.mark_table[canon] != 0) continue;        // [ctx+0x128 未访问]
                ctx.mark_table[canon] = 1;                       // [1309dcf]
                next.push_back(canon);                           // [层级向量 push]
            }
        }
        if (next.empty()) break;
        worklist.swap(next);                          // [1309efc-1309f27 两向量互换重扫]
    }

    stcut_replay_apply(ctx, trial_ledger);            // [1309f37 重放收尾：rsi=rsp+0x90 局部账本]

    // 边界导出 #1 [1309f3c]：seed 邻接（add集[rsp+0x14]）的 flags&4 边，
    //   对端越界或 ownership[canon]≠新号 → 导出对端句柄 [1309fa5 jbe / 1309fb6 jne → push]
    auto export_rel = [&](uint32_t rel) {
        uint32_t canon = ctx.weight_table[rel].canon;
        if (canon >= ctx.ownership.size() || ctx.ownership[canon] != new_rec_id)
            exported.push_back(ctx.weight_table[rel].idx);      // [0x853eb0 push &表[rel].@4]
    };
    for (uint32_t rel : ctx.add_set[seed])
        if (ctx.weight_table[rel].flags & 4) export_rel(rel);
    // 边界导出 #2（冷段 [130a050-130a1ca]，out 非空时）：全部已迁节点邻接
    for (uint32_t node : out)
        for (uint32_t rel : ctx.add_set[node])
            if (ctx.weight_table[rel].flags & 4) export_rel(rel);

    return est;                                       // [返回 0x1309940 估量]
}

// ---- 0x130b9a0 — 工作序稳定划分修正器（M30 §2 全解）----
// 区间最终形状 [父保留前缀 | 迁移尾巴]；实现按解码不变量以稳定划分落地
// （.so 为相1跳前缀 + 相2双游标压实打洞 + 相3轮转填洞三步，见 M30 §2 逐指令）。
void stcut_cut_fix_range(StCutContext &ctx, size_t best, std::vector<uint32_t> &out, uint16_t rec_id)
{
    StCutRec &rec = ctx.journal[best];
    // ★调用序实证（130dc5b 落地段 → 130b600 分裂器之后才 [130b698] 父 span−=cut_count）：
    //   修正器运行时父区间尚未收缩，扫描全长 = span_len 本身。曾误写 span_len+|out|
    //   多扫 |out| 个元素 → 越界读 → 工作序混入脏值（非排列 FAIL 的根因）。
    uint32_t range_len = rec.span_len;
    uint32_t boundary = rec.range_start + (rec.span_len - (uint32_t)out.size()); // [130ba3b]
    (void)boundary;
    // 相1 [130ba50+]：跳过前缀 = 从 range_start 起所有权≠新记录号的节点（父保留段）；
    // 相2/3：其余位置打洞压实，迁入新记录的节点（ownership==rec_id）轮转填到尾巴。
    // 几何约束（与 0x130b600 子记录继承父区间尾自洽）：尾巴段 = [boundary, +|out|)。
    std::vector<uint32_t> keep, move;
    for (uint32_t k = 0; k < range_len; ++k) {                  // 扫全区间
        uint32_t id = ctx.working_order[rec.range_start + k];
        if (id < ctx.ownership.size() && ctx.ownership[id] == rec_id) move.push_back(id);
        else keep.push_back(id);
    }
    uint32_t w = rec.range_start;
    for (uint32_t id : keep) ctx.working_order[w++] = id;       // [父保留前缀]
    for (uint32_t id : move) ctx.working_order[w++] = id;       // [迁移尾巴]
}

// ---- 0x130d050 — 分裂判定器（M30 §3 全解，返回 u8）----
bool stcut_cut_verdict(StCutContext &ctx, size_t best, const std::vector<uint32_t> &cut_set)
{
    // 波层完备界同 build_local_tree（概要级调用方护栏；.so 原语无界）：
    // 执行器围栏/记录选择替身可能造出跨切界不收敛锥，截断 = 部分锥。
    stcut_mark_levels(ctx, ctx.journal[best].add_entry,
                      (uint32_t)ctx.node_count() + 2);          // [130d07c 0x1305130 打层标记]
    std::map<uint32_t, uint64_t> hist;                          // 计数向量（键=标记值）
    for (uint32_t rel : cut_set) {                              // [130d0ce 对切集每条 rel]
        if (ctx.weight_table[rel].flags & 4) continue;          // flags&4 → 跳过
        uint32_t far = ctx.weight_table[rel].idx;               // far = 表[rel].@4
        uint32_t canon = ctx.weight_table[far].canon;           // canon = 表[far].@0
        if (canon >= ctx.mark_table.size()) continue;
        hist[ctx.mark_table[canon]]++;                          // 计数向量[mark]++
    }
    if (hist.size() < STCUT_MIN_BUCKETS) return false;          // [130d123 <8 桶 → 返 0]
    // [130d18c] 清零前 ≤5 桶规则：反汇编未完全理解（M30 原文「清零前 ≤5 桶 → 前缀和」
    // 的精确筛选未拆）——此处直接对全直方图做前缀和。
    uint64_t max_prefix = 0, total = 0, prefix = 0;
    for (auto &kv : hist) {
        prefix += kv.second * STCUT_VERDICT_FIX;                // 前缀和 ×0x2AAAAAAA（1/6 定点）
        max_prefix = std::max(max_prefix, prefix);              // rdx = 最大前缀和
        total += kv.second;                                     // ecx/r10 = 总和
    }
    if (total == 0) return false;
    uint64_t rax = (hist.size() * max_prefix) / total;          // [130d348 (桶数×最大前缀)/总数]
    return rax < STCUT_VERDICT_THR;                             // [130d366 cmpq $0x340000000 → setl]
}

// ---- 0x130b600 — 记录分裂器（M30 §4 + 本日直接复核 asm 130b600-130b80f）----
// 子记录布局（记录基址 = .so 栈 rsp+0x30，实测）：
//   @0x00=父.link_fwd(链后继) @0x08=best_idx(父回链) @0x10=A 槽(sub_entry)
//   @0x14=B 槽(add_entry) @0x18=cut_count @0x20=判定 u8 [130b6c5 movb]
//   @0x24=r9d(第6参,语义未解) @0x28=父区间尾(@0x18+@0x28 [130b6d1-130b6df])
//   @0x30=元素计数 @0x38=新 u16 id [130b6ed-130b6f5]
//   ★M30 报告表格中「@0x20=B_idx/@0x24=A_idx」为栈帧偏移误读，以 asm 为准。
void stcut_cut_split_record(StCutContext &ctx, size_t best,
                            const std::vector<uint32_t> &cut_ids,
                            uint32_t cut_count, uint8_t verdict)
{
    // [130b646-130b676] 计数器指针（arg8, 0xb8(%rsp)）取旧值后 +1，两次连调：
    //   0x1301040(ctx, *counter, 0, flags=2) → A；再取（已 +1）→ flags=1 → B。
    //   ★asm 复核：两次注册的节点值是 counter 的两个连续值（非同值），counter 净 +2。
    uint32_t v1 = ctx.next_rec_id;                        // [130b646 movq (%r14),%rsi]
    ctx.next_rec_id = (uint16_t)(v1 + 1);                 // [130b649-130b64d]
    uint32_t node_a = stcut_register_node(ctx, v1, 2);    // [130b653 edx=2 / 130b65a call]
    uint32_t v2 = ctx.next_rec_id;                        // [130b662 movq (%r14),%rsi]
    ctx.next_rec_id = (uint16_t)(v2 + 1);                 // [130b665-130b669]
    uint32_t node_b = stcut_register_node(ctx, v2, 1);    // [130b66f edx=1 / 130b676 call]

    StCutRec &parent = ctx.journal[best];
    uint64_t old_fwd = parent.link_fwd;                    // [130b68f r14 = journal[best].@0]
    uint32_t child_idx = (uint32_t)ctx.journal.size();     // [130b69d-130b6db child_index = size]

    // 防御钳位：cut_count > 父区间长时无符号减法下溢（.so 依输入不变式不会发生）
    if (cut_count > parent.span_len) cut_count = parent.span_len;
    parent.span_len -= cut_count;                          // [130b698 ★父区间长收缩]

    StCutRec child{};                                      // 子记录（栈 64B @rsp+0x30）
    child.link_fwd = old_fwd;                              // [130b6a0/130b6f4-130b6fa @0x30=r14]
    child.link_back = best;                                // [130b6a5 @0x38=rsi=best_idx]
    child.sub_entry = node_a;                              // [130b6aa/130b6af @0x10=A]
    child.add_entry = node_b;                              // [130b6b4/130b6b8 @0x14=B]
    child.span_len = cut_count;                            // [130b6bc @0x18=cut_count]
    child.field20 = verdict;                               // [130b6c1-130b6c5 @0x20=判定 u8]
    child.field24 = 0;                                     // @0x24=r9d（语义未解，置 0）
    child.range_start = parent.range_start + parent.span_len; // [130b6d1-130b6df @0x28=@0x18+@0x28]
    child.counter = 0;
    // [130b6ed movzwl 0xc0(%rsp)] 子记录 u16 id 是调用方栈参——develop 预分配的
    // 新记录号（= 计数器入口值 v1，与执行器归属改写所用 new_id 同源）。
    child.id = v1;
    ctx.journal.push_back(child);                          // [130b6f2→130b6f0 0x130b840 压入]

    // push_back 可能使 journal 重分配——此处必须回索引访问（parent 引用已失效）
    ctx.journal[best].link_fwd = child_idx;                // [130b70b journal[best].@0=child]
    if (old_fwd < ctx.journal.size())
        ctx.journal[old_fwd].link_back = child_idx;        // [130b718 journal[旧后继].@8=child（链拼接）]

    // [130b71d-130b7d3] 对切集每 id 双侧重接线 + 清理
    // ★asm 复核（130b765-130b771）：call1 的源 = idx 链对端槽的 canon
    //   （weight_table[weight_table[ebx].idx].canon），非槽自身 canon——M30 表格笔误。
    //   call2 的目的 = 槽自身 canon [130b78d edx=r12d=@0]。
    uint64_t wsum = 0;
    for (uint32_t ebx : cut_ids) {
        uint64_t w = ctx.weight_table[ebx].weight;         // [130b760 @8=weight]
        uint32_t n_far = ctx.weight_table[ctx.weight_table[ebx].idx].canon; // [130b769-130b771 idx→对端@0]
        uint32_t n_near = ctx.weight_table[ebx].canon;     // [130b765 @0]
        stcut_connect_nodes(ctx, n_far, node_a, w, nullptr);   // [130b774 0x1300cc0(ctx, 对端canon, A, w, 0)]
        stcut_connect_nodes(ctx, node_b, n_near, w, nullptr);  // [130b789 0x1300cc0(ctx, B, 自身canon, w, 0)]
        stcut_remove_edge(ctx, ebx, nullptr);              // [130b7b6 0x13087f0 逐 id 撤除]
        wsum += w;                                         // [130b740 r15 += weight]
    }
    ctx.journal[child_idx].counter = wsum;                 // [130b7ec journal[child].@0x30 = Σ权重]
}

// ---- 0x130a560 — 净重计算器（M30 §5 全解；终段排序键来源）----
void stcut_net_weight(StCutContext &ctx)
{
    size_t n = ctx.node_count();                           // [130a566 N=(ctx+0x70−ctx+0x68)>>3]
    ctx.net_weight.resize(n);                              // [resize：大→扩 / 小→截断]
    for (size_t i = 0; i < n; ++i) {
        uint64_t w = 0;
        for (uint32_t rel : ctx.add_set[i])                // [130a5ff-130a62b 加集]
            if (!(ctx.weight_table[rel].flags & 4)) w += ctx.weight_table[rel].weight;
        for (uint32_t rel : ctx.subtract_set[i])           // [130a630-130a65b 减集]
            if (!(ctx.weight_table[rel].flags & 4)) w -= ctx.weight_table[rel].weight;
        ctx.net_weight[i] = w;                             // [130a5f0]
    }
}

// ---- 0x130a330 — 局部权重树构建器（M30 §6 + M32 复核：两趟种子不同）----
// 签名实测（M32 从提案器调用点+本体重核）：f(ctx, arg2=esi, arg3=edx, tree=rcx)
//   第一趟 0x1305130 打标种子 = arg2（best.add_entry）[130a354]；
//   第二趟 0x1308c50 打标种子 = arg3（journal[best.链前向].sub_entry）
//   [130a341 movl %edx,%ebx → 130a38f movl %ebx,%edx → 130a391]；
//   count = (新标记−1)×旧标记 [130a3e6-130a3ff]——两锚间双向可达节点的层积。
void stcut_build_local_tree(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                            std::map<uint32_t, uint64_t> &tree)
{
    // 波层完备界：收敛波每层至少定 1 节点 ⇒ 层数 ≤ 节点数；n+2 即完备。
    // .so 原语无界（真实调用语境保证收敛）；替身的执行器归属围栏/记录选择
    // （M29/M30 概要级）可能造出跨切界的不收敛锥——以截断等价「部分锥」，
    // 锥外节点 mark=0（树值 0，不参与择点）。反汇编未完全理解处。
    const uint32_t wave_cap = (uint32_t)ctx.node_count() + 2;
    stcut_mark_levels(ctx, arg2, wave_cap);                // [130a354 0x1305130 先打标（软边正向）]
    std::vector<uint16_t> new_marks = ctx.mark_table;      // [130a36a-130a382 assign 副本（长度继承）]
    stcut_level_mark_bfs(ctx, arg3, new_marks, wave_cap);  // [130a391 0x1308c50 重打标（&4 反向，写副本）]
    for (uint32_t ebx = 0; ebx < ctx.node_count(); ++ebx) { // [130a426 对每节点]
        uint16_t m_new = (ebx < new_marks.size()) ? new_marks[ebx] : 0;
        uint16_t m_old = (ebx < ctx.mark_table.size()) ? ctx.mark_table[ebx] : 0;
        if (m_new != 0 && m_old != 0 && (ctx.group_flags[ebx] & 3) == 0)
            tree[ebx] = (uint64_t)(m_new - 1) * m_old;     // [130a3e6-130a3ff count=(新−1)×旧]
    }
}

// ---- 0x130bc70 — 随机重启（全解）：撤销一刀，把区间归还给链前向记录 ----
// 结构：① 链摘除判死 → ② 端点两侧集合副本 → ③ 关联树 → ④⑤⑥ 预删两个特殊槽 →
// ⑦ 重合并外循环（内层快照对账 + 双计数门）→ ⑧ B 剩余重接 → ⑨ 尾声 →
// ⑩ 归属改写/区间合并 → ⑪ 两端点残余清除 → ⑫ retire 收尾。
void stcut_random_restart(StCutContext &ctx, size_t rec_idx)
{
    auto rm = [](std::vector<uint32_t> &v, uint32_t x) {   // swap-with-last 删除
        for (size_t j = 0; j < v.size(); ++j)
            if (v[j] == x) { v[j] = v.back(); v.pop_back(); return; }
    };
    auto count_live = [](StCutContext &c, const std::vector<uint32_t> &set, bool snap) {
        uint64_t n = 0;                                    // [130c270-130c28c adcq 计数]
        for (uint32_t s : set)
            if (((snap ? c.snap_weights[s].flags : c.weight_table[s].flags) & 4) == 0) ++n;
        return n;
    };

    StCutRec &rec = ctx.journal[rec_idx];
    const size_t fwd_idx = rec.link_fwd;                   // [130bcb7 ×64 → rbp/[rsp+0x28]
    StCutRec &fwd = ctx.journal[fwd_idx];

    // ① 双向跨接摘链 + 判死 [130bcb3/130bcc0/130bcc8]
    ctx.journal[rec.link_back].link_fwd = rec.link_fwd;    // [130bcb3 前驱跨接]
    ctx.journal[fwd_idx].link_back = rec.link_back;        // [130bcc0 后继回链]
    rec.link_fwd = 0;                                      // [130bcc8 @0/@8 同清判死]
    rec.link_back = 0;

    // ② 端点两侧集合副本（栈位 B=[rsp+0xf0] / add_c=[rsp+0xd0]）[130bce1-130be0b]
    std::vector<uint32_t> B = ctx.subtract_set[rec.sub_entry];   // [130bcd6 读 @0x10 → ctx+0xf8 拷]
    std::vector<uint32_t> add_c = ctx.add_set[rec.add_entry];    // [130bd87 读 @0x14 → ctx+0xe0 拷]

    // ③ 局部关联树 [130be2e-130be84]：对 B 每个非 &4 槽 s，key = wt[wt[s].idx].canon
    //   （一级 idx 间接——伙伴 canon，非槽自身 canon），已存在则 payload 覆写为 s [130be74]
    std::map<uint32_t, uint32_t> rel_tree;
    for (uint32_t s : B)
        if (!(ctx.weight_table[s].flags & 4))
            rel_tree[ctx.weight_table[ctx.weight_table[s].idx].canon] = s;

    // ④ aslot0 = rel_tree[fwd.add_entry]（树查 key=[rsp+0x48]=journal[fwd].@0x14，
    //    未命中插入 payload=0 [130c371-130c3c5]，命中读 payload [130bfd8-130c066]）
    uint32_t aslot0 = rel_tree[fwd.add_entry];
    // ⑤ r15d = add_c 中首个 wt[slot].canon==fwd.sub_entry 的槽 [130c086-130c0b4]
    uint32_t r15d = 0;
    for (uint32_t slot : add_c)
        if (ctx.weight_table[slot].canon == fwd.sub_entry) { r15d = slot; break; }
    // ⑥ 预删：aslot0 从 B [130c0c5-130c102]、r15d 从 add_c [130c114-130c14d]
    if (aslot0 != 0) rm(B, aslot0);
    if (r15d != 0) rm(add_c, r15d);

    // ⑦ 重合并外循环（每 rel）[130c150-130c74e]
    for (uint32_t rel : add_c) {
        if (ctx.weight_table[rel].flags & 4) continue;      // [130c1ba]
        uint32_t far = ctx.weight_table[rel].canon;         // [130c1c1]
        if (far == fwd.sub_entry) continue;                 // [130c1c4（[rsp+0x58]=&fwd.@0x10）]

        // 内循环：对快照 subtract[far]（ctx+0x208）逐 srel 对账 [130c1da-130c43c]
        if (far < ctx.snap_subtract.size()) {
            for (uint32_t srel : ctx.snap_subtract[far]) {
                if (ctx.snap_weights[srel].flags & 4) continue;          // [130c2cf]
                uint64_t w = ctx.snap_weights[srel].weight;              // [130c2e3]
                uint32_t canon2 =
                    ctx.snap_weights[ctx.snap_weights[srel].idx].canon;  // [130c2e8-130c2f0 伙伴链]
                uint32_t aslot = rel_tree[canon2];         // [130c2f4-130c3ce find-or-insert，未命中 payload=0]
                ctx.weight_table[aslot].weight -= w;       // [130c3e7 减账①（aslot，含 0 槽）]
                ctx.weight_table[rel].weight -= w;         // [130c3f1 减账②（rel 自身）]
                StCutReplayRec r;                          // [130c3f6-130c412 24B {op=srel, 0, type=4}]
                r.op = srel; r.value = 0; r.type = 4;
                stcut_replay_push(ctx.replay_ledger, r);
                stcut_replay_apply(ctx, ctx.replay_ledger); // [130c43c 0x13080a0：type4 四链表重建]

                // add 侧计数门（每 srel）[130c453-130c5b6]：
                //   count(add_set[far] 非&4) > count(snap_add[far] 非&4) [130c50e jbe 跳过]
                if (far >= ctx.add_set.size() || far >= ctx.snap_add.size()) continue;
                if (count_live(ctx, ctx.add_set[far], false)
                    <= count_live(ctx, ctx.snap_add[far], true)) continue;
                // 找 aslot 于 add_set[far]（130c524；未命中走两圈 LOG4 诊断 [130c5d9-130c6d1]，
                // 诊断后同样落到 remove_edge [130c669 je 130c559]——重实现省日志，行为等价）
                stcut_remove_edge(ctx, aslot, nullptr);    // [130c565/130c559]
                rm(B, aslot);                              // [130c5a0-130c5b6 从 B 副本撤]
            }
        }

        // aslot0 补偿减账（每 rel，内循环之后）[130c207-130c226]
        if (aslot0 != 0)
            ctx.weight_table[aslot0].weight -= ctx.weight_table[rel].weight; // [130c226]

        // subtract 侧计数门（每 rel）[130c22b-130c749]：
        //   count(subtract_set[far] 非&4) > count(snap_subtract[far] 非&4) → 撤 rel
        //   [130c71f jbe → 否则 130c73f]；否则 rel 重指向 fwd 的 add 端点 [130c17b-130c18a]
        if (far < ctx.subtract_set.size() && far < ctx.snap_subtract.size()) {
            if (count_live(ctx, ctx.subtract_set[far], false)
                > count_live(ctx, ctx.snap_subtract[far], true))
                stcut_remove_edge(ctx, rel, nullptr);      // [130c73f]
            else
                stcut_repoint_b(ctx, rel, fwd.add_entry);  // [130c18a（[rsp+0x48]=&fwd.@0x14）]
        }
    }

    // ⑧ 块2：B 剩余非 &4 成员逐个减 r15d 账并重指向 fwd.sub_entry [130c74e-130c7c9]
    for (uint32_t s : B) {
        if (ctx.weight_table[s].flags & 4) continue;       // [130c7b3]
        if (r15d != 0)
            ctx.weight_table[r15d].weight -= ctx.weight_table[s].weight; // [130c7bf-130c7c4]
        stcut_repoint_a(ctx, s, fwd.sub_entry);            // [130c790-130c797（r12=&fwd.@0x10）]
    }
    // ⑨ 尾声 [130c7cb-130c80a]：aslot0 未被置 0x80 才重指向；r15d 撤除
    if (aslot0 != 0 && !(ctx.weight_table[aslot0].flags & 0x80))  // [130c7e0 testb $-0x80]
        stcut_repoint_a(ctx, aslot0, fwd.sub_entry);       // [130c7f8]
    if (r15d != 0)
        stcut_remove_edge(ctx, r15d, nullptr);             // [130c80a]

    // ⑩ 归属改写 + 区间合并 [130c80f-130c8d7]：rec 区间内节点归属改记 fwd.id（u16），
    //    fwd.span += rec.span，rec.span 清零（死记录）
    for (uint64_t k = 0; k < rec.span_len; ++k) {
        uint32_t node = ctx.working_order[rec.range_start + k];
        ctx.ownership[node] = fwd.id;                      // [130c83b-130c8c6 u16 归属]
    }
    fwd.span_len += rec.span_len;                          // [130c8d2]
    rec.span_len = 0;                                      // [130c8d7]

    // ⑪ 块3/4：两端点残余槽清除——flags&0x84==0 才撤 [130c8dc-130cad1]
    std::vector<uint32_t> C = ctx.add_set[rec.add_entry];  // [130c9b4 testb $-0x7c]
    for (uint32_t slot : C)
        if ((ctx.weight_table[slot].flags & 0x84) == 0)
            stcut_remove_edge(ctx, slot, nullptr);
    std::vector<uint32_t> D = ctx.subtract_set[rec.sub_entry]; // [130ca8b testb $-0x7c]
    for (uint32_t slot : D)
        if ((ctx.weight_table[slot].flags & 0x84) == 0)
            stcut_remove_edge(ctx, slot, nullptr);

    // ⑫ 收尾 [130ca97-130cad1]：两端点退役 + 记录三字段清零
    stcut_retire_node(ctx, rec.add_entry);                 // [130caa7 0x1307aa0]
    stcut_retire_node(ctx, rec.sub_entry);                 // [130cab3]
    rec.field24 = 0;                                       // [130cac2]
    rec.add_entry = 0;                                     // [130caca]
    rec.sub_entry = 0;                                     // [130cad1]
}

// ============================================================================
// M31-6 调度层
// ============================================================================

// ---- 0x130cea0 — aux 前缀和填充器（M29 §4 全解）----
void stcut_aux_fill(StCutContext &ctx, std::vector<uint64_t> &aux, uint32_t start, uint32_t len)
{
    stcut_grain_tree_build(ctx);                             // [130ceb1 懒建权重树守卫]
    uint64_t acc = (start != 0) ? aux[start - 1] : 0;        // [r13 = aux[start-1] / 0]
    for (uint32_t k = start; k < start + len; ++k) {         // [上界 = start+len，130cece]
        uint32_t id = ctx.working_order[k];
        acc += stcut_grain_of(ctx, id);                      // [权重树 find-or-insert；r13 += count]
        if (aux.size() <= k) aux.resize(k + 1, 0);
        aux[k] = acc;                                        // [aux[k] = r13 前缀和曲线]
    }
}

// ---- 0x13065f0 — 评分器（M25 原名 stcut_delay_dma；M28 全解）----
// 权重树懒建；沿工作序前缀和 r13 += count，取运行最大 = 内存峰值（grain 数）。
// 重试循环以返回值×2048=字节 作 flow 量纲（M27 §4 [rsp+0x18]）。
uint64_t stcut_measure_peak(StCutContext &ctx)
{
    stcut_grain_tree_build(ctx);                             // [懒建守卫（与 0x130cea0 同款）]
    uint64_t acc = 0, peak = 0;
    for (uint32_t id : ctx.working_order) {
        acc += stcut_grain_of(ctx, id);                      // [r13 += count]
        peak = std::max(peak, acc);                          // [rax = max(rax, r13)]
    }
    return peak;
}

// ---- 0x1306a20 — 工作序合法性验证器（M28 全解）----
// 逆扫工作序；节点已在依赖闭包中 → "%s:1782:WARNING:invalid in this ordering: %u %llx"
// → 返 0（Bad Schedule 来源）；否则沿快照邻接（ctx+0x1f0）BFS 收集闭包，
// 跳 flags&4 / 快照组标志（ctx+0x1a8）&2 / 已访问。
bool stcut_validate_order(StCutContext &ctx)
{
    size_t n = ctx.working_order.size();
    std::vector<uint8_t> vis(ctx.node_count(), 0);
    for (size_t i = n; i-- > 0; ) {                          // [逆扫]
        uint32_t id = ctx.working_order[i];
        if (id >= vis.size()) continue;
        if (vis[id]) {
            // [invalid 日志（优先级警告级）] —— 实现略日志本体，返回失败位
            return false;
        }
        std::vector<uint32_t> stack = { id };                // [BFS 闭包收集]
        while (!stack.empty()) {
            uint32_t u = stack.back(); stack.pop_back();
            if (vis[u]) continue;
            vis[u] = 1;
            if (u >= ctx.snap_add.size()) continue;
            for (uint32_t rel : ctx.snap_add[u]) {           // [快照邻接 ctx+0x1f0]
                if (ctx.snap_weights[rel].flags & 4) continue;      // [flags&4 跳过]
                uint32_t v = ctx.snap_weights[rel].canon;
                if (v >= vis.size()) continue;
                if (ctx.snap_flags[v] & 2) continue;                // [快照组标志 ctx+0x1a8 &2]
                if (vis[v]) continue;
                stack.push_back(v);                                // [push closure]
            }
        }
    }
    return true;
}

// ---- 0x1306750 — 本轮调度准备（M28；M27 §1 调用点 1304104-130412a）----
// 首轮拍五表快照（0x1307780，guard [ctx+0x220]）→ 0x130cea0(0,n) 全量 aux 曲线
// → 0x130d3e0 develop_schedule。
void stcut_prep_round(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                      std::vector<uint64_t> &aux, uint64_t budget, int32_t opt1)
{
    if (!ctx.snapshot_taken)
        stcut_snapshot_five(ctx);                            // [1306782 call 0x1307780（首轮 guard）]
    uint32_t n = (uint32_t)ctx.working_order.size();
    stcut_aux_fill(ctx, aux, 0, n);                          // [0x130cea0(0, n) 全量]
    std::vector<uint32_t> out_vec;
    stcut_develop_schedule(ctx, arg2, arg3, out_vec, aux, budget, opt1);  // [→ 0x130d3e0]
}

// ---- 0x130d3e0 — develop_schedule（M29 全解 + M30 六件套）----
void stcut_develop_schedule(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                            std::vector<uint32_t> &out_vec,
                            std::vector<uint64_t> &aux,
                            uint64_t inner_budget, int32_t max_iters)
{
    std::vector<uint32_t> local_order = ctx.working_order;   // [130d454-130d48b 局部深拷贝 → [rsp+0x140]]

    // ---- journal 初始化 [130d4d3-130d618] ----
    uint16_t id2 = ++ctx.next_rec_id;                        // [[ctx+8] 计数器 +1 → 事件 id]
    StCutRec r1{};                                           // #1 kind=0（init）
    StCutRec r2{};                                           // #2 kind=2、@0x14=arg2、区间长=n、区间起=0
    r2.link_fwd = 2;                                         // [M29 §7：初始记录 kind=2 与链哨兵同值，两用性未定论]
    r2.add_entry = arg2;
    r2.span_len = (uint32_t)ctx.working_order.size();
    r2.range_start = 0;
    r2.id = id2;
    StCutRec r3{};                                           // #3 kind=0、@0x10=arg3、@0x30=n
    r3.sub_entry = arg3;
    r3.counter = ctx.working_order.size();
    ctx.journal.clear();
    ctx.journal.push_back(r1);
    ctx.journal.push_back(r2);
    ctx.journal.push_back(r3);

    // [130d61d-130d7ab] ctx+0x98 归属表扩容到节点数并全填记录#2 的事件 id
    ctx.ownership.assign(ctx.node_count(), id2);

    int32_t iters = max_iters;                               // [130d7ad ebp = arg7 迭代上限]
    size_t cursor = 0;                                       // [[rsp+0x30] 游标，130df50 每轮 ++]
    // [rsp+0x8] 轮初哨兵 0x7FFFFFFFFFFFFFFC [130d7f9-130d814]；Path A 提案器返回值
    //   盖写 [130db11]；落地段取低 32 位 push 进 out_vec [130de5b-130de60]
    uint32_t last_prop = (uint32_t)0x7FFFFFFFFFFFFFFCu;
    std::set<uint32_t> proposed_pivots;                      // 提案器 arg5 排除集（跨轮持久）

    // ---- 主循环 [130d820-130e0c6] ----
    while (true) {
        if (ctx.inner_loop_count >= inner_budget) {          // [130d7d1/130e117 超时门]
            // log "%s:3331:WARNING:TIMEOUT: develop_schedule() hit inner-loop timeout
            //      with %d iterations remaining"（st_cut.cc）——日志本体略
            break;
        }
        if (iters <= 0) break;                               // [ebp≤0 → 直达终段 130e143]

        // 2a. 每 5 轮进度日志 [130d890 前置]（"%s:3101:Schedule-cutting has %d
        //     remaining iterations or %zu inner-loops"）——略

        // 2b. 评分扫描 [130d890-130daa8]
        // 阈值（.so 逐指令 [130d8ec-130d94b]）：
        //   v = (double)(u64)[ctx+0x278]（uitofp 展开：unpcklps 拼低/高 32 位为
        //   2^52/2^84 两段精确 double 相加 [130d8f5/130d8fc]，常数 0x3970340/
        //   0x3970350 = {0x43300000,0x45300000} 已从 .so 提取验证）
        //   × [[ctx+0x270]+0x128] → cvttsd2si [130d91f] → 负数 +0x7ff 修正
        //   [130d924-130d92e] → sar 11 = 截断除 2048 [130d94b]
        const double thr_prod = (double)ctx.threshold_base * ctx.config.threshold_mult;
        int64_t thr_q = (int64_t)thr_prod;                   // [130d91f cvttsd2si]
        int64_t thr = (thr_q >= 0 ? thr_q : thr_q + 0x7ff) >> 11; // [130d92e/130d94b]
        // 反汇编未完全理解：[ctx+0x278] 语义（位模式经 utofp 参与运算，字段
        // 疑为总可用内存字节——其 C++ 源型应为整型；此处按 u64→double 忠实计算）
        ptrdiff_t best = -1;
        uint64_t best_score = 0;
        for (size_t i = 0; i < ctx.journal.size(); ++i) {    // [130d995 64B 步进遍历]
            const StCutRec &rec = ctx.journal[i];
            if (rec.link_fwd == 0) continue;                 // [130d9a1 rec@0==0 死记录跳过]
            uint64_t score = 0;
            if (rec.span_len > 0) {                          // [130d9b7 span≤0 → 评分 0]
                uint32_t start = rec.range_start;            // [130d9b1 movslq ebx=起]
                for (uint32_t k = 0; k < rec.span_len; ++k) { // [130d9e0-130da3a 2× 展开]
                    uint64_t a = aux[start + k];
                    score += a + ((int64_t)a > thr ? 3 * a : 0); // [130d9ec cmovle：>thr 计 4 倍]
                }
            }
            // [130d970-130d98f] 更优门：无 best（r11==0）或严格大于 → 登记索引与分
            if (best < 0 || (int64_t)score > (int64_t)best_score) {
                best_score = score;
                best = (ptrdiff_t)i;
            }
        }
        if (best < 0) break;
        StCutRec &brec = ctx.journal[best];

        // 2b'. 历史切点复用 Path B [130da40-130da8e]（out_vec 元素数 ≠ 游标时）
        //   node = out_vec[cursor]（arg4 数据 [130da40]）；
        //   rid = ownership[node]（ctx+0x98 u16 表 [130da50]——节点→持有记录事件 id）；
        //   从链头 1 沿 link_fwd 步进 [130da7e]，rec.id==rid 命中即 best=该记录
        //   [130da77/130da7c]；步到 link==2 → best=2 兜底 [130da82-130da88]。
        //   ★Path B 直接替换 Path A 的评分结果，且跳过提案器 [130dadf jne→130db16]。
        std::vector<uint32_t> cut_before, cut_after;          // 提案器 arg6/arg7 切集两半
        if (out_vec.size() != cursor) {
            uint32_t node = out_vec[cursor];                 // [130da40 u32 数组取元素]
            uint16_t rid = ctx.ownership[node];              // [130da50 归属表读取]
            best = 1;                                        // [130da54 r15d=1 链头]
            while ((size_t)best < ctx.journal.size()) {
                if (ctx.journal[best].id == rid) break;      // [130da77 cmpw 命中]
                uint64_t nxt = ctx.journal[best].link_fwd;   // [130da7e 链步进]
                if (nxt == 2) { best = 2; break; }           // [130da82 链尾哨兵 → 兜底 2]
                best = (ptrdiff_t)nxt;
            }
            brec = ctx.journal[best];
        } else {
            // 2c. 提案 [130db08]（arg5=排除集，arg6/arg7=切集两半——M32）
            //   返回值（切点）盖写 [rsp+0x8] [130db11]，落地段低 32 位入 out_vec
            last_prop = stcut_cut_propose(ctx, (size_t)best, proposed_pivots, cut_before, cut_after);
        }

        // 2c'. 执行（rdtsc 夹时 [130db16-130dba0]）→ [ctx+0x280] += 耗时
        //   参数 [130db34-130db76]：esi=add_entry(best)；edx=journal[best.link_fwd]
        //   .sub_entry（fwd 记录）；ecx=cursor（游标，[rsp+0x18] 经 subq 位移读取）
        uint64_t t0 = StCutContext::tick();
        uint16_t new_id = ++ctx.next_rec_id;                 // [新记录号（分裂器将盖到子记录 @0x38）]
        std::vector<uint32_t> out, exported;
        stcut_cut_execute(ctx, brec, ctx.journal[brec.link_fwd].sub_entry,
                          (uint32_t)cursor, new_id, cut_before, cut_after,
                          out, exported);
        ctx.cut_time += StCutContext::tick() - t0;           // [130dbaf-130dbc3]

        // 2d. 落地 [130dbc3-130ddc0]
        // |out| == rec.span_len → 逐 id 写归属；否则先 0x130b9a0 修正工作序再写。
        // ★M29 记「写 journal[best].id@0x38」与执行器写「新记录号」并存；唯一自洽读法
        //   （0x130b9a0 以「属新记录号」为划分键，M30 §2 实证）= 统一写新记录号。
        if (out.size() != brec.span_len)
            stcut_cut_fix_range(ctx, (size_t)best, out, new_id);  // [130dc5b]
        for (uint32_t id : out)
            ctx.ownership[id] = new_id;                      // [逐 id 写 ctx+0x98]

        stcut_aux_fill(ctx, aux, brec.range_start, brec.span_len); // [130dc7f 重填被切区间]

        if (ctx.config.deep_copy_gate) {                     // [130dc9b 门 [[ctx+0x270]+0x160]]
            // [130dcf0] 区间内找 aux 非降前缀末点
            uint32_t end = brec.range_start + brec.span_len;
            uint32_t p = brec.range_start;
            while (p + 1 < end && aux[p + 1] >= aux[p]) ++p;
            if (aux[p] > STCUT_PREFIX_SENTINEL) {            // [前缀越过水位 [rsp+0x38]]
                local_order = ctx.working_order;             // [0x8dc340 深拷同步局部工作序]
                for (uint32_t k = brec.range_start; k <= p && k < end; ++k)
                    ctx.ownership[ctx.working_order[k]] = new_id;  // [再补一次归属标记（细节概要级）]
            }
        }

        bool verdict = stcut_cut_verdict(ctx, (size_t)best, exported);  // [130dddc 0x130d050]
        stcut_cut_split_record(ctx, (size_t)best, exported, (uint32_t)out.size(),
                               verdict ? 1 : 0);             // [130de22 0x130b600]
        // [130de2b-130de6b] 仅当 out_vec 元素数 == 游标时 push [rsp+0x8] 低 32 位
        //   （= 提案器返回值；Path B 轮为轮初哨兵 0x7FFFFFFFFFFFFFFC 的截断值，
        //   即 0xFFFFFFFC）——.so 与 Path 判定同用 count==cursor 这一等式
        if (out_vec.size() == cursor)
            out_vec.push_back(last_prop);
        ++cursor;                                            // [130df50 游标++]
        --iters;

        // 2e. 随机重启 [130df50-130e06a]
        if (ctx.config.restart_gate >= 2 &&
            StCutContext::rand_next(ctx) % ctx.config.restart_gate == 0) {
            uint64_t r = StCutContext::rand_next(ctx);       // [再取一 rand]
            // [130dfcb-130e035 链上等分布选一条活记录；记录死（link_fwd==0）则放弃]
            size_t pick = (size_t)(r % ctx.journal.size());
            if (ctx.journal[pick].link_fwd != 0)
                stcut_random_restart(ctx, pick);             // [130e053 0x130bc70]
        }
        // 2f. 循环尾 [130e060-130e0c6]：释放临时向量 → 回边
    }

    // ---- 终段：journal → 工作序发射 [130e143-130e801] ----
    if (!ctx.config.emit_terminal)                           // [门 [[ctx+0x270]+0x110]]
        return;
    stcut_net_weight(ctx);                                   // [130e15a 0x130a560 收尾步]

    for (size_t i = 0; i < ctx.journal.size(); ++i) {        // [逐 journal 记录，步长 0x40]
        const StCutRec &rec = ctx.journal[i];
        if (rec.span_len == 0) continue;                      // [130e1f8 cmpq start,end; jae → span≤0 跳过]
                                                             // （死记录 #1 全零模板 span=0 → 自然落此门）
        struct Item { uint32_t id; uint64_t net; };
        std::vector<Item> batch;
        std::set<uint32_t> live;
        for (uint32_t k = 0; k < rec.span_len; ++k) {        // [区间收集]
            uint32_t id = ctx.working_order[rec.range_start + k];
            uint64_t net = (id < ctx.net_weight.size()) ? ctx.net_weight[id] : 0;  // [[ctx+0x140][id]]
            batch.push_back({ id, net });                    // [批向量 push {u32 id, net}（16B/条）]
            live.insert(id);                                 // [set<u32>（32B 树节点）]
        }
        // [130e5a8-130e600] 0x13125e0 std::sort（深度限 2·log2n），比较器 0x12fffb0：
        //   [12fffb2 cmpq/12fffb5 jl] 净重 payload 严格升序在前；
        //   [12fffba cmpl/12fffc1 setb+and] 平值（payload_a≤payload_b 且相等）→ 节点号升序。
        std::sort(batch.begin(), batch.end(),
                  [](const Item &a, const Item &b) {
                      return a.net < b.net || (a.net == b.net && a.id < b.id);
                  });

        // 多趟 Kahn 拓扑 [130e620-130e7ee]：flags&4 依赖的 canon 仍在 set → 本趟跳过
        uint32_t write = rec.range_start;
        while (!live.empty()) {
            uint32_t before = write;                        // 本趟发射起点（趟内无进展检测）
            for (const Item &item : batch) {
                if (!live.count(item.id)) continue;
                bool blocked = false;
                for (uint32_t rel : ctx.add_set[item.id]) {  // [ctx+0xe0 邻接]
                    if (!(ctx.weight_table[rel].flags & 4)) continue;  // [非&4 忽略]
                    uint32_t canon = ctx.weight_table[rel].canon;
                    if (live.count(canon)) { blocked = true; break; }  // [canon 仍在 set → 跳过]
                }
                if (blocked) continue;
                live.erase(item.id);                         // [可发射：删出 set（130e700-130e78e）]
                ctx.working_order[write++] = item.id;        // [130e79b ★原地写回 range_start++ = id]
            }
            if (write == before && !live.empty()) {
                // [130e810 eb fe unreachable 自旋]（批空而 set 非空的理论不可达分支）
                // —— 重实现以按序直发剩余节点代替无限自旋（不阻塞宿主进程）
                for (const Item &item : batch)
                    if (live.count(item.id)) { live.erase(item.id); ctx.working_order[write++] = item.id; }
                break;
            }
        }
    }
}

// ---- 迭代预算公式（M27 §2；130408b 与 1304363 同式，操作数不同）----
// 预算 = (opt×1e6 + 3e9 − n×530000) / (n/3226 + 26)
// 常量：0xf4240=1e6、0xB2D05E00=3e9(3GHz)、0x81650=530000、除数修正 +0x1a(26)。
// [1304393-130439e] 除数项 = (n×0x1450f0)>>32 + 0x1a：0x1450f0 是 n/3226 的
// 免修正上取整魔数（3226×0x1450f0 − 2^32 = 258144 → 解析上界 n<16639 精确；
// 暴力验证首个失配点 n=19355）。等价实现直接用 n/3226。
uint64_t stcut_iteration_budget(int32_t opt, size_t n)
{
    uint64_t n64 = (uint64_t)n;
    uint64_t denom = ((n64 * 0x1450f0ULL) >> 32) + 26;       // [1304393 imul 0x1450f0 / 130439a shr 32 / 130439e add 0x1a]
    int64_t numer = (int64_t)opt * 1000000 + 3000000000LL
                  - (int64_t)n64 * 530000;                    // [0xf4240 / 0xB2D05E00 / 0x81650]
    if (numer < 0) numer = 0;
    return (uint64_t)numer / (int64_t)denom;
}

// ============================================================================
// full_schedule 重试循环（M27 §1/§3 逐指令实证）—— st_cut.cc 主入口
// 三份节点序拷贝分工（M27 §0 表）：
//   本实现: original_order（快照#2 [rsp+0x90]）/ ctx.working_order（工作序
//   [rsp+0x1140]）/ best_snap（快照#1 [rsp+0x10e0] 最优序）
// ============================================================================
void stcut_full_schedule(StCutContext &ctx, const StCutOptions &opt,
                         const std::vector<uint32_t> &original_order,
                         std::vector<uint32_t> &best_order,
                         std::vector<uint64_t> &flows,
                         std::vector<uint64_t> &cycles,
                         uint32_t s_seed, uint32_t t_seed)
{
    ctx.working_order = original_order;                      // [1303ca4 进循环前快照#2，仅一次]
    std::vector<uint32_t> best_snap;                         // [rsp+0x10e0 快照#1]
    uint64_t best_flow = std::numeric_limits<uint64_t>::max(); // [[rsp+0x18] 历史最优 flow]
    // [rsp+0x68] flow 预算上限 = u64(this+0x6830) × double(this+0x55f8=TR)（M26 §1.3/M27 §4）
    uint64_t flow_cap = (uint64_t)((double)opt.budget_base * opt.tr);
    // [rsp+0x1f0] 继续/达标标志 [1303ff0 写点已解]：cont = (opt_get(0x130ebe0).rax & 1)
    //   ? (其 rdx ≠ 0) : (this+0x5618 ≠ 0) [1303fd7-1303ff0 setne/cmovnel]。
    //   —— 选项 getter 的有状态返回未建模，重实现取默认 this+0x5618≠0 路径 → 恒 1。
    uint8_t cont_flag = 1;
    // [rsp+0x1530] 已用量 = ctx+0x2a0（帧内嵌 ctx 基 [rsp+0x1290]，0x1290+0x2a0=0x1530）：
    //   构造清零 [12fbc0f]、每次迭代原语 +1 [13093f1]、develop 预算门读 [130d7d1/130d820]、
    //   重试循环头 budget+used 传 prep_round [13040fc]、循环尾 used≤budget2 续轮 [13043b3]。
    uint64_t used = 0;
    std::vector<uint64_t> aux(ctx.node_count() + 1, 0);
    uint32_t round = 1;                                      // [r14 轮次，1 起]
    bool stop = false;
    uint64_t t_round0 = StCutContext::tick();                // [[rsp] 本轮计时起点]

    for (;;) {
        // ① [1303ee9-1304060] 选项装载：6×csv_int + 1×csv_float + 布尔，按 (轮次−1) 字段重载，
        //   未置位/解析失败回退 this+0x55d4/0x55d8/… 默认值。
        //   —— 本实现以 StCutOptions 固定值代替（CSV 逐轮重载未重放，地址注释留档）。
        size_t n = ctx.node_count();                         // [130409b n = 工作表#1 元素数]

        // ② [130408b] 迭代预算（第一遍，opt = IT[+0x55d8]）+ [rsp+0x1530] 已用量
        uint64_t budget = stcut_iteration_budget(opt.it, n) + used;

        // ③ [1304104-130412a] 0x1306750（装载本轮参数 + 首轮拍五表快照 + develop_schedule）
        stcut_prep_round(ctx, s_seed, t_seed, aux, budget, opt.rg);  // [选项#1 Rg 作 r9d]

        // ④ [1304136] 0x1306a20 执行体验证（M28：验证序合法性）
        bool ok = stcut_validate_order(ctx);
        if (!ok) {                                           // [130413b al≠0 → 跳过失败路]
            ctx.working_order = original_order;              // [130413f copy(快照#2→工作序)]
            // [1304181] 日志 "%s:948:WARN: Bad Schedule Detected (seed=%u)"（seed=轮次号）
            if (opt.am) {                                    // [1304186 cmpb this+0x55e8 → fatal 路]
                // [1304b48] "%s:950:Fatal error" —— 直接中止（AM=fatal-on-bad-schedule）
                best_order = best_snap.empty() ? original_order : best_snap;
                return;
            }
        }

        // ⑤ [1304193] 0x13065f0 → 本轮 flow 量（grain 数；×2048=字节）
        uint64_t flow = stcut_measure_peak(ctx);
        // ⑥ [13041a5/13042ce] FLOWS/CYCLES 序列记录
        flows.push_back(flow);                               // [vector#3 push r15]
        uint64_t elapsed = StCutContext::tick() - t_round0;  // [13042ad rdtsc/16 − 本轮起点]
        cycles.push_back(elapsed);                           // [13042ce 0x853d80 push 耗时]
        // [13042e9] FlowRetry 日志（优先级 11）"%s:959:FlowRetry: %llx"（%llx = flow<<0xb 字节）

        // ⑦ [130431e/1304325] 更优 → 保存最优序到快照#1
        if (flow < best_flow) {
            best_flow = flow;
            best_snap = ctx.working_order;                   // [copy(工作序→快照#1)]
        }

        // ⑧ [1304347/130434f] 判定：重试标志 = (flow×2048 > 预算上限) or (标志==0)
        bool over_budget = (flow * STCUT_GRAIN_BYTES) > flow_cap;   // [setg cl: r15<<0xb > [rsp+0x68]]
        bool flag_zero = (cont_flag == 0);                           // [sete sil: [rsp+0x1f0]==0]
        bool retry = over_budget || flag_zero;                       // [13043f9 ebp = cl|sil]

        // ⑨ [1304363] 迭代预算（第二遍，opt = Rt[+0x55dc] 选项#3）→ 与已用量比较判超时
        used = ctx.inner_loop_count;                         // [[rsp+0x1530] 更新（等价代入，见上注）]
        uint64_t budget2 = stcut_iteration_budget(opt.rt, n);
        if (used > budget2) {                                // [13043b3 cmp %rax,%rdi; jbe 正常]
            // [13043e7] 日志 "%s:971:WARNING:TIMEOUT: sched_outer_timeout=%llu
            //      after stcut #%d"（%llu=选项#3 值；#%d=0 基轮次）
            stop = true;                                     // [ebp=0 停止重试]
            retry = false;
        }

        // ⑩ [13043ff/1304434-13044a8] 工作序重置 ← 原始序；ctx 五表重置 ← 快照表
        ctx.working_order = original_order;                  // [copy(快照#2→工作序)]
        stcut_restore_five(ctx);                             // [1304434-13044a8 五表恢复]

        // ⑪ [13044b2] 轮次 ≥ Rt 上限 → 退出；否则 r14++、记下轮计时起点
        if ((int32_t)round >= opt.rt) break;
        ++round;
        t_round0 = StCutContext::tick();                     // [13044bb [rsp]=r12 下轮起点]

        // [13044c5 test %ebp,%ebp; jne 1303ee0] ← 回边
        if (!retry || stop) break;
    }

    // ⑫ [13044cb/13044e3] 退出：工作序 ← 最优序（最终输出）
    best_order = best_snap.empty() ? original_order : best_snap;

    // ---- 后循环段（M27 §3b，实测执行序）----
    // [13044e8-1304510] 计时 push("stcut_full_schedule")
    // [1304515-130453d] 计时 push("stcut_delay_dma_again")
    // [1304542] call 0x12fc740(rdi=ctx, rsi=[rsp+0x50], rdx=&工作序)
    //   —— delay_dma_again 内部未全解（M24 收尾块有部分注），未重放；
    // [130455c] rdtsc/16 新基准
    // [130456f-1304591] 计时 push("stcut_convert_to_ids") —— convert_to_ids 已收窄为
    //   完全内联于统计块区间（M27 §5-6），未划出指令段，未重放。
    // [1304596] this+0x5634 统计块总开关（SCHED_OPTIONS/SCHED_GRAINS_* 打印）——纯日志，未重放。
}

} // namespace hnnx
