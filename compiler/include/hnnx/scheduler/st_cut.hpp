#pragma once
// ============================================================================
// st_cut.hpp — libHtpPrepare.so「stcut 调度/切分」管线重实现（M17–M30 结论可执行化）
//
// 源对象: real_so/libHtpPrepare.so（SDK 2.48.40.260702，x86-64）
// 宿主函数: GraphPrepare::schedule_for_alloc @ 0x1302900（st_cut.cc 内联全链）
//
// 管线全景（每个函数体首行注释 = 反汇编地址 + 里程碑出处）：
//   schedule_for_alloc @0x1302900
//     └ full_schedule 重试循环 @1303ee0–13044c5 (M27)
//         每轮: 0x1306750 准备(首轮快照五表 + aux 全量曲线) (M28)
//           → 0x1306a20 执行体(验证序合法性) (M28)
//               → 0x130d3e0 develop_schedule (M29/M30)
//                   ① 0x130ab30 切点提案   ② 0x1309a60 泛洪迁移(归属改写)
//                   ③ 0x130b9a0 工作序修正 ④ 0x130d050 分裂判定
//                   ⑤ 0x130b600 记录分裂    ⑥ 0x130bc70 随机重启
//                   终段: 0x130a560 净重 → 批排序(0x12fffb0) + flags&4 Kahn 拓扑
//                         → 原地写回工作序
//           → 0x13065f0 评分(内存峰值 = grain 前缀和运行最大)
//         循环退出 → 恢复最优序 → 0x12fc740 delay_dma_again → convert_to_ids
//
// 约定（沿 DISASM_PLAN 验收标准）：
//   * 每个逻辑段以 [0x地址] 注释对应反汇编指令；
//   * 未定论处显式标注「反汇编未完全理解」，不臆断、不编造；
//   * 结构体字段名后标注其在 .so 中的 ctx 偏移/记录偏移。
// ============================================================================

#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <string>

namespace hnnx {

// 常量（.so 中逐指令实测）
constexpr uint64_t STCUT_MAX_WEIGHT   = 99999999;   // 0x5f5e0ff [1300e0a/1309886/130100d]
constexpr uint64_t STCUT_BATCH_CAP    = 100000000;  // 1e8 冷热路径分界 [1309ab6]
constexpr uint64_t STCUT_SMALL_STEP   = 10000;      // 0x2710 小步提前收敛 [13097d0]
constexpr uint64_t STCUT_GRAIN_BYTES  = 2048;       // grain→字节换算 [13042e9 <<0xb / M29 §2b]
constexpr uint64_t STCUT_MIN_BUCKETS  = 8;          // 判定器最少直方图桶数 [130d123]
constexpr uint64_t STCUT_VERDICT_FIX  = 0x2AAAAAAA; // 1/6 定点乘子 [130d18c]
constexpr uint64_t STCUT_VERDICT_THR  = 0x340000000;// 判定阈值 [130d366]
constexpr uint64_t STCUT_PREFIX_SENTINEL = 0x5f5e0ff; // 深拷门水位哨兵 [130d7ef]

// ============================================================================
// 数据结构
// ============================================================================

// ---- 权重表槽（ctx+0xb0，24B/槽）[M23 §2.4 / M24 §3] ----
// 一条「替换对」= 互指的 A/B 两槽（0x1300cc0 建立）：
//   slot A: canon=对端节点号, idx=B, weight=阈值参数, flags=[ctx+0xc]
//   slot B: canon=源节点号,   idx=A, weight=99999999, flags=[ctx+0xc]|4
struct StCutSlot {
    uint32_t canon;   // @0x00 field0：对端（B 侧为源节点号）
    uint32_t idx;     // @0x04 互指对侧槽号（A.idx=B，B.idx=A）
    uint64_t weight;  // @0x08 权重（grain 量纲；B 侧恒 0x5f5e0ff）
    uint32_t flags;   // @0x10 filter：bit2(0x4)=跨记录关系（切面边/硬依赖）
};
static_assert(sizeof(StCutSlot) == 24, "weight slot must be 24B [1300e02-1300e13]");

// ---- 重放账本记录（24B）[M22 §3 / M23 §2.7] ----
// 0x1300cc0 命中路径 type=0（改权重）、未命中 type=2（移出 add 集）；
// 0x130bc70 重启用 type=4（重新加入 add 集）；0x1307eb0 建记录 type=4 交 0x13080a0 重放。
struct StCutReplayRec {
    uint32_t op;      // @0x00 权重表槽号（或节点号，随 type）
    uint64_t value;   // @0x08 旧值/数量
    uint32_t type;    // @0x10 6-case 分派号（见 replay_ledger）
};
static_assert(sizeof(StCutReplayRec) == 24, "replay record must be 24B [1307f84 movups]");

// ---- journal 记录（64B，0x130b840 push_back）[M29 §1 / M30 §4 + asm 复核] ----
// 一条记录 = 工作序的一个「区间」及其切分历史。@0 与 @8 同清 = 死记录。
struct StCutRec {
    uint64_t link_fwd;   // @0x00 链后继记录索引；清零即死 [130b702/130bcc8]
    uint64_t link_back;  // @0x08 父回链 [130b70c/130bcc8]
    uint32_t sub_entry;  // @0x10 subtract 集槽号（分裂器写 A 槽 [130b6aa-130b6b8]）
    uint32_t add_entry;  // @0x14 add 集槽号（初始记录#2 为 arg2；分裂器写 B 槽）
    uint64_t span_len;   // @0x18 区间长 u64（0x130b600: −= cut_count [130b698 subq]）
    uint32_t field20;    // @0x20 判定 u8（130b6c5 movb，M30 表格「@0x20=B_idx」为栈帧误读）
    uint32_t field24;    // @0x24 分裂器第 6 参 r9d（语义未解，置 0）
    uint32_t range_start;// @0x28 区间起（子记录继承父区间尾 [130b6d1-130b6df @0x18+@0x28]）
    uint32_t pad_2c;     // @0x2c
    uint64_t counter;    // @0x30 元素计数（子记录 = Σ切集权重 [130b7ec movq r15]）
    uint16_t id;         // @0x38 u16 事件 id（[ctx+8]++ 分配）[130d4d3/130b6ed]
    uint16_t pad_3a;
    uint32_t pad_3c;
};
static_assert(sizeof(StCutRec) == 64, "journal record must be 64B [130b840 步长 0x40]");

// ---- 调度配置（[ctx+0x270] 对象的已解字段）[M29 §2b/§2e/§3] ----
struct StCutConfig {
    bool     emit_terminal = true;  // +0x110 终段发射门 [130e143]
    double   threshold_mult = 1.0;  // +0x128 评分阈值乘子 [130d890]
    uint32_t restart_gate = 0;      // +0x158 随机重启门（≥2 才启用）[130df50]
    uint32_t jitter = 0;            // +0x15c 提案器抖动门（≠0 启用）[M30 0x130ab30]
    bool     deep_copy_gate = false;// +0x160 非降前缀深拷门 [130dc9b]
};

// ---- CSV 选项表（schedule_for_alloc this+0x55d4…，SCHED_OPTIONS）[M26/M27 §4b] ----
struct StCutOptions {
    int32_t  ot = 0;          // +0x55e4 OT
    int32_t  it = 0;          // +0x55d8 IT（预算公式选项#2 分子乘子 ×1e6）
    int32_t  rg = 0;          // +0x55d4 Rg（选项#1，传 0x1306750）
    int32_t  rt = 0;          // +0x55dc Rt（最大轮数；超时公式选项#3）
    int32_t  ab = 0;          // +0x55e0 AB
    uint8_t  am = 0;          // +0x55e8 AM（Bad Schedule→Fatal 开关 [1304186 cmpb]）
    uint8_t  eo = 0;          // +0x5618 EO
    uint8_t  hd = 0;          // +0x5619 HD
    uint8_t  dd = 0;          // +0x561a DD
    double   tr = 0.0;        // +0x55f8 TR（超时预算乘子：u64(+0x6830)×TR）
    double   lt = 0.0;        // +0x5620 LT
    int32_t  rc = 0;          // +0x5628 RC
    int32_t  rp = 0;          // +0x562c RP
    uint8_t  db = 0;          // +0x5630 DB
    uint64_t budget_base = 0; // this+0x6830（TR 乘子基數，grain）
    uint32_t stats_gate = 0;  // this+0x5634 统计块总开关
};

// ============================================================================
// StCutContext —— 0x12fba20 构造 / 0x13073c0 析构（M24 §3 / M25 §2c）
// 成员注释中的 [ctx+0xXX] = .so 中该字段的偏移；快照表 [ctx+0x190…] 与工作表
// [ctx+0x68…] 一一对应（0x1307780 拍 / 0x1307170 还原）。
// ============================================================================
class StCutContext {
public:
    // ---- 基础状态 ----
    uint16_t next_rec_id = 0;      // +0x008 u16 记录 id 计数器 [130d4d3 +1 / 0x130b600 取新 id]
    uint64_t call_count = 0;       // +0x010 调用计数 [130981e +=1]
    uint32_t mode = 0;             // +0x00c 0x40=建表期(12fba54) / 0=复用期(1307812,130d1c2)
    bool     tree_built = false;   // +0x018 权重树懒建守卫 [0x130ccd0]
    std::map<uint32_t, uint64_t> grain_tree; // +0x020 节点号→grain 权重（懒建，只读）
    std::vector<uint32_t> delay_dma_src;     // +0x038 delay_dma_again 投影源

    // ---- 五张工作表（0x1307780 快照 / 0x1307170 还原）----
    std::vector<uint32_t> node_slots;    // +0x068 节点槽表（元素=节点号；.so 存节点指针，8B/条）
    std::vector<uint8_t>  group_flags;   // +0x080 组标志字节表（0x83=退役 [1307af2]；bit0x2=冻结组 [1309429]；&3==0=可泛洪 [1309d60]）
    std::vector<StCutSlot> weight_table; // +0x0b0 权重表（24B/槽）
    std::vector<std::vector<uint32_t>> add_set;      // +0x0e0 每节点 add 集（元素=权重槽号）
    std::vector<std::vector<uint32_t>> subtract_set; // +0x0f8 每节点 subtract 集

    // ---- 快照五表（0x1307780 拍自工作表；guard [ctx+0x220]）----
    std::vector<uint32_t> snap_nodes;      // +0x190
    std::vector<uint8_t>  snap_flags;      // +0x1a8
    std::vector<StCutSlot> snap_weights;   // +0x1c0
    std::vector<std::vector<uint32_t>> snap_add;      // +0x1f0
    std::vector<std::vector<uint32_t>> snap_subtract; // +0x208
    bool snapshot_taken = false;           // +0x220 快照 guard（首轮拍，此后只还原）

    // ---- 其它表 ----
    std::vector<uint64_t> threshold_ledger; // +0x0c8 阈值账本（u64/权重槽 [13097b3]）
    std::vector<uint32_t> bitmap;           // +0x110 每槽 u32 位图（[1309433 置位 / 130975b bt]）
    std::vector<uint16_t> mark_table;       // +0x128 三态标记表（u16/节点 [1309b38 memset]）
    std::vector<uint64_t> net_weight;       // +0x140 净内存增量表（终段前 0x130a560 整体覆写）
    std::vector<uint16_t> ownership;        // +0x098 节点→记录号归属表（u16 [13061d-130d7ab 扩容填 #2 id]）
    std::vector<uint32_t> working_order;    // 经 [ctx+0x158] 指针暴露（重试循环三拷贝之一）
    std::vector<uint32_t> free_node_slots;  // +0x160 节点槽空闲栈（0x1307aa0 push / 0x1301040 pop）
    std::vector<uint32_t> free_weight_slots;// +0x178 权重槽空闲栈（0x1300cc0 pop 2）

    // ---- 配置/计时/统计 ----
    StCutConfig config;            // +0x270 已解字段见 StCutConfig
    uint64_t threshold_base = 0;   // +0x278 评分阈值被乘数（语义未完全理解，疑似总可用内存字节）
    uint64_t cut_time = 0;         // +0x280 累计切分执行耗时（rdtsc/16 拍）[130dbaf]
    uint64_t batch_time = 0;       // +0x288 批量搬运耗时 [13098bb]
    uint64_t probe_time = 0;       // +0x290 探测耗时 [130991d]
    uint64_t inner_loop_count = 0; // +0x2a0 inner-loop 计数：ctor 清零 [12fbc0f]、迭代原语 +1
                                   // [13093f1]、develop 预算门 [130d7d1]、重试循环已用量
                                   // [rsp+0x1530]=[ctx+0x2a0]（循环头 +预算 [13040fc] /
                                   // 循环尾 ≤budget2 续轮 [13043b3]）
    uint64_t rng_state = 0;        // +0x2a8 RNG 状态（0xd79830 rand 消费）

    // ---- 供 develop_schedule / 重试循环使用的运行态 ----
    std::vector<StCutRec> journal;             // journal（0x130b840 push）
    std::vector<StCutReplayRec> replay_ledger; // 24B 账本（0x1307f60 push）
    // +0x228 map A：组键（张量级强关联键，load_replacement_plan 装载期填充，
    //   插入方在 0x1299ff4 主体内、当前 asm 转储范围之外）→ 成员节点表。
    //   语义实证：strong_relevel@0x12fd600 遍历每组取成员 u16 层级 [12fd6cc]
    //   做组内层级归并 —— 同组节点必须一起分层。
    std::map<uint64_t, std::vector<uint32_t>> relation_groups;
    std::vector<std::vector<uint32_t>> cand_tables; // +0x240 提案器候选表（stcut_read_nodes 填充）
    std::vector<uint64_t> group_list_keys;          // +0x258 组表收集（.so 存表指针，此处存组键）

    // ---- 计时原语：.so 用 rdtsc/16（M23 §3.4 双采样 >>4 与 >>3&~1，净效果 /16）----
    static uint64_t tick();
    static uint64_t rand_next(StCutContext &ctx); // 0xd79830 rand@plt，状态 ctx+0x2a8

    // 节点数（[ctx+0x70−ctx+0x68]>>3 [130a566 N 计算 / 130409b 预算公式 n]）
    size_t node_count() const { return node_slots.size(); }
};

// ============================================================================
// 基础层（M31-3）
// ============================================================================
void   stcut_snapshot_five(StCutContext &ctx);            // 0x1307780 五表快照 + [ctx+0xc]=0
void   stcut_restore_five(StCutContext &ctx);             // 0x1307170 五表还原（工作表←快照表）
uint64_t stcut_grain_of(StCutContext &ctx, uint32_t id);  // 0x130ccd0/+0x20 树懒建 + 查询
void   stcut_grain_tree_build(StCutContext &ctx);         // 0x130ccd0 主体（guard +0x18）

// ============================================================================
// 建图层（M31-4）
// ============================================================================
// 0x1300cc0 (M25: stcut_connect_nodes)。cb = 调用者提供的记账账本（第 5 参）：
//   建对路推 {op=A, 0, type2}（撤销建对）；命中路推 {op=A, 旧权重, type0}（恢复权重）。
//   M34 破译：脚手架经此记账，重放 type2 从四条 per-node 链表整体撤除。
uint32_t stcut_connect_nodes(StCutContext &ctx, uint32_t src, uint32_t dst,
                             uint64_t threshold, std::vector<StCutReplayRec> *cb = nullptr);
uint32_t stcut_register_node(StCutContext &ctx, uint32_t node, uint32_t flags); // 0x1301040
void   stcut_retire_node(StCutContext &ctx, uint32_t slot);             // 0x1307aa0
void   stcut_add_push(StCutContext &ctx, uint32_t node, uint32_t wslot);    // 0x1307820
void   stcut_sub_push(StCutContext &ctx, uint32_t node, uint32_t wslot);    // 0x1307960
StCutReplayRec *stcut_replay_push(std::vector<StCutReplayRec> &ledger,
                                  const StCutReplayRec &r);        // 0x1307f60（返回新记录指针）
void   stcut_replay_apply(StCutContext &ctx, std::vector<StCutReplayRec> &log); // 0x13080a0 6-case 重放
void   stcut_repoint_a(StCutContext &ctx, uint32_t wslot, uint32_t new_partner); // 0x1307c90
void   stcut_repoint_b(StCutContext &ctx, uint32_t wslot, uint32_t new_partner); // 0x1307d90
uint64_t stcut_batch_accounting(StCutContext &ctx, uint32_t arg2, uint32_t arg3, uint32_t arg4,
                                std::vector<uint32_t> *arg5, std::vector<uint32_t> *arg6,
                                std::vector<StCutReplayRec> *cb = nullptr); // 0x1309940
uint64_t stcut_iterate(StCutContext &ctx, uint32_t seed, uint32_t target); // 0x1309810 迭代驱动
uint64_t stcut_flood_collect(StCutContext &ctx, uint32_t seed, uint32_t target); // 0x1309230
uint64_t stcut_weight_pull(StCutContext &ctx, uint32_t seed, uint32_t target,
                           uint64_t cap, uint32_t depth);              // 0x13096b0
// 0x1305130 — stcut_add_dependencies（M34 全解）：软边正向分层，标记表=ctx+0x128。
//   mark：0=未达 1=在队 ≥2=层号（首层 2）；软前驱全定层才就绪，就绪者扩散软后继。
//   max_levels：.so 原语无界（默认 ∞ 忠实）。仅概要级调用方（build_local_tree，
//   执行器围栏/记录选择未全解，可能造出不收敛锥）传有限值截断——锥留部分标记。
void   stcut_mark_levels(StCutContext &ctx, uint32_t seed,
                         uint32_t max_levels = UINT32_MAX);
// 0x1308c50 — 0x1305130 的 &4 孪生（M34 全解）：f(ctx, 死参, seed, &外部标记表)。
//   只走 flags&4 槽（B 半边 canon=前驱 + 硬边）反向分层；memset 的是外部表。
void   stcut_level_mark_bfs(StCutContext &ctx, uint32_t seed,
                            std::vector<uint16_t> &marks,
                            uint32_t max_levels = UINT32_MAX);
void   stcut_remove_edge(StCutContext &ctx, uint32_t slot,
                         std::vector<StCutReplayRec> *replay = nullptr); // 0x13087f0（全解：
                        // 四链表 swap-remove add_set[p2]←slot/add_set[p1]←pslot/
                        // subtract_set[p2]←pslot/subtract_set[p1]←slot；replay≠0 追加 {slot,0,4}）

// ---- 建图输入（早期阶段的产物，见 st_cut.cpp 头注）----
// 替换对不变式（M34 经 replay type2/type4 四链表撤销反推敲定）：
//   A{canon=dst 后继, 非&4} ∈ add_set[前驱] + subtract_set[后继]
//   B{canon=src 前驱, &4}   ∈ add_set[后继] + subtract_set[前驱]
//   即 add_set[n] 槽的 canon=后继（扩散直取），subtract_set[n] 槽的对端 canon=前驱。
struct StCutGraphInput {
    // hard：输入期表达（M34 修正：不落槽 flags——A&4 会令 0x1308c50 就绪门在硬
    // 前驱上自锁死循环；硬依赖序约束由 Kahn 终段经恒 &4 的 B 半边消费）。
    struct Relation { uint32_t src, dst; uint64_t weight; bool hard; };
    uint32_t node_count = 0;
    std::vector<Relation> relations;
    std::vector<uint32_t> initial_order;   // 初始工作序（节点号排列）
    // 强关联组（map A 的装载期内容）：组键 → 成员节点表（无需有序，read_nodes 会排序）
    std::map<uint64_t, std::vector<uint32_t>> groups;
};
void stcut_build_initial_state(StCutContext &ctx, const StCutGraphInput &in);

// ============================================================================
// 0x12fc820 — stcut_read_nodes（M33 逐指令解码）
// 计时器名实证（st_cut.cc 段名池 0x55b3de5）；调用点 schedule_for_alloc [13030e0]
// （14 段建图准备链的第 1 段，位于选项重试循环之前）。
// ①扁平索引 [rsp+0x60]：vector<pair<payload,id>> 16B 槽，0x12fcdd0 reserve(n)
//   + 0x12fce80 push —— 槽位 i 恒存节点 i（按 id 顺序构建，id 字段冗余仅用于
//   越界告警文案）。
// ②遍历 map A 每组：收集成员的扁平槽指针 → std::sort（0x130ed40，比较器
//   0x12fd0d0 按 pair.id —— 同基址指针比较优化）→ 有序 id 写回组成员表
//   [12fcbf0-12fcc51，4 路展开]；越界成员（id≥n）告警跳过 [12fcac8
//   "Node index %u is larger than size %zu"]。
// ③cand_tables[id] = 该组成员表 [12fcc79]（.so 存指针全组共享；本实现按值
//   拷贝——提案器对表只读，观测等价）。
// ④组成员表登记进 group_list_keys（.so：表指针 push 进 ctx+0x258 [12fc916→0x12fcfa0]）。
void stcut_read_nodes(StCutContext &ctx);

// ============================================================================
// 切分层（M31-5，M30 六件套 + 辅助）
// ============================================================================
// 0x130ab30 提案器（M32 逐指令解码）：
//   arg5 = 排除集（已提切点，跨轮持久；.so 节点键@0x1c/32B），arg6/arg7 = 切集两半
//   （cand_tables[pivot] 过滤后按切点劈开：前半→before / 后半→after，
//    即执行器 batch_accounting 的 v5/v6 —— 循环2/循环3 的输入）。
uint32_t stcut_cut_propose(StCutContext &ctx, size_t best,
                           std::set<uint32_t> &exclusion,
                           std::vector<uint32_t> &out_before,
                           std::vector<uint32_t> &out_after);
// 0x1309a60 迁移执行器（M35 参数语义破译）：
//   seed = journal[best.link_fwd].sub_entry（[130db41 edx]；BFS 种子 [1309b6c rsp+0x14]、
//   导出循环扫描基 [1309f3c]）；cursor = develop 游标（[130db51 ecx ← rsp+0x18]，
//   记账 connect 的对端与 add集 扫描下标 [130996e/130997c]）。M30 旧注把它们
//   误记为 pivot/arg4——提案器的 pivot 只回到 develop [rsp+0x8]，从不进执行器。
uint64_t stcut_cut_execute(StCutContext &ctx, const StCutRec &rec, uint32_t seed,
                           uint32_t cursor, uint16_t new_rec_id,
                           std::vector<uint32_t> &cands_before,
                           std::vector<uint32_t> &cands_after,
                           std::vector<uint32_t> &out,
                           std::vector<uint32_t> &exported);
void   stcut_cut_fix_range(StCutContext &ctx, size_t best,
                           std::vector<uint32_t> &out, uint16_t rec_id); // 0x130b9a0 修正器
bool   stcut_cut_verdict(StCutContext &ctx, size_t best,
                         const std::vector<uint32_t> &cut_set);          // 0x130d050 判定器
void   stcut_cut_split_record(StCutContext &ctx, size_t best,
                              const std::vector<uint32_t> &cut_ids,
                              uint32_t cut_count, uint8_t verdict);      // 0x130b600 分裂器
void   stcut_net_weight(StCutContext &ctx);                              // 0x130a560 净重
void   stcut_build_local_tree(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                              std::map<uint32_t, uint64_t> &tree);       // 0x130a330（arg3=journal[best.链前向].sub_entry）
void   stcut_random_restart(StCutContext &ctx, size_t rec_idx);          // 0x130bc70（全解：链摘除→
                        // 副本/关联树→重合并双计数门→repoint→归属改写→retire）

// ============================================================================
// 调度层（M31-6）
// ============================================================================
void   stcut_aux_fill(StCutContext &ctx, std::vector<uint64_t> &aux,
                      uint32_t start, uint32_t len);                     // 0x130cea0
uint64_t stcut_measure_peak(StCutContext &ctx);                          // 0x13065f0 (M25: delay_dma)
bool   stcut_validate_order(StCutContext &ctx);                          // 0x1306a20 验证器
void   stcut_prep_round(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                        std::vector<uint64_t> &aux, uint64_t budget,
                        int32_t opt1);                                   // 0x1306750
void   stcut_develop_schedule(StCutContext &ctx, uint32_t arg2, uint32_t arg3,
                              std::vector<uint32_t> &out_vec,
                              std::vector<uint64_t> &aux,
                              uint64_t inner_budget, int32_t max_iters); // 0x130d3e0

// 重试循环（M27 §3 十二步）——返回最优工作序（退出前恢复快照#1）
// s_seed/t_seed = 进循环前预载的 S/T 锚点（.so 中 [rsp+0x20]/[rsp+0x30]，来自早期阶段）
void   stcut_full_schedule(StCutContext &ctx, const StCutOptions &opt,
                           const std::vector<uint32_t> &original_order,
                           std::vector<uint32_t> &best_order,
                           std::vector<uint64_t> &flows,
                           std::vector<uint64_t> &cycles,
                           uint32_t s_seed = 0, uint32_t t_seed = 0);

// 迭代预算公式（M27 §2，130408b/1304363 同式）
uint64_t stcut_iteration_budget(int32_t opt, size_t n);

// 反汇编未完全理解段的集中声明（不实现，见 st_cut.cpp 尾部说明）：
//   0x1305130 内部（层标记怎么算）、0x1308c50 收尾段 1308f80–130922f、
//   0x12fffb0 比较器方向、0x130bc70 中后段 130c519–130cb4d、
//   [ctx+0x278] 语义、除数魔数 0x1450f0、[rsp+0x1530] 已用量更新点。

} // namespace hnnx
