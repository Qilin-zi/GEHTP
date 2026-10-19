#include <cstdio>
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/pass_view_fold.hpp"  // 005: dump_view_census
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/host_decode_state.hpp"
#include "hnnx/mcast/mcast_optimizer.hpp"
#include "hnnx/scheduler/dp_sequencer.hpp"
#include "hnnx/vtcm/dp_group_graph.hpp"
#include "hnnx/vtcm/supertile.hpp"
#include "hnnx/ir/graph_deps.hpp"
#include "hnnx/schedule/scheduler.hpp"  // Scheduler(ST-Cut 计划序, do_prepare2 计算)
#include "hnnx/tiling/conv_tiling.hpp"  // 阶段6: conv 空间分块(halo 推导)
#include <algorithm>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <cstring>

namespace hnnx {

// Graph transformations: remove_dead_code / order_nodes / common-subexpr /
//   const_prop / const_prop_and_cse / eliminate_split_nodes / rebuild_consumers

// remove_dead_code: REAL implementation
// Source: graph_prepare.cc, 402 bytes at 0xF68D90
int GraphPrepare::remove_dead_code(bool) {
    // Source: graph_prepare.cc, 402 bytes at 0xF68D90
    // Algorithm (from decompilation):
    // 1. Iterate all ops, find those with no consumers (dead)
    // 2. Mark them dead (flags |= 3)
    // 3. Collect and delete
    // 4. Repeat for newly-dead ops (cascade)
    // +0x6ea4 flag tracks if any dead code was found

    int deleted_count = 0;
    bool changed = true;

    // Iterate to handle cascading dead code
    while (changed) {
        changed = false;
        std::vector<op_id_t> to_delete;

        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled()) continue;
            if (opdef->is_dead()) continue;
            if (id == output_node_id_) continue; // Don't delete Output node
            if (id == input_node_id_) continue;  // Don't delete Input node

            // Check if this op has any live consumers
            bool has_live_consumer = false;
            for (op_id_t consumer_id : opdef->consumers) {
                auto it = opdef_map_.find(consumer_id);
                if (it != opdef_map_.end() && it->second &&
                    it->second->is_enabled() && !it->second->is_dead()) {
                    has_live_consumer = true;
                    break;
                }
            }

            if (!has_live_consumer) {
                // Mark as dead: flags |= 3 (OP_ENABLED | OP_DEAD)
                opdef->flags |= OP_DEAD;
                to_delete.push_back(id);
                if (getenv("GEHTP_EXDIAG"))
                    std::fprintf(stderr, "[dce] delete %llu:%s\n",
                                 (unsigned long long)id,
                                 opdef->name_tag && opdef->name_tag->name() ? opdef->name_tag->name() : "?");
                changed = true;
            }
        }

        // Delete dead ops
        for (op_id_t id : to_delete) {
            // Remove this op from its sources' consumer lists
            auto it = opdef_map_.find(id);
            if (it != opdef_map_.end() && it->second) {
                for (const auto& conn : it->second->inputs) {
                    auto src_it = opdef_map_.find(conn.src_id);
                    if (src_it != opdef_map_.end() && src_it->second) {
                        auto& consumers = src_it->second->consumers;
                        consumers.erase(
                            std::remove(consumers.begin(), consumers.end(), id),
                            consumers.end());
                    }
                }
            }
            opdef_map_.erase(id);
            deleted_count++;
        }
    }

    if (deleted_count > 0) {
        graph_dirty_ = true;
    }

    return deleted_count;
}

// order_nodes: topological sort via iterative DFS (stupid_fast_topo_sort)
// Source: stupid_fast_topo_sort @ 0x1301c70 (verified)
// Algorithm: iterative DFS with bitmap visited, explicit stack, post-order output.
// "stupid fast" = no recursion, no priority queue, bitmap O(1) visited check.
// 本函数只做 post-order 反转得到拓扑序，无随机打散 / tie-break。
// (PCG32 常数 0x5851f42d4c957f2d / 0x14057b7ef767814f 属 schedule_for_alloc @ 0x1302900
//  的调度键哈希，非 order_nodes 的同层 tie-break —— M17 已纠正。)
int GraphPrepare::order_nodes(bool) {
    if (opdef_map_.empty()) return 0;

    // Build node list and id->index map
    std::vector<op_id_t> nodes;
    std::unordered_map<op_id_t, size_t> id_to_idx;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || opdef->is_dead()) continue;
        id_to_idx[id] = nodes.size();
        nodes.push_back(id);
    }
    size_t node_count = nodes.size();
    if (node_count == 0) return 0;

    // Build successor adjacency list: edge from input source -> consumer
    std::vector<std::vector<size_t>> adj(node_count);
    for (size_t i = 0; i < node_count; i++) {
        auto it = opdef_map_.find(nodes[i]);
        if (it == opdef_map_.end() || !it->second) continue;
        for (const auto& conn : it->second->inputs) {
            auto src_it = id_to_idx.find(conn.src_id);
            if (src_it != id_to_idx.end()) {
                adj[src_it->second].push_back(i);
            }
        }
    }

    // Bitmap visited: (node_count + 31) / 32 uint32 words, zero-initialized
    // Source: 0x1301ce8-0x1301f74, movdqu批量�?
    std::vector<uint32_t> visited((node_count + 31) / 32, 0);
    auto test_bit = [&](size_t i) -> bool {
        return (visited[i / 32] >> (i % 32)) & 1u;
    };
    auto set_bit = [&](size_t i) {
        visited[i / 32] |= (1u << (i % 32));
    };

    // DFS explicit stack (linked-list stack in real binary, vector here)
    // Source: 0x60(%rsp) and 0x78(%rsp) two linked-list heads
    // Each stack node 24 bytes: +0x00 next, +0x08 prev, +0x10 node_id
    struct StackEntry { size_t node_idx; size_t neighbor_pos; };
    std::vector<StackEntry> dfs_stack;
    std::vector<op_id_t> post_order;
    post_order.reserve(node_count);

    for (size_t start = 0; start < node_count; start++) {
        if (test_bit(start)) continue;
        set_bit(start);
        dfs_stack.push_back({start, 0});

        while (!dfs_stack.empty()) {
            auto& top = dfs_stack.back();
            size_t idx = top.node_idx;

            // Find next unvisited neighbor (depth-first: break on first found)
            bool pushed = false;
            while (top.neighbor_pos < adj[idx].size()) {
                size_t neighbor = adj[idx][top.neighbor_pos++];
                if (!test_bit(neighbor)) {
                    set_bit(neighbor);
                    dfs_stack.push_back({neighbor, 0});
                    pushed = true;
                    break;
                }
            }

            if (!pushed) {
                // No unvisited neighbors: post-order output
                post_order.push_back(nodes[idx]);
                dfs_stack.pop_back();
            }
        }
    }

    // Reverse post-order = topological order
    std::reverse(post_order.begin(), post_order.end());

    int changed = (post_order != ordering_) ? static_cast<int>(post_order.size()) : 0;
    if (changed > 0) {
        ordering_ = std::move(post_order);
        graph_dirty_ = true;
    }
    return changed;
}

// ===== schedule_for_alloc 的 17 个文件内 helper（反汇编确证，M21+M22+M23+M24）=====
// 调度器本体 schedule_for_alloc @ 0x1302900（10286B，M17 已定位），内含若干无符号
// 静态局部 helper（落在 dynsym 间隙）。以下 10 个已逐指令反汇编确证，供后续重实现
// schedule_for_alloc 时按地址回填。ctx 结构布局见 M18–M23。
//
// 【M25 原名恢复】源文件 = st_cut.cc（.rodata 0x55b3db9 日志宏字符串证实）。
//   二进制 strip 掉 .symtab（仅 5758 个 .dynsym 导出），但 qnndsp 日志宏把 __func__
//   字符串嵌进 .rodata（0x55b3de5–0x55b402c，18 个 stcut_* 名）。计时器 0x130ea30
//   已反汇编证实为 vector<pair<const char*,u64>>::push_back({name, rdtsc>>4})：
//   名字标注其**后**即将执行的 stage 调用（SCHED_CYCLS 差分的数据源）。
//   已配对原名（名字 lea → 计时 call → 下一个非计时 call；M26 修正：配对需
//   距离≤0x30 且目标在 helper 簇 0x12fc000–0x1307000，12 个成立）：
//     stcut_read_nodes=0x12fc820  stcut_collate_sibs=0x12fd0e0
//     stcut_node_hash=0x1300210   stcut_quick_early_sort=0x1300600
//     stcut_block_relate=0x1300a40  stcut_connect_nodes=0x1300cc0 ←（M23 误标更正）
//     stcut_add_dependencies=0x1305130  stcut_strong_relevel=0x12fd600
//     stcut_arrange_sibs=0x12fffd0  stcut_clean_sibs=0x13056f0
//     stcut_delay_dma=0x13065f0   stcut_delay_dma_again=0x12fc740（内部已解，见
//       下方 M24/M25 收尾块：清出向量→按 flags&0xB==0 过滤入参向量→投影
//       [ctx+0x38 向量][id] 压回）
//   内联/未定：stcut_relate_by_tensor / stcut_topo_presort / stcut_measure_peak
//     （初版误配 0x130ebe0，实为 CSV 整数解析器：',' 跳字段→strtol→cmova 取值/0；
//     同族 0x130eb50=CSV 浮点解析 strtod 失败取默认 [rsi+0x128]）/ stcut_full_schedule
//     （SCHED_ITERS 重试循环，schedule_for_alloc 体内）/ stcut_convert_to_ids。
//     SCHED_OPTIONS 的 OT/IT/... 14 参数即由解析器家族在 1303ee9–1304060 从 CSV 读入
//     （条件：GraphPrepare+0x5638..0x56b0 optional 字段步长 0x18 已置位）。
//   其余叶子 helper（0x1309230/0x13096b0/0x1307820/0x1307960/0x1307f60/0x1307aa0/
//   0x1301040/0x12fba20/0x1307780/0x1307c90/0x1307d90 等）无日志调用 → 二进制内
//   无原名，保留本文件工作名。
//
// 【M26 schedule_for_alloc 内联准备段（13039e5–1303ee9，进重试循环前）】
//   1303a04 节点数=(end-begin)>>4/4；1303a27 NODE_ID_LIST BEGIN 日志（行927，优先级3）。
//   1303a2c 三个空 vector：rsp+0x10e0/0x1100/0x1120（各 24B）。
//   1303a5a rdtsc 原始 TSC（r14d 低/[rsp] 高）；1303a63-1303a8e 预算 double =
//     u64(this+0x6830) × double(this+0x55f8)（punpcklps/subpd@0x3970340/0x3970350
//     为 u64→double 改写）→ [rsp+0x40]；1303e80 常量 0xB2D05E00=3e9（3GHz TSC 基准）。
//   1303a94 节点序快照#1：_Znwm n×4B → vector@rsp+0x10e0，movups 64B/轮 拷自
//     [rsp+0x1140/0x1148]（有序节点 u32 列表）。
//   1303c4a call 0xf6f1f0(&rsp+0xa8, [rsp+0x38]) = 配置对象深拷贝（+0 u64、六个
//     24B string @+0x8/0x20/0x38/0x50/0x68/0x90、+0x80 u64、+0x88 u32、+0xb0 byte）。
//     ⚠ objdump 标签 sap_reduce_bandwidth 是回退标签（真身 0xf6ed80 止于 0xf6efdc）。
//   1303c5c 节点序快照#2 → vector@rsp+0x90（同源同法）。
//   1303e0f 门1：this+0x55dc(i32) ≤ 0 → 跳 13044cb 绕过循环；1303e1c-1303e58 门2：
//     [rsp+0x18]<<0xb ≤ double(r15)×[rsp+0x10] → 跳 13044cb（r15/[rsp+0x10] 来源未解）。
//   1303e5e TSC/16 起点 → [rsp]；1303e74 预算 int64 → [rsp+0x68]（deadline）。
//   1303e86 预取 6 个 optional 地址（this+0x5639/0x5651/0x5669/0x5681/0x5699/0x56b1）。
//   1303ee9- 选项解析循环：optional{flag@+0x0, csv 指针@+0x10}（0x18 步长）；
//     call 0x130ebe0 csv_int(idx) 后 test $0x1,%al 当布尔；假/未置位 → 取默认
//     this+0x55d4→[rsp+0x1ac]、this+0x55d8→[rsp+0x1b0]。1303fb2 另有 csv float
//     （0x130eb50）。即 SCHED_OPTIONS 14 参数的装载（OT/IT/…/DB，打印@1304642）。
//
// 【M27 内联 full_schedule 重试主循环（1303ee0–13044e3，逐指令实证）】
//   多轮试探—择优：回边 13044c5 jne→1303ee0。r14=轮次(1 起)，r15=r14−1=CSV 字段号。
//   三份节点序分工闭环：[rsp+0x10e0]=最优序（1304325 本轮 r15 更优时 copy 工作序入，
//     [rsp+0x18]=r15 历史最优 grain 数；退出后 13044cb-13044e3 copy 回工作序=最终输出）；
//     [rsp+0x90]=原始序（进循环前 1303c5c 拍；坏调度 130413f 与每轮尾 13043ff 重置用）；
//     [rsp+0x1140]=工作序（每轮被调度修改）。
//   每轮序列：
//   ① 1303ee9-1304060 按 (r14−1) 从各选项 CSV 重载参数（[rsp+0x1ac/0x1b0/0x1bc]，
//     默认 this+0x55d4/0x55d8/0x55dc）；1304062 记录区 [rsp+0x1500/0x1538=0x330e tag/
//     0x153a/0x153c=轮次两半/0x1540=本轮耗时]。
//   ② 130408b 迭代预算 = (opt2×1e6 + 3e9 − n×530000) / ((n×0x1450f0)>>32 + 26)，
//     n=([rsp+0x1300]−[rsp+0x12f8])>>3（工作表#1 元素数）；+ [rsp+0x1530] 已用量 →
//     作 r8 传 0x1306750。
//   ③ 1304104 call 0x1306750(ctx, [rsp+0x20], [rsp+0x30], &[rsp+0x1148], 预算,
//     [rsp+0x1ac], [rsp+0x30])＝本轮准备（内部 1306782 call 0x1307780 首轮五表快照）。
//   ④ 1304136 call 0x1306a20(ctx)→al＝本轮调度执行体（M21-M24 helper 链驱动者）。
//     失败(130413f)：恢复原始序+「Bad Schedule Detected (seed=%u)」(行 948,seed=r14d)；
//     1304186 this+0x55e8≠0 → 1304b48 Fatal(行 950)。
//   ⑤ 1304193 call 0x13065f0(ctx)→r15＝本轮 flow（grain 数；×2048=字节）。
//     13041a5 push 到 [rsp+0x1120] 向量(SCHED_FLOWS)；13042ad rdtsc/16−[rsp]→
//     [rsp+0x1540]，13042ce push 到 [rsp+0x1100] 向量(SCHED_CYCLES)；13042e9
//     FlowRetry 日志(行 959,优先级 11,%llx=r15<<0xb)。
//   ⑥ 130431e r15<[rsp+0x18] → 1304325 copy(工作序→最优序)、[rsp+0x18]=r15。
//   ⑦ 判定：1304347 cl=(r15<<0xb > [rsp+0x68] deadline)；130434f sil=([rsp+0x1f0]==0)；
//     1304363 预算公式第二遍（opt3=[rsp+0x1bc]）；13043b3 [rsp+0x1530]≤预算 →
//     13043f9 ebp=cl|sil；否则 13043b8 TIMEOUT 日志(行 971,%llu=opt3 原值,
//     #%d=[rsp+0x10] 0基轮次)+ebp=0 停。
//   ⑧ 13043ff 恢复原始序→工作序；1304434-13044a8 五表恢复（快照表[+0x190..0x208]
//     =栈 rsp+0x1420..0x14a0 → 工作表 rsp+0x12f8/0x1310/0x1340/0x1370/0x1388；
//     call ef9d90/13111b0/13112e0/8f02c0×2）。
//   ⑨ 13044b2 r14d ≥ this+0x55dc(Rt=最大轮数) → 退出；否则 r14d++、[rsp]=r12、
//     13044c5 ebp≠0 → 回①。
//   后循环：13044cb 恢复最优序；1304510 push 名"stcut_full_schedule"（⚠计时归属
//     异常见 M27 报告 §5a——push 在循环之后）；130453d push"stcut_delay_dma_again"；
//     1304557 call 0x12fc740(ctx,[rsp+0x50],&工作序)；1304591 push"stcut_convert_to_ids"
//     （无独立 call 可配对——完全内联于其后的统计/收尾段，区间未划出）。
//   统计块 1304596-1304ab9 受 this+0x5634 总闸：1304642 SCHED_OPTIONS 打印（行 999，
//     实参逐一重建出字段映射：OT=+0x55e4, IT=+0x55d8, Rg=+0x55d4, Rt=+0x55dc,
//     AB=+0x55e0, AM=+0x55e8(byte), EO=+0x5618, HD=+0x5619, DD=+0x561a(byte),
//     TR=+0x55f8(double,即预算乘数), LT=+0x5620, RC=+0x5628, RP=+0x562c,
//     DB=+0x5630(byte)；第 10 栈参=0x4628a0e 空串""宏尾参）；13046a2/13046d4/1304706
//     GRAINS_CUT/FLOW/RELEVEL ← [rsp+0x1510/0x1518/0x1520]（行 1000-1002）；
//     1304770 差分 STAT 循环（行 694,"%s=%lld",16B 元素表）；13047c3 call 0x13065f0
//     最终 delay_dma 复算（最优序上）→ rbx 再打印。
//   未完全理解：[rsp+0x1530]/[rsp+0x1f0] 写入点；0x1450f0 等效除数；CSV 值是否写回
//     this；统计块尾部 13047e0-1304900 打印未逐条对齐。详见
//     audit_verify/reports/M27_full_schedule_retry_loop.md。
//
// 【M28 循环体三大件（全解，见 M28_loop_body_three_functions.md）】
// 0x13065f0 — stcut_delay_dma = 目标函数（337B 全解）：
//   1306601 懒建权重树（cmpb ctx+0x18 → 0x130ccd0，与 0x1307200 同款）；
//   [ctx+0x158] = 工作序 vector 对象指针（★0x1306750:130677b 存入）。
//   遍历 u32 id：权重树 find-or-insert（48B 节点 key@+0x20/count@+0x28，
//   0x868c50 再平衡，树大小 ctx+0x30）；1306664 r13+=count；1306668 rax=max(rax,r13)。
//   返回 = 前缀和运行最大 = 内存峰值（grain 数，×2048=字节）。
//   对照 0x1307200（M22）：同一遍历，Σ前缀和（调试用）；本函数 max（调度目标）。
// 0x1306750 — 本轮准备（0x272B 全解）：[ctx+0x158]=&工作序；call 0x1307780 首轮
//   五表快照；v30=vector<u64>(n) 清零，call 0x130cea0(ctx,&v30,0,n) 填充；
//   call 0x130d3e0(ctx,arg2,arg3,&v10空,&v30,预算,opt1,arg7) = ★真引擎（副作用
//   写 ctx，两局部向量随后释放）。
// 0x1306a20 — 排序合法性验证器（0x750B 全解，st_cut.cc:1782）：
//   局部 BFS 工作表 v30@rsp+0x30 + visited 集合@rsp+0x48（32B 节点 key@+0x1c）。
//   倒序扫工作序：id 已在集合 → 130708c node=[[ctx+0x68]][id]，log(1,
//   "%s:1782:WARNING:invalid in this ordering: %u %llx") → 返 0（=Bad Schedule）；
//   否则入表入集，BFS 展开：邻接=[ctx+0x1f0]（快照#4=add-set 每 id 24B 记录
//   {u32 边表}）；过滤 [ctx+0x1c0]（快照#3=+0xb0 记录 flags&4）与 [ctx+0x1a8]
//   （快照#2=+0x80 字节表 &2）；canon=记录.id；未访问则入表入集。扫完返 1。
//   语义：替换/吸收关系传递闭包，被吸收 id 出现于工作序 → 非法。
// 0x1307170 — ctx 五表恢复（新识别，35B）：(+0x68,+0x80,+0xb0,+0xe0,+0xf8) ←
//   (+0x190,+0x1a8,+0x1c0,+0x1f0,+0x208)。循环尾 1304434-13044a8 即其内联展开。
//   ★更正：五张工作表 = +0x68/+0x80/+0xb0/+0xe0/+0xf8（M27 报告括注
//   +0x98/+0x110/+0x128 系换算笔误，栈槽没错）。
// 0x130d3e0 — develop_schedule()（M29 全解，st_cut.cc:3101/3331 日志自证；
//   0x130d3e0-0x130e87f，0x168 帧，唯一 retq@130e87f）：
//   签名 (ctx, arg2, arg3, out_vec=&v10, aux=&v30, inner_budget=r9, max_iters=arg7)。
//   序言：深拷工作序→[rsp+0x140]；[ctx+8]=u16 事件 id 计数；journal（rsp+0x120，
//   64B/条：kind@0/u32@0x10/u32@0x14/区间长@0x18/byte@0x20/u32@0x24/区间起@0x28/
//   u16 id@0x38）压 3 条初始记录（#2 kind=2 覆盖全 [0,n)）；ctx+0x98 逐节点 u16
//   归属表扩到节点数并全填记录#2 id（0x13114b0 扩容）。
//   主循环 130d820-130e0c6：门=[ctx+0x2a0](inner 计数)≥r9 预算 → 130e117
//   log "st_cut.cc:3331:WARNING:TIMEOUT: develop_schedule() hit inner-loop
//   timeout with %d iterations remaining"（★M24 遗留 [ctx+0x2a0] 读方关闭）；
//   arg7≤0 跳主循环；每 5 轮（模 5 魔数）log "…:3101:Schedule-cutting has %d
//   remaining iterations or %zu inner-loops"。
//   ①评分扫描 130d890-130daa8：阈值 r12 = ((u64)[ctx+0x278] 位技巧转 double) ×
//   [[ctx+0x270]+0x128] ÷ 2048（grain）；score = Σ_{k∈记录区间}(aux[k]+3·aux[k]·
//   [aux[k]>r12])（超阈值 3 倍惩罚面积），取最大记录。out_vec 数≠游标时走 130da40
//   历史切点复用（沿记录链 field0 步进，==2 停）。
//   ②130db08 0x130ab30 提案（切点 u32→[rsp+0x8]）；rdtsc×3 夹 130db76 0x1309a60
//   执行（M24/M25 迭代驱动）；[ctx+0x280] += 执行耗时（/16 拍粒度累计）。
//   ③落地：输出表条数==记录长 → 逐 id 写 ctx+0x98[id]=记录 id；否则 0x130b9a0
//   修正 journal 再写；130dc7f 0x130cea0(ctx,aux,区间起,区间长) 重填被切区间 aux；
//   门 [[ctx+0x270]+0x160]：扫 aux 非降前缀末点，越过水位 [rsp+0x38] 则 0x8dc340
//   把 [ctx+0x158] 拷回局部同步；130dddc 0x130d050 分裂判定(u8) → 130de22
//   0x130b600 分裂记录压回 journal（新 id=[ctx+8]++）；out_vec.push_back(切点)。
//   ④随机重启 130df50：门 [[ctx+0x270]+0x158]≥2 且 0xd79830(&ctx+0x2a8)取模==0
//   → 链上等分布选活记录 → 0x130bc70 扰动。
//   终段 130e143-130e801：门 [[ctx+0x270]+0x110]；130e15a 0x130a560 收尾；
//   逐记录：id=工作序[k]、node=[ctx+0x140][id]，收集 (id,node*) 批 + set<u32>
//   （32B 节点同验证器）；130e600 0x13125e0 按比较器 0x12fffb0 = (node指针,id)
//   字典序（=节点创建序）std::sort；多趟 Kahn：ctx+0xe0 邻接中 flags&4 关系的
//   canon（ctx+0xb0[rel].id）仍在 set → 本趟跳过；可发射则删出 set 并
//   130e79b ★写回 [[ctx+0x158]].begin[range_start++] = id —— 工作序原地重排。
//   130e810 eb fe = unreachable 自旋（编译器生成）。
//   未解：六 helper 内部（M30）、[ctx+0x2a0] 递增指令、[ctx+0x278] 语义、
//   field0==2 常量两用性。
// 0x130cea0 — aux 前缀和填充器（M29 全解，0x1a1B）：(ctx, aux_vec, start, len)；
//   130ceb1 懒建权重树（守卫 ctx+0x18→0x130ccd0）；r13 = start? aux[start-1] : 0；
//   for k∈[start,start+len)：id=工作序[k]，权重树 find-or-insert（48B 节点，
//   0x868c50 再平衡，树大小 ctx+0x30；防御插入 count=0），r13 += count，
//   aux[k] = r13。★aux = 内存累计曲线（grain）；M28 记的第 4 参"终点 n"应更正为
//   "长度 n"（130cece addq：上界=start+len；终段 130e1c6 同式互证）。
// 0x130ccd0 — 权重树懒初始化 = ★count 写入方（M28 遗留#1 关闭）：guard ctx+0x18
//   置 1；遍历 [ctx+0x68] 节点表建树节点（key=节点号，0x868c50，[ctx+0x30]++），
//   count = Σ_{rel∈ctx+0xe0[id] 邻接} ctx+0xb0[rel].field@+8（grain 量纲权重，
//   spill/DMA 内存量）。写一次后只读。
//
// 【M30 切分六件套内部（见 audit_verify/reports/M30_cut_helpers.md）】
// 0x130ab30 — 切点提案器（M30 全解）：外部候选路（journal[min(best,链next)].@0x20
//   置位 → ctx+0x240[rec.@0x24] 候选表过滤到 min 记录归属（ctx+0x98[node]==min_id），
//   切点=中位元素）；通用路 0x130a330(ctx,best@0x14,next@0x10,&局部树) 后扫区间
//   （跳 ctx+0x80[id]&4），weight=候选数×子树.count，jitter 门 [[ctx+0x270]+0x15c]
//   ≠0 时 +=(rand%(cfg+1)×weight)>>6；排除集（32B 节点树）校验；无候选→随机兜底
//   working_order[rand%区间长+区间起]；候选表在切点劈两半（前段→rsp+0x68、
//   后段→rsp+0x88 交执行器），返回切点 u32。
// 0x1309a60 — 迁移执行器（M30 全解）：0x1309940 估量 ≥0x5f5e0ff(1e8) → 先
//   0x13080a0 迭代慢路径再带输出向量重调；out.clear()；memset([ctx+0x128])；
//   工作列表={pivot(edx)}，换缓冲逐层泛洪：非 pivot 节点写 ctx+0x98[id]=u16
//   新记录号（归属改写）+ push 进 out；邻接展开门 = ctx+0x80[canon]&3==0 且
//   ctx+0x128 未标记，flags&4 边另需 ctx+0xb0[rel.@4].@8 > ctx+0xc8[rel.@4]；
//   收尾两遍把「对方不属本记录」的 flags&4 边 @4 字段导出到导出向量
//   （pivot 邻接一遍 + 全部已迁节点邻接一遍）；返回 0x1309940 估量。
// 0x130b9a0 — 工作序稳定划分修正器（M30 全解）：boundary = 区间起+区间长−|out|；
//   跳过前缀已属本记录节点；自首个属本记录位起压实（属→留、不属→存临时向量、
//   被换入源位写 0 打洞）；再单遍将临时向量内容填回尾部空洞（非洞先收入临时
//   向量再后填）= 区间内 [父保留前缀 | 迁移尾巴] 稳定划分，与 0x130b600 的
//   parent.@0x18 -= cut_size 完全自洽。
// 0x130d050 — 分裂判定器（M30 全解，返回 u8）：0x1305130(ctx, journal[best].@0x14)
//   打层标记进 ctx+0x128；对切集每条非 flags&4 关系取远端标记值做直方图计数；
//   <8 桶 → 返 0（不切）；≥8 桶 → 清零前 ≤5 桶 → 前缀和 ×0x2AAAAAAA（1/6 定点）
//   → verdict = (桶数×最大前缀和)/总数 < 0x340000000（≈最大前缀占比 < 19/桶数，
//   分布够平才切）。
// 0x130b600 — 记录分裂器（M30 全解）：计数器++×2 经 0x1301040 建 kind=2/1 两新
//   槽；journal[best].@0x18(区间长) -= cut_size；子记录=父区间尾段（@0x28 终点
//   继承、长=cut_size、判定 u8@0x50、id@0x68）经 0x130b840 压入 journal；
//   journal[best].@0(链后继)=新记录索引、子.@0x8=自身索引（链表式链接）；
//   对切集每 id：0x1300cc0(ctx, related→A, weight,0) + 0x1300cc0(ctx,B→node,
//   weight,0) 双侧重接线（0x13087f0 逐 id 清理）；journal[A].@0x30=元素计数。
// 0x130a560 — ★净重计算器（M30 全解；M29 曾注"收尾"，实为终段排序键来源）：
//   ctx+0x140 表 resize 到节点数 N（0x1311b50 扩/截），每项 =
//   Σ(add集 非 flags&4 的 @8) − Σ(subtract集 非 flags&4 的 @8) = 节点净内存
//   增量（grain）。⇒ M29 终段批排序比较器 0x12fffb0 的第一键 [ctx+0x140][id]
//   是净重而非"节点指针/创建序"（★M29 修正）。
// 0x130a330 — 局部权重树构建器（提案器用，M30 全解）：0x1305130 打标前快照
//   ctx+0x128 副本；0x1308c50 重打标（=层级标记 BFS，见下方收尾块，标记=层级
//   号≥2）；双标记（旧≠0 且新≠0）且 ctx+0x80&3==0 的节点插入调用方树，
//   count = (新标记−1)×旧标记（新标记为层级号故 ≥2，公式自洽）。
// 0x130bc70 — 随机重启（0x130bc70-0x130ccbf，~4.3KB）：记录从链摘除（前驱.@0 =
//   本记录.@0）、@0/@8 清零判死；拷贝其 @0x10→subtract 集与 @0x14→add 集；
//   局部关联树（40B 节点 key@0x1c/count@0x20）收 add 集非 flags&4 关系；
//   130c0a0 在 subtract 副本找 canon==@0x10 槽并双向修剪；重合并循环
//   （130c150-130c74e）快照对账：远端快照 subtract 集（ctx+0x208）逐成员入树，
//   130c3e7/130c3f1 活跃权重表 ctx+0xb0 两处减账，130c3f6 建 24B 重放记录
//   {op,0,type=4} 经 0x13080a0 重派发（type4=重新加入 add 集，M22），
//   再对远端活跃 add 集（ctx+0xe0）/快照 add 集（ctx+0x1f0）数非 flags&4 成员
//   （130c490/130c4f0 adcq 同构计数）——把该记录的关系编辑回滚向快照态
//   （撤销这刀重来）。
// 未完全理解：0x130bc70 中后段 0x130c519-0x130cb4d（同构对账/删除/重放循环
//   展开，未逐条注释）；0x1305130(stcut_add_dependencies) 内部与 0x1308c50
//   收尾段(1308f80-130922f)；[ctx+0x278] 语义；0x12fffb0 升/降序方向。
//
// 【M24/M25 遗留收尾（本日，转储 asm/M25_parser_variants_*.asm、
//   asm/M25_ctx_dtor_probe_*.asm、asm/M24_slowpath_*.asm、asm/CU_range_*.asm）】
// 0x130ec60/0x130ec65 — ★修正：非 CSV 解析器变体（M25/M26 误判），是树节点
//   递归析构（后序：call([node])×2 → delete node）：节点 {left@0, right@8, POD}，
//   即 ctx 权重树节点析构（0x30B：key@0x20、count@0x28）。
// 0x130eca0/0x130eca5 — 同族但节点带 vector 载荷 {left@0, right@8, …, buf@0x28,
//   end@0x30}：每节点先 end=buf、delete[] buf 再 delete 节点 = map<key,vector> 析构。
// 0x13073c0 — ctx 析构函数（0x13073c0–0x13076de）：依次释放 ctx+0x258/0x240/0x228
//   (树)/0x208(vector<vector>)/0x1f0/0x1d8/0x1c0/0x1a8/0x190/0x178/0x160/0x140/
//   0x128/0xe0/0xf8/0x110/…/0x98/0x68/0x50/0x38，末尾 this=ctx+0x20、root=[ctx+0x28]
//   尾跳 0x130ec60 销毁权重树。其异常孪生内联在 0x12fba20 尾部 12fc07b-12fc229
//   （同序，_Unwind_Resume@12fc229）→ 0x12fba20 全函数=0x12fba20-12fc046。
//   ctx 新字段：+0x228/0x230 map<key,vector> 树；+0x240/0x248、+0x258/0x260、
//   +0x1d8 = vector；加/减集与其快照(0xe0/0xf8/0x1f0/0x208) = vector<vector<u32>>
//   （析构逐 24B 元素删各自缓冲区）。
// 0x12fc740 — stcut_delay_dma_again 内部（调用点 1304557）：清出向量(vecA)、
//   按入参 vecB 大小 reserve，遍历 vecB 每个 id：[ctx+0x80][id]&0xB≠0 跳过；
//   否则经 0x105bd40 把 [ctx+0x38 向量][id]（u32）压入 vecA。
// 0x1308c50 — ★层级标记 BFS（M30 未解项半关）：memset(rcx 标记表)；worklist=
//   {edx 种子}；层级计数 ebp 自 2 递增（1308cdf）；每层每节点 idx：扫
//   subtract集[idx] 找「伙伴(表[rel].@4→表[far].@0) 标记≤1 且 表[rel].flags&4」
//   的关系 → idx 入输出向量；再扫 add集[idx]：flags&4 边的 canon 未标记 →
//   下一层 push + 打标。收尾段 1308f80-130922f 未逐条。
//   ⇒ 标记值=层级号(≥2)，解释 0x130a330 的 (新−1)×旧 公式。
// 0x1305130 = stcut_add_dependencies（M25 配名）全部 4 调用方：1303766
//   (schedule_for_alloc 阶段调用)、1305738(stcut_clean_sibs 内)、130a354
//   (0x130a330)、130d07c(0x130d050)；内部仍未拆。
// 0x130ed00/0x130ed10 — __cxa_throw 小包装（throw length_error/reserve 溢出，
//   vector push_back 家族的越界路径共用）。
//
// 0x1307200 — 权重树查找/插入 + 前缀和累加（rdi=ctx，返回 rax=u64 累加值）
//   1307211 cmpb $0,0x18(%rdi) / jne 130721f：未初始化 → call 130ccd0 建树。
//   1307222 起遍历 [ctx+0x158] vector<u32>，对每个索引在红黑树（root 在 ctx+0x28）
//   查 node->count（节点 +0x28，u64 权重）；命中 1307274 add 0x28(%rbx),%r13 /
//   add %r13,%rax。未命中（13072de）→ new 0x30 字节节点：+0x20=key、+0x28=count=0、
//   +0x10=parent、1307302 链接；130725b call 868c50（__tree_balance_after_insert）；
//   1307260 addq $1,0x30(%rbp) 树大小+1（+0x30 是 ctx 树大小计数，非节点域）。
//   返回 rax = Σ_k (Σ_{j≤k} weight_j)（前缀和之和）。消费方（M22 已证）：schedule_for_alloc
//   本体 1304801 / 13049f3 两处 call 0x1307200，返回值经 rbx 作 r8/rcx 传入 qnndsp_log
//   打调试统计（优先级 2 / 0），非 0x130d3e0 消费，不参与调度计算。ctx = &[rsp+0x1290]
//   （栈局部结构体，13047f9/13049eb lea 0x1290(%rsp),%rdi）。
//
// 0x1307780 — 一次性物化 5 个源 vector 到工作副本（rdi=ctx，void）
//   1307781 cmpb $0,0x220(%rdi) / je 130778c：一次性门，已执行则 ret；
//   130778f movb $1,0x220(%rdi) 置门。随后 5 次深拷贝（源 → 工作区）：
//     [ctx+0x68]→[ctx+0x190]（helper ef9d90）、[ctx+0x80]→[ctx+0x1a8]（13111b0）、
//     [ctx+0xb0]→[ctx+0x1c0]（13112e0）、[ctx+0xe0]→[ctx+0x1f0]（8f02c0）、
//     [ctx+0xf8]→[ctx+0x208]（8f02c0）。1307812 movl $0,0xc(%rbx) 清零 [ctx+0xc]。
//   说明：存在「源表 vs 工作表」两套权重表，由本函数一次性同步。
//
// 0x13087f0 — 从 add/subtract 集移除 op + 置 filter=0x80 + 追加输出（rdi=ctx, esi=op索引, rdx=回调）
//   1308821 起 r12=esi*3（24B/slot）；经权重表[esi].field0(+0x0) 与 .idx(+0x4) 链式定位
//   到 add 集与 subtract 集，swap-with-last 移除该 op（1308864–130887a 等）。
//   130895d test %rdx,%rdx：回调非空 → call 1307f60；否则 13089a7/13089b7
//   movl $0x80,0x10(权重表) 置 filter 位 0x80（已移除），并 13089d7 push op 索引到
//   [ctx+0x178] 输出 vector。1308ad6 call 853eb0 = vector<u32>::push_back
//   （1308acf add $4,%rsi 指向 weight_table[esi].idx，push 该 u32，M22 已证）。
//
// 0x1309a60 — spill 候选筛选（M20 遗留 [ctx+0x128] u16 三态表的写入方）
//   1309b38–1309b4b memset([ctx+0x128], 0, size) 全表清零；
//   1309bf8–1309bff 写 tag 数组 [ctx+0x98]（u16，tag[ebp]=cx）；
//   第二遍 1309d30 起遍历 add 集：1309d8e testb $0x4,0x10(权重表)（纳入位）、
//   1309da9 weight(+0x8) > [ctx+0xc8] 阈值、1309db6 cmpw $0,0x128(表)==0 且
//   1309d67 testb $0x3,[ctx+0x80]&0x3==0，则 1309dcf movw $1,(%rax,%r14,2) 置 1 并收集。
//   三态：0=未访问、1=本函数标记候选、≥2=「已最终处理」。M22 已证：全区域（0x1302900–
//   0x1313ac0）对 u16 表只有 movw $0x1（1305427/1308f3d/13092e9/130939a/1309dcf）与
//   memset 清零（1309b4b），无 incw/addw；cmpw $0x1;ja（>1 跳过，130529b/1308dbb/1309352）
//   的 ≥2 状态无生产者，属死/防御性代码（ctx 为栈局部，外部无法写）。
//
// 0x130ebe0 — 逗号分隔整数域解析器（rdi=域索引, rsi=逗号串，返回 eax=bool）
//   130ec00–130ec19 跳过 edi 个逗号；130ec23 mov $0xa,%edx / 130ec28 call strtol 基数 10；
//   130ec2f cmp %rbx,(%rsp)（endptr vs 原指针）→ seta 得 bool。非「6 选项 flag 检查器」
//   （M17 已纠正）；真 flag 检查是调用点前的 testb $0x1,[this+0x5638/0x5650/0x5668/0x5680/0x5698/0x56b0]。
//
// 权重表 [ctx+0xb0] 24B/slot 全字段：+0x0 field0(组id) +0x4 idx +0x8 weight(u64)
//   +0x10 filter（bit0x4=纳入、bit0x80=已移除）。新增 ctx 字段：+0xc8 阈值数组(u64)、
//   +0x80 组标志字节数组、+0x98 u16 tag 数组、+0x128 u16 三态表。
//
// 0x1309940 — 替换对批量记账 wrapper（M22；M23 改名：非 load_replacement_plan 包装器）
//   1309969 mov $0x5f5e0ff,%ecx 哨兵阈值（无阈值/恒接受）。4 次 call 1300cc0：
//     1309974(ctx,arg2,arg4,99999999,arg7)、13099d3(ctx,field0,arg3,99999999,arg7)、
//     1309a00(ctx,arg2,元素,99999999,arg7)、1309a30(ctx,元素,arg3,99999999,arg7)。
//   循环1 1309990–13099d8 遍历 add集[arg4] 跳过 filter0x4 或 field0==arg3 者；
//   循环2 13099da–1309a0c 遍历 arg5 向量；循环3 1309a0e–1309a3c 遍历 arg6 向量。
//   1309a55 jmp 1309810 尾调用（ctx,arg2,arg3）。被 0x1309a60 调用 2 次（1309ab1/1309afd）。
//
// 0x1300cc0 — 替换对记录核心（M23，无符号，gap 0x12f9a9e–0x1301c70）
//   【M25 原名 = stcut_connect_nodes】0x130368b lea "stcut_connect_nodes" → 130369a
//   计时 call 130ea30 → 130371d call 0x1300cc0（另 1303866/1303887 重试点）。
//   名字与 M23 功能解读一致：连接节点 arg2→arg3 并设阈值权重（建/查替换对）。
//   签名 u32(ctx*, u32 arg2, u32 arg3, u64 threshold, void* callback)，返回映射到 arg3 的槽索引。
//   objdump 标签 load_replacement_plan@@Base+0x72b0 是回退标签（非调用关系）：M23 已证
//   load_replacement_plan(DataReader&) @0x12f9820 不调用本函数，其反序列化进 this->[0x72e0]。
//   1300cec arg2==arg3 时 1300cf4 取 [ctx+0x68][arg2] 打日志（qnndsp_log 优先级1）。
//   1300d31–1300d77 在 add集[arg2] 遍历，找 权重表[idx].field0==arg3：
//     命中 1300fc9：callback 非空记 {idx,旧weight,type=0}（1300ff9 call 0x1307f60），
//       1301008–130101d 权重表[idx].weight = min(threshold+旧weight, 99999999)，返回 idx。
//     未命中：按 [ctx+0xc] 分路——快路径(==0) 1300da7/1300dab 从 [ctx+0x178] 空闲槽栈
//       pop 2 索引(ebp=slotB, ebx=slotA)，1300dc5/1300e02 复用写两槽；慢路径(!=0)
//       1300e6f/1300eb8 call 0x1308b10 push_back 2 个 24B 槽 + 1300e8c/1300ed3 call
//       0x12fc610 push 2 个 u64 0 到 [ctx+0xc8]。（两 callee 已解，M24 收尾：
//       0x1308b10=vector<24B>::push_back 缺省扩容版——快路径 24B 原地拷、满则
//       max(2n,n+1)×0x18 扩容 memcpy，元素数按字节>>3×0xAAAAAAAAAAAAAAAB ÷3；
//       0x12fc610=vector<u64>::push_back 同构（8B 元素，上限 0x1fffffffffffffff）；
//       另 0x12fc4d0 为第三份 24B 拷贝，仅 0x12fba20 尾部 @12fc00d 调用。）
//       两路结果一致：
//         slotA(idx=ebx): field0=arg3, idx=ebp, weight=threshold, filter=0
//         slotB(idx=ebp): field0=arg2, idx=ebx, weight=99999999, filter=[ctx+0xc]|4
//       （slotA/B 的 +0x4 idx 互指，构成双向替换对；[ctx+0xc8] 对应两项清零）
//   1300ee0/1300eed call 0x1307820(ctx,arg2,slotA)/(ctx,arg3,slotB)；
//   1300efa/1300f07 call 0x1307960(ctx,arg2,slotB)/(ctx,arg3,slotA)（建双向链接）。
//   1300f0c–1300f75 同步 [ctx+0x110] 每槽 u32 数组到槽数（扩充 call 0xcf27b0 / 收缩改 end）。
//   1300f7c callback 非空记 {slotA,0,type=2}（1300fa3 call 0x1307f60）；1300fb8 返回 slotA。
//
// 0x1309810 — 调度迭代驱动（M23，rdi=ctx, esi=arg2, edx=arg3，返回 u64 累计处理数）
//   130981e addq $1,0x10(%rdi) 计数；1309829–130983f memset([ctx+0xc8],0,size) 清阈值数组。
//   130984d call 1309230(ctx,arg2,arg3) 首次探测，1309852 test/je 130992d（0 → 返回 0）。
//   循环A 1309860–13098c8：1309897 call 13096b0(ctx,arg2,arg3,99999999,0) 批量处理，
//     13098c2 r12+=rax；rdtsc 计时 13098bb [ctx+0x288]+=(TSC_end/16-TSC_start/16)；
//     13098c8 jne 1309860（处理数≠0 续）。
//   循环B 13098ca–1309927：13098f9 call 1309230 探测，130991d [ctx+0x290]+=耗时，
//     1309927 jne 1309860（非0 回循环A）。130992d 返回 r12。
//   （rdtsc 双采样 TSC>>4 与 (TSC>>3)&~1 净效果为 /16，属计时改写，非调度计算。）
//
// 0x1309230 — 组可达传播/收集（M24；rdi=ctx, esi=seed, edx=target，返回 u64 收集计数）
//   130926d memset([ctx+0x128] u16 三态表,0)；1309285 memset([ctx+0x110] u32 位数组,0)。
//   13092b4 _Znwm(4) 建 frontier vector<u32>=[seed]（13092c0 mov %ebp,(%r10)）；
//   13092e9 movw $0x1,(%rax,%rcx,2) 给 seed 打 tag=1。代计数 [rsp+0x14] 初值 2（13092fc）。
//   外层循环（1309330–13095ea）逐代处理 frontier：130934f ecx=frontier[i]，
//   1309352 cmpw $0x1（tag!=1 跳过，含 M22 的 >1 跳过）；130935d 写本代 tag=[rsp+0x14]。
//   内层循环（13093ed–13093e7）遍历 add集[ecx]（130936c r12/r13=[ctx+0xe0][ecx*3]）：
//     ebp=成员；13093f1 [ctx+0x2a0]++ 访问计数；r15d=权重表[ebp].field0；
//     1309408 cmpw $0,(%rax,%r15,2)==0（未访问才进）；1309417 rdx=[ctx+0xc8][ebp] 与
//     权重表[ebp].weight(+0x8) 比较，jge 跳过（阈值已≥权重，无可取）；1309429
//     testb $0x2,[ctx+0x80][r15]（组标志 bit2：置位时须 r15==target，1309430 cmp/jne）；
//     通过则 1309388 推入结果 vector（[r9]=[rsp+0x60]，扩容 1309443–130958e 内联 2x）、
//     130939a movw $1 给 r15 打 tag=1（下代 frontier）、13093b8 置位 [ctx+0x110] 位图
//     （rdx=ebp>>5, or 1<<(ebp&31)）、13093bb r8++、13093d3 r11d|=（field0==target）。
//   本代结束（13095a0）：13095a0 testb $0x1,[rsp+0x38]（已触达 target 标志）非 0 →收尾；
//   否则 13095c8 代计数++、结果/新frontier 向量互换（13095c8–13095db），回 1309330。
//   13095ef–1309643 释放两 vector；130962f mov %r8,%rax 返回收集计数。
//   语义：从 seed 沿 add 集逐代洪泛，收集「权重>阈值」的可达槽，触达 target 即停。
//
// 0x13096b0 — 递归容量受限权重搬运（M24；rdi=ctx, esi=seed, edx=arg3, rcx=cap, r8d=flags，
//   返回 u64 实际搬运量）
//   13096ca seed==arg3 → 13097e6 返回 0。13096dd 取 add集[seed]（[ctx+0xe0][seed*3]），
//   空 → 返回 0。13096ff 深度计数 [rsp+0xc]++。130970d 累计器 [rsp+0x10]=0。
//   循环（1309742–13097e1）遍历 add集[seed]：ebx=成员；130974c r12=ebx>>5 取
//   [ctx+0x110] 位图字；130975b bt %ebx,%edx 测试 pending 位——
//     位 0 → 1309733 not/and 清位跳过（已处理）；位 1 → 1309760：
//     r14=权重表[ebx].weight(+0x8) − [ctx+0xc8][ebx]（可用量=权重−阈值）；
//     130977b cmp/cmovge %r8,%r14 → r14=min(可用量,cap)；
//     130978f–130979e 递归 call 13096b0(ctx, 权重表[ebx].field0, arg3, r14, [rsp+0xc])；
//     返回 0 → 1309720 清位继续；非 0 → 13097ac [ctx+0xc8][ebx]+=rax、
//     13097b7 累计器+=rax、13097c8 [ctx+0xc8][权重表[ebx].idx]-=rax（对槽扣减）；
//     13097d0 cmp $0x2710(10000),%r14：<10000 → 13097ed 返回累计（小量即停）；
//     否则 13097d9 cap-=rax 续。
//   13097ed 返回 [rsp+0x10]。
//   语义：自 seed 向下游递归「抽取」至多 cap 的权重，写入阈值账本 [ctx+0xc8]；
//   单步 <10000 提前返回（防长链微搬运）。
//
// 0x1307820 — add 集追加（M24；rdi=ctx, esi=slot, edx=val，void）
//   1307830 rbx=[ctx+0xe0]；1307843 rdx=add集[slot*3].end；1307848 cmp 0x10(容量)：
//   快路径 130784f–1307856 *end=val, end+=4；扩容 1307868–1307930（2x+memcpy+free，
//   溢出 130794f/1307954 length_error/bad_array_new_length）。
//
// 0x1307960 — subtract 集追加（M24；rdi=ctx, esi=slot, edx=val，void）
//   与 0x1307820 同构，作用表 [ctx+0xf8]：1307970 rbx=[ctx+0xf8]，快路径 130798f–1307996，
//   扩容 13079a8–1307a70。
//
// 0x1307f60 — 24B 账本记录 push_back（M24；rdi=vector<24B>* , rsi=记录*，返回新记录指针）
//   快路径 1307f7c–1307f96：end<cap 时拷 24B（movups 16B+u64），end+=0x18。
//   扩容 1307f9b–1308076：1307fb2 imul 0xAAAAAAAAAAAAAAAB（无符号 /3 魔数）算旧容量，
//   1307fd9–1307fe0 新容量=max(旧+1, 2x)，上限 0x555555555555555；1307ff5 容量 0 →
//   1308020；130800f _Znwm(24*cap)；memcpy 旧数据；free 旧。
//   130807a rbx=end−0x18 → rax 返回 pushed 记录指针（非 void，供调用方就地改写）。
//   溢出 130808b call 13114a0 / 1308093 call 855470。
//
// 0x1307aa0 — 节点槽退役（M24；rdi=ctx, esi=slot，void）
//   1307ab3/1307aca 清空 slot 的 add 集（[ctx+0xe0][slot*3]）与 subtract 集
//   （[ctx+0xf8][slot*3]）：mov (%rcx,%rdx),%rsi; mov %rsi,0x8(...) 即 end=begin（不释放）。
//   1307adf [ctx+0x68][slot*8]=0（节点指针清空）；1307af2 movb $0x83,[ctx+0x80][slot]
//   （组标志=0x83）；1307b06 push slot 进空闲槽向量 [ctx+0x160/0x168/0x170]（快路径
//   1307b06–1307b0d，扩容 1307b23–1307bf3）。注意：这是节点槽空闲表，与权重槽空闲表
//   [ctx+0x178/0x180]（0x1300cc0 pop）是两个不同 vector。
//
// 0x1301040 — 节点注册（M24；rdi=ctx, rsi=节点指针, dl=flags；返回 eax=槽索引；
//   [ctx+0xc] 的唯一「读路」消费者）
//   1301068 cmpl $0,[ctx+0xc]：!=0（构造期 0x40）→ 追加路 130106e：节点指针 push 进
//   [ctx+0x68/0x70/0x78] vector<u64>（扩容 13010e2–13011a7），槽=旧 size；
//   ==0 → 复用路 130109d：从 [ctx+0x160/0x168] pop 槽号 r13d（13010ad mov -0x4(%rax)，
//   end-=4），13010c4 [ctx+0x68][r13]=节点，13010cb [ctx+0x80][r13]=flags 字节，
//   13010d6 movw $0,[ctx+0x128][r13*2]（三态表复位）。
//   13011af orl [ctx+0xc],%r12d（flags|当前模式字）→ 13011c6 把该字节 push 进
//   [ctx+0x88/0x90] 字节 vector（flags 账本）。返回槽索引。
//
// 0x12fba20 — ctx 构造函数（M24；rdi=ctx, rsi=GraphPrepare*，schedule_for_alloc
//   1302a3b 调用）—— [ctx+0xc] 的初始化方
//   12fba44 [ctx+0x0]=[rsi+0x7468]（自 GraphPrepare 取初值）；12fba4e [ctx+0x8]=0(u16)；
//   12fba54 movl $0x40,0xc(%rdi) ← [ctx+0xc]=0x40（构造即非 0！bit0x40=建表/追加期）；
//   12fba5b [ctx+0x10]=0；12fba63 [ctx+0x18]=0；随后 0x20–0x2a8 全部向量/数组清零。
//   [ctx+0xc] 全部写方（0x12f9a10–0x1312b85 范围 grep 确证）：初始化 12fba54=0x40、
//   清零 1307812（0x1307780 快照函数内）与 130d1c2；无其他写指令。
//
// 0x1307780 — ctx 阶段快照（M24；rdi=ctx；[ctx+0x220] 一次性守卫）
//   1307781 cmpb $0,0x220(%rdi) 已做过→ret；130778f 置 1。随后 5 次快照调用：
//   13077a5 call ef9d90（[ctx+0x68/0x70] 节点指针 vector → [ctx+0x190]）、
//   13077bf call 13111b0（[ctx+0x80/0x88] flags 字节 vector → [ctx+0x1a8]）、
//   13077d9 call 13112e0（[ctx+0xb0/0xb8] 权重表 → [ctx+0x1c0]）、
//   13077f3 call 8f02c0（[ctx+0xe0/0xe8] add 集表 → [ctx+0x1f0]）、
//   130780d call 8f02c0（[ctx+0xf8/0x100] subtract 集表 → [ctx+0x208]）。
//   1307812 movl $0,0xc(%rbx) 清模式字 → 进入「复用空闲槽」阶段（0x1301040/0x1300cc0
//   的快路径自此生效）。调用方：0x1306750 helper @1306782。
//
// 0x1307c90 — 替换对重指向 A 侧（M24；rdi=ctx, esi=slot, edx=newPartner，void）
//   eax=权重表[slot].field0（旧伙伴）；1307cd0 在 subtract集[旧伙伴] 线性找 slot，
//   命中 1307ce4–1307cfa swap-with-last 移除；1307d20 在 add集[旧伙伴] 找
//   权重表[slot].idx(+0x4)，命中 1307d34–1307d4a 同法移除；1307d53 call 0x1307960
//   (ctx,newPartner,slot)（subtract集[newPartner] push slot）；1307d71 call 0x1307820
//   (ctx,newPartner,权重表[slot].idx)（add集[newPartner] push idx）；
//   1307d7d 权重表[slot].field0=newPartner。
//   调用方（M24 遗留收尾，全簇 grep 确证）：仅 0x130bc70 随机重启 @130c797/130c7f8
//   ——重指向是重启回滚的接线工具（与 M30「撤销这刀重来」互证）。
//
// 0x1307d90 — 替换对重指向 B 侧（M24；rdi=ctx, esi=slot, edx=newPartner，void；
//   0x1307c90 的镜像，作用于 idx 侧）
//   ecx=权重表[slot].idx(+4)；eax=权重表[idx].field0（对侧旧伙伴）；1307de0 在
//   add集[对侧旧伙伴] 找 slot，命中 1307df4–1307e0f swap 移除；1307e30 在
//   subtract集[对侧旧伙伴] 找 idx，命中 1307e44–1307e5d swap 移除；1307e66 call
//   0x1307820(ctx,newPartner,slot)（add集[newPartner] push slot）；1307e84 call
//   0x1307960(ctx,newPartner,权重表[newPartner].idx)（subtract集[newPartner] push）；
//   1307e98 权重表[权重表[newPartner].idx].field0=newPartner。
//   调用方（M24 遗留收尾）：仅 0x130bc70 随机重启 @130c18a。
//
// 0x13080a0 — 6-case 记录重放 dispatch（M22，rdi=ctx, rsi=24B 记录 vector，逆序 pop）
//   记录 24B：+0x0 op索引(u32,r14d) +0x8 value(u64) +0x10 type(u32,ecx)。
//   13080f7 end-=0x18 逆序 pop；130810b cmp $0x5 / ja 1308136（>5 错误）；
//   1308115 movslq 0x0(%r13,%rcx,4) 跳转表（r13=0x55b3d04，6 个 32-bit 相对偏移）→ jmp *%rcx。
//   type→case（跳转表已提取）：
//     0→130811f 置权重 weight_table[op].weight(+0x8)=value（130812f）；
//     1→1308136 日志(优先级100, fmt@0x55b42b0)；2→1308151 从 add 集移除(swap-with-last)；
//     3→13080d4 [ctx+0x70]-=8 且 [ctx+0x88]-=1（pop 计数）；4→13081a8 加入 add 集(push)；
//     5→13081fe 日志(优先级100, fmt@0x55b4288)。type 1/5=日志记录、3=pop 记录、0/2/4=权重/add 更新。
//
// 0x853eb0 — vector<u32>::push_back 冷路径（M22，rdi=vector*, rsi=const u32*）
//   853ec1 mov 0x8(%rdi),%rdx（end）/ 853ec5 cmp 0x10(%rdi)（capacity）/ 853ec9 je 扩容：
//   快路径 853ecb–853ed4 *end=*element, end+=4；扩容路径 853ee7–853faa 2x 增长+memcpy+free，
//   溢出 853fbd call 8553f0(length_error)/853fc5 call 855470(bad_array_new_length)。
//   7 处调用点（13057e5/1308ad6/1309f6b/130b311/130b328/130bb50/130bc01），均 push 一个 u32
//   （典型 weight_table[].idx）。
//
// common_subexpr_eliminate: CSE
// Source: graph_prepare.cc, 4156 bytes at 0xF6A5B0
int GraphPrepare::common_subexpr_eliminate(bool) {
    // Source: graph_prepare.cc, 4156 bytes at 0xF6A5B0
    // Algorithm:
    // 1. For each enabled op, build a signature:
    //    signature = (name_tag_hash, sorted_input (src_id, out_idx) pairs)
    // 2. Hash the signature
    // 3. Look up in hash table
    // 4. If match found with identical signature: supersede (replace) this op
    // 5. Return number of ops eliminated

    int eliminated = 0;

    // signature -> existing opdef
    std::unordered_map<uint64_t, OpDef*> cse_map;

    // Helper: compute CSE signature for an op
    auto compute_signature = [](const OpDef* opdef) -> uint64_t {
        uint64_t sig = opdef->hash_key();
        // Mix in input connections (sorted for order-independence)
        std::vector<InputConn> sorted_inputs = opdef->inputs;
        std::sort(sorted_inputs.begin(), sorted_inputs.end(),
            [](const InputConn& a, const InputConn& b) {
                if (a.src_id != b.src_id) return a.src_id < b.src_id;
                return a.out_idx < b.out_idx;
            });
        for (const auto& conn : sorted_inputs) {
            sig ^= (conn.src_id * 0x9E3779B97F4A7C15ULL);
            sig ^= (static_cast<uint64_t>(conn.out_idx) << 32);
            sig = (sig << 1) | (sig >> 63);
        }
        /* tensor_param 也是语义的一部分: 同名同输入但 ranges/perm 不同的
         * op(probe rope Slice 半切)不得合并 */
        std::vector<op_id_t> tp = opdef->tensor_param_ids;
        std::sort(tp.begin(), tp.end());
        for (op_id_t tpid : tp) {
            sig ^= (static_cast<uint64_t>(tpid) * 0x9E3779B97F4A7C15ULL);
            sig = (sig << 1) | (sig >> 63);
        }
        /* scalar params(op_data)同样参与签名 —— Split 三副本同名字同输入,
         * 仅 split_index 不同, 不混 params 会被 CSE 合并(probe qkv 实锤:
         * 副本 1/2 被删 → Reshape_2 输入重连到 q 段 → 整链数值错) */
        for (uint8_t b8 : opdef->op_data) {
            sig ^= (static_cast<uint64_t>(b8) + 0x9E3779B97F4A7C15ULL);
            sig = (sig << 8) | (sig >> 56);
        }
        /* 输出形状也是语义的一部分 —— 同输入无参数但输出形状不同的 op
         * (GDN Unsqueeze_25/26: 同 CumSum 输入, axes 烘焙, 输出
         * [1,16,1,64,1] vs [1,16,1,1,64])不得合并 */
        sig ^= (static_cast<uint64_t>(opdef->output_def.rank) * 0x9E3779B97F4A7C15ULL);
        sig ^= static_cast<uint64_t>(opdef->output_def.dtype) << 48;
        for (uint32_t d = 0; d < opdef->output_def.rank && d < 5; ++d) {
            sig ^= (static_cast<uint64_t>(opdef->output_def.dims[d]) + 0x9E3779B97F4A7C15ULL);
            sig = (sig << 5) | (sig >> 59);
        }
        return sig;
    };

    std::vector<op_id_t> to_remove;

    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_const() || opdef->is_dead())
            continue;
        if (id == input_node_id_ || id == output_node_id_) continue;
        /* 多输入图: 各 Input 节点(cos/sin 同形状)不得 CSE 合并 */
        if (opdef->name_tag && opdef->name_tag->name() &&
            std::string(opdef->name_tag->name()) == "Input")
            continue;

        uint64_t sig = compute_signature(opdef.get());

        auto it = cse_map.find(sig);
        if (it != cse_map.end()) {
            // Found potential match - verify inputs are identical
            OpDef* existing = it->second;
            bool match = (existing->inputs.size() == opdef->inputs.size());
            if (match) {
                for (size_t i = 0; i < opdef->inputs.size(); ++i) {
                    if (existing->inputs[i].src_id != opdef->inputs[i].src_id ||
                        existing->inputs[i].out_idx != opdef->inputs[i].out_idx) {
                        match = false;
                        break;
                    }
                }
            }

            if (match) {
                // Replace this op's consumers to point to existing op
                // Update consumers of this op to use existing->op_id instead
                for (op_id_t consumer_id : opdef->consumers) {
                    auto cit = opdef_map_.find(consumer_id);
                    if (cit != opdef_map_.end() && cit->second) {
                        for (auto& conn : cit->second->inputs) {
                            if (conn.src_id == id) {
                                conn.src_id = existing->op_id;
                            }
                        }
                        // Add to existing op's consumer list
                        existing->consumers.push_back(consumer_id);
                    }
                }
                to_remove.push_back(id);
                eliminated++;
            }
        } else {
            cse_map[sig] = opdef.get();
        }
    }

    // Remove eliminated ops
    for (op_id_t id : to_remove) {
        opdef_map_.erase(id);
    }

    if (eliminated > 0) graph_dirty_ = true;
    return eliminated;
}

// const_prop: constant propagation
// Source: graph_prepare.cc, 1667 bytes at 0xF74E30
void GraphPrepare::const_prop(HexagonNNEnv& env, bool aggressive) {
    // Source: graph_prepare.cc, 1667 bytes at 0xF74E30
    // Algorithm:
    // 1. Find all const ops (OpDef_Const)
    // 2. For each consumer of a const op:
    //    - If all inputs are const: the op can be constant-folded
    //    - replace_opdef_with_opconst: replace with OpDef_Const
    // 3. aggressive=true: fold through more op types (e.g., Concat, Reshape)
    // 4. const_prop_extract_outputs: evaluate the op with const inputs
    // 5. zero_fill_outputs_for_const_prop: if can't evaluate, zero-fill
    // 6. try_reduce_const: simplify const expressions

    bool changed = true;
    while (changed) {
        changed = false;

        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_const() || opdef->is_dead())
                continue;
            if (id == input_node_id_ || id == output_node_id_) continue;

            // Check if all inputs are const
            bool all_const = !opdef->inputs.empty();
            for (const auto& conn : opdef->inputs) {
                auto src_it = opdef_map_.find(conn.src_id);
                if (src_it == opdef_map_.end() || !src_it->second ||
                    !src_it->second->is_const()) {
                    all_const = false;
                    break;
                }
            }

            if (all_const) {
                // 折叠必须算出值(replace_opdef_with_opconst 语义);
                // 无值只标 OP_CONST 会让 emit 跳过该 op → 消费方读空槽
                // → 数值错(M3c probe 实锤: 掩码子图折叠后 emit 空 blob)。
                // 求值需接 TypicalOp host 参考核(run_host 通路已存在,
                // 见 optimize 区 tensor_map 执行链), 暂缓 —— 设备执行
                // 常量子图成本可忽略, 正确性优先。
                (void)aggressive;
            }
        }
    }

    if (changed) graph_dirty_ = true;
}

void GraphPrepare::const_prop_and_cse(HexagonNNEnv& env, bool aggressive, bool* changed) {
    // Source: graph_prepare.cc, 244 bytes at 0xF6A4B0
    const_prop(env, aggressive);
    int cse_changed = common_subexpr_eliminate(true);
    if (changed) *changed = (cse_changed != 0);
}

// eliminate_split_nodes: fold Split ops by replacing Split outputs with
// injected placeholder const tensors (type=4, dims=[1]).
// HtpPrepare's graph import eliminates Split nodes: each Split output is
// replaced by an injected placeholder tensor (no real producer). Downstream
// ops reference this placeholder, which appears as type=4 dims=[1] in
// before_graph.json.
void GraphPrepare::eliminate_split_nodes() {
    // Create one shared placeholder per Split GROUP (identified by 'grouping').
    // QNN uses dt=2147483647 (undefined quantization) with dims=[1] for
    // Split output placeholders (e.g., k/v tensors in attention).
    // All Split outputs from the same original Split share one placeholder.
    
    // First pass: create one placeholder per unique grouping
    std::map<std::string, op_id_t> group_placeholders;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
        if (!opdef->name_tag) continue;
        std::string nm = opdef->name_tag->name() ? opdef->name_tag->name() : "";
        if (nm != "Split") continue;
        std::string grp = opdef->grouping;
        if (group_placeholders.count(grp)) continue;
        
        op_id_t placeholder_id = next_op_id_++;
        OutputDef pod{};
        pod.rank = 1;
        pod.dtype = 2147483647;  // QNN undefined quantization datatype
        pod.dims[0] = 1;
        pod.element_size = 4;
        uint8_t zero[4] = {0, 0, 0, 0};
        append_const_node(placeholder_id, pod, zero, 4);
        group_placeholders[grp] = placeholder_id;
    }
    
    // Second pass: redirect all Split outputs to their group's placeholder
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
            if (!opdef->name_tag) continue;
            std::string nm = opdef->name_tag->name() ? opdef->name_tag->name() : "";
            if (nm != "Split") continue;
            std::string grp = opdef->grouping;
            auto pit = group_placeholders.find(grp);
            if (pit == group_placeholders.end()) continue;
            op_id_t placeholder_id = pit->second;

            // Redirect all consumers: replace input pointing to this Split
            // with the placeholder const.
            for (auto& [cid, cdef] : opdef_map_) {
                if (!cdef || !cdef->is_enabled() || cdef->is_dead()) continue;
                for (auto& conn : cdef->inputs) {
                    if (conn.src_id == id) {
                        conn.src_id = placeholder_id;
                    }
                }
                auto& cs = cdef->consumers;
                cs.erase(std::remove(cs.begin(), cs.end(), id), cs.end());
            }

            // Mark Split as dead
            opdef->flags |= OP_DEAD;
            changed = true;
            break; // restart iteration after modification
        }
    }
    // Rebuild consumer lists after redirecting inputs to placeholders
    rebuild_consumers();
    // Directly erase dead Split nodes from opdef_map_ (no cascade DCE).
    // QNN's before_graph keeps upstream nodes even after Split elimination,
    // so we must NOT cascade-delete the Split's input producer.
    for (auto it = opdef_map_.begin(); it != opdef_map_.end(); ) {
        if (it->second && it->second->is_dead()) {
            it = opdef_map_.erase(it);
        } else {
            ++it;
        }
    }
}

// rebuild_consumers: reconstruct consumer lists from input connections.
// Needed after build_graph because param const nodes (created with high IDs)
// may be created AFTER the op that references them, so append_node's consumer
// registration misses them. This walks all ops' inputs and rebuilds consumers.
void GraphPrepare::rebuild_consumers() {
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        opdef->consumers.clear();
    }
    int links = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled()) continue;
        for (const auto& conn : opdef->inputs) {
            auto src_it = opdef_map_.find(conn.src_id);
            if (src_it != opdef_map_.end() && src_it->second) {
                auto& cs = src_it->second->consumers;
                if (std::find(cs.begin(), cs.end(), id) == cs.end()) {
                    cs.push_back(id);
                    links++;
                }
            }
        }
    }
}
} // namespace hnnx
