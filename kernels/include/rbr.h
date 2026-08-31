/* rbr.h — U18 recurrent 状态部分接受回退 (快照→回拨→重放)
 *
 * 问题 (docs/P0-recurrent-state-rollback.md): 树形投机解码部分接受时,
 * recurrent 状态 (conv/recurrent, 定长无历史) 已被幻影 token 前移且 F 不可逆。
 * 解法族 A: 树评估后快照 in-state, 部分接受时 恢复→回拨→重放接受链。
 *
 * 本模块是引擎无关抽取 (§11.1): 引擎侧提供三件事 ——
 *   1. 状态注册表   rbr_register(group, in_state, bytes)
 *   2. setup hook   rbr_setup_hook(n_past) → CLEAR/COPY/SKIP 三分支
 *   3. 时机锚点     rbr_note_process(n_tokens) 每次 process 返回点调用
 * 对话层只调 rbr_snapshot / rbr_restore / rbr_rewind。
 *
 * 三个不变量 (模块合同, 违反即错):
 *   INV-1 快照时机  snapshot() 必须在 process() 返回之后 (in 此刻 = 本轮真实
 *                   起点; process 内部 setup 有隐藏 out→in 写)。模块用世代号
 *                   强制: process 后未重拍快照而 restore → 拒绝 (陈旧)。
 *   INV-2 KV 回拨   重放前 KV 行写位置必须回到本轮起点 (覆写而非追加)。
 *                   rbr 镜像 n_past 账本; 两套账 (对话层 token 记账 vs 引擎
 *                   行记账) 每帧 setup 对账, root 计数口径差在此暴露。
 *   INV-3 copy 抑制 restore 与 skip 成对: 恢复后的第一次 setup 必须走 SKIP
 *                   分支 (一次性), 否则恢复值被常规 out→in copy 立即覆盖。
 *
 * per-token 状态 (full-attn KV 行) 不需要本模块: 重放以相同 embeds+位置覆写
 * 相同行, bitwise 幂等 (§7.3)。
 */
#ifndef HVXHMX_V23_RBR_H
#define HVXHMX_V23_RBR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RBR_MAX_GROUPS 4    /* 状态一致性单元 (同图变体族) */
#define RBR_MAX_STATES 2    /* 每 group 的 recurrent in-state (conv + rec) */

#define RBR_OK            0
#define RBR_ERR_PARAM    -1
#define RBR_ERR_NOSNAP   -2   /* restore/rewind 无快照 */
#define RBR_ERR_STALE    -3   /* 快照后发生过 process (世代不符, INV-1) */
#define RBR_ERR_FULL     -4   /* 注册表满 */
#define RBR_ERR_FROZEN   -5   /* 已有快照后注册新状态 (布局变了, 须重拍) */

enum rbr_setup_mode {
    RBR_CLEAR = 0,   /* n_past==0: 引擎清零状态 */
    RBR_COPY,        /* 常规: 引擎 out→in copy */
    RBR_SKIP,        /* INV-3: 保留 in (刚 restore 过), 一次性 */
};

struct rbr {
    uint32_t n_groups;
    struct {
        uint32_t n_states;
        void*    in[RBR_MAX_STATES];      /* 引擎 recurrent in-state buffer */
        uint32_t bytes[RBR_MAX_STATES];
        uint8_t* shadow[RBR_MAX_STATES];  /* 逐字节影子 (malloc, 引擎零解释) */
    } grp[RBR_MAX_GROUPS];

    uint32_t  frame;         /* process 世代计数 (note_process +1) */
    uint32_t  snap_frame;    /* 快照拍摄时的 frame */
    int       has_snapshot;  /* 单代: 重拍覆盖 */
    int       skip_next;     /* INV-3 一次性标志 */

    uint32_t  n_past;        /* 引擎 KV 行位置镜像 (已提交行数) */
    uint32_t  last_tokens;   /* 最近一次 process 消费 token 数 */

    /* 统计 (取证/门) */
    uint32_t n_snapshot, n_restore, n_skip_used, n_rewind;
};

int  rbr_init(struct rbr* r);
/* 注册一个 recurrent in-state。group = 状态一致性单元 (恢复以 group 原子)。
 * 有快照后注册 → RBR_ERR_FROZEN。 */
int  rbr_register(struct rbr* r, uint32_t group, void* in_state, uint32_t bytes);
/* INV-1: 调用方保证在 process() 返回后调用。逐字节 in→shadow, 覆盖旧快照。 */
int  rbr_snapshot(struct rbr* r);
/* 前置: 有快照且世代新鲜 (快照后无 process)。后置: in←shadow; skip_next=1。 */
int  rbr_restore(struct rbr* r);
/* 引擎每帧 setup 调用; 返回分支并驱动一次性 skip 消耗。
 * n_past==0 → CLEAR 且作废挂起 skip (引擎无条件清零状态, restore 对 fresh
 * start 无意义 — leak 回归: 否则标志穿透到下一帧把常规 COPY 错吃成 SKIP)。
 * 调用方应同时校验 engine_n_past == rbr_n_past() (两账本对账, 陷阱1)。 */
enum rbr_setup_mode rbr_setup_hook(struct rbr* r, uint32_t engine_n_past);
/* process 返回点: frame++, n_past += n_tokens, last_tokens = n_tokens。 */
int  rbr_note_process(struct rbr* r, uint32_t n_tokens);
/* INV-2: 部分接受回拨。n_selected = 接受链长 (含 root)。
 * 语义: n_past 回到本轮起点 (树评估增量的 last_tokens 全部扣除),
 * 随后 replay 的 note_process(n_selected) 把行位置落在
 * [round_start, round_start+n_selected) — 覆写而非追加。
 * 若引擎账本含 root 补偿 (root 计两次), 调用方先自行对齐再回拨。 */
int  rbr_rewind(struct rbr* r, uint32_t n_selected);
uint32_t rbr_n_past(const struct rbr* r);
/* 取证等式2: 全部注册 in-state 与 shadow 逐字节相等 (restore 生效直接证据) */
int  rbr_shadow_equals(const struct rbr* r);
void rbr_close(struct rbr* r);    /* 释放影子缓冲, 注册表清零 */

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_RBR_H */
