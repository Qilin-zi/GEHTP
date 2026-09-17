# api_v23_rbr — U18 recurrent 状态部分接受回退 (快照→回拨→重放)

源: `src/runtime/rbr.c` · 头: `include/rbr.h` · 例: `32_rbr` (8 门 PASS)
· 需求源: `/data/Qwen30724/qwen3.5_4b_mtp/docs/P0-recurrent-state-rollback.md`

## 问题

树形投机解码部分接受时, recurrent 状态 (conv_state / recurrent_state, 定长、无历史)
已被幻影 token 整体前移且 F 不可逆。解法族 A: 树评估后快照 in-state, 部分接受时
**恢复 → 回拨 KV 行位置 → 重放接受链**。全接受轮零开销 (仅一次 memcpy 快照)。

## API

```c
#define RBR_MAX_GROUPS 4      /* 状态一致性单元 (同图变体族) */
#define RBR_MAX_STATES 2      /* 每 group 的 recurrent in-state (conv + rec) */
#define RBR_OK 0 / RBR_ERR_PARAM -1 / RBR_ERR_NOSNAP -2
#define RBR_ERR_STALE -3 / RBR_ERR_FULL -4 / RBR_ERR_FROZEN -5

enum rbr_setup_mode { RBR_CLEAR = 0, RBR_COPY, RBR_SKIP };

struct rbr { 注册表 (group×state: in 指针/bytes/shadow), frame/snap_frame,
             has_snapshot, skip_next, n_past/last_tokens, 4 组统计 };

int  rbr_init(struct rbr* r);
int  rbr_register(struct rbr* r, uint32_t group, void* in_state, uint32_t bytes);
      /* 有快照后注册 → RBR_ERR_FROZEN (布局稳定假设) */
int  rbr_snapshot(struct rbr* r);       /* INV-1: process() 返回后调用 */
int  rbr_restore(struct rbr* r);        /* 有快照且世代新鲜; 后置 skip_next=1 */
enum rbr_setup_mode rbr_setup_hook(struct rbr* r, uint32_t engine_n_past);
int  rbr_note_process(struct rbr* r, uint32_t n_tokens);  /* process 返回点 */
int  rbr_rewind(struct rbr* r, uint32_t n_selected);      /* INV-2 */
uint32_t rbr_n_past(const struct rbr* r);
int  rbr_shadow_equals(const struct rbr* r);  /* 取证等式2 */
void rbr_close(struct rbr* r);
```

## 引擎接线三锚点 (P0 §11.1)

| 锚点 | 引擎侧 | rbr 侧 |
|------|--------|--------|
| 状态注册表 | 启动时枚举 recurrent in-state | `rbr_register` |
| setup hook | 每帧 setup 调 `rbr_setup_hook(n_past)` 取分支: CLEAR(清零)/COPY(out→in)/SKIP(保留) | 一次性 skip 消耗 |
| 时机锚点 | `process()` 返回点 | `rbr_note_process(n)` |

对话层只调 `rbr_snapshot` (每轮树评估后) 与 部分接受时
`rbr_restore → rbr_rewind(n_selected) → 引擎 replay`。

## 三个不变量 (模块合同)

- **INV-1 快照时机**: snapshot 必须在 process() 返回后 (引擎 setup 有隐藏 out→in 写)。
  世代号强制: 快照后发生过 process 而 restore → `RBR_ERR_STALE`。
- **INV-2 KV 回拨**: 重放前 `rbr_rewind` 把行位置回本轮起点 (覆写非追加);
  `note_process(n_selected)` 落位 `[round_start, round_start+n_selected)`。
  引擎账本含 root 补偿时调用方先对齐。每帧 `engine_n_past == rbr_n_past()` 对账。
- **INV-3 copy 抑制**: restore 与 skip 成对; SKIP 一次性。**CLEAR 分支作废挂起 skip**
  (fresh start 引擎无条件清零, 标志不得穿透到下一帧 — 见坑 6)。

## 陷阱防御 ↔ 错误码

| P0 陷阱 | 防御 |
|---------|------|
| 陷阱1 双账本口径差 | 每帧对账 + `rewind` 语义 = 回本轮起点 |
| 陷阱2 快照时机差一阶段 | 世代号 `RBR_ERR_STALE` |
| 陷阱3 restore 被覆盖 | `skip_next` 一次性标志 |
| 布局变化 | `RBR_ERR_FROZEN` |

per-token 状态 (full-attn KV 行) 不需本模块: 相同 embeds+位置覆写相同行, bitwise 幂等 (P0 §7.3)。

## 验证 (例 32, 金字塔)

| 层 | 门 | 断言 |
|----|-----|------|
| L1 | l1_snapshot_roundtrip | 污染后恢复逐字节 + shadow_equals |
| L1 | l1_skip_onshot_then_copy_clear | SKIP 一次性 / COPY 恢复 / **CLEAR 作废 skip (m4)** |
| L1 | l1_stale_frozen_badrewind_rejected | NOSNAP / STALE / FROZEN / rewind 参数域 |
| L2 | defect_fingerprint_detected | 等式1: 缺陷引擎 replay_in == phantom_out (可检出且偏离基线) |
| L2 | restore_effective_equation2 | 等式2: replay_in == snapshot_src 逐字节 |
| L2 | replay_kv_overwrite_idempotent | 重放覆写行 == 树评估行 (P0 §7.3) |
| L3 | e2e_rounds_bitexact_vs_baseline | 16 轮混合 (j=0 强拒绝…全接受): 终态+91 KV 行+双账本 **bitwise == 基线** |
| L3 | counters_exact_fullaccept_zero_touch | snap=16 restore=rewind=6 skip=5 (round-0 重放=CLEAR 不耗 skip) |

host: `host/gtest_v23.c` t_rbr 5 门 (往返/skip 算术/陈旧冻结/rewind 账本/shadow_eq)。

## 已知限制 (P0 §13 同源)

- 单代快照 (重拍覆盖); 温度>0 的 RNG 对齐不在范围; 逐字节 shadow 假设 buffer 布局稳定。
- `restore` 后紧跟 fresh-start forward (n_past==0) 无意义 — 引擎清零, 快照作废。
