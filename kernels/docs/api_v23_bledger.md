# api_v23_bledger — U19 Buffer Ledger 数据流溯源审计

源: `src/runtime/bledger.c` · 头: `include/bledger.h` · 例: `33_bledger` (9 门 PASS)
· 需求源: `/data/Qwen30724/qwen3.5_4b_mtp/docs/P1-uninitialized-read-dataflow.md`

## 问题

管线下游消费的 buffer 在某条执行路径上从未被生产者写入 (或错误写入者/错误偏移)
—— 读出"结构合法的垃圾"。静默、不崩溃、几乎无现成检测器 (读的是自己的合法内存),
只在下游统计 (接受率/准确率/PPL) 露出异常 (P1 §1)。

## 核心思想 (P1 §11)

每个 buffer 每 row 一本**台账**: 谁写的 (writer) / 第几代 (seq) / 什么量化标签
(qtag) / 现在什么状态。消费点声明期望 (`bl_expect`) 并核账 (`bl_verify`):
断链从"统计异常晚发现"变成**消费即报错** (L2)。canary/release 把"从未写过"
变成"读已知模式" (E3 判别实验)。

## API

```c
#define BL_MAX_BUFS 4
#define BL_ANY_WRITER 0 / BL_NO_QTAG 0
BL_OK 0 / ERR_PARAM -1 / ERR_NOSUCH -2 / ERR_RANGE -3
BL_ERR_NEVER -4    /* ★ 3.1/3.2 该行从未被写 (断链实锤) */
BL_ERR_CANARY -5   /* ★ 3.1/E3 读到 canary 模式 (未初始化读实锤) */
BL_ERR_RELEASED -6 /* ★ 3.4 归还后消费 */
BL_ERR_WRITER -7   /* ★ H3 写入者与期望不符 */
BL_ERR_QTAG -8     /* ★ 3.5 量化标签写读不一致 */

enum bl_state { UNWRITTEN, WRITTEN, READ, CANARY, RELEASED };

int bl_init(struct bledger* b);
int bl_register(struct bledger* b, uint32_t slot, void* base, uint32_t bytes,
                uint32_t row_bytes, const char* name);  /* bytes 整倍于 row_bytes */
int bl_write(struct bledger* b, uint32_t slot, uint16_t writer,
             uint32_t row0, uint32_t nrows, uint16_t qtag);  /* 盖戳, 同调用共享 seq */
int bl_canary(struct bledger* b, uint32_t slot);   /* memset 0xAA + 全行 CANARY */
int bl_release(struct bledger* b, uint32_t slot);  /* 归还前 canary + 全行 RELEASED */
int bl_expect(struct bledger* b, uint32_t slot, uint32_t row,
              uint16_t writer_filter, uint16_t qtag);  /* 消费点声明期望 */
int bl_verify(struct bledger* b, uint32_t slot, uint32_t row);
      /* 核账: 失败 n_breaks++ + BL_ERR_*; 成功 n_reads++, 行→READ (幂等) */
uint32_t bl_seq(const struct bledger* b);
uint32_t bl_timeline(const struct bledger* b, uint32_t slot, char* out, uint32_t cap);
void bl_close(struct bledger* b);
```

## 根因 → 防御映射 (P1 §3)

| 根因 | 防御 |
|------|------|
| 3.1 分配未初始化 | verify → NEVER; canary 后 → CANARY + 0xAA 位样 |
| 3.2 路径依赖跳过 | 新路径消费点 verify 即时报错 (T5) |
| 3.3 行号错位 | verify → NEVER (读的行从未写) + 例 33 E2 rank 反查定位 |
| 3.4 生命周期/别名 | release 后 verify → RELEASED; expect writer → WRITER |
| 3.5 dtype/scale | qtag 写读对账 → QTAG |
| 3.6 双写竞争 | 未读行被异 writer 覆写 → `n_double_write` 计数 |

状态机: `UNWRITTEN --write--> WRITTEN --verify--> READ`; 任意 `--canary/release-->`
对应态; CANARY/RELEASED `--write-->` WRITTEN (重新拿回写权)。

## 双写计数语义

`bl_write` 覆写**未读** (WRITTEN) 行且 writer 不同 → `n_double_write++`
(last-writer-wins 风险, §3.6)。读后覆写合法 (刷新); 同 writer 覆写未读行不计。
注意: 读后覆写会把行变回 WRITTEN —— 其后再被异 writer 覆写仍计 (例 33 T7 实证)。

## 三个使用强度 (P1 §11.2)

- **L1 取证**: `bl_canary` + `bl_timeline` (排查时挂上; timeline = P1 §4.4 时序图)
- **L2 断言**: 消费点 `bl_expect` + `bl_verify` (CI / 例 33; 统计 `n_breaks`)
- **L3 生产**: 编译期定义 `BL_DISABLED` → 全 API static inline no-op, 零开销,
  无需分配 bledger 对象

反模式 (P1 §8.1): 本模块不做消费点兜底/静默修数, 只报错与取证。

## 验证 (例 33, P1 §12 模板)

| 门 | 模板 | 断言 |
|----|------|------|
| bl_contract_table | — | 参数域/状态机/统计 (writes/reads/breaks/seq) 精确 |
| t1_coldstart_immediate_error | T1 | 断链 verify=NEVER 即时报错; 写入收敛修复后 OK |
| t2_row_misalign_rank_localize | T2/E2 | 行错位: verify=NEVER + 消费值恰在错位行 rank 0 |
| t3_canary_fingerprint_h2 | T3/E3 | canary: verify=CANARY + 0xAA 位样 + 全行深 rank (与 H1a 区分) |
| t4_release_forbidden_read | T4 | release 后 verify=RELEASED + 归还前 canary 位样 |
| t7_double_write_detected | T7 | 未读异写覆写计数精确; 读后覆写合法 |
| t5t6_newpath_qtag_guard | T5/T6 | 新路径未写 → NEVER; qtag 不一致 → QTAG; 修复后 OK |
| t8_e2e_stethoscope_anchor | T8 | P1 §5 复刻: d1 垃圾 (rank 127/128) + d2 正确 (部分正确模式) → 每轮 break + 接受率崩塌 (hist 全 1); 修复引擎 breaks=0 + **回归锚 argmax(首树提议)==argmax(prefill 应写分布)** + hist 全 3 |
| l1_timeline_text | L1 | timeline 文本非空且含统计 (writes/breaks) |

host: `host/gtest_v23.c` t_bledger (契约/WRITER/double/release/canary/timeline)。
