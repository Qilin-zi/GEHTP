/* bledger.h — U19 Buffer Ledger 数据流溯源审计 (P1 未初始化读/断链)
 *
 * 问题 (docs/P1-uninitialized-read-dataflow.md): 管线下游消费的 buffer 在某条
 * 执行路径上从未被生产者写入 (或被错误写入者/错误偏移写入) —— 读出"结构合法的
 * 垃圾"。静默、不崩溃、只在下游统计 (接受率/准确率) 露出异常。
 *
 * 本模块把 P1 §11 的 Buffer Ledger 固化: 每 buffer 每 row 一本台账 ——
 * 谁写的 (writer) / 第几代 (seq) / 什么量化标签 (qtag) / 现在什么状态。
 * 消费点声明期望 (expect) 并核账 (verify): 断链从"统计异常晚发现"变成
 * "消费即报错" (L2)。canary/release 把"从未写过"变成"读已知模式" (E3)。
 *
 * 根因覆盖 (P1 §3):
 *   3.1 分配未初始化   → verify 报 NEVER / CANARY
 *   3.2 路径依赖跳过   → 新路径消费点 verify 即时报错 (T5)
 *   3.3 行号错位       → verify 报 NEVER (读的行从未写过) + 例 33 rank 反查定位
 *   3.4 生命周期/别名  → release 后 verify 报 RELEASED (归还前自动 canary)
 *   3.5 dtype/scale    → qtag 写读对账 (verify 报 QTAG)
 *   3.6 双写竞争       → 未读行被异 writer 覆写 → n_double_write 计数
 *
 * 三个使用强度 (P1 §11.2):
 *   L1 取证: canary + timeline (排查时挂上)
 *   L2 断言: 消费点 expect+verify (CI / 例 33)
 *   L3 生产: 编译期定义 BL_DISABLED → 全 API 变 no-op, 零开销
 *
 * 反模式 (P1 §8.1): 本模块不做消费点兜底/静默修数; 只报错与取证。
 */
#ifndef HVXHMX_V23_BLEDGER_H
#define HVXHMX_V23_BLEDGER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BL_MAX_BUFS 4

#define BL_ANY_WRITER 0u   /* expect 通配: 任意 writer 均可 */
#define BL_NO_QTAG    0u   /* qtag 不校验 (expect 或 write 侧均可置 0) */

#define BL_OK            0
#define BL_ERR_PARAM    -1   /* 参数非法 / 重复注册 / writer==0 */
#define BL_ERR_NOSUCH   -2   /* slot 未注册 */
#define BL_ERR_RANGE    -3   /* 行号越界 */
#define BL_ERR_NEVER    -4   /* ★ 3.1/3.2: 该行从未被写 (断链实锤) */
#define BL_ERR_CANARY   -5   /* ★ 3.1/E3: 读到 canary 模式 (未初始化读实锤) */
#define BL_ERR_RELEASED -6   /* ★ 3.4: 已归还 (pool) 后消费 */
#define BL_ERR_WRITER   -7   /* ★ 3.4/H3: 写入者与期望不符 */
#define BL_ERR_QTAG     -8   /* ★ 3.5: 量化标签 (scale/offset) 写读不一致 */

/* 行状态机: UNWRITTEN --write--> WRITTEN --verify--> READ
 *            任意 --canary--> CANARY --release--> RELEASED
 *            CANARY/RELEASED --write--> WRITTEN (重新拿回写权) */
enum bl_state {
    BL_ST_UNWRITTEN = 0,
    BL_ST_WRITTEN,      /* 已写未读 */
    BL_ST_READ,         /* 已写且至少消费过一次 */
    BL_ST_CANARY,       /* canary 填充过 (数据 = 0xAA 模式) */
    BL_ST_RELEASED,     /* 已归还池 (数据 = 0xAA 模式) */
};

struct bl_row {
    uint32_t seq;            /* 写入代数 (全局单调, 0 = 从未写) */
    uint16_t writer;         /* 最后写入者 (注册的 writer id) */
    uint16_t qtag;           /* 量化标签 (scale/offset 打包; 0 = 无) */
    uint8_t  state;          /* enum bl_state */
    uint8_t  has_expect;     /* 消费点已声明期望 */
    uint16_t exp_writer;     /* 期望 writer (BL_ANY_WRITER=任意) */
    uint16_t exp_qtag;       /* 期望 qtag (BL_NO_QTAG=不查) */
};

struct bl_buf {
    char     name[16];
    uint8_t* base;           /* canary/release 时 memset 用 */
    uint32_t bytes, row_bytes, n_rows;
    struct bl_row* rows;     /* malloc, close 释放 */
    uint32_t n_writes, n_reads, n_breaks;   /* 事件数 / 成功 verify / 报错 */
};

struct bledger {
    uint32_t n_bufs;
    uint32_t seq;                    /* 全局写入代数 */
    uint32_t n_double_write;         /* 未读行被异 writer 覆写 (3.6) */
    struct bl_buf buf[BL_MAX_BUFS];
};

int bl_init(struct bledger* b);
/* 注册 buffer: slot ∈ [0, BL_MAX_BUFS); bytes 须为 row_bytes 整数倍。
 * name 仅用于 timeline/日志 (可 NULL)。重复注册 → PARAM。 */
int bl_register(struct bledger* b, uint32_t slot, void* base, uint32_t bytes,
                uint32_t row_bytes, const char* name);
/* 写入记账: [row0, row0+nrows) 行盖戳 (同一次调用共享一个 seq)。
 * 未读行 (WRITTEN) 被不同 writer 覆写 → n_double_write++ (3.6)。
 * 覆写 CANARY/RELEASED 行合法 (重新拿回写权)。 */
int bl_write(struct bledger* b, uint32_t slot, uint16_t writer,
             uint32_t row0, uint32_t nrows, uint16_t qtag);
/* E3 canary 判别: base 填 0xAA, 全行状态 → CANARY (seq/writer 保留供 timeline)。 */
int bl_canary(struct bledger* b, uint32_t slot);
/* 归还池: base 填 0xAA (§8 归还前 canary), 全行 → RELEASED。此后消费报 RELEASED。 */
int bl_release(struct bledger* b, uint32_t slot);
/* 消费点声明期望 (T5): writer_filter=BL_ANY_WRITER 任意; qtag=BL_NO_QTAG 不查。 */
int bl_expect(struct bledger* b, uint32_t slot, uint32_t row,
              uint16_t writer_filter, uint16_t qtag);
/* 消费点核账: 状态机判定 + 期望比对; 失败 n_breaks++ 并返回 BL_ERR_*;
 * 成功 n_reads++, 行 → READ (幂等可复核)。 */
int bl_verify(struct bledger* b, uint32_t slot, uint32_t row);
uint32_t bl_seq(const struct bledger* b);
/* L1 取证: 文本时序图 (P1 §4.4) 写入 out (cap 截断), 返回字符数 (不含 NUL)。 */
uint32_t bl_timeline(const struct bledger* b, uint32_t slot,
                     char* out, uint32_t cap);
void bl_close(struct bledger* b);   /* 释放 rows, 全结构清零 */

/* ---- L3 生产关闭: 编译期定义 BL_DISABLED → 全 no-op, 零开销 ---- */
#ifdef BL_DISABLED
struct bledger;   /* 不完整类型: 生产构建无需分配对象, 传 NULL 即可 */
static inline int bl_init(struct bledger* b) { (void)b; return BL_OK; }
static inline int bl_register(struct bledger* b, uint32_t s, void* p,
                              uint32_t y, uint32_t r, const char* n) {
    (void)b;(void)s;(void)p;(void)y;(void)r;(void)n; return BL_OK; }
static inline int bl_write(struct bledger* b, uint32_t s, uint16_t w,
                           uint32_t r0, uint32_t n, uint16_t q) {
    (void)b;(void)s;(void)w;(void)r0;(void)n;(void)q; return BL_OK; }
static inline int bl_canary(struct bledger* b, uint32_t s) { (void)b;(void)s; return BL_OK; }
static inline int bl_release(struct bledger* b, uint32_t s) { (void)b;(void)s; return BL_OK; }
static inline int bl_expect(struct bledger* b, uint32_t s, uint32_t r,
                            uint16_t w, uint16_t q) { (void)b;(void)s;(void)r;(void)w;(void)q; return BL_OK; }
static inline int bl_verify(struct bledger* b, uint32_t s, uint32_t r) {
    (void)b;(void)s;(void)r; return BL_OK; }
static inline uint32_t bl_seq(const struct bledger* b) { (void)b; return 0u; }
static inline uint32_t bl_timeline(const struct bledger* b, uint32_t s,
                                   char* o, uint32_t c) { (void)b;(void)s;(void)o;(void)c; return 0u; }
static inline void bl_close(struct bledger* b) { (void)b; }
#endif

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_BLEDGER_H */
