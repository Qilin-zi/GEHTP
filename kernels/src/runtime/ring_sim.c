/* ring_sim.c — U20 引擎 FSM 仿真 + Law1-8 断言层 (GENERIC_A §5)
 *
 * 纯软件离散事件模型, 单线程虚拟时钟 (µs, double) 惰性推进:
 *   - 引擎不主动跑; 每次 poll/wait/advance 按当前时刻结算完成事件
 *   - done 位置位 = 数据 memcpy 落槽时刻 (F4 注入可撕开两者)
 *   - RETIRE 窗 = [链尾done+delay, +w_retire): dmpoll 谎报 IDLE (灵魂, 不许修)
 *   - 门铃 (dmstart/dmlink 唤醒) 软件成本 t_doorbell 计入虚拟时钟
 *
 * 断言 (§3 速查表): L1/L4/L5/L6/L7/L8 → FATAL 冻结; L2/L3 是 FSM 语义。
 */
#include "engine_adapter.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef enum { S_IDLE = 0, S_RUN, S_RETIRE } eng_state;

static const sim_params DEF = { 55.0, 0.5, 2.0, 1.0, 0.3, 2.0 };

const sim_params *sim_default_params(void) { return &DEF; }

static struct {
    sim_params  p;
    eng_state   st;
    double      now;
    eng_desc   *head;          /* 当前链头 (dmstart / 唤醒起点) */
    eng_desc   *cur;           /* 正在执行的描述符 */
    double      cur_t_end;     /* cur 完成时刻 */
    double      retire_end;    /* RETIRE 窗关闭时刻 */
    eng_desc   *parked_tail;   /* 上次 park 时的链尾 (law2 锚) */
    double      last_doorbell_t;
    double      last_dmwait_done_t;
    eng_desc   *pool_lo, *pool_hi;
    unsigned    pool_n;
    uint32_t    fault;         /* SIM_FAULT_* */
    sim_report  rep;           /* rep.law != 0 → 冻结 */
    int         frozen;
    int         single_shot;   /* law2: IDLE dmlink 唤醒只执行一条即再 park */
} E;

static int in_pool(const eng_desc *d) {
    return d >= E.pool_lo && d < E.pool_hi
        && ((uintptr_t)d & 7u) == 0u
        && ((uintptr_t)d - (uintptr_t)E.pool_lo)
               % (unsigned)sizeof(eng_desc) == 0u;   /* L5: 池内且 8B 对齐 */
}

static void fatal(int law, const char *op) {
    if (E.frozen) return;
    E.frozen = 1;
    E.rep.law = law;
    E.rep.t_us = E.now;
    snprintf(E.rep.op, sizeof E.rep.op, "%s", op);
}

/* 搬运耗时 µs: bw_eng[GB/s] = B/ns → bytes/bw = ns → /1000 = µs */
static double xfer_us(uint32_t bytes) {
    return (double)bytes / (E.p.bw_eng * 1000.0);
}

/* 结算 (t_done ≤ t) 的完成事件; 可能进入/退出 RETIRE */
static void settle(double t) {
    for (;;) {
        if (E.st != S_RUN) return;
        if (t < E.cur_t_end) return;
        /* cur 完成: done + 数据落地 (F4 注入撕开一次) */
        double t_done = E.cur_t_end;
        if (E.fault & SIM_FAULT_F4_EARLY_DONE) {
            E.cur->done = 1;                 /* 假 done, 数据未拷 */
            E.fault &= ~SIM_FAULT_F4_EARLY_DONE;
        } else {
            memcpy((void *)(uintptr_t)E.cur->dst, (void *)(uintptr_t)E.cur->src,
                   E.cur->bytes);
            E.cur->done = 1;
        }
        E.rep.n_bytes += E.cur->bytes;
        if (E.cur->next && !E.single_shot) {     /* law3 续链自然执行 */
            E.cur = E.cur->next;
            E.cur_t_end = t_done + E.p.t_desc + xfer_us(E.cur->bytes);
            continue;
        }
        /* 链走完(或 law2 单发) → RETIRE 窗 [t_done+delay, +w_retire) */
        E.single_shot = 0;
        E.parked_tail = E.cur;
        E.retire_end = t_done + E.p.done_to_retire + E.p.w_retire;
        E.st = S_RETIRE;
        return;
    }
}

double sim_now(void) { return E.now; }

double sim_advance(double us) {
    E.now += us;
    settle(E.now);
    if (E.st == S_RETIRE && E.now >= E.retire_end) {
        E.st = S_IDLE;                       /* RETIRE 关闭, 真 IDLE */
        E.rep.t_busy_us = E.now;             /* 结算点近似 */
    }
    return E.now;
}

static uint32_t op_poll(void) {
    settle(E.now);
    if (E.st == S_RETIRE) return ENG_STATUS_IDLE;   /* 说谎 (law8 灵魂) */
    return E.st == S_RUN ? ENG_STATUS_RUN : ENG_STATUS_IDLE;
}

static void op_wait(void) {
    settle(E.now);
    if (E.st == S_RETIRE) sim_advance(E.retire_end - E.now);  /* 阻塞至真空闲 */
    E.last_dmwait_done_t = E.now;
}

static int doorbell_gap_ok(void) {           /* law4 */
    return (E.now - E.last_doorbell_t) >= E.p.w_accept
        || E.last_dmwait_done_t > E.last_doorbell_t;
}

static void note_doorbell(void) {
    E.last_doorbell_t = E.now;
    E.rep.n_doorbells++;
    sim_advance(E.p.t_doorbell);              /* 门铃软件成本 */
}

static int op_start(eng_desc *d) {
    if (E.frozen) return -1;
    E.single_shot = 0;                      /* law1 全链 walk, 清单发标志 */
    if (!in_pool(d)) { fatal(5, "dmstart"); return -1; }        /* L5 */
    if (E.st == S_RETIRE) { fatal(6, "dmstart"); return -1; }   /* L6: RETIRE 窗门铃 */
    if (E.st != S_IDLE) { fatal(1, "dmstart"); return -1; }     /* L1 */
    if (!doorbell_gap_ok()) { fatal(4, "dmstart"); return -1; } /* L4 */
    note_doorbell();
    if (E.fault & SIM_FAULT_F2_DOORBELL_LOST) {
        E.fault &= ~SIM_FAULT_F2_DOORBELL_LOST;   /* 门铃已响但引擎未醒 */
        return 0;                                 /* 孤儿: 描述符永不 done */
    }
    E.head = d;
    E.cur = d;
    E.cur_t_end = E.now + E.p.t_desc + xfer_us(d->bytes);
    E.st = S_RUN;
    return 0;
}

static int op_link(eng_desc *cur, eng_desc *nx) {
    if (E.frozen) return -1;
    if (!in_pool(cur) || !in_pool(nx)) { fatal(5, "dmlink"); return -1; }  /* L5 */
    if (E.st == S_RETIRE) { fatal(6, "dmlink"); return -1; }    /* L6 */
    cur->next = nx;                                          /* 原子存储语义 */
    if (E.st == S_RUN) {
        /* L7 slack: tail(cur) 与其前驱都未 done */
        if (cur->done) { fatal(7, "dmlink"); return -1; }
        eng_desc *pred = NULL;
        for (eng_desc *d = E.head; d && d != cur; d = d->next) pred = d;
        if (pred && pred->done) { fatal(7, "dmlink"); return -1; }
        if (!doorbell_gap_ok()) { fatal(4, "dmlink"); return -1; }  /* L4 */
        note_doorbell();
        return 0;                          /* law3: 续链, 引擎走到时执行 */
    }
    /* 真 IDLE: law2 —— 唤醒只执行一条即再 park (nx 的预链在此路径不生效;
     * 全链 walk 只有 law1 dmstart)。cur 已 done(正常 parked tail)则执行
     * nx;cur 未 done 则重做 cur 自身(幂等拷贝, "restart at cur" 语义)。 */
    if (E.fault & SIM_FAULT_F2_DOORBELL_LOST) {
        E.fault &= ~SIM_FAULT_F2_DOORBELL_LOST;   /* 吞唤醒 → nx 成孤儿 */
        return 0;
    }
    if (!doorbell_gap_ok()) { fatal(4, "dmlink"); return -1; }
    note_doorbell();
    {
        eng_desc *run = cur->done ? nx : cur;
        E.single_shot = 1;
        E.head = run;
        E.cur = run;
        E.cur_t_end = E.now + E.p.t_desc + xfer_us(run->bytes);
        E.st = S_RUN;
    }
    return 0;
}

static eng_desc *op_pool_alloc(unsigned n_slots, unsigned dp) {
    unsigned n = (unsigned)(n_slots * dp);
    eng_desc *p = calloc(n ? n : 1u, sizeof(eng_desc));
    if (!p) return NULL;
    E.pool_lo = p; E.pool_hi = p + n; E.pool_n = n;
    return p;
}
static void op_pool_free(eng_desc *p) {
    if (p) { E.pool_lo = E.pool_hi = NULL; E.pool_n = 0; }
    free(p);
}

static void op_spin(uint32_t us) { sim_advance((double)us); }

static uint64_t op_clock(void) { return (uint64_t)E.now; }

int sim_adapter_init(const sim_params *p, engine_adapter *out) {
    memset(&E, 0, sizeof E);
    E.p = p ? *p : DEF;
    E.st = S_IDLE;
    E.last_doorbell_t = -1e9;               /* 首门铃不误触 L4 */
    out->dmstart = op_start;
    out->dmlink = op_link;
    out->dmpoll = op_poll;
    out->dmwait = op_wait;
    out->pool_alloc = op_pool_alloc;
    out->pool_free = op_pool_free;
    out->spin = op_spin;
    out->clock_us = op_clock;
    return 0;
}

void sim_adapter_report(sim_report *rep) { *rep = E.rep; }

void sim_adapter_fault(uint32_t mask) { E.fault |= mask; }
