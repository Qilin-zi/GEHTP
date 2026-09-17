/* ring_policy.c — U20 通用预取环策略库实现 (GENERIC_A §6)
 *
 * canonical submit (wt-bp-ring build52 消化, §6.4 纪律):
 *   claim  锁内填描述符, span 预链接 (d[c].next=&d[c+1]), 尾 next=NULL,
 *          深度门 n_qsub < n_fired + depth ∧ n_qsub - n_rel < N (I2/I3);
 *   fire   锁外逐 entry: IDLE→dmwait+dmstart (L8/L1); RUN→dmlink 仅当
 *          slack≥2 (L7), 否则推迟 (所有等待路径重驱动 → 无死锁);
 *   re-kick 仅 dmwait 确认真空闲后 dmstart (幂等, F2 孤儿救援)。
 * 组 g 恒落槽 g%N; 条目状态 PENDING→QUEUED→SUBMITTED→READY→GONE。
 */
#include "ring_policy.h"

#include <stdlib.h>
#include <string.h>

#define RP_PENDING 0u
#define RP_QUEUED  1u   /* 描述符已填 (claimed) */
#define RP_SUBMIT  2u   /* 已发门铃 (引擎可见) */
#define RP_READY   3u
#define RP_GONE    4u

struct ring_ent {
    uint8_t  state;
    uint32_t n_descs;
    eng_desc *dfirst, *dlast;   /* 槽内 span 首/尾 (池内固定区段) */
    uint64_t src, slot;
    uint32_t bytes;
};

struct ring_policy {
    engine_adapter eng;                  /* 值拷贝 */
    unsigned n_slots, dp_slot;
    uint32_t slot_bytes;
    uint8_t *slots;
    eng_desc *pool;
    struct ring_ent ent[RING_MAX_GROUPS];
    uint32_t n_enq, n_qsub, n_fired, n_ready, n_rel;
    eng_desc *fire_tail, *fire_tail_pred;   /* L7/串行化器维护 */
    uint32_t fire_depth_boost;              /* serial 相位提升 (临时) */
    volatile int lock;                      /* __sync 自旋 (host/DSP 通用) */
    int failed;
    ring_stats st;
};

static void lk(volatile int *m) { while (__sync_lock_test_and_set(m, 1)) { } }
static void ulk(volatile int *m) { __sync_lock_release(m); }

static unsigned cur_depth(const ring_policy *r) {
    return r->fire_depth_boost ? r->n_slots : RING_FIRE_DEPTH;
}

/* ---- claim: 填描述符 (锁内, 无引擎操作) ---- */
static void claim(ring_policy *r) {
    while (r->n_qsub < r->n_enq
        && r->n_qsub - r->n_rel < r->n_slots                 /* I2 */
        && r->n_qsub < r->n_fired + cur_depth(r)) {          /* 深度窗 */
        struct ring_ent *e = &r->ent[r->n_qsub];
        unsigned slot = r->n_qsub % r->n_slots;
        eng_desc *base = r->pool + (size_t)slot * r->dp_slot;
        uint32_t rowb = e->bytes ? e->bytes : 1u;
        /* span 拆分: 每 desc ≤ slot_bytes (2D 语义收窄为字节总量) */
        uint32_t nd = (rowb + r->slot_bytes - 1u) / r->slot_bytes;
        if (nd < 1u) nd = 1u;
        if (nd > r->dp_slot) { r->failed = 1; return; }      /* 拒收降级 */
        uint64_t off = 0;
        for (uint32_t i = 0; i < nd; i++) {
            eng_desc *d = base + i;
            d->next = (i + 1u < nd) ? (base + i + 1u) : NULL; /* span 预链 */
            d->done = 0;
            uint32_t chunk = e->bytes - (uint32_t)(off);
            if (chunk > r->slot_bytes) chunk = r->slot_bytes;
            d->src = e->src + off;
            d->dst = e->slot + off;
            d->bytes = chunk;
            d->nrows = 1u; d->row_size = chunk;
            d->src_stride = d->dst_stride = chunk;
            d->flags = 0;
            off += chunk;
        }
        e->dfirst = base;
        e->dlast = base + (nd - 1u);
        e->n_descs = nd;
        e->state = RP_QUEUED;
        r->n_qsub++;
    }
}

static int desc_done(const eng_desc *d) { return d->done != 0u; }

static void fire(ring_policy *r);

static void claim_and_fire(ring_policy *r) {
    lk(&r->lock);
    claim(r);
    ulk(&r->lock);
    fire(r);
}

/* ---- fire: 发门铃 (锁外) ---- */
static void fire(ring_policy *r) {
    if (r->failed) return;
    for (;;) {
        if (r->n_fired >= r->n_qsub) return;
        struct ring_ent *e = &r->ent[r->n_fired];
        if (e->state != RP_QUEUED) return;
        uint32_t st = r->eng.dmpoll();
        if (st == ENG_STATUS_RUN) {
            int slack2 = r->fire_tail && !desc_done(r->fire_tail)
                && (!r->fire_tail_pred || !desc_done(r->fire_tail_pred));
            if (!slack2) return;               /* 推迟; 等待路径重驱动 */
            if (r->eng.dmlink(r->fire_tail, e->dfirst) != 0) {
                r->failed = 1; return;         /* 断言层冻结 */
            }
            r->st.n_doorbells++;
        } else {
            r->eng.dmwait();                   /* L8: 关 retire 窗 */
            if (r->eng.dmstart(e->dfirst) != 0) { r->failed = 1; return; }
            r->st.n_doorbells++;
        }
        r->fire_tail_pred = e->n_descs > 1u ? e->dlast - 1 : r->fire_tail;
        r->fire_tail = e->dlast;
        e->state = RP_SUBMIT;
        r->n_fired++;
    }
}

/* READY 前缀推进 (done 位由引擎结算后可见) */
static void advance_ready(ring_policy *r) {
    while (r->n_ready < r->n_fired) {
        struct ring_ent *e = &r->ent[r->n_ready];
        if (e->state > RP_SUBMIT) { r->n_ready++; continue; }
        const eng_desc *d = e->dfirst;
        int all = 1;
        for (uint32_t i = 0; i < e->n_descs; i++, d = d->next)
            if (!d->done) { all = 0; break; }
        if (!all) break;
        e->state = RP_READY;
        r->n_ready++;
        r->st.bytes_prefetched += e->bytes;
    }
}

ring_policy *ring_create(const engine_adapter *eng, unsigned n_slots,
                         unsigned max_descs_per_slot, uint32_t slot_bytes,
                         uint8_t *slots_base) {
    if (!eng || !n_slots || !max_descs_per_slot || !slot_bytes || !slots_base)
        return NULL;
    ring_policy *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->eng = *eng;
    r->n_slots = n_slots;
    r->dp_slot = max_descs_per_slot > RING_MAX_DESCS_SLOT
        ? RING_MAX_DESCS_SLOT : max_descs_per_slot;
    r->slot_bytes = slot_bytes;
    r->slots = slots_base;
    r->pool = r->eng.pool_alloc(n_slots, r->dp_slot);
    if (!r->pool) { free(r); return NULL; }
    return r;
}

void ring_destroy(ring_policy *r) {
    if (!r) return;
    if (r->pool) r->eng.pool_free(r->pool);
    free(r);
}

int ring_enqueue(ring_policy *r, uint64_t src, uint32_t bytes,
                 uint32_t nrows, uint32_t row_size, uint32_t src_stride) {
    if (!r || !bytes) return RING_E_ARG;
    if (r->failed) return RING_E_FATAL;
    lk(&r->lock);
    if (r->n_enq >= RING_MAX_GROUPS) { ulk(&r->lock); return RING_E_RANGE; }
    struct ring_ent *e = &r->ent[r->n_enq];
    memset(e, 0, sizeof *e);
    e->src = src;
    e->bytes = bytes;
    e->slot = (uint64_t)(uintptr_t)r->slots
            + (size_t)(r->n_enq % r->n_slots) * r->slot_bytes;
    e->state = RP_PENDING;
    r->n_enq++;
    r->st.n_groups++;
    (void)nrows; (void)row_size; (void)src_stride;   /* 几何透传位 */
    claim(r);
    ulk(&r->lock);
    fire(r);
    return r->failed ? RING_E_FATAL : RING_OK;
}

ring_need_result ring_need(ring_policy *r, uint32_t g,
                           uint64_t deadline_us, uint64_t *slot_out) {
    if (!r) return RING_DEGRADED;
    if (r->failed) { if (slot_out) *slot_out = 0; return RING_DEGRADED; }
    if (g >= r->n_enq) { if (slot_out) *slot_out = 0; return RING_NOT_READY; }
    uint64_t t0 = r->eng.clock_us();
    uint64_t deadline = t0 + deadline_us;
    int rekicked = 0;
    for (;;) {
        r->eng.dmpoll();                      /* 结算引擎事件到当前时刻 */
        advance_ready(r);
        if (r->ent[g].state >= RP_READY) {
            if (slot_out) *slot_out = r->ent[g].slot;
            r->st.stall_us_total += (double)(r->eng.clock_us() - t0);
            r->fire_depth_boost = 0;          /* bulk 到来, 深度回落 */
            return RING_READY;
        }
        if (r->failed) {
            r->st.stall_us_total += (double)(r->eng.clock_us() - t0);
            if (slot_out) *slot_out = 0; return RING_DEGRADED;
        }
        fire(r);                              /* 等待即驱动 */
        claim_and_fire(r);
        uint64_t now = r->eng.clock_us();
        if (now >= deadline) {
            /* re-kick: dmwait 确认真空闲后 dmstart 幂等救援 (F2) */
            if (!rekicked && r->ent[g].state == RP_SUBMIT) {
                r->eng.dmwait();
                if (r->eng.dmstart(r->ent[g].dfirst) == 0) {
                    r->st.n_doorbells++;
                    r->st.n_rekicks++;
                    r->fire_tail = r->ent[g].dlast;
                    r->fire_tail_pred = r->ent[g].n_descs > 1u
                        ? r->ent[g].dlast - 1 : NULL;
                    rekicked = 1;
                    deadline = r->eng.clock_us() + deadline_us;  /* 宽限一周期 */
                    continue;
                }
            }
            r->st.stall_us_total += (double)(now - t0);
            r->st.n_degrades++;
            if (slot_out) *slot_out = 0;
            return RING_NOT_READY;
        }
        r->eng.spin(2u);                      /* 自旋节拍 (虚拟/真实) */
    }
}

void ring_release(ring_policy *r, uint32_t g) {
    if (!r || g >= r->n_enq) return;
    lk(&r->lock);
    if (r->ent[g].state == RP_READY) {
        r->ent[g].state = RP_GONE;
        while (r->n_rel < r->n_enq && r->ent[r->n_rel].state == RP_GONE)
            r->n_rel++;                       /* GONE 前缀推进 */
    }
    claim(r);                                 /* RELEASE 先于 CLAIM */
    ulk(&r->lock);
    fire(r);
}

void ring_on_serial_phase(ring_policy *r, uint32_t est_us) {
    if (!r) return;
    (void)est_us;
    lk(&r->lock);
    r->fire_depth_boost = 1;                  /* 空窗填满环 */
    claim(r);
    ulk(&r->lock);
    fire(r);
    fire(r);                                  /* 两次: IDLE 起步后续链 */
}

int ring_drain_to(ring_policy *r, uint32_t g, uint64_t watchdog_us) {
    if (!r || g >= r->n_enq) return RING_E_ARG;
    for (uint32_t i = r->n_rel; i <= g; i++) {
        uint64_t slot = 0;
        ring_need_result rc = ring_need(r, i, watchdog_us, &slot);
        if (rc != RING_READY) return RING_E_FATAL;
        ring_release(r, i);
    }
    return RING_OK;
}

void ring_get_stats(const ring_policy *r, ring_stats *out) {
    if (out) *out = r->st;
}

int ring_check_invariants(const ring_policy *r) {
    if (!r) return 1;
    if (!(r->n_rel <= r->n_ready && r->n_ready <= r->n_fired
          && r->n_fired <= r->n_qsub && r->n_qsub <= r->n_enq)) return 1; /* I1 */
    if (r->n_qsub - r->n_rel > r->n_slots) return 2;                     /* I2 */
    if (r->fire_tail
        && !(r->fire_tail >= r->pool
             && r->fire_tail < r->pool
                  + (size_t)r->n_slots * r->dp_slot)) return 5;           /* I5 */
    return 0;
}
