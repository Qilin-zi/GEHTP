/* dcache.c — U21 版本化派生格式缓存实现 (GENERIC_B §5)
 *
 * seqlock 形态: 写者 (bt_mark_*) 只原子推进版本, 绝不阻塞;
 * 读者 校验版本, 失败丢弃重试。转换只读私有快照拷贝 (§5.2),
 * 缓存产物写库私有区 → 不产生新脏块、不干扰 B2 (§5.4, F-B7)。
 * 发布原子性 (AX5): 条目数据完整填充后才置 valid。
 */
#include "dcache.h"

#include <stdlib.h>
#include <string.h>

#define DC_MAX_FORMATS 8u

struct dc_entry {
    int      valid;
    uint32_t buf_id;
    uint64_t ver;                 /* 全宽 64 位 */
    uint32_t fmt_id;
    uint8_t *data;
    uint64_t size;
    uint64_t lru;
};

struct dcache_ctx {
    btrack_ctx *bt;
    dc_format fmt[DC_MAX_FORMATS];
    uint32_t n_fmt;
    uint32_t max_entries;
    uint64_t max_total_bytes;
    int      replacement;         /* 0=LRU 1=直接映射 */
    struct dc_entry *ent;
    uint64_t bytes_used, lru_tick;
    volatile int lock;
    dc_stats st;
};

static void lk(volatile int *m) { while (__sync_lock_test_and_set(m, 1)) { } }
static void ulk(volatile int *m) { __sync_lock_release(m); }

static uint64_t ver_mask(const btrack_ctx *bt) {
    uint32_t bits = bt_ver_bits(bt);
    return bits >= 64u ? ~0ULL : ((1ULL << bits) - 1u);
}

static const dc_format *find_fmt(dcache_ctx *dc, uint32_t fmt_id) {
    for (uint32_t i = 0; i < dc->n_fmt; i++)
        if (dc->fmt[i].fmt_id == fmt_id) return &dc->fmt[i];
    return NULL;
}

static void drop_entry(dcache_ctx *dc, uint32_t i) {
    struct dc_entry *e = &dc->ent[i];
    if (!e->valid) return;
    dc->bytes_used -= e->size;
    free(e->data);
    memset(e, 0, sizeof *e);
    dc->st.evictions++;
}

/* 尸体 (版本过期) 优先, 否则 LRU; 返回槽号或 max_entries */
static uint32_t pick_victim(dcache_ctx *dc) {
    uint32_t lru_i = dc->max_entries;
    uint64_t best = ~0ULL;
    for (uint32_t i = 0; i < dc->max_entries; i++) {
        struct dc_entry *e = &dc->ent[i];
        if (!e->valid) continue;
        if (e->ver != bt_version(dc->bt, e->buf_id)) return i;  /* 尸体 */
        if (e->lru < best) { best = e->lru; lru_i = i; }
    }
    return lru_i;
}

dcache_ctx *dc_create(btrack_ctx *bt, uint32_t max_entries,
                      uint32_t max_total_bytes, int replacement) {
    if (!bt || !max_entries) return NULL;
    dcache_ctx *dc = calloc(1, sizeof *dc);
    if (!dc) return NULL;
    dc->bt = bt;
    dc->max_entries = max_entries;
    dc->max_total_bytes = max_total_bytes;
    dc->replacement = replacement ? 1 : 0;
    dc->ent = calloc(max_entries, sizeof(struct dc_entry));
    if (!dc->ent) { free(dc); return NULL; }
    return dc;
}

void dc_destroy(dcache_ctx *dc) {
    if (!dc) return;
    for (uint32_t i = 0; i < dc->max_entries; i++)
        if (dc->ent[i].valid) free(dc->ent[i].data);
    free(dc->ent);
    free(dc);
}

int dc_register_format(dcache_ctx *dc, const dc_format *fmt) {
    if (!dc || !fmt || !fmt->convert || !fmt->fmt_id || dc->n_fmt >= DC_MAX_FORMATS)
        return -1;
    if (find_fmt(dc, fmt->fmt_id)) return -1;
    dc->fmt[dc->n_fmt++] = *fmt;
    return 0;
}

int dc_get_or_convert(dcache_ctx *dc, uint32_t buf_id,
                      const uint8_t *src, uint64_t src_size, uint32_t fmt_id,
                      const uint8_t **out, uint64_t *out_size) {
    if (!dc || !src || !src_size || !out) return -1;
    const dc_format *f = find_fmt(dc, fmt_id);
    if (!f) return -1;

    for (int attempt = 0; attempt < 4; attempt++) {
        uint64_t v0 = bt_version(dc->bt, buf_id);
        uint64_t mask = ver_mask(dc->bt);
        int retry = 0;

        lk(&dc->lock);
        for (uint32_t i = 0; i < dc->max_entries; i++) {
            struct dc_entry *e = &dc->ent[i];
            if (!e->valid || e->buf_id != buf_id || e->fmt_id != fmt_id)
                continue;
            if ((e->ver & mask) != (v0 & mask)) continue;
            if (e->ver != v0) {
                dc->st.wrap_misses++;           /* F-B5: 掩码撞车保守未命中 */
                continue;
            }
            if (bt_version(dc->bt, buf_id) != v0) { retry = 1; break; }
            e->lru = ++dc->lru_tick;
            *out = e->data;
            if (out_size) *out_size = e->size;
            dc->st.hits++;
            dc->st.bytes_saved += e->size;
            ulk(&dc->lock);
            return 1;
        }
        ulk(&dc->lock);
        if (retry) { dc->st.retries++; continue; }

        /* ---- miss: 拷贝快照 → 转换 → 复核版本 → 原子发布 ---- */
        uint64_t cap = src_size * (f->dst_cap_per_src_byte
                                   ? f->dst_cap_per_src_byte : 1u);
        uint8_t *snap = malloc(src_size);
        uint8_t *dst = malloc(cap);
        if (!snap || !dst) { free(snap); free(dst); return -1; }
        memcpy(snap, src, src_size);            /* 快照拷贝 (§5.2) */
        uint64_t dsize = 0;
        int crc = f->convert(snap, src_size, dst, cap, &dsize, f->user);
        free(snap);
        if (crc != 0 || dsize > cap) { free(dst); return -1; }

        if (bt_version(dc->bt, buf_id) != v0) { /* AX4: 期间源被写 */
            free(dst);
            dc->st.retries++;
            return -1;                          /* 调用方必须重试 */
        }

        uint32_t slot = (uint32_t)((buf_id * 1315423911u + v0
                                    + (uint64_t)fmt_id * 2654435761u)
                                   % dc->max_entries);
        lk(&dc->lock);
        uint32_t chosen = dc->max_entries;
        for (uint32_t i = 0; i < dc->max_entries; i++)
            if (!dc->ent[i].valid) { chosen = i; break; }
        if (chosen == dc->max_entries)
            for (uint32_t i = 0; i < dc->max_entries; i++) {
                struct dc_entry *e = &dc->ent[i];
                if (e->valid && e->buf_id == buf_id && e->fmt_id == fmt_id
                    && e->ver != v0) {                       /* 同 key 尸体 */
                    drop_entry(dc, i);
                    chosen = i;
                    break;
                }
            }
        while (chosen == dc->max_entries) {
            uint32_t v = dc->replacement == 1 ? slot : pick_victim(dc);
            if (v >= dc->max_entries) break;                 /* 空/无 victim */
            drop_entry(dc, v);
            chosen = v;
            if (dc->replacement == 1) break;
        }
        while (dc->bytes_used + dsize > dc->max_total_bytes) {
            uint32_t v = pick_victim(dc);
            if (v >= dc->max_entries) break;
            drop_entry(dc, v);
        }
        if (dc->bytes_used + dsize > dc->max_total_bytes) {  /* 单条超容量 */
            ulk(&dc->lock);
            free(dst);
            return -1;
        }
        if (chosen == dc->max_entries)
            for (uint32_t i = 0; i < dc->max_entries; i++)
                if (!dc->ent[i].valid) { chosen = i; break; }
        if (chosen == dc->max_entries) {                     /* 竞态中被占满 */
            uint32_t v = dc->replacement == 1 ? slot : pick_victim(dc);
            if (v >= dc->max_entries) { ulk(&dc->lock); free(dst); return -1; }
            drop_entry(dc, v);
            chosen = v;
        }
        struct dc_entry *e = &dc->ent[chosen];
        if (e->valid) drop_entry(dc, chosen);
        e->data = dst;                          /* 完整填充后才 valid (AX5) */
        e->size = dsize;
        e->buf_id = buf_id;
        e->ver = v0;
        e->fmt_id = fmt_id;
        e->lru = ++dc->lru_tick;
        __sync_synchronize();
        e->valid = 1;
        dc->bytes_used += dsize;
        dc->st.misses++;
        *out = e->data;
        if (out_size) *out_size = e->size;
        ulk(&dc->lock);
        return 0;
    }
    return -1;                                  /* 重试耗尽 */
}

int dcache_invalidate(dcache_ctx *dc, uint32_t buf_id) {
    if (!dc) return -1;
    lk(&dc->lock);
    for (uint32_t i = 0; i < dc->max_entries; i++)
        if (dc->ent[i].valid && dc->ent[i].buf_id == buf_id)
            drop_entry(dc, i);
    ulk(&dc->lock);
    return 0;
}

void dc_get_stats(const dcache_ctx *dc, dc_stats *out) {
    if (out && dc) *out = dc->st;
}
