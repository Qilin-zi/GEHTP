/* btrack.c — U21 写跟踪位图库实现 (GENERIC_B §3)
 *
 * 位集单调性定理 (ATOMIC 快照): mark 只做 OR, clear 只作用于已快照位;
 * 两遍读 OR 得到的并集 ⊇ 第一遍开始时刻的脏集 ⇒ 超集性质成立 (§3.3)。
 */
#include "btrack.h"

#include <stdlib.h>
#include <string.h>

struct bt_buf {
    uint64_t base, size;
    uint64_t ver;
    int      valid;
};

struct bt_tok {
    uint64_t addr, size;
    int      state;               /* 0=free 1=open 2=closed */
};

struct btrack_ctx {
    uint64_t mem_bytes;
    uint32_t blk_bytes, n_blocks, n_words;
    uint64_t *bits;
    int      mode;
    volatile int lock;            /* BT_MODE_LOCK */
    volatile int slock[64];       /* BT_MODE_SHARD (按位图字分片) */
    struct bt_buf buf[BT_MAX_BUFFERS];
    uint32_t n_buf;
    struct bt_tok tok[BT_MAX_DMA_TOKENS];
    int      suspect;
    uint32_t ver_bits;            /* 64 = 全宽 (缺省); 8/16 = F-B5 注入 */
};

static void lk(volatile int *m) { while (__sync_lock_test_and_set(m, 1)) { } }
static void ulk(volatile int *m) { __sync_lock_release(m); }

btrack_ctx *bt_create(uint64_t mem_bytes, uint32_t blk_bytes,
                      int concurrency_mode) {
    if (!mem_bytes || !blk_bytes) return NULL;
    btrack_ctx *bt = calloc(1, sizeof *bt);
    if (!bt) return NULL;
    bt->mem_bytes = mem_bytes;
    bt->blk_bytes = blk_bytes;
    bt->n_blocks = (uint32_t)(mem_bytes / blk_bytes);
    bt->n_words = (bt->n_blocks + 63u) / 64u;
    bt->mode = concurrency_mode;
    bt->ver_bits = 64u;
    bt->bits = calloc(bt->n_words ? bt->n_words : 1u, sizeof(uint64_t));
    if (!bt->bits) { free(bt); return NULL; }
    return bt;
}

void bt_destroy(btrack_ctx *bt) {
    if (!bt) return;
    free(bt->bits);
    free(bt);
}

int bt_register_buffer(btrack_ctx *bt, uint64_t base, uint64_t size,
                       uint32_t *buf_id_out) {
    if (buf_id_out) *buf_id_out = 0xffffffffu;      /* 失败防御: 不留垃圾 */
    if (!bt || !size || base + size > bt->mem_bytes) return -1;
    uint32_t id = 0;
    int found = 0;
    for (uint32_t i = 0; i < BT_MAX_BUFFERS; i++) {
        struct bt_buf *b = &bt->buf[i];
        if (!b->valid) { if (!found) { id = i; found = 1; } continue; }
        if (base < b->base + b->size && b->base < base + size) return -1;
    }
    if (!found) return -1;
    if (id >= bt->n_buf) bt->n_buf = id + 1u;
    bt->buf[id].base = base;
    bt->buf[id].size = size;
    bt->buf[id].ver = 0;
    bt->buf[id].valid = 1;
    if (buf_id_out) *buf_id_out = id;
    return 0;
}

int bt_unregister_buffer(btrack_ctx *bt, uint32_t buf_id) {
    if (!bt || buf_id >= BT_MAX_BUFFERS || !bt->buf[buf_id].valid) return -1;
    bt->buf[buf_id].valid = 0;
    return 0;
}

/* ---- 位图置脏: 三条钩子共用 ---- */
static void set_bits_range(btrack_ctx *bt, uint64_t addr, uint64_t size) {
    if (addr >= bt->mem_bytes || size == 0) return;
    uint64_t end = addr + size;
    if (end > bt->mem_bytes) end = bt->mem_bytes;
    uint32_t b_lo = (uint32_t)(addr / bt->blk_bytes);
    uint32_t b_hi = (uint32_t)((end - 1u) / bt->blk_bytes);
    if (b_hi >= bt->n_blocks) b_hi = bt->n_blocks - 1u;

    if (bt->mode == BT_MODE_LOCK) lk(&bt->lock);
    for (uint32_t b = b_lo; b <= b_hi; ) {
        uint32_t w = b >> 6;
        uint32_t i0 = b & 63u;
        uint32_t i1 = (b_hi - (w << 6) < 63u) ? (b_hi - (w << 6)) : 63u;
        uint64_t mask = (i1 >= 63u) ? (~0ULL << i0)
                                    : (((1ULL << (i1 - i0 + 1u)) - 1u) << i0);
        if (bt->mode == BT_MODE_ATOMIC)
            __sync_fetch_and_or(&bt->bits[w], mask);
        else if (bt->mode == BT_MODE_SHARD) {
            volatile int *sl = &bt->slock[w & 63u];
            lk(sl); bt->bits[w] |= mask; ulk(sl);
        } else
            bt->bits[w] |= mask;
        b = ((w + 1u) << 6);
    }
    if (bt->mode == BT_MODE_LOCK) ulk(&bt->lock);
}

/* 区间重叠的每个缓冲版本 +1 (批量接口: 一次区间一版本, §3.5) */
static void bump_versions(btrack_ctx *bt, uint64_t addr, uint64_t size) {
    for (uint32_t i = 0; i < bt->n_buf; i++) {
        struct bt_buf *b = &bt->buf[i];
        if (!b->valid) continue;
        if (addr < b->base + b->size && b->base < addr + size)
            __sync_add_and_fetch(&b->ver, 1u);   /* 版本恒原子 (各模式通用) */
    }
}

static void mark_write(btrack_ctx *bt, uint64_t addr, uint64_t size) {
    set_bits_range(bt, addr, size);
    bump_versions(bt, addr, size);
}

void bt_mark_cpu_write(btrack_ctx *bt, uint64_t addr, uint64_t size) {
    if (!bt) return;
    mark_write(bt, addr, size);
}

void bt_mark_peer_write(btrack_ctx *bt, uint64_t addr, uint64_t size) {
    if (!bt) return;
    mark_write(bt, addr, size);
}

int bt_mark_dma_write(btrack_ctx *bt, uint64_t addr, uint64_t size,
                      uint64_t *token_out) {
    if (!bt || !size || addr + size > bt->mem_bytes) return -1;
    /* 铁律 §3.4: 先标记后提交 —— 提交前保守置脏 */
    mark_write(bt, addr, size);
    if (bt->mode == BT_MODE_LOCK) lk(&bt->lock);
    uint64_t tok = 0;
    for (uint32_t i = 0; i < BT_MAX_DMA_TOKENS; i++)
        if (bt->tok[i].state == 0) {
            bt->tok[i] = (struct bt_tok){ addr, size, 1 };
            tok = (uint64_t)i + 1u;
            break;
        }
    if (bt->mode == BT_MODE_LOCK) ulk(&bt->lock);
    if (!tok) return -1;               /* 无空闲 token; 位已置 (保守) */
    if (token_out) *token_out = tok;
    return 0;
}

int bt_dma_complete(btrack_ctx *bt, uint64_t token) {
    if (!bt || token == 0 || token > BT_MAX_DMA_TOKENS) return -1;
    struct bt_tok *t = &bt->tok[token - 1u];
    if (t->state != 1) return -1;
    t->state = 2;
    return 0;
}

uint64_t bt_version(const btrack_ctx *bt, uint32_t buf_id) {
    if (!bt || buf_id >= BT_MAX_BUFFERS || !bt->buf[buf_id].valid) return 0;
    return bt->buf[buf_id].ver;
}

int bt_snapshot_dirty(btrack_ctx *bt, bt_snapshot *snap_out) {
    if (!bt || !snap_out) return -1;
    uint64_t *w = calloc(bt->n_words ? bt->n_words : 1u, sizeof(uint64_t));
    if (!w) return -1;
    switch (bt->mode) {
    case BT_MODE_ATOMIC:
        /* 两遍读 OR: 并集 ⊇ 第一遍开始时刻脏集 (超集性质) */
        for (uint32_t i = 0; i < bt->n_words; i++) w[i] = bt->bits[i];
        for (uint32_t i = 0; i < bt->n_words; i++)
            w[i] |= __atomic_load_n(&bt->bits[i], __ATOMIC_ACQUIRE);
        break;
    case BT_MODE_SHARD:                       /* stop-the-world */
        for (int i = 0; i < 64; i++) lk(&bt->slock[i]);
        memcpy(w, bt->bits, (size_t)bt->n_words * sizeof(uint64_t));
        for (int i = 0; i < 64; i++) ulk(&bt->slock[i]);
        break;
    default:
        lk(&bt->lock);
        memcpy(w, bt->bits, (size_t)bt->n_words * sizeof(uint64_t));
        ulk(&bt->lock);
        break;
    }
    snap_out->words = w;
    snap_out->n_words = bt->n_words;
    return 0;
}

void bt_snapshot_free(bt_snapshot *snap) {
    if (!snap) return;
    free(snap->words);
    snap->words = NULL;
    snap->n_words = 0;
}

static void clear_masked(btrack_ctx *bt, const uint64_t *clr, uint64_t addr,
                         uint64_t size) {
    if (addr >= bt->mem_bytes || size == 0) return;
    uint64_t end = addr + size;
    if (end > bt->mem_bytes) end = bt->mem_bytes;
    uint32_t b_lo = (uint32_t)(addr / bt->blk_bytes);
    uint32_t b_hi = (uint32_t)((end - 1u) / bt->blk_bytes);
    if (b_hi >= bt->n_blocks) b_hi = bt->n_blocks - 1u;
    if (bt->mode == BT_MODE_LOCK) lk(&bt->lock);
    for (uint32_t b = b_lo; b <= b_hi; ) {
        uint32_t w = b >> 6;
        uint32_t i0 = b & 63u;
        uint32_t i1 = (b_hi - (w << 6) < 63u) ? (b_hi - (w << 6)) : 63u;
        uint64_t mask = (i1 >= 63u) ? (~0ULL << i0)
                                    : (((1ULL << (i1 - i0 + 1u)) - 1u) << i0);
        uint64_t keep = ~(clr[w] & mask);   /* 只清已快照过的位 */
        if (bt->mode == BT_MODE_ATOMIC)
            __sync_and_and_fetch(&bt->bits[w], keep);
        else
            bt->bits[w] &= keep;
        b = ((w + 1u) << 6);
    }
    if (bt->mode == BT_MODE_LOCK) ulk(&bt->lock);
}

int bt_clear(btrack_ctx *bt, const bt_snapshot *snap, uint64_t addr,
             uint64_t size) {
    if (!bt || !snap || snap->n_words != bt->n_words) return -1;
    clear_masked(bt, snap->words, addr, size);
    return 0;
}

int bt_clear_all_flushed(btrack_ctx *bt, const bt_snapshot *snap) {
    if (!bt || !snap || snap->n_words != bt->n_words) return -1;
    if (bt->mode == BT_MODE_LOCK) lk(&bt->lock);
    for (uint32_t i = 0; i < bt->n_words; i++) {
        uint64_t keep = ~snap->words[i];
        if (bt->mode == BT_MODE_ATOMIC)
            __sync_and_and_fetch(&bt->bits[i], keep);
        else
            bt->bits[i] &= keep;
    }
    if (bt->mode == BT_MODE_LOCK) ulk(&bt->lock);
    return 0;
}

int bt_merge(btrack_ctx *bt, const bt_snapshot *external) {
    if (!bt || !external) return -1;
    uint32_t n = external->n_words < bt->n_words ? external->n_words
                                                 : bt->n_words;
    if (bt->mode == BT_MODE_LOCK) lk(&bt->lock);
    for (uint32_t i = 0; i < n; i++) {
        if (bt->mode == BT_MODE_ATOMIC)
            __sync_fetch_and_or(&bt->bits[i], external->words[i]);
        else
            bt->bits[i] |= external->words[i];
    }
    if (bt->mode == BT_MODE_LOCK) ulk(&bt->lock);
    return 0;
}

void bt_flag_suspect(btrack_ctx *bt) {
    if (!bt) return;
    __sync_synchronize();
    bt->suspect = 1;
}

int bt_is_suspect(const btrack_ctx *bt) { return bt ? bt->suspect : 1; }

int bt_dma_pending(const btrack_ctx *bt) {
    if (!bt) return 0;
    int n = 0;
    for (uint32_t i = 0; i < BT_MAX_DMA_TOKENS; i++)
        if (bt->tok[i].state == 1) n++;
    return n;
}

uint64_t bt_dirty_bytes(const btrack_ctx *bt) {
    if (!bt) return 0;
    uint64_t bits = 0;
    for (uint32_t i = 0; i < bt->n_words; i++)
        bits += (uint64_t)__builtin_popcountll(bt->bits[i]);
    return bits * bt->blk_bytes;
}

uint32_t bt_ver_bits(const btrack_ctx *bt) {
    return bt ? bt->ver_bits : 64u;
}

uint32_t bt_n_blocks(const btrack_ctx *bt) {
    return bt ? bt->n_blocks : 0u;
}

uint32_t bt_blk_bytes(const btrack_ctx *bt) {
    return bt ? bt->blk_bytes : 0u;
}

void bt_set_ver_bits(btrack_ctx *bt, uint32_t bits) {
    if (!bt) return;
    if (bits < 8u) bits = 8u;
    if (bits > 64u) bits = 64u;
    bt->ver_bits = bits;
}

int bt_debug_bump_version(btrack_ctx *bt, uint32_t buf_id, uint32_t n) {
    if (!bt || buf_id >= BT_MAX_BUFFERS || !bt->buf[buf_id].valid) return -1;
    __sync_add_and_fetch(&bt->buf[buf_id].ver, n);
    return 0;
}
