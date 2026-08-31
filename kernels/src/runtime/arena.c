/* arena.c — V2.3 U10 双池对齐 arena (boundary-tag, 地址序 first-fit) */
#include <string.h>
#include "arena.h"

/* 区域布局: [tag 16B][对齐垫][data req][pad][ftr 8B], 全区域 16B 倍数 */
struct tag { uint32_t size, free_, req, data_off; };
struct ftr { uint32_t size, free_; };

static uint32_t r16(uint32_t x) { return (x + 15u) & ~15u; }

static struct tag* TAG(uint8_t* rs) { return (struct tag*)rs; }
static struct ftr* FTR(uint8_t* rs) { return (struct ftr*)(rs + TAG(rs)->size - 8u); }
static void mk_ftr(uint8_t* rs) {
    struct ftr* f = FTR(rs);
    f->size = TAG(rs)->size; f->free_ = TAG(rs)->free_;
}

int arena_init(struct arena* a, void* ddr, uint32_t ddr_bytes,
               void* vtcm, uint32_t vtcm_bytes) {
    if (!a) return ARENA_ERR_PARAM;
    memset(a, 0, sizeof(*a));
    if (ddr  && ddr_bytes  >= 64u) { a->base[ARENA_DDR]  = ddr;  a->cap[ARENA_DDR]  = r16(ddr_bytes); }
    if (vtcm && vtcm_bytes >= 64u) { a->base[ARENA_VTCM] = vtcm; a->cap[ARENA_VTCM] = r16(vtcm_bytes); }
    for (int loc = 0; loc < 2; loc++) {
        if (!a->base[loc]) continue;
        struct tag* t = TAG(a->base[loc]);
        t->size = a->cap[loc]; t->free_ = 1; t->req = 0; t->data_off = 0;
        mk_ftr(a->base[loc]);
    }
    return ARENA_OK;
}

void* arena_alloc(struct arena* a, uint32_t bytes, uint32_t align, int loc) {
    if (!a || (loc != ARENA_DDR && loc != ARENA_VTCM) || !a->base[loc]) return NULL;
    if (align == 0) align = ARENA_ALIGN_HVX;
    if (align < 16u || align > 4096u || (align & (align - 1u))) return NULL;
    uint32_t req = r16(bytes ? bytes : 1u);
    uint8_t* base = a->base[loc];
    uint8_t* end = base + a->cap[loc];
    for (uint8_t* rs = base; rs < end; rs += TAG(rs)->size) {
        struct tag* t = TAG(rs);
        if (!t->free_) continue;
        uint32_t data_off = (uint32_t)((((uintptr_t)rs + 16u + align - 1u) & ~(uintptr_t)(align - 1u)) - (uintptr_t)rs);
        uint32_t need = data_off + req + 8u;              /* tag + 垫 + data + ftr */
        if (need > t->size) continue;
        uint32_t new_sz = r16(need);
        if (new_sz > t->size) continue;                   /* 对齐圆整溢出本区域 */
        uint32_t tail = t->size - new_sz;
        if (tail >= 32u) {                                /* 尾部余量切成 free 区域 */
            uint8_t* trs = rs + new_sz;
            struct tag* tt = TAG(trs);
            tt->size = tail; tt->free_ = 1; tt->req = 0; tt->data_off = 0;
            mk_ftr(trs);
        } else {
            new_sz = t->size;                             /* 吸收零头 */
        }
        t->size = new_sz; t->free_ = 0; t->req = req; t->data_off = data_off;
        mk_ftr(rs);
        a->used[loc] += req;
        return rs + data_off;
    }
    return NULL;
}

void arena_free(struct arena* a, void* p) {
    if (!a || !p) return;
    for (int loc = 0; loc < 2; loc++) {
        if (!a->base[loc]) continue;
        uint8_t* base = a->base[loc];
        uint8_t* end = base + a->cap[loc];
        if ((uint8_t*)p < base + 16u || (uint8_t*)p >= end) continue;
        for (uint8_t* rs = base; rs < end; rs += TAG(rs)->size) {
            struct tag* t = TAG(rs);
            if (!t->free_ && rs + t->data_off == (uint8_t*)p) {
                a->used[loc] -= (t->req <= a->used[loc]) ? t->req : a->used[loc];
                t->free_ = 1; t->req = 0;
                /* 前向合并 */
                uint8_t* nxt = rs + t->size;
                if (nxt < end && TAG(nxt)->free_) {
                    t->size += TAG(nxt)->size;
                }
                mk_ftr(rs);
                /* 后向合并 (尾标签) */
                if (rs > base) {
                    struct ftr* pf = (struct ftr*)(rs - 8u);
                    if (pf->free_) {
                        uint8_t* prev = rs - pf->size;
                        TAG(prev)->size += t->size;
                        mk_ftr(prev);
                    }
                }
                return;
            }
        }
    }
}

void arena_reset(struct arena* a, int loc) {
    if (!a || !a->base[loc]) return;
    struct tag* t = TAG(a->base[loc]);
    t->size = a->cap[loc]; t->free_ = 1; t->req = 0; t->data_off = 0;
    mk_ftr(a->base[loc]);
    a->used[loc] = 0;
}

uint32_t arena_used(const struct arena* a, int loc) {
    return a ? a->used[loc] : 0;
}

uint32_t arena_largest_free(const struct arena* a, int loc) {
    if (!a || !a->base[loc]) return 0;
    uint32_t best = 0;
    uint8_t* base = a->base[loc];
    uint8_t* end = base + a->cap[loc];
    for (uint8_t* rs = base; rs < end; rs += TAG(rs)->size)
        if (TAG(rs)->free_ && TAG(rs)->size > best) best = TAG(rs)->size;
    return best;   /* 含 tag/垫的毛尺寸, 归零门用 == cap 判 */
}
