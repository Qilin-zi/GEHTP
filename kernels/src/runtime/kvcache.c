/* kvcache.c — U15 KV 槽缓存管理 (见 include/kvcache.h) */
#include "kvcache.h"
#include "fence.h"
#include <string.h>

int kvc_init(struct kvc* c, void* base, uint32_t slot_bytes, uint32_t nslots,
             uint32_t* posmap) {
    if (!c || !base || !posmap) return -1;
    if (slot_bytes < 256 || (slot_bytes & 127u) || ((uintptr_t)base & 127u)) return -1;
    c->base = (uint8_t*)base;
    c->slot_bytes = slot_bytes;
    c->nslots = nslots;
    c->posmap = posmap;
    c->consumer = FC_CPU;
    c->mem = FM_DDR;
    c->n_append = 0; c->n_evict = 0;
    for (uint32_t i = 0; i < nslots; i++) posmap[i] = KVC_POS_NONE;
    return 0;
}

void kvc_set_consumer(struct kvc* c, int consumer, int mem) {
    if (!c) return;
    c->consumer = consumer; c->mem = mem;
}

uint8_t* kvc_k_face(struct kvc* c, uint32_t slot) {
    return (slot < c->nslots) ? c->base + (size_t)slot * c->slot_bytes : 0;
}
uint8_t* kvc_v_face(struct kvc* c, uint32_t slot) {
    return (slot < c->nslots) ? c->base + (size_t)slot * c->slot_bytes + c->slot_bytes / 2 : 0;
}
uint32_t kvc_pos(const struct kvc* c, uint32_t slot) {
    return (slot < c->nslots) ? c->posmap[slot] : KVC_POS_NONE;
}

static void kvc_write(struct kvc* c, uint32_t slot, const void* k, const void* v,
                      uint32_t pos) {
    uint8_t* kf = kvc_k_face(c, slot);
    uint8_t* vf = kvc_v_face(c, slot);
    uint32_t half = c->slot_bytes / 2;
    memcpy(kf, k, half);
    memcpy(vf, v, half);
    fence_handoff(kf, half, FC_CPU, c->consumer, c->mem);
    fence_handoff(vf, half, FC_CPU, c->consumer, c->mem);
    if (c->posmap[slot] != KVC_POS_NONE && c->posmap[slot] != pos) c->n_evict++;
    c->posmap[slot] = pos;
}

int kvc_append(struct kvc* c, const void* k, const void* v, uint32_t pos) {
    if (!c || !k || !v) return -1;
    uint32_t slot = pos % c->nslots;
    kvc_write(c, slot, k, v, pos);
    c->n_append++;
    return (int)slot;
}

int kvc_scatter(struct kvc* c, uint32_t slot, const void* k, const void* v,
                uint32_t pos) {
    if (!c || !k || !v || slot >= c->nslots) return -1;
    kvc_write(c, slot, k, v, pos);
    return 0;
}

int kvc_lookup(const struct kvc* c, uint32_t pos) {
    if (!c) return -1;
    uint32_t slot = pos % c->nslots;
    return (c->posmap[slot] == pos) ? (int)slot : -1;
}

int kvc_read(const struct kvc* c, uint32_t slot, void* k, void* v) {
    if (!c || !k || !v || slot >= c->nslots) return -1;
    uint32_t half = c->slot_bytes / 2;
    memcpy(k, kvc_k_face((struct kvc*)c, slot), half);
    memcpy(v, kvc_v_face((struct kvc*)c, slot), half);
    return 0;
}
