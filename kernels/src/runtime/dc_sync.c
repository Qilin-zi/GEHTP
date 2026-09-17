/* dc_sync.c — 同步原语: qurt_barrier / qurt_mutex / VTCM 自旋旗标
 *
 * 旗标契约 (P4 实测修订): 该芯片 VTCM 的 CPU 读写跨线程天然一致
 * (P4 四象限 {FLUSH}×{INVALIDATE} 全 100/100 通过) — 旗标用纯 volatile
 * 读写即可; set/wait 的 FLUSH/INVALIDATE 是纯开销 (C4 实测拖慢 ~10x)。
 * DMA/HMX 写的面仍需按各自契约处理 (见 dc_parts.c)。
 */
#include "dc_threads.h"

#include <qurt.h>

void dc_barrier_init(dc_barrier_t* b, int n) {
    qurt_barrier_init(b, (unsigned int)n);
}
void dc_barrier_wait(dc_barrier_t* b) {
    qurt_barrier_wait(b);
}
void dc_mutex_init(dc_mutex_t* m) {
    qurt_pimutex_init(m);
}
void dc_mutex_lock(dc_mutex_t* m) {
    qurt_pimutex_lock(m);
}
void dc_mutex_unlock(dc_mutex_t* m) {
    qurt_pimutex_unlock(m);
}

void dc_flush(void* p, uint32_t bytes) {
    qurt_mem_cache_clean((qurt_addr_t)p, bytes, QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}
void dc_invalidate(void* p, uint32_t bytes) {
    qurt_mem_cache_clean((qurt_addr_t)p, bytes, QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

void dc_flag_set(volatile uint32_t* f, uint32_t v) {
    *f = v;
    dc_flush((void*)f, 4);
}

uint32_t dc_flag_wait(volatile uint32_t* f, uint32_t v) {
    uint32_t cur;
    for (;;) {
        dc_invalidate((void*)f, 4);
        cur = *f;
        if (cur == v) return cur;
    }
}
/* 流水线用: 等待 >= v。消费者可领先任意步, 相等等待会永久挂死 (C4 实测)。 */
uint32_t dc_flag_wait_ge(volatile uint32_t* f, uint32_t v) {
    uint32_t cur;
    for (;;) {
        dc_invalidate((void*)f, 4);
        cur = *f;
        if (cur >= v) return cur;
    }
}
