/* host 模拟用 qurt 桩 (仅 fence.c 所需) */
#ifndef HOST_QURT_STUB_H
#define HOST_QURT_STUB_H
#include <stdint.h>
typedef uintptr_t qurt_addr_t;
enum { QURT_MEM_CACHE_FLUSH = 1, QURT_MEM_CACHE_INVALIDATE = 2,
       QURT_MEM_CACHE_FLUSH_INVALIDATE = 3, QURT_MEM_DCACHE = 0 };
static inline int qurt_mem_cache_clean(qurt_addr_t a, unsigned n, unsigned f, unsigned c) {
    (void)a; (void)n; (void)f; (void)c; return 0;
}
#endif
