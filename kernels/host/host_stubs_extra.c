#include <stdint.h>
#include <stdlib.h>

typedef unsigned long long qurt_addr_t;

void qurt_mem_cache_clean(qurt_addr_t addr, uint32_t size, int op, int type) {
    (void)addr; (void)size; (void)op; (void)type;
}
