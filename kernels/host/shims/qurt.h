/* qurt.h — host 构建 SDK 桩 (仅声明; 定义在 host_stubs.c)
 * 设备侧本头来自 Hexagon SDK; host 侧只缺 fence.c 用的两个符号。 */
#ifndef HOST_SHIM_QURT_H
#define HOST_SHIM_QURT_H
#include <stdint.h>
typedef uintptr_t qurt_addr_t;
enum { QURT_MEM_CACHE_FLUSH = 1, QURT_MEM_CACHE_INVALIDATE = 2,
       QURT_MEM_CACHE_FLUSH_INVALIDATE = 3, QURT_MEM_DCACHE = 0 };
void qurt_mem_cache_clean(uint64_t addr, uint32_t size, int op, int type);
#endif
