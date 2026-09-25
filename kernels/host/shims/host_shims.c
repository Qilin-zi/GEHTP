/* host_shims.c — host 构建缺的函数补丁 (host_stubs.c 未覆盖的 dc_parts 面)
 * host 无 DSP cache/DMA 语义: 全部空操作或直拷。 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>  /* posix_memalign */

/* 铁律①在 host 无 cache, 空操作 (设备侧 dc_parts.c 真实现) */
void dc_clean_ddr(const void* p, uint32_t bytes) { (void)p; (void)bytes; }

/* memalign: macOS 缺失 (设备侧 Hexagon libc 原生) */
void* memalign(size_t align, size_t size) {
    void* p = NULL;
    if (align == 0) align = 1;
    return posix_memalign(&p, align, size) ? NULL : p;
}
