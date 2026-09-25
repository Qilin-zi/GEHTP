/* stdlib_compat.h — host 构建补丁 (force-include):
 * macOS 无 memalign (Hexagon libc 有); 定义在 host_shims.c。 */
#ifndef HOST_SHIM_STDLIB_COMPAT_H
#define HOST_SHIM_STDLIB_COMPAT_H
#include <stdlib.h>
void* memalign(size_t align, size_t size);
#endif
