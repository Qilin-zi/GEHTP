/* stdlib_compat.h — host 构建补丁 (force-include):
 * macOS 无 memalign (Hexagon libc 有); 定义在 host_shims.c。
 * x86 gcc 无 __fp16 (arm64 clang / hexagon 原生有): host 侧声明仅为
 * 字节缓冲, 存储等价 typedef 即可 (头文件声明不进算术)。 */
#ifndef HOST_SHIM_STDLIB_COMPAT_H
#define HOST_SHIM_STDLIB_COMPAT_H
#include <stdlib.h>
void* memalign(size_t align, size_t size);
#if !defined(__hexagon__) && !defined(__aarch64__) && !defined(__arm__)
typedef unsigned short __fp16;
#endif
#endif
