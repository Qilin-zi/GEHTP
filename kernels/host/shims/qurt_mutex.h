/* qurt_mutex.h — host 构建 SDK 桩 (host 单线程, mutex 全空) */
#ifndef HOST_SHIM_QURT_MUTEX_H
#define HOST_SHIM_QURT_MUTEX_H
#include <stdint.h>
typedef struct { uint32_t d[2]; } qurt_mutex_t;
#endif
