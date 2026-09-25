/* qurt_barrier.h — host 构建 SDK 桩 (dc_threads.h 仅需 typedef) */
#ifndef HOST_SHIM_QURT_BARRIER_H
#define HOST_SHIM_QURT_BARRIER_H
#include <stdint.h>
typedef struct { uint32_t d[2]; } qurt_barrier_t;
#endif
