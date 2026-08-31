/*
 * hmx_add.c — 元素级 fp16 加法 + bias + ReLU (残差连接)
 * Module: hmx-elementwise
 * Math:   out[i] = max(0, a[i] + b[i] + bias[i % N])
 * Note:   标量实现 (本设备 V81 FUSA CDSP 的 fp16 HVX vadd_hf 路径不稳,
 *         详见 docs/data_layout.md "fp16 HVX 约束" 节).
 */
#include "hmx_kernels.h"
#include "hmx_common.h"

void hmx_add(const __fp16 * __restrict__ a,
             const __fp16 * __restrict__ b,
             const __fp16 * __restrict__ bias,
             __fp16       * __restrict__ out,
             uint32_t M, uint32_t N)
{
    const uint32_t total = M * N;
    for (uint32_t i = 0; i < total; ++i) {
        float s = (float) a[i] + (float) b[i] + (float) bias[i % N];
        if (s < 0.0f) s = 0.0f;
        out[i] = (__fp16) s;
    }
}
