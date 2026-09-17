/*
 * hvx_activation.h — V81 HVX 激活函数 (intrinsic form)
 * =====================================================================
 * 物理意义:
 *  - HardSwish (MobileNetV3):
 *      f(x) = x * clamp(x + 3, 0, 6) / 6
 *    在 x ∈ [-3, 3] 是 x²/6 + x/2 的分段线性近似.
 *  - PReLU (Parametric ReLU):
 *      f(x) = x        if x > 0
 *           = slope*x  if x <= 0
 *    slope ∈ (0, 1).
 *
 * 数值:
 *  - HardSwish 用 Q12 定点 (除以 6 用 Q12 乘法, 误差 ≤ 1 LSB)
 *  - PReLU 滑点为 Q15 形式 (-32768..32767)
 */
#ifndef HVXHMX_HVX_ACTIVATION_H
#define HVXHMX_HVX_ACTIVATION_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 标量参考 */
static inline int16_t hvhx_hardswish_scalar(int16_t x) {
    /* Q12: x + 3 偏移 12288, clamp [0, 24576] (=6<<12) */
    const int32_t offs = (int32_t) x + 12288;
    const int32_t cl   = offs < 0 ? 0 : (offs > 24576 ? 24576 : offs);
    /* f_q12 = x_q12 * cl_q12 / 24576 (= 6*4096), 四舍五入 */
    int64_t prod = (int64_t) x * (int64_t) cl;
    int32_t r = (int32_t) ((prod + 12288) / 24576);
    if (r > INT16_MAX) r = INT16_MAX;
    if (r < INT16_MIN) r = INT16_MIN;
    return (int16_t) r;
}

static inline uint8_t hvhx_prelu_scalar_u8(int8_t x, uint8_t slope_q7) {
    /* u8 域零点 = 0x80: 正 → 原值+0x80, 负 → slope*x+0x80 */
    if (x >= 0) return (uint8_t)((int)x + 0x80);
    int16_t y = ((int16_t) x * (int16_t) slope_q7) >> 7;
    int16_t r = y + 0x80;
    return (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

/* ----------------------------------------------------------------
 *  HVX 路径
 * ---------------------------------------------------------------- */

/* HardSwish: flat u16 → u16 (Q12 内部精度, 输出 Q12) */
void hvhx_hardswish_flat_u16(const uint16_t *in, uint16_t *out, uint32_t n);

/* HardSwish: crouton 32x32 → 32x32 (u16), 跨多 batch 布局 */
void hvhx_hardswish_crouton_u16(const uint16_t *in, uint32_t batches,
                                uint16_t *out);

/* PReLU: u8 in, slope Q7 (0..255) → u8 out */
void hvhx_prelu_u8(const uint8_t *in, uint8_t slope_q7,
                   uint8_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_HVX_ACTIVATION_H */
