/*
 * hvx_divide.h — V81 HVX element-wise division (intrinsic form)
 * =====================================================================
 * 物理意义:  out[i] = a[i] / b[i]
 *  - u8 / u8:  u8 结果, 除零饱和到 0xFF
 *  - u16 / u16: u16 结果 (Q16 表示), 除零饱和到 0xFFFF
 *  - i32 / i32: i32 结果, 除零饱和到 ±INT32_MAX
 *  - floor variant: 朝 -∞ 方向取整 (用于 floor_div)
 *
 * 实现策略:
 *  - u8/u16: 倒数近似 (Newton-Raphson 1 次迭代) + vmpy 还原
 *  - i32: long-division 4-cycle loop (vector shift-subtract)
 *  - 所有路径 explicit 标量版本用于 ground-truth 校验
 *
 * 注意: 这里 HVX 版本与 libQnnHtpV81 反汇编 1:1 复算,
 * 但用 intrinsic 写出, 可读性、可校验性、可溢出保护优先.
 */
#ifndef HVXHMX_HVX_DIVIDE_H
#define HVXHMX_HVX_DIVIDE_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
 *  标量参考实现 (用于精度验证)
 * ---------------------------------------------------------------- */
static inline uint8_t hvhx_div_u8_scalar(uint8_t a, uint8_t b) {
    return HVHX_GUARD_DIV_U8(a, b);
}

static inline uint16_t hvhx_div_u16_scalar(uint16_t a, uint16_t b) {
    return HVHX_GUARD_DIV_U16(a, b);
}

static inline int32_t hvhx_div_i32_scalar(int32_t a, int32_t b) {
    return HVHX_GUARD_DIV_I32(a, b);
}

/* floor_div: 朝 -∞ 取整; 公式: a/b 同号 → a/b; 否则 → (a-b+1)/b */
static inline uint8_t hvhx_floor_div_u8_scalar(uint8_t a, uint8_t b) {
    if (b == 0) return UINT8_MAX;
    return (uint8_t) (a / b);
}

static inline uint16_t hvhx_floor_div_u16_scalar(uint16_t a, uint16_t b) {
    if (b == 0) return UINT16_MAX;
    return (uint16_t) (a / b);
}

/* ----------------------------------------------------------------
 *  HVX u8 整除 — 128 元素/向量
 *   输入: a, b 各 N 字节 (N 是 128 倍数)
 *   输出: out[i] = a[i] / b[i], b=0 → 0xFF
 *
 *  数值路径 (来自 disasm):
 *   1. vabsdiff(a.ub, b.ub)  → max-min 形态
 *   2. normamt + vasl       → 归一化到 [0x4000, 0x7fff]
 *   3. 倒数 LUT 1 次迭代     → reciprocal
 *   4. vmpy reciprocal*被除数 → 结果
 * ---------------------------------------------------------------- */
void hvhx_divide_u8(const uint8_t *a, const uint8_t *b,
                    uint8_t *out, uint32_t n);

/* u16 整除 — 64 元素/向量, Q16 表示 (左移 16) */
void hvhx_divide_u16(const uint16_t *a, const uint16_t *b,
                     uint16_t *out, uint32_t n);

/* i32 整除 — 32 元素/向量, 4-cycle shift-subtract */
void hvhx_divide_flat_i32(const int32_t *a, const int32_t *b,
                          int32_t *out, uint32_t n);

/* floor 变体 */
void hvhx_floor_divide_u8(const uint8_t *a, const uint8_t *b,
                          uint8_t *out, uint32_t n);

void hvhx_floor_divide_u16(const uint16_t *a, const uint16_t *b,
                           uint16_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_HVX_DIVIDE_H */
