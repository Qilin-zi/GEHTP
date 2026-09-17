# HVX Activation (激活函数) API 参考

声明: [`include/hvx_activation.h`](../include/hvx_activation.h)

## HardSwish (MobileNetV3)

```
f(x) = x * clamp(x + 3, 0, 6) / 6
```
Q12 定点实现 (除以 6 用 Q12 乘法, 误差 ≤1 LSB).

### hvhx_hardswish_flat_u16
```c
void hvhx_hardswish_flat_u16(const uint16_t *in, uint16_t *out, uint32_t n);
```
flat 布局 (行主序) u16 in/out (内部 Q12 精度). n 建议 64 倍数.

### hvhx_hardswish_crouton_u16
```c
void hvhx_hardswish_crouton_u16(const uint16_t *in, uint32_t batches, uint16_t *out);
```
crouton 32×32 布局, 跨 `batches` 个 batch.

## PReLU (Parametric ReLU)

```
f(x) = x        if x > 0
     = slope*x  if x <= 0
```

### hvhx_prelu_u8
```c
void hvhx_prelu_u8(const uint8_t *in, uint8_t slope_q7, uint8_t *out, uint32_t n);
```
u8 in, **slope 为 Q7 定点** (0..255, 表示 0..~1). u8 域零点 = 0x80 (128): 正 → 原值+0x80,
负 → `slope*x + 0x80`. n 建议 128 倍数.

## 标量参考
```c
static inline int16_t hvhx_hardswish_scalar(int16_t x);          /* Q12 */
static inline uint8_t hvhx_prelu_scalar_u8(int8_t x, uint8_t slope_q7);
```

## 容差
HardSwish ≤1 LSB (Q12), PReLU exact.

example: [examples/09_activation](../examples/09_activation/main.c)
