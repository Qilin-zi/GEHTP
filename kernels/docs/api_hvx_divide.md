# HVX Divide (整除) API 参考

声明: [`include/hvx_divide.h`](../include/hvx_divide.h)

逐元素整除 `out[i] = a[i] / b[i]`, 除零饱和. HVX intrinsic 实现 (Newton-Raphson 倒数 +
shift-subtract).

## 函数

### hvhx_divide_u8
```c
void hvhx_divide_u8(const uint8_t *a, const uint8_t *b, uint8_t *out, uint32_t n);
```
u8 / u8 → u8. `b[i]=0 → out[i]=0xFF`. n 建议 128 倍数.

数值路径: `vabsdiff` → `normamt+vasl` 归一化 → 倒数 LUT 1 次迭代 → `vmpy` 还原.

### hvhx_divide_u16
```c
void hvhx_divide_u16(const uint16_t *a, const uint16_t *b, uint16_t *out, uint32_t n);
```
u16 / u16 → u16 (Q16 表示, 左移 16). `b=0 → 0xFFFF`. n 建议 64 倍数.

### hvhx_divide_flat_i32
```c
void hvhx_divide_flat_i32(const int32_t *a, const int32_t *b, int32_t *out, uint32_t n);
```
i32 / i32 → i32, **round-to-nearest** (四舍五入, 不是截断!). `b=0 → ±INT32_MAX` (按符号).
4-cycle shift-subtract. n 建议 32 倍数.

> ⚠️ 原版 (libQnnHtpV81 反汇编) 用 qf32 浮点倒数 = round-to-nearest. 不要改成截断,
> 否则与生产 .so 结果不符. 标量 golden 也用 round-to-nearest (`hvhx_div_i32_scalar`).

### hvhx_floor_divide_u8 / hvhx_floor_divide_u16
```c
void hvhx_floor_divide_u8 (const uint8_t  *a, const uint8_t  *b, uint8_t  *out, uint32_t n);
void hvhx_floor_divide_u16(const uint16_t *a, const uint16_t *b, uint16_t *out, uint32_t n);
```
朝 -∞ 取整的 floor 变体 (无符号 = 普通截断). `b=0 → MAX`.

## 标量参考 (验证用)

头文件内联提供:
```c
static inline uint8_t  hvhx_div_u8_scalar (uint8_t a, uint8_t b);   /* HVHX_GUARD_DIV_U8 */
static inline uint16_t hvhx_div_u16_scalar(uint16_t a, uint16_t b);
static inline int32_t  hvhx_div_i32_scalar(int32_t a, int32_t b);   /* round-to-nearest */
static inline uint8_t  hvhx_floor_div_u8_scalar (uint8_t a, uint8_t b);
static inline uint16_t hvhx_floor_div_u16_scalar(uint16_t a, uint16_t b);
```

## 容差
全部 exact (err=0) — 与标量 golden 逐元素对比.

example: [examples/08_divide](../examples/08_divide/main.c)
