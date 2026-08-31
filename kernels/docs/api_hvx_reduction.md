# HVX Reduction (归约) API 参考

声明: [`include/hvx_reduction.h`](../include/hvx_reduction.h)

沿 depth 维 (通道) 做归约: argmin/argmax, find_max, top1, reducesum.

## 结果类型

```c
typedef struct {
    uint16_t min_val, max_val;       /* crouton u16 形式 */
    uint32_t min_idx, max_idx;       /* depth 索引 */
} hvhx_argminmax_t;

typedef struct {
    uint16_t val;
    uint32_t idx;
} hvhx_top1_t;
```

## Argmin/Argmax (5 个)

沿 depth 维找每行 (每空间位置) 的 min/max 值与索引.

### hvhx_argminmax_depth_crouton_b
```c
void hvhx_argminmax_depth_crouton_b(const uint8_t *in, uint32_t hw, uint32_t d,
                                    hvhx_argminmax_t *out);
```
u8 crouton 输入, `hw` 个空间位置, depth `d`. out 长度 = hw.

### hvhx_argminmax_depth_crouton_h
```c
void hvhx_argminmax_depth_crouton_h(const uint16_t *in, uint32_t hw, uint32_t d,
                                    hvhx_argminmax_t *out);
```
u16 crouton 版本.

### hvhx_argminmax_depth_dLE32_crouton_b
```c
void hvhx_argminmax_depth_dLE32_crouton_b(const uint8_t *in, uint32_t hw, uint32_t d,
                                          hvhx_argminmax_t *out);
```
depth ≤ 32 的优化版 (单 tile).

### hvhx_argminmax_depth_flat_h
```c
void hvhx_argminmax_depth_flat_h(const uint16_t *in, uint32_t hw, uint32_t d,
                                 hvhx_argminmax_t *out);
```
flat (行主序) u16 输入.

### hvhx_argminmax_depth_short_b
```c
void hvhx_argminmax_depth_short_b(const uint8_t *in, uint32_t hw, uint32_t d,
                                  hvhx_argminmax_t *out);
```
短 depth 的 u8 版.

## Find Max (2 个)

### hvhx_find_max_and_index_in_depth_b / _h
```c
void hvhx_find_max_and_index_in_depth_b(const uint8_t  *in, uint32_t hw, uint32_t d,
                                        uint32_t *sumbuf, uint8_t *out);
void hvhx_find_max_and_index_in_depth_h(const uint16_t *in, uint32_t hw, uint32_t d,
                                        uint32_t *sumbuf, uint16_t *out);
```
找每行 depth 维最大值及索引. `sumbuf` 是 scratch (长度 ≥ hw, 128B 对齐).

## Top-1

### hvhx_top1_qu8_dLE32_cr2flt
```c
void hvhx_top1_qu8_dLE32_cr2flt(const uint8_t *in, uint32_t hw, uint32_t d,
                                hvhx_top1_t *out);
```
全局 top-1 (整个 [hw][d] 里最大的一个值+索引). depth ≤ 32, crouton→flat.

## ReduceSum

### hvhx_reducesum_depth_u8
```c
void hvhx_reducesum_depth_u8(const uint8_t *in, uint32_t hw, uint32_t d,
                             uint32_t *sumbuf);
```
沿 depth 求和 (每行一个和), 写到 sumbuf (长度 hw).

### reduce_sum_u8_case_{1,2,4,6,8} / reduce_sum_u16_case_{1,2,4,6,8}
```c
void hvhx_reduce_sum_u8_case_4 (const uint8_t  *in, uint32_t hw, uint8_t  *out);
void hvhx_reduce_sum_u16_case_4(const uint16_t *in, uint32_t hw, uint16_t *out);
```
按 depth 分组求和, case N = 每 N 个通道一组求和. case ∈ {1,2,4,6,8}.

## 容差
全部 exact (err=0).

example: [examples/10_reduction](../examples/10_reduction/main.c)
