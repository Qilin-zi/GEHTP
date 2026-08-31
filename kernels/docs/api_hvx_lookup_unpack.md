# HVX Lookup / Unpack API 参考

声明: [`include/hvx_lookup_unpack.h`](../include/hvx_lookup_unpack.h)

查表 (table lookup) 与权重解包 (unpack).

## Table Lookup

### hvhx_table_lookup_flat_u8
```c
void hvhx_table_lookup_flat_u8(const uint8_t *idx, const uint8_t *table,
                               uint8_t *out, uint32_t n);
```
`out[i] = table[idx[i]]`. flat 布局. `table` 长度 = 256 (完整 u8 索引空间).
n 建议 128 倍数.

### hvhx_table_lookup_crouton_u8
```c
void hvhx_table_lookup_crouton_u8(const uint8_t *idx, const uint8_t *table,
                                  uint32_t batches, uint8_t *out);
```
crouton 32×32 布局, 跨 `batches` 个 batch.

## Unpack (权重解包)

### hvhx_unpack_weights
```c
void hvhx_unpack_weights(const uint8_t *packed, uint8_t *out, uint32_t n);
```
标准权重解包 (4-bit → 8-bit 等, 按库内部约定格式). n 建议 128 倍数.

### hvhx_unpack_custom_weights
```c
void hvhx_unpack_custom_weights(const uint8_t *packed, const uint8_t *table,
                                uint8_t *out, uint32_t n);
```
带自定义 table 的解包 (table 可为 NULL 用默认).

## 容差
全部 exact (err=0).

example: [examples/11_lookup_unpack](../examples/11_lookup_unpack/main.c)
