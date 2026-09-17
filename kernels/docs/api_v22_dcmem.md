# U2 dcmem — VTCM arena / 文件 / DMA 流 / W4 引擎封装 功能手册

声明: [`include/dc_parts.h`](../include/dc_parts.h) · 源: `src/v22/dc_parts.c`
(源模块 dualcore_v81 · 4C 12/12 闭合) · 测试: examples/17, 20, 21, 22

本单元不新写数值 kernel — W4A16 用 t10 已闭合件原样; norm/dot 是确定性整数元素运算。

## API

### VTCM arena (主线程 carve, 指针分发)

```c
void dc_arena_init(struct dc_arena* a, uint8_t* base, uint32_t size);
uint8_t* dc_arena_alloc(struct dc_arena* a, uint32_t bytes, uint32_t align);
```

典型接法 (example 17/22): arena 建在 wtcache pin 区之后:
```c
wtcache_layout(wc, &vb, &vs, &pb, &pc);
uint32_t off = (pc + 2047u) & ~2047u;          /* HMX 面 2KB 对齐起点 */
dc_arena_init(&ar, (uint8_t*)vb + off, vs - off);
```

### 文件读入 (DDR staging)

```c
uint8_t* dc_read_file(const char* path, uint32_t* bytes);
```
128B 对齐堆缓冲。**读回即 CPU 写 → 作 DMA 源前 `dc_clean_ddr` (铁律①)**。

### HVX 负载与确定性整数运算

| 函数 | 说明 |
|------|------|
| `uint32_t dc_hvx_load(uint8_t* scratch4k, uint32_t iters)` | 4 链 HVX intrinsic 负载, 返回折叠校验和。scratch 4KB VTCM, 读初值参与累加 |
| `void dc_norm_i16(const int16_t* x, int16_t* y, uint32_t n)` | `clip((x>>2)+64)` 自动向量化, bit-exact |
| `uint64_t dc_dot_u64(const int16_t* a, const int16_t* b, uint32_t n)` | 确定性整数点积 |

### DMA 流 (P2/C2/C4/M)

```c
int  dc_dma_init(struct dc_dma* d, uint8_t* src, uint8_t* dst, uint32_t bytes,
                 dc_mutex_t* mu);
void dc_dma_destroy(struct dc_dma* d);
int  dc_dma_once(struct dc_dma* d);        /* 单次 DDR→VTCM */
void dc_dma_clean_src(struct dc_dma* d);   /* src 一次性清 cache (每轮清会把 CPU 开销算进带宽) */
void dc_clean_ddr(const void* p, uint32_t bytes);
uint64_t dc_dma_checksum(const struct dc_dma* d);  /* dst 校验和 (cache 安全读) */
void dc_dma_fence(void);                   /* 引擎 IDLE + 清 g_last_desc */
```

约定 (R-D1 对策, 全部内置): submit 恒在 mutex 内且 poll 到 IDLE 才 submit →
`dma_desc_submit` 只走 dmstart 分支, 跨线程 `g_last_desc` 别名不可能; desc 每流持久
持有。**跨线程/多流共用一把 `dc_mutex_t`**。

### W4A16 引擎封装 (P3/C2/C3/C4/M)

```c
int  dc_w4_carve(struct dc_w4* e, struct dc_arena* a, uint32_t m, uint32_t k,
                 uint32_t n, const uint8_t* atbl_ddr, const uint8_t* otbl_ddr);
int  dc_w4_invoke(struct dc_w4* e);        /* 表回填+FLUSH+kernel (t10 原样) */
void dc_w4_read_out(const struct dc_w4* e, void* recv);   /* INVALIDATE+memcpy */
```

- 每 `dc_w4` 独立 VTCM 面 (act/out/wt/bias 全 **2KB 对齐** — HMX mxmem 硬约束)。
- invoke 每次 memcpy 表回填 (**表面被破坏性重写**), act/wt/bias 此刻必须已在面上。
- `dc_w4_read_out` 内置铁律③。
- 详见 [api_v22_w4a16.md](api_v22_w4a16.md) (数值契约/形状约束/perf)。

## 使用范例 (example 17 摘)

```c
uint8_t* wt = dc_read_file(".../packed_weight.raw", &bw);  dc_clean_ddr(wt, bw);
wtcache_open(&wc, 4096);  /* KB */
/* arena 建在 pin 区后 (见上) */
dc_w4_carve(&e, &ar, 256, 256, 256, atbl, otbl);
cpu_to_vtcm(e.wt, wt, bw);       /* memcpy + QURT FLUSH */
cpu_to_vtcm(e.act, act, ba);
dc_w4_invoke(&e);
dc_w4_read_out(&e, out);         /* 铁律③内置 */
wtcache_close(wc);               /* 铁律④ */
```

## 坑 (4C 实测)

- HMX 面对 UserDMA 读回是 **400× 慢** — 输出回 DDR 走 `dc_w4_read_out` 的 memcpy 路径,
  不要给 HMX 面配 DMA 读。
- 引擎单 DMA 引擎 + HMX 全局锁: 双线程双引擎**并发不缩放** (正确性无损, example 22
  有 byte-exact 门); 要吞吐走 U7 双域两进程。
- `m` 必须 256 倍数; 小 M 见 [api_v22_smallm.md](api_v22_smallm.md)。
