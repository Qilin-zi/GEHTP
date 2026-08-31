# api_v23_fence — U9 方向对偶 cache fence (决策表)

源: `src/runtime/fence.c` · 头: `include/fence.h` · 例: `23_fence` (7 门 PASS)

## 为什么存在

V2.2 四铁律散落在 wtcache/dc_*/4C 各处的裸 `qurt_mem_cache_clean` 调用里, 每处都要
重新推理 "谁写谁读哪块内存该 FLUSH 还是 INVALIDATE"。U9 把这张决策表收敛成纯函数,
一处定义, 全库复用 (kvcache / wpool / gemm_dispatch 已接)。

## API

```c
enum { FC_CPU, FC_HVX, FC_DMA, FC_HMX };      /* 写者/读者 */
enum { FM_DDR, FM_VTCM };                     /* 内存域 */
enum { FO_NONE=0, FO_FLUSH, FO_INVALIDATE, FO_FLUSH_INVALIDATE, FO_INVALID };

int fence_op_for(int writer, int reader, int mem);   /* 纯函数, host 可对拍 */
int fence_handoff(void* ptr, uint32_t bytes,
                  int writer, int reader, int mem);  /* 查表 + 真调用 */
```

## 决策表 (语义)

| writer | reader | DDR | VTCM |
|--------|--------|-----|------|
| CPU    | DMA    | **FLUSH_INVALIDATE** (铁律①) | FLUSH |
| CPU    | HMX    | *(HMX 不访 DDR)* → 拒绝 | FLUSH |
| CPU    | HVX/CPU| NONE (同为 dcache 代理) | NONE |
| DMA    | CPU/HVX| INVALIDATE | INVALIDATE |
| DMA    | HMX    | *(HMX 不访 DDR)* → 拒绝 | INVALIDATE |
| DMA    | DMA    | FLUSH | FLUSH |
| HMX    | CPU/HVX| — | INVALIDATE |
| HMX    | DMA    | — | FLUSH |
| HVX    | *      | 同 CPU 行 (HVX store 走 dcache, 与 CPU 同代理) | 同 |

- HMX 只访 VTCM (mxmem 物理限制, DDR 直访 fault) → 任何含 HMX 的 DDR 组合返回
  `FO_INVALID` = 非法, `fence_handoff` 返回 `FENCE_ERR_COMBO` 不调用。
- 铁律① "CPU 写 DDR → DMA bypass 读" 是 FLUSH_INVALIDATE: 写者 FLUSH 落内存,
  读者侧 INVALIDATE 防陈旧行。

## 设备门 (23_fence, 2026-08-16)

- `fence_decision_table_32`: 4×4×2 全组合与表一致 (host 对拍)
- `fence_cpu_dma_ddr_200iters`: 铁律① 200 轮真 DMA bypass 读回逐字节
- `fence_cpu_hmx_vtcm_20iters` / `fence_hmx_input_sensitivity`
- `fence_cpu_hvx_vtcm_visible` / `fence_cpu_dma_vtcm_moveback_100iters`

## 纪律

- fence 只解决**可见性**, 不解决**互斥**; 并发结构用 wpool/dc_threads。
- 长度非 0 才调; `bytes==0` 直接 OK (与 cache line 无关的短路径)。
