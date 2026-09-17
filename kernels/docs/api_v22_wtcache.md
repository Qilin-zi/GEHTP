# U1 wtcache — VTCM 权重缓存 (pin / ring) 功能手册

声明: [`include/wtcache.h`](../include/wtcache.h) · 源: `src/v22/wtcache_impl.c`
(源模块 wtcache_pin_v81, T1-T9 设备 9/9 闭合) · 测试: [examples/16_wtcache_pin](../examples/16_wtcache_pin/main.c)

## 功能

把 16MB VTCM 管成三段, 服务 LLM decode 的权重供给:

```
[pin_pool 12MB]  热权重一次 DMA 进 VTCM, 整会话稳定指针 (tcm_migration 等价)
[ring_in/out]    prefetch 4 + move_back 4 槽 (svf0_dma_cfg 等价), DMA×compute 重叠 0.968
[temp]           剩余自由区 (供 U2 arena 使用)
```

物理实测: UserDMA DDR→VTCM 65.5 GB/s; VTCM→DDR 69.9 GB/s; DMA 与 HMX 独立硬件单元。

## API

### 会话

| 函数 | 说明 |
|------|------|
| `int wtcache_open(struct wtcache_ctx** out, uint32_t pin_cap_bytes)` | acquire VTCM + HVX/HMX power on + hmx_lock。`pin_cap_bytes=0` → 12MB。**V2.2 起末尾全 VTCM FLUSH (铁律②修复)** |
| `int wtcache_close(struct wtcache_ctx* ctx)` | 收尾 (铁律④), 会话末必调 |
| `void wtcache_layout(ctx, &vtcm_base, &vtcm_size, &pin_base, &pin_cap)` | 诊断: 布局四元组 |
| `wtcache_hmx_lock/unlock(ctx)` | 持锁线程实验开关 (4C 假设B) |

### pin (1-C-α)

| 函数 | 说明 |
|------|------|
| `int wtcache_pin_weight(ctx, ddr_src, bytes, align, void** vtcm_out)` | 一次 DMA 入 pin 区, 返回稳定 VTCM 指针。align: 0=128B。内部 DDR src FLUSH_INVALIDATE → DMA → dst INVALIDATE |
| `int wtcache_pin_verify(ctx, ddr_src, vtcm, bytes, uint32_t* first_bad_off)` | pin 区 vs DDR bit-exact 自检 |

### ring (1-C-β)

| 函数 | 说明 |
|------|------|
| `int wtcache_ring_init(ctx, struct wtcache_ring** r, tile_bytes, depth_pf, depth_mb)` | tile_bytes 必须 128 倍数; 默认 4+4 |
| `int wtcache_ring_prime(r, const void* ddr_tiles[], n)` | 预填 n ≤ depth_pf 个 tile (批量 prefetch, 等全完成) |
| `int wtcache_ring_next(r, ddr_next, ddr_out_target, invalidate, &cur_in, &cur_out)` | 稳态主循环原语: 等当前槽 → 提交下一 prefetch → 回搬已消费输出。`invalidate=1` 读前清输入槽 |
| `void wtcache_ring_set_overlap(r, enable)` | 1=非阻塞菊花链 (depth≥2 在途, 会话内禁 move_back); 0=阻塞式 (默认) |
| `int wtcache_ring_drain(r)` | 回搬残留输出 + 等全 idle |
| `wtcache_ring_destroy(r)` / `wtcache_ring_stats(r, &pf, &mb, &peak)` | 释放 / 诊断 |

host 侧另有 `wtcache_warm_file(path)` (posix_fadvise page-cache warm, 设备外用)。

## 使用范例 (完整见 example 16)

```c
struct wtcache_ctx* wc;
wtcache_open(&wc, 12u << 20);            /* 铁律②内置 */
void* vt;  wtcache_pin_weight(wc, w_ddr, 32768, 0, &vt);   /* 权重常驻 */
struct wtcache_ring* r;
wtcache_ring_init(wc, &r, 16u << 10, 4, 4);
wtcache_ring_prime(r, tiles, 4);
for (i = 0; i < N; i++) {
    wtcache_ring_next(r, next_tile(i), back_ddr(i), 1, &vin, &vout);
    /* HMX 读 vin / 写 vout (绕 dcache, 无需 flush) */
}
wtcache_ring_drain(r);                   /* 铁律③: CPU 读回 DDR 前 INVALIDATE */
wtcache_close(wc);                       /* 铁律④ */
```

## 契约与坑

- **CPU 模拟 HMX 写 vout 再 move_back**: memcpy 后必须 `QURT_MEM_CACHE_FLUSH(vout)`
  (example 16 第 3 段)。真 HMX 写绕 dcache, 不需要。
- pin 指针整会话稳定, ring/temp 活动不破坏 pin 区 (T6 证明; example 16 有回归门)。
- WTC_ERR_* 错误码见 wtcache.h; `pin_verify` 失败时首坏偏移编码在 `*first_bad_off`。
- V2.2 对源码的唯一修改 = open 末尾全 VTCM FLUSH (dualdomain run3 / wt_repack W4
  同族根因), 其余逐字节同源模块。
