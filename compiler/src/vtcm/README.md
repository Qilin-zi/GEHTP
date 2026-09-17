# src/vtcm/ — VTCM 内存分配器

对应真实 `vtcm_alloc.cc`。VTCM (Vector Tightly Coupled Memory) 是 HVX 的高速片上内存。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `fancy_allocator.cpp` | `vtcm_alloc.cc` @ 0xF4B5B0 等 | FancyAllocator: **allocate/deallocate** (线性 bump), allow_tensor_overlap, setup_heap_info, make_persistent_pools, serialize_pools, block 映射, MCRecv 连续分配 |

## 关键函数

| 函数 | 真实地址 | 实现 |
|------|----------|------|
| `allocate` | — | 线性 bump, 4B 对齐, 记录 block_map_ |
| `setup_heap_info` | 0xF4B5B0 (723B, ELF st_size) | 计算 VTCM 用量写堆元数据 |
| `make_persistent_pools` | 0xF453D0 (8477B, ELF st_size) | 权重/常量分类成池 |
| `rewrite_to_physical_offset` | 0xF40DD0 (506B, ELF st_size) | 虚拟偏移 -> 物理地址 |
| `force_contiguous_allocate_mcrecv_blocks` | 0xF823A0 (178B, ELF st_size) | MCRecv 需连续 DMA 缓冲 |

> 注：size 字段以 libHtpPrepare.so (x86_64-linux-clang) 的 ELF `.dynsym` st_size 为准。
> 旧注释中的大 size（如 53533/6889/3480）是反编译器对 stripped 二进制函数边界识别失败
> 造成的虚胖值——实际范围里包含多个独立命名函数（如 make_persistent_pools 的旧 53533B
> 范围内含 40 个 FancyAllocator helper）。

## VtcmCacheInstance

每 NSP 一个 VTCM 实例 (test 用 8 MB),内含 FancyAllocator。
