# HTP 编译算法逆向工程结果

## 1. f2 Hash 算法 (从 libHtpPrepare.so 反汇编确认)

**函数**: `gen_Const_int32_common` @ 0xF86600 (offset 0xA0-0x130)

**算法** (反汇编 @ 0xF866AE-0xF86728):
```
hash = 0x51BE1035              ; 初始值 (mov $0x51be1035, %r13d)
hash += dims[0]                ; add 0x8(%rbx), %r13d
if rank > 1:
    hash *= 0x01003123          ; imul $0x1003123, %r13d, %r13d
    hash += dims[1]            ; add 0x10(%rbx), %r13d
if rank > 2:
    hash *= 0x01003123
    hash += dims[2]
... (最多8维)
```

**Python实现**:
```python
def compute_f2_hash(dims):
    h = 0x51BE1035
    h = (h + dims[0]) & 0xFFFFFFFF
    for i in range(1, len(dims)):
        h = (h * 0x01003123) & 0xFFFFFFFF
        h = (h + dims[i]) & 0xFFFFFFFF
    return h
```

**验证**: 7/7 样本全部匹配 (simple_linear + 6个对照模型)

**输入**: `dims = extras[1:]` (op记录的extras数组去掉第一个元素)

**常量来源**:
- `0x51BE1035` = 初始哈希值 (在代码中硬编码)
- `0x01003123` = 乘法因子 (在代码中出现7次,对应最多8维)

## 2. DMA Tag (f2 for DMA ops)

**函数**: `make_dma_checkpoint_op(op_id, param, is_set)` @ 0xD958D0

**反汇编** (@ 0xD9591E): `mov %r12d, 0x8(%rbx)` — param 直接写入 op+0x8

DMA tag 不是计算出来的,是调度器传入的参数。
调用者分配 tag 值 (0x11, 0x16, 0x1A 等)。

## 3. 19步展开规则 (7样本对比)

| QNN op | 贡献步骤 | 步数 |
|--------|---------|------|
| 图框架 | C0(null)+C0(shape)+C1(Slice)+M2(perm)+尾部[C0(Out)+S+M2+C3+C10] | 9 |
| Transpose | M3(layout)+C4(flat)+C4(prep)+D | 4 |
| Reshape | C3 或合并 | 1-2 |
| FC | D(W)+D(b)+C(MatMul)+C(bias)+D(out) | 5 |
| 第二层FC | C(MatMul)+C(bias)+D(out) | 3 |

**验证**:
- fc_only: 9+5=14 ✅
- trans_only: 9+4=13 ✅
- reshape_only: 9+0=9 ✅
- two_fc: 9+5+3=17 ✅
- simple_linear: 9+4+5+3-2=19 ✅

## 4. block_ref (VTCM block index)

跨4个shape样本确认: shape无关,结构决定。
VTCM block index 在编译时由 VTCM allocator 分配。
具体值 (0x1003, 0x1019, 0x101A...) 对应特定的 tensor/block。

## 5. Kernel Name Table

跨4个shape样本确认: shape无关。
11个kernel名固定 (Shape, *InputSlice@Ff.s4*6., @DmaCheckpointSet, ...)
不同op组合会使用不同的kernel子集。

## 6. op 记录格式 (从 serialize_op @ 0x12EC630 确认)

```
[0-3]   marker    0x1303EE{record_id} (LE u32)
[4]     counter   tensor_id (1 byte)
[5-6]   padding   0x0000
[7]     type      0x10/0x20/0x30/0x40 (1 byte)
[8-11]  f2        hash或DMA_tag (LE u32)
[12-15] block_ref 0x1000|vtcm_block (LE u32)
[16+]   extras[]  变长 (LE u32 array)
```

**f2 来源** (serialize_op @ 0x12EC793):
- type = (extra_info[0x18] >> 6) & 0xf
- f2 = extra_info[0x1c] & 0xffffff
- block_ref = extra_info[0x20] & 0xffffff

## 7. extra_info 结构体布局 (从 set_extra_info @ 0xD293E0 + update_extra_info_map @ 0xD29720)

**set_extra_info** 复制 24 字节的 OpExtraInfo:
- `[0x00-0x0F]`: 16 bytes (xmm0, 来自 OpExtraInfo 前16字节)
- `[0x10-0x17]`: 8 bytes (r14, 来自 OpExtraInfo+0x10)
- `[0x18]`: type flags (serialize_op 读取 (val>>6)&0xf)
- `[0x1C]`: f2 / DMA tag (serialize_op 读取 & 0xffffff)
- `[0x20]`: block_ref (serialize_op 读取 & 0xffffff)
- `[0x24-0x2F]`: 额外字段 (被 set_extra_info 清零)
- `[0x30+]`: tensor_ids 列表

**update_extra_info_map** 使用 fibonacci_hash:
- `fibonacci_hash(op_id)` = (hi32 * 0x192E2101) ^ (full * 0x740F1DE9) 的 hi32
- 哈希表条目大小: 0x58 字节 (88 bytes)
- 用 op_id 作为 key 查找 extra_info

## 8. Graph::compile 调用序列 (从 0xD2B930 反汇编)

`Graph::compile()` 依次调用:
1. `compile_init(env, ...)` — 初始化
2. `compile_exec_list(env, runlist_set, ListType=0, ...)` — 主执行列表
3. `compile_exec_list(env, runlist_set, ListType=1, ...)` — 后台列表
4. (继续, 多次 compile_exec_list)
5. `compile_finish(env, ...)` — 完成

**compile_exec_list** 是构建执行步骤的核心函数。
ListType 0 = 主列表, 1 = 后台, 等。

## 9. make_dma_checkpoint_op 结构 (从 0xD958D0 反汇编)

```
struct DmaCheckpointOp : Op {
    // vtable @ offset 0 (0x5EC2488 for SET, 0x5EC2568 for WAIT)
    // Op base @ offset 0-7
    uint32_t param;    // @ offset 0x08: DMA tag (直接从参数传入)
    uint64_t reserved; // @ offset 0x10: 清零
};
```

DMA tag 不是计算出来的,是 make_dma_checkpoint_op 的第2个参数。
tag 值由调用者(调度器/优化pass)分配。

## 10. make_SyncOp 结构 (从 0xDAC250 反汇编)

```
struct SyncOp : Op {
    // vtable @ offset 0 (0x5EC31F8)
    // 只设置 vtable,无额外字段
};
```

SyncOp 无参数,只是一个同步屏障标记。

## 11. phys_alloc_in_runlist (从 0xF72910 反汇编)

遍历 runlist 中的每个 Op:
1. 获取 op->id (通过 Op::id(graph))
2. 在 VTCM allocator hash map 中查找 op_id
3. hash map 使用 popcount 计算 bucket
4. 查找对应的物理分配信息
5. 设置 extra_info 中的 block_ref

block_ref 来自 VTCM allocator 的 hash map 查找结果。
物理分配在 FancyAllocator 中完成。
