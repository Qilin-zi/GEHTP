# schedule_for_alloc 反汇编分析

## 函数信息
- 符号: `GraphPrepare::schedule_for_alloc(map<...>&, vector<...>&, uint64&)`
- 地址: `0x1302710`  大小: `0x282e` (10286 B)
- 全量反汇编: `schedule_for_alloc.disasm` (2608 行)
- 库: `libQnnHtp.so` (x86_64-linux-clang, QAIRT 2.48.0.260626)

## 关键结论: schedule_for_alloc 不生成 SET/WAIT

反汇编证明 `schedule_for_alloc` **不直接调用** `make_dma_checkpoint_op` 或 `make_SyncOp`。
它的调用目标（从 2608 行反汇编统计）:

| 调用次数 | 目标 | 含义 |
|---|---|---|
| 48 | `schedule_for_alloc` (自身) | **递归**调度 |
| 32 | `_ZdlPv` (operator delete) | 内存释放 |
| 22 | `qnndsp_log` | 日志 |
| 15 | `load_replacement_plan` | 加载替换计划 |
| 1 | `hexagon_nn_deserialize_graph` | 反序列化图 |
| 1 | `create_init_worker_thread` | 初始化工作线程 |
| 1 | `perform_graph_check` | 图检查 |
| 1 | `stupid_fast_topo_sort` | 拓扑排序 |
| 1 | `sap_reduce_bandwidth` | 带宽优化 |

`schedule_for_alloc` 的职责是**递归分配 VTCM 块并组织加载顺序**，
SET/WAIT 的实际生成在 `deserialize_graph` 内部（通过 `make_dma_checkpoint_op`）。

## make_dma_checkpoint_op 反汇编 (0xd958d0)

签名 (demangle): `GraphPrepare::make_dma_checkpoint_op(uint64 tensor_id, uint32 param, bool is_set)`

```asm
d958d0: push rbp; ...; sub rsp,0x20
d958dc: mov ebp, ecx          ; ebp = is_set (第3参数, bool)
d958de: mov r12d, edx         ; r12 = param
d958e1: mov r14, rsi           ; r14 = tensor_id
d958f5: mov edi, 0x18; call operator new(24)  ; 分配 24B op 对象
d95902: test ebp, ebp         ; if (is_set)
d95904: je  d9592c            ;   → SET 分支
                            ; else → WAIT 分支

; SET 分支 (is_set=true):
d95914: lea rax, [rip+0x512cb6d]  ; rax = 0x5ec2488 (SET vtable)
d9591b: mov [rbx], rax            ; op->vtable = SET_vtable
d9591e: mov [rbx+0x8], r12d       ; op->param = param
d95922: mov qword [rbx+0x10], 0   ; op->extra = 0
d9592a: jmp d95950                ; → insert_op

; WAIT 分支 (is_set=false):
d9593a: mov [rbx+0x8], r12d       ; op->param = param
d9593e: mov qword [rbx+0x10], 0   ; op->extra = 0
d95946: lea rax, [rip+0x512cc1b]  ; rax = 0x5ec2568 (WAIT vtable)
d9594d: mov [rbx], rax            ; op->vtable = WAIT_vtable

; 共同: insert_op(graph, tensor_id, op, false)
d95967: call insert_op
d9599c: ret
```

### SET vs WAIT 的 C++ 类区别
- SET vtable @ `0x5ec2488` → `N4hnnx16DmaCheckpointSetE`
- WAIT vtable @ `0x5ec2568` → `N4hnnx17DmaCheckpointWaitE`
- 两者是**不同的 C++ 类** (不同 typeinfo)
- 但 `serialize_internal` 虚函数指针**相同** (都指向 `0xd969a0`)
- → **context binary 字节级无法区分 SET 和 WAIT**

### 文档字符串证据 (.rodata)
```
@0x039B26D0 "Records a DMA operation's completion tag in a table, so that
             DmaCheckpointWait can wait for it. Some Ops, such as SpillOp
             and FillOp, can record their own tag, and don't need this."
             → DmaCheckpointSet 语义

@0x039B2690 "Waits for a previously started DMA to complete before proceeding"
             → DmaCheckpointWait 语义
```

## serialize_op 反汇编 (0x12ec630) — type 字段编码

`Serializer::serialize_op(const Op&, uint32)` 写入 op 记录。

### 0x1303EE 标记
```asm
12ec688: mov r13d, 0x1303ee71   ; 硬编码标记值 (记录ID起始)
```
后续递增 r13d 生成每个记录的 `0x1303EE{XX}` 标记。

### type 字段计算 (0x12ec788-0x12ec7d1)
```asm
12ec793: mov eax, [rcx+0x18]    ; eax = extra_info[0x18] (op flags)
12ec796: shr eax, 6             ; >> 6
12ec799: and eax, 0xf           ; & 0xf  → op_class (bits 6-9)
12ec79c: mov ebp, 1             ; default = 1
12ec7a1: cmovne ebp, eax        ; if op_class != 0: ebp = op_class
12ec7a4: cmp bp, 1
12ec7a8: jbe 12ec815            ; if op_class <= 1: skip HMX check
12ec7b0: call *[r15+0x70]       ; op->get_hmx_info() (vtable+0x70)
12ec7be: test eax, 0x4000000    ; HMX threaded?
12ec7c3: mov ebp, 0xb           ; default = 11
12ec7c8: cmove ebp, ecx         ; if not HMX: ebp = min(op_class,8)
12ec7cb: lea eax, [rbp+0x1000000]  ; type = op_class | (0x01 << 24)
12ec7d1: mov [rsp+0x34], eax    ; 存 type
12ec7df: call emit_u32          ; 写入 type 字段
```

→ **type 字节 [7] = op_class** (从 extra_info[0x18] bits[6-9] 提取):
- `0x01` = compute (默认/≤1)
- `0x02` = memory (extra_info[0x1c] & 0xFFFFFF != 0, `or 0x2000000` @0x12ec834)
- `0x03` = sync (extra_info[0x20] != 0, `or 0x3000000` @0x12ec880)
- `0x04` = DMA (DmaCheckpoint, 由 make_dma_checkpoint_op 创建时设定)
- `0x0b` = HMX threaded compute

记录格式最终为: `[0x1303EE{XX}][counter:u32<<0][type:u32<<24 | flags][block_ref:u32][...]`

## make_SyncOp 反汇编 (0xdac250)

签名: `GraphPrepare::make_SyncOp(uint64 tensor_id)`

```asm
dac250: push r15; ...; sub rsp,0x20
dac26a: call Graph::new_id()      ; 生成新 op_id
dac277: mov edi, 8; call operator new(8)  ; 分配 8B SyncOp
dac288: call Op::Op(Graph&, uint64)
dac28d: lea rax, [rip+0x5116f64]  ; rax = 0x5ec31f8 (SyncOp vtable)
dac294: mov [r15], rax            ; op->vtable = SyncOp_vtable
dac2b1: call insert_op(graph, id, op, true)
dac2c4: ret
```

SyncOp vtable @ `0x5ec31f8` → `N4hnnx6SyncOpE`

## VTCM 块引用编码

记录的 `[12-15]` block_ref 字段 = `0x10{idx}`:
- 高 16 位 `0x0010` = pool_id (VTCM pool)
- 低 16 位 = block 索引

`serialize_op` 写入 block_ref:
```asm
12ec8b7: and eax, [rbx+0x24]     ; eax = extra_info[0x24] & 0xFFFFFF (block_id)
12ec8ba: or eax, 0x3000000       ; | (0x03 << 24)  → pool标记
12ec8bf: mov [rsp+0x34], eax
12ec8cd: call emit_u32
```

→ block_ref 高字节 `0x03` 不是 pool_id 而是字段类型标记 (0x03=block_ref)。
实际 pool_id 在低 16 位的 `0x10` 部分。

## counter 字段 = tensor id

经字节验证 (见 re_op_final.py):
- counter 值 0-10 完美匹配 `simple_linear_net.json` 的 tensor id 1-10
- counter=0 = 图输入/输出节点 (特殊值)
- counter=2,9 = perm 张量 (在 node tensor_params 里, 不在顶层 tensors)

这是反汇编间接确认: `serialize_op` 读取 op 关联的 tensor id 作为 counter。

## 仍待确认 (需进一步反汇编)

1. `deserialize_graph` (0xcf28c0) 内部如何决定插入 SET vs WAIT
   - 需反汇编 `deserialize_graph` 的 ~16KB 代码
2. `schedule_for_alloc` 递归调用的终止条件和分配策略
   - 已反汇编但逻辑复杂, 需结合 `sap_reduce_bandwidth` 分析
3. 4 条 DMA 记录哪些是 SET 哪些是 WAIT
   - 字节不可区分, 需运行时 profiler 或反汇编 deserialize_graph 的决策逻辑

## 相关文件
- 全量反汇编: `schedule_for_alloc.disasm`
- bin 格式分析: `bin_format_analysis.md`
- 本文件: `schedule_analysis.md`
