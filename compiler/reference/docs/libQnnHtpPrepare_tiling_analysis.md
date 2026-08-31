# libQnnHtpPrepare.so Tiling 完整流程分析

> 本文结论来自对 `libQnnHtpPrepare.so`(即俗称 `libqnnhtpprepare.so`)的反汇编整理,全部关键点标注反汇编地址出处,并区分可信度:
> - ✅ **指令级验证**:带具体指令比对(`mulsd`/`orb`/`cmpl` 等),来自 `ALGORITHM_PRINCIPLES_VERIFIED.md`(V2.48)
> - ⚠️ **早期推断**:Ghidra/objdump 早期产物,来自 `ARCHITECTURE_PRINCIPLES.md` / `PASS_PSEUDOCODE.md`,部分结论可能被证伪
>
> ⚠️ **地址版本差异**:同一函数在不同 .so 版本(V73 / V81 / V2.48)地址不同。例如 `create_supertiles` 在符号表中是 `0x1313ac0`,在 PASS_PSEUDOCODE 中是 `0x13be970`。核对时请以手头 .so 实际反汇编为准。

---

## 0. Tiling 在 prepare pipeline 中的位置

`do_prepare2` @ `0xf633e0`(✅)内部按 `mark_prepare_stage` 编号顺序执行,tiling 分布在 Stage 0–5 中:

| Stage | 函数 | 地址 | 与 tiling 的关系 |
|-------|------|------|------------------|
| 0 | `initial_sequencer` | — | InitialSequencerDP 把图切成可独立分配 VTCM 的子图(DP 分组) |
| 2 | `sap_reduce_bandwidth` | `0xf6ee78` | **计算 VTCM 可用预算 = budget × 0.75** ✅ |
| 4 | **`create_supertiles`** | `0x13be970` (5196B) / 符号表 `0x1313ac0` | **★ SuperTile DP 合并相邻 op** |
| 5 | `post_spill_fill_design_pass` | — | spill/fill 插入后处理 |
| (P18) | `tcm_migration` | `0x13219c0` | VTCM 超预算则 spill 到 DDR ✅ |

完整 pass 编号见 `PASS_PSEUDOCODE.md:145-176`:

```
mark_prepare_stage({"initial_sequencer", 0});
mark_prepare_stage({"run_predication_pass", 1});
mark_prepare_stage({"sap_reduce_bandwidth", 2});
mark_prepare_stage({"do_unshare_interfaces", 3});
mark_prepare_stage({"create_supertiles", 4});
mark_prepare_stage({"post_spill_fill_design_pass", 5});
```

tiling 整体分为 **四个层次**:

```
1. Per-op 选 tile      —— Central Tiler + Hextimate 代价驱动
2. Op 间合并          —— SuperTile DP (create_supertiles)
3. VTCM 分配 / 溢出    —— DP 分组 + 图着色 First-Fit + tcm_migration
4. 权重侧重排         —— BlockTable tile-major 重排
```

---

## 1. Per-op Tiling:Central Tiler + Hextimate 代价驱动

### 1.1 Central Tiler 配置参数(⚠️,来自字符串提取)

`ARCHITECTURE_PRINCIPLES.md:1064-1074`:

| 参数 | 含义 |
|------|------|
| `central_tiler_tcm_limit` | VTCM 容量限制 |
| `central_tiler_tcm_limit_conv` | Conv 专用 VTCM 限制 |
| `central_tiler_frac` | 分块比例(如 0.5 = 使用 50% VTCM) |
| `central_tiler_height` | 高度维度分块大小 |
| `central_tiler_kb` | K 维度分块(KB) |
| `central_tiler_min_tiling` | 最小分块大小 |
| `central_tiler_min_tiling_frac` | 最小分块比例 |
| `central_tiler_concurrency` | 并发度(多 NSP) |
| `central_tiler_split_batch` | 是否拆分 batch 维度 |

### 1.2 选择算法(§12.3,⚠️)

遍历候选 tile 配置,用 **Hextimate** 估代价,取最小:

```
for each candidate_tile_config:
    cost = hextimate_estimate(graph, candidate_tile_config)
    if cost < best_cost:
        best_config = candidate_tile_config
        best_cost = cost
return best_config
```

### 1.3 代价模型(§12.2,⚠️带公式)

**MatMul** `[M,K] @ [K,N]`:

```
num_tiles        = ceil(K / K_tile) * ceil(N / N_tile)
per_tile_compute = K_tile * N_tile * MAC_PER_CYCLE
per_tile_dma     = K_tile * N_tile * dtype_size / DDR_BYTES_PER_CYCLE
total_cycles     = num_tiles * max(per_tile_compute, per_tile_dma)   # 双缓冲取 max,否则相加
```

**Conv2D**:

```
macs        = K * K * C_in * C_out * H_out * W_out
throughput  = LOOKUP(op_type, dtype)          # HMX INT8 ~32 MAC/cyc, HVX FP16 ~64 MAC/cyc
compute     = macs / throughput
dma_bytes   = weight_size + input_size + output_size
dma_cycles  = dma_bytes / DDR_BYTES_PER_CYCLE * DMA_EFFICIENCY
            # DMA_EFFICIENCY: tile-major 0.95 vs row-major 0.30
total       = USE_DOUBLE_BUFFER ? max(compute, dma) : compute + dma
```

### 1.4 tile 指纹(✅指令级验证)

`ALGORITHM_PRINCIPLES_VERIFIED.md:58`:

- `LayoutTensor<...>::tile_support_bits` @ `0xd06d4a` 等 **6 处**
- `make_tensor_template` @ `0xd1446d`
- 对 tensor 元数据做 **LFSR 指纹**,用作缓存/去重键

---

## 2. SuperTile:Op 间 DP 合并(减少 DDR 往返)

### 2.1 核心思想(§11.2)

```
传统 tiled execution:
  Op1(tile0) → DDR → Op2(tile0) → DDR → Op1(tile1) → DDR → ...
  每个 tile 都通过 DDR 传递中间结果

SuperTile execution:
  SuperTile(Op1, Op2):
    for each tile:
      DMA_IN: 加载 Op1.weight_tile + Op2.weight_tile → VTCM
      Compute: Op1(tile) → Op2(tile)  (中间结果留在 VTCM!)
      DMA_OUT: 写回最终结果 → DDR
  只有最终的 tile 结果写回 DDR —— 中间结果不离开 VTCM!

决策(HeuristicDP 驱动):
  if (VTCM(Op1+Op2_weight) < VTCM_free AND DDR_saved > threshold):
      合并为 SuperTile
  else:
      保持独立
```

### 2.2 create_supertiles 伪代码(`0x13be970`,5196B)

`PASS_PSEUDOCODE.md:802` / `ALLCONTAINED.md:3572`:

```c
// @ 0x13be970, 文件: supertile.cc
GraphStatus GraphPrepare::create_supertiles() {
    // Step 1: 构建 Op 邻接图
    DPGroupGraph group_graph;
    build_dp_group_graph(group_graph);

    // Step 2: 对每个 group 尝试合并
    for (auto& group : group_graph.groups) {
        HeuristicDP dp_solver(group, this->vtcm_size);

        // 代价模型适配器
        auto cost_adapter = make_unique<CostCalculatorAdapter>(
            group, this->ddr_bandwidth, this->vtcm_size);
        auto tcm_adapter  = make_unique<TCMCalculatorAdapter>(
            group, this->vtcm_size);
        dp_solver.set_cost_calculator(move(cost_adapter));
        dp_solver.set_tcm_calculator(move(tcm_adapter));

        // Step 3: DP 求解
        auto solution = dp_solver.solve();

        // Step 4: 应用解
        for (auto& st : solution.supertiles)
            make_one_supertile(st.op_ids, st.split_history);
    }
    return GRAPH_STATUS_OK;
}
```

### 2.3 DP 递推式

```
dp[i][k] = min(dp[j][k-1] + cost(j+1..i))   for all j < i
```
即:把 `op[0..i]` 分成 `k` 个 supertile 的最优代价。

- **CostCalculatorAdapter**:`cost = DDR_saved(merge) - VTCM_penalty(merge)`
  - `DDR_saved = (intermediate_tensor_size / DDR_bandwidth) * access_count`
  - `VTCM_penalty = merged_vtcm_requirement - max(individual_vtcm_requirements)`
- **TCMCalculatorAdapter**:`can_merge(op1..opN) = sum(vtcm_need_i) <= vtcm_available`

### 2.4 make_one_supertile(`0x13bfdc0`,591B)

`ALLCONTAINED.md:3611`:

```c
GraphStatus GraphPrepare::make_one_supertile(
    vector_view<uint64_t> op_ids, splithist_t split_history)
{
    // 1. 验证所有 op 无外部 consumer(可安全合并)
    // 2. 计算合并后 VTCM 需求
    size_t total_vtcm = 0;
    for (id : op_ids)
        total_vtcm += get_op(id)->estimated_vtcm_requirement();

    // 3. 超 VTCM → 按 split_history 递归切分
    if (total_vtcm > this->vtcm_free) {
        auto split_point = split_history.next_split(total_vtcm, this->vtcm_free);
        return make_one_supertile(op_ids.subspan(0, split_point), split_history);
    }

    // 4. 合并:分配连续 VTCM/DD,单次批量 DMA
    SuperTile st;
    st.op_ids           = op_ids;
    st.vtcm_base        = allocate_vtcm(total_vtcm);
    st.ddr_base         = allocate_ddr(total_vtcm);
    st.num_dma_roundtrips = 1;   // ★ 从 N 次 DDR 往返降到 1 次
    this->supertiles.push_back(st);
    return GRAPH_STATUS_OK;
}
```

### 2.5 SuperTile 标志位(✅)

`ALGORITHM_PRINCIPLES_VERIFIED.md:174`:

- `this+0x5578` 置 `0x100000001`(SuperTile 标志)

---

## 3. VTCM 分配:DP 分组 + 图着色 First-Fit + bank

### 3.1 四阶段(§10,`ARCHITECTURE_PRINCIPLES.md:800-862`)

| Stage | 算法 | 作用 | 复杂度 |
|-------|------|------|--------|
| A 生命期 | `build_graph_deps` @ `0xfac410` → 拓扑序扫描 | 每 tensor 的 `first_use / last_use / access_count` | O(V+E) |
| B 图分组 | `InitialSequencerDP`(HeuristicDP) | 把图切成可独立分配 VTCM 的子图 | O(n²log n) |
| C 图着色 | `fa_alloc.cc` First-Fit | 按 size 降序分配,落不下则 spill,bank conflict 避免 | O(n²) |
| D spill/fill | `insert_spill_fill` → `grdep_spillfill` | 在 first_use 前/last_use 后插 DMA_IN/OUT 节点 | — |

### 3.2 图着色(Stage C)算法

```
1. 按 size 降序排列 tensor(大 tensor 先分配)
2. First-Fit:
   for each tensor t:
       for each free_vtcm_chunk:
           if free_vtcm_chunk.size >= t.size:
               assign(t, free_vtcm_chunk.base); break
       if not assigned:
           spill(t)   // 标记为 DDR spill
3. Spill 决策:
   spill_cost = t.size / max(t.access_count, 1)
   // 大但访问少  → 优先 spill(spill cost 低)
   // 小但访问多  → 留在 VTCM(高收益)
4. Bank conflict 避免:
   将经常同时访问的 tensor 分配到不同 bank
   (8 banks × 2 MB,Weight A Bank0-1 / Weight B Bank2-3 / States Bank4-5 / Scratch Bank6-7)
```

### 3.3 DP vs 图着色关系(§10.3)

```
DP(Stage B):     "图应该怎样被切分为子图?"        → 全局最优,较慢
图着色(Stage C): "子图内 tensor 放在 VTCM 哪个地址?" → 局部最优,快速

协作:DP 确定 "哪些 op 在一个 VTCM context 内执行"
     图着色确定 "VTCM context 内 tensor 的物理地址"
```

---

## 4. VTCM 预算与溢出:tcm_migration(✅指令级验证)

### 4.1 预算字段(`ALGORITHM_PRINCIPLES_VERIFIED.md:162-199`)

| 字段/常量 | 含义 |
|-----------|------|
| `this+0x6338` | 当前 VTCM 预算(字节) |
| `this+0x5cc8` | 用户配置值 A;`val << 10` 得字节(KB 级) |
| `this+0x5ccc` | 用户配置值 B;`val << 20` 得字节(MB 级) |
| `this+0x6304` | 默认预算来源 |
| `this+0x5fc8 / +0x5fd0` | 实际生效的 VTCM 预算(高/低 64 位) |
| `this+0x62c0` | NSP 数;`cmpq $0x2` 判断多 NSP |
| `this+0x5578` | SuperTile 标志;置 `0x100000001` |
| `this+0x74c0` | MXFP4 相关计数 |
| `0x7ff` | 2KB 对齐掩码(`test $0x7ff` 检测未对齐) |
| `0x3fffff` / `0x400000` | 4MB 上限(两处都判 ≥4MB 触发 spill) |

### 4.2 4MB 上限逻辑(`0xf671f4`–`0xf67249`,✅)

```
budget = this->0x6338
if budget == -1:                    // 未设置
    A = this->0x5cc8 << 10           // KB 配置
    B = this->0x5ccc << 20           // MB 配置
    budget = (A == -1) ? B : A       // cmove 优先 MB 配置
if budget >= 0x400000 (4MB):         // 超上限 → spill 路径
    goto spill
```

即:VTCM 预算由 `+0x5cc8`(KB)或 `+0x5ccc`(MB)配置项换算,**硬上限 4MB**,超过则走 spill-to-DDR。

### 4.3 3/4 预算策略(✅)

常量 `0x3fe8000000000000` = double 0.75:

- 出现在 `0xd62d64`(写入结构体 `+0x150`)与 `0x10f1982`(写入结构体 `+0x2c8`)
- **乘法点**:`sap_reduce_bandwidth` @ `0xf6ee78`:`mulsd 0x5a68(%r15), %xmm1`
  - 从 `0x7468(%r15)` 读 VTCM 预算
  - `unpckhpd` + `addsd` 合并两个 double
  - **`mulsd 0x5a68(%r15), %xmm1`** 乘以 0.75
  - `cvttsd2si` 截断为整数 → **实际可用 VTCM = budget × 0.75**
- 即:**VTCM 预算的 3/4 作实际可用,1/4 留给系统/堆栈**
- 消费点:`do_prepare2` @ `0xf6416a` 同样读 `0x5a68(%r15)` 做预算校验

### 4.4 spill 机制(`tcm_migration` @ `0x13219c0`,✅)

**SPILL_TO_DDR 标志**(`0x1321fc0`):
```
1321fc0: orb $0x40, 0x20(%r12)     # tensor->flags |= 0x40 (SPILL_TO_DDR)
```
- 偏移 `+0x20` 是 tensor 标志位字段
- `0x40 = SPILL_TO_DDR`

**优先级 Score**(`+0x98`,✅):
- `tcm_migration` **从不计算优先级**,只读 tensor 对象 `+0x98` 处的 **预计算 32 位无符号整数**
- 比较(`0x132213f`–`0x1322245`)用 `jae/jb`(无符号)→ **max-heap**(值越大越先 spill)
- 优先级写入分散在 20+ 处 tensor 构造代码(`0x6f3d48`/`0x860f6d`/`0x89293d`/`0x8934ef`/`0x89ac98`/`0x89adf3`/`0x89f921`/`0x8e195e` …)
- 推断公式:`priority ∝ size / (access_frequency × lifetime)`,即 **大 + 少用 + 短命 = 高优先级 spill**
- 无单一公式,权重分散在各 tensor 构造/更新路径

**Heap Sort**(`0x1322118`–`0x1322270`,✅):
```
栈上 max-heap @ 0x98(%rsp),元素 24 字节:
  +0x00: tensor_ptr
  +0x08: aux_ptr
  +0x10: key (与 +0x98 优先级比较)

sift_down:
  1. 比较 parent->key 与 children->priority(+0x98)
  2. 若 child.priority > parent.key,交换并继续下沉
  3. 最终堆顶是优先级最高的 tensor
```

**预算逻辑**(✅):
- `0x168(%rsp)` = VTCM 预算上限(来自 `get_vtcm_tile_size`)
- 遍历所有 tensor,累加 `tile_size × access_frequency` 计算总需求
- 若需求 > 预算:按优先级从高到低标 `0x40` 标志
- 超 16MB spill 报错 `GRAPH_STATUS_SPILL_EXCEEDED`

---

## 5. 权重 Tile 重排:BlockTable(§13)

### 5.1 tile-major 重排原理

```
标准 Row-Major 权重 W[K][N]:
  MatMul tile 访问 Tile(0,0) = W[0:64][0:32]
  需读 W[0][0:32], W[1][0:32], ..., W[63][0:32]
  DDR 中分散在 64 行,每行 stride = N
  → 64 次小 burst → DDR 效率 ~30%

Tile-Major 重排(编译期):
  W 重排为 Tile(0,0), Tile(0,1), ..., Tile(1,0), ...
  每个 tile 内部连续,单次 2048-byte burst → DDR 效率 ~95%

即:"DDR 效率从 30% 到 95%" 是编译器重排,不是硬件改进
```

### 5.2 BlockTableEntry 数据结构(§13.2,`ARCHITECTURE_PRINCIPLES.md:1124-1134`)

```c
struct BlockTableEntry {
    uint64_t ddr_offset;          // tile 在 DDR 中的起始地址(编译期已知)
    uint64_t vtcm_addr;           // tile 在 VTCM 中的目标地址(编译期已知)
    uint32_t original_size;       // 原始 tile 大小(bytes)
    uint32_t compressed_size;     // LZ4 压缩后大小(0 = 未压缩)
    uint8_t  compression_type;    // 0=none, 1=LZ4
    uint8_t  dma_channel;         // DMA 通道:0=load, 1=store
    uint16_t prefetch_distance;   // 预取距离(当前 tile 计算时提前 N 个 tile 发 DMA)
    uint32_t k_tile;              // K 维分块大小
    uint32_t n_tile;              // N 维分块大小
};
```

### 5.3 blocktable_reduce(§13.3)

合并 DDR/VTCM 中都连续的相邻 tile,N 个 DMA 描述符 → 1 个:

```
缩减前:
  Tile 0: W[0:64][0:32],  DMA → VTCM[0xA00000], size=2048
  Tile 1: W[0:64][32:64], DMA → VTCM[0xA00800], size=2048
  Tile 2: W[0:64][64:96], DMA → VTCM[0xA01000], size=2048
  → 3 个 DMA 描述符

缩减后(DDR+VTCM 都连续):
  Merged: W[0:64][0:96], DMA → VTCM[0xA00000], size=6144
  → 1 个 DMA 描述符

优点:更少 DMA setup overhead + 更大 burst + 更高 DDR 效率
```

---

## 6. 指令选择:HMX vs HVX(`Graph::compile`,§14)

`if op.M >= 32: select_hmx_kernel`(32×32 tile 矩阵乘)`else: select_hvx_kernel`(8-thread 512-bit 向量)。

**tile 大小直接决定走哪个微内核**。

`Graph::compile` 流程:
1. 指令选择(HMX/HVX,查 `lookup_kernel_table`)
2. 指令调度(列表调度,每 cycle 为 5 引擎 [HVX, HMX, Scalar, DMA0, DMA1] 各找一条指令)
3. 软件流水(小循环展开)
4. 寄存器分配(图着色,生命期不重叠复用)
5. DMA 流构建(从 BlockTableEntry 发 DMA 指令,含 prefetch)
6. 二进制编码 → `.serialized.bin`

---

## 7. 整体调用图

```
do_prepare2 @ 0xf633e0
├── [P1-P10] do_prepare1 阶段:图构建/初始优化/常量传播/IO 分配
├── run_optimize_passes @ 0xf730b0          (8 阶段 pass,带节点数阈值)
├── build_graph_deps @ 0xfac410             (依赖图 + tensor 生命期)
├── [P11] initial_sequencer                  (InitialSequencerDP 图分组)
├── [P12] sequencing_stage("initial_sequencer")
├── [P13] run_predication_pass
├── [P14] sap_reduce_bandwidth @ 0xf6ee78   (★ VTCM × 0.75 预算)
├── [P15] do_unshare_interfaces
├── [P16] create_supertiles @ 0x13be970     (★ SuperTile DP 合并)
│   └── make_one_supertile @ 0x13bfdc0       (per supertile)
├── [P17] post_spill_fill_design_pass
├── [P18] tcm_migration @ 0x13219c0         (★ VTCM 超限 → spill 0x40)
├── [P19] const_tracking_after_prep
├── Graph::compile                           (指令选择 + 调度 + 编码)
└── do_prepare2_late @ 0x1062150
```

---

## 8. 关键地址速查表

| 函数 | 地址 | 大小 | 可信度 |
|------|------|------|--------|
| `do_prepare1` | `0xf66550` | ~8.5K | ✅ |
| `do_prepare2` | `0xf633e0` | ~6K | ✅ |
| `run_optimize_passes_single_registry` | `0xf730b0` | 7060 | ✅ |
| `sap_reduce_bandwidth`(VTCM×0.75) | `0xf6ee78` | — | ✅ |
| `build_graph_deps` | `0xfac410` | ~8K | ✅ |
| `schedule_for_alloc` | `0x1302900` | 10286 | ✅ |
| `create_supertiles` | `0x13be970` / `0x1313ac0` | 5196 | ⚠️/✅ |
| `make_one_supertile` | `0x13bfdc0` | 591 | ⚠️ |
| `tcm_migration` | `0x13219c0` | 10141 | ✅ |
| `sequencing_stage` | `0x12b2770` | 19068 | ⚠️ |
| `GraphPrepare::main` | `0x11e10a0` | ~20K | ⚠️ |
| `tile_support_bits` | `0xd06d4a` 等 6 处 | — | ✅ |
| `make_tensor_template` | `0xd1446d` | — | ✅ |
| SPILL 标志写入 | `0x1321fc0` | — | ✅ |
| sift_down(heap) | `0x1322118`–`0x1322270` | — | ✅ |
| checksum_bytes(LFSR) | `0xdacd20` | — | ✅ |
| 0.75 系数常量 | `0x3fe8000000000000` | double | ✅ |

---

## 9. 可信度提示

### ✅ 指令级验证(来自 `ALGORITHM_PRINCIPLES_VERIFIED.md`,V2.48)

- VTCM 4MB 硬上限 + 0.75 系数(`mulsd 0x5a68(%r15), %xmm1`)
- `tcm_migration` 的 0x40 SPILL 标志(`orb $0x40, 0x20(%r12)`)+ max-heap(`jae/jb`)
- 8 阶段 pass 节点数阈值(3000/10190/11900/12500/21101/22000/24999/∞)
- LFSR tile 指纹(`tile_support_bits` / `make_tensor_template`)
- SuperTile 标志 `this+0x5578 = 0x100000001`

### ⚠️ 早期推断(来自 `ARCHITECTURE_PRINCIPLES.md` / `PASS_PSEUDOCODE.md`)

- Central Tiler 配置参数(字符串提取)
- Hextimate 代价模型公式
- SuperTile DP 伪代码与 `make_one_supertile` 细节
- 图着色四阶段(Stage A-D)
- `Graph::compile` 指令选择流程

README 明确指出:部分早期结论已被主文档证伪(LFSR 被误记 CRC32、调度被误记图着色、1.3/1.4 双种子不存在)。**阅读时以 `ALGORITHM_PRINCIPLES_VERIFIED.md` 为准**。

### 地址版本差异

同一函数在不同 .so 版本地址不同:
- `create_supertiles`:`0x13be970`(PASS_PSEUDOCODE) vs `0x1313ac0`(符号表)
- `do_prepare1`:`0xf66550` 等

核对时请以手头 .so 实际反汇编为准。

---

## 10. 待进一步反汇编验证的点

1. `HeuristicDP` + `Simplex LP` 的完整实现(`heuristic_dp.cc` / `clp_simplex.cc`,6 个文件)
2. `build_dp_group_graph` 的具体切分逻辑
3. `CostCalculatorAdapter` / `TCMCalculatorAdapter` 的精确代价公式
4. Central Tiler 候选配置枚举的具体代码位置
5. tensor `+0x98` 优先级在各构造点的实际计算逻辑(20+ 处分散写入)
6. `blocktable_encode` / `blocktable_reduce` 的字节级编码格式

> 如需对某个具体函数做逐指令核对,提供手头 .so 的版本/架构,可用 WSL 或 Python+capstone 对该函数定向反汇编验证。
