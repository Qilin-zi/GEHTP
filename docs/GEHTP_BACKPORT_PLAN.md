# GEHTP 编译器反向移植清单（本地 REQNNFRAME → 104 GEHTP）

> 生成日期 2026-09-15。对比基线：本地 `/Users/apple/Documents/ge/REQNNFRAME/REQNN`
> vs 104 `/disk2/GEHTP/compiler`（B线编译器，8-31 从 `/disk2/REQNNFRAME/REQNN` 拷贝后持续开发）。
> 逐文件 md5 对比结论：82 文件一致，26 文件分叉，7 仅本地，11 仅 104。

## 0. 迁移原则（先读）

- 本地是**字节级保真复刻** libHtpPrepare.so 2.48.40.260702：裸指针 `u8*`、placement new、
  裸地址偏移、每个函数标 `.so` 地址；104 是**功能等价**实现：STL 容器、可读类、跑通模型即可。
  两者**风格不兼容，不能直接搬代码**。
- 因此本清单交付的是**语义约束**（偏移 / 字段布局 / 枚举 / 打包规则 / 错误路径），
  让 104 在其功能实现里逐条对齐，用本地实现作为**校验 oracle**（数值/布局对拍）。
- 每个 `.so` 地址是本地反汇编锚点，104 侧可用同一份 libHtpPrepare.so `objdump` 复核。
- 优先级：**P0** = 104 明确缺、且是其 M7「编译期内存规划」必需；**P1** = 对齐官方语义；
  **P2** = 参考级（本地自身也含垫片，或 104 走自研引擎不直接需要）。

---

## 1. 分配器层级（P0）—— 104 注释自己承认缺的中间层

### 1.1 恢复 `fa::RuntimeAllocator` 中间层

- **本地结论**：真实层级是三层 `hnnx::Allocator → fa::RuntimeAllocator → fa::FancyAllocator`。
  证据：`fa::FancyAllocator ctor @0xf3f2c0` 首调
  `_ZN2fa16RuntimeAllocatorC2EN4hnnx9Allocator4ModeER5Graph@plt`（基类 ctor），
  RuntimeAllocator `C2/C1 @0xd8a880`，虚表 `_ZTVN2fa16RuntimeAllocatorE @0x5ec1ec0`。
- **104 现状**：`include/hnnx/vtcm/fancy_allocator.hpp:62` 是 `class FancyAllocator {`（独立类，无继承），
  把 RuntimeAllocator 的职责扁平化/砍掉了。其 `fancy_allocator.cpp:78` 注释自己写
  「跨组复用在 .so 中由 `RuntimeAllocator::make_allocator` 负责，本仓只能顺序 bump」——
  **明确知道自己缺这个层**。
- **建议动作**：新增 `hnnx/alloc/runtime_alloc.hpp` + `src/alloc/runtime_alloc.cpp`（本地两个文件可直接做结构模板），
  让 `FancyAllocator : public RuntimeAllocator`。收益：跨组复用语义、槽位工厂、记录面访问都有正确归属。

### 1.2 `make_allocator` 槽位工厂（`@0xf4e3b0`）

- **本地结论**：`RuntimeAllocator::make_allocator(Graph&, bool fancy)` 按 fancy 标志分派——
  `fancy ? new(0x3a0)+FancyAllocator : new(0xb8)+RuntimeAllocator`，两分支 `Mode` 均传 `esi=0`
  （`AllocVirtual`），写入 `*(Allocator**)this = p` 并返回 this。
  `0xb8 == sizeof(RuntimeAllocator)` 精确；`0x3a0` 是 FancyAllocator 的 .so 真分配尺寸
  （本树对象仅 0x238，尾 0x168 未解码，placement 构造不越界）。
- **104 现状**：无此工厂（已见 1.1 注释自认缺）。
- **建议动作**：实现 `make_allocator` + `make_allocator_at(&slot, graph, fancy)`（调用方 Graph ctor
  `@0xd23509` 以栈槽为 this，随后写 `this+0x1d8` 并对旧者调 `vptr+0x08` deleting dtor）。

### 1.3 `RuntimeAllocator` 字段布局（`sizeof == 0xb8`）

| 偏移 | 字段 | 本地证据 |
|---|---|---|
| +0x08 | `graph`（继承 Allocator） | vptr 后首成员 |
| +0x10 | `mode` | ctor `0xd8a8a2 movl %esi,0x10(%rdi)` |
| +0x14 | `f_14`（bool） | `0xd8a8b3` |
| +0x40 | `tcm_pool_base` | static_assert |
| +0x48 | `tcm_pool_size` | static_assert |
| +0x50 | `largest_memory_alloc_size` | static_assert |
| +0x88 | `shared_tensors` | 见 1.4 记录面 |
| +0xa0 | `shared_spillfill_base` | static_assert |

- **建议动作**：104 侧若已有 FancyAllocator 字段，对齐上述偏移；`FancyAllocator` 自有状态应自 `+0xb8` 起。

### 1.4 `map_block_reference` / `deserialize_blocks` 记录面

- **本地结论**（`runtime_alloc.cpp`，`@0xd8d640` / `@0xd8d6e0`）：两函数共用的记录 vector
  `this+0x88` / `this+0x90` = 48 字节记录 vector 的 begin/end；记录数 =
  `((end-begin) >> 4) × 0xAAAAAAAAAAAAAAAB`（`0xAAAA...AB` 是 3 的模逆，`3×m ≡ 1 mod 2^64`，
  故是「(字节数/16)/3」的精确除，向量字节数恒为 48 的倍数）。
- **建议动作**：104 若实现块引用映射/反序列化，用此记录步长（48B）和除 3 语义对齐，避免逐项遍历错位。

### 1.5 错误处理两型

- **本地结论**：两类错误路径——
  ① `qnndsp_log(0, "%s:<行>::ERROR:<msg>", "runtime_alloc.cc")` + 可选写 `dctx->errstr` + **整函数返回**（不抛），
  行号/消息逐串核对（964 bad offset / 1012 bad encoding / 1036 / 1064 / 1078 bad encoding / 1092 bad format-4 / 1132 bad offset）；
  ② 块号越界 → 冷径 `@0xd8ed50` 抛 `std::out_of_range("vector")`（`__cxa_allocate_exception` + typeinfo
  `_ZTISt12out_of_range`）。
- **建议动作**：104 的分配器错误语义按此两型对齐（返回值 vs 抛异常），别把可恢复错误一律抛。

### 1.6 `FancyAllocator` 尾区（`ctor @0xf3f2c0` 尾）

- **本地结论**（M36-e 记账，`audit_verify/reports/M36e_FancyAllocator_tail_map.md`）：
  `+0x298` 非零 `u32[4]{0x4000,0xec00,0,0}`；`+0x360` 长串 `"DMA/L2 Access Statistics"`；
  `+0x378 u32=1`；`+0x380 shared_ptr{new(0x30) 节点}`；末触 `+0x388`，`+0x390..0x3a0` 为对齐填充；
  60 方法中 33 个消费尾部位移（分簇表见该 report）。
- **建议动作**：104 若对齐 FancyAllocator 真尺寸/统计段，参考此尾区布局（否则 0x3a0 分配会越界或浪费）。

---

## 2. SFCD spill/fill 写侧（P0）—— 104 只有读侧，M7 阶段二必需

**整体差距**：`spillfill_g4.cpp` 本地 779 行 vs 104 388 行。104 只到 §A（SFCD dump）+ §B（slc 序列化，读侧），
**缺 §C 分配面 / §D 检查点 op / §E 写侧真体 / §F-G-H 欠账批模型**。要产出真实 spill/fill 记录（M7 阶段二
「VTCM 驻留 + DMA 算子 runlist」），必须补写侧。

### 2.1 SFCD tcm 块记录 = 两词头（最高优先）

- **本地结论**：tcm 块记录头两词 `{w0 占位→回填 (pool<<16)|nblocks, w1=blob 偏移}`，子项在其后；
  终结器 `@0x1012460` 是「push {0, blob_off}」两词。**只压一词会让读侧 `@0x1013a90` 整体错位**
  （bad_length 级联）。
- **建议动作**：写侧 emit 每个块记录时固定两词头，第一词先占位后回填 `(pool<<16)|nblocks`。

### 2.2 wait 记录窄/宽式打包

- **本地结论**（`@0x10210f0`）：窄式 = 单词打包（bit16 清零，`字 = (val<<24)|mb`），
  条件 **iff 全部 `val ≤ 0xff` 且 `mb ≤ 0xffffff`**；宽式 = 对形态（bit16 置位，`low16 = n<<1`）。
- **建议动作**：104 写 wait 记录时按此判定窄/宽，别一律宽式（会多占字节、读侧长度对不上）。

### 2.3 `:78` 字段 = 残差

- **本地结论**：`:78 net = running − arr[2n−1]`（末次释放后的净剩），不是 running 本身；
  `find_peak_tcm_usage @0x1018c50` 的 `blocks = bytes >> 11`（2KB）。
- **建议动作**：写侧峰值统计按此残差语义，别把 running 直接当 net 写进记录。

### 2.4 常量池 ID（写侧/分配面共用）

- **本地结论**（`spillfill_g4.hpp`）：`G4_SPILLFILL_POOL_ID = 2`；`G4_COPYLESS_WEIGHTS_POOL = 0xf`；
  `G4_POOL_SHARED = 0x1`；`G4_POOL_FAR = 0x10`。相关判定 `g4_is_shared_spillfill` / `g4_can_mempool_be_far`。
- **建议动作**：104 侧池标志位/ID 与此对齐，避免 spill/fill 池选错。

### 2.5 写侧方法清单（本地已落地，104 缺）

| 方法 | 段 | 职责 |
|---|---|---|
| `g4_set_spillfill_size` / `g4_set_spillfill_shared_size` | §C | 池尺寸设置 |
| `g4_fill_slots_multi` / `g4_fill_slots_single` | §C | 三槽填充（`slot_sizes[3]` @+0xa8/+0xac/+0xb0） |
| `g4_insert_spill_fill` | §C | 插入 spill/fill 记录 |
| `g4_make_dma_checkpoint_op`（`G4CheckpointOp` vtable 0x5ec2488 set / 0x5ec2568 wait） | §D | DMA checkpoint op |
| `g4_dlbc_spill_fill_setup` / `g4_fill_mgroup_check` | §E | DLBC 写侧 setup + mgroup 校验 |
| `g4_sfcd_write_set_progress` / `g4_sfcd_write_waits` | §E | 记录级写入 |
| `g4_sfcd_finalize_copies`（合并子项）/ `g4_sfcd_finish_header`（回填 record_count） | §E | 拷贝记录收尾 |

- **建议动作**：按依赖顺序补：§C 分配面 → §E 写侧 → §D checkpoint（插在 runlist 正确位置）。
  §F/G/H（SLC 缓存模拟器 `G4SlcCacheSim`、`g4_slc_stat_init` 等）是**欠账批模型（近似非真体）**，
  仅供理解缓存行为，不要求 bit-exact。

### 2.6 读侧往返针脚（做往返断言时必看）

- **本地结论**：读侧打印 `mb` 用 **%u 十进制**、tcm 偏移用 **%X 大写**（`"tcm 0xABC0"` 不是 `"0xabc0"`）；
  `g4_err_2095` 头格式 `"%s:2095::ERROR:Bad SFCD record header %08X"`。
- **建议动作**：104 若做「写侧产记录 → 读侧 dump」的往返测试，字符串/格式与此对齐才能对拍。

---

## 3. Graph 类方法（P1 / P2）

**整体差距**：`src/ir/graph.cpp` 本地独有 **4191 行**，104 无此文件；`graph.hpp` 104 缺执行引擎方法声明。
这是 libHtpPrepare Graph 对象的完整字节级落地。

### 3.1 构造/析构子对象布局（P1）

- **本地结论**（`Graph::Graph @0xd22c60` / `~Graph @0xd263d0`）：
  - `+0xc8` 块：`new(0x1e0)+memset+0xd360b0`（内含 map@+0x138、Crate C1@+0x150、零段 +0x198..）
  - `+0x200`：`DMA_Manager C2 @0xd7f5e0`（`reset @0xd7f690` / `wait_all @0xd7fe00` / `wait_for @0xd7f710`）
  - `+0x54d0`：选项对象 `ctor @0xd627e0`（96 string + 2 vector{"core"} + 2 string_view + 151 原始存 + 16B blob×16 + memset×34）
  - `+0x6810`：`VtcmReq`（`clear @0xddc2b0` / `set+0x40 @0xddc3a0`）；`+0x68a0 = &no_op_delete_vtcm_save_buffer`（函数指针，非 vptr）
  - `+0x6b00`：`ctor @0xd3c5c0 {0, rsi=+0x6948, rdx=+0x5e08}`
- **建议动作**：104 若对齐官方 Graph 内存布局（尤其选项对象、VtcmReq），按此偏移表；否则默认容器布局即可。

### 3.2 选项键表 483 键（P1）

- **本地结论**（`Graph::set_option_finalize @0xd27680`）：setter/getter 由 483 键表驱动；另有 5 个列表键
  （regex `",+"` 分隔 / 逗号 join）；getter **U32 有符号读不对称**（signed 读，别按 unsigned 建模）。
- **建议动作**：104 若要对齐官方 option 语义（键名/类型/列表分隔），参考本地 `options_set_by_key` /
  `options_get_by_key`。

### 3.3 `soc_name_to_tab_index` 99 对（P1）

- **本地结论**：99 对 name→tab_index（miss=0）；其中四名（Matrix / Kaanapali / NordAU / SXR2330P）
  在**非连续 rodata**，盲走会错位。`soc_id_to_tab_index` 同域同 SoC 表。
- **建议动作**：104 若实现 SoC 名→表索引，注意这四名的非连续布局。

### 3.4 `ubwcd_set_prepare_options` 33 项描述符表（P1）

- **本地结论**：两 vector（`0x5db0`/`0x5dc8`）校验 + helper（`@0xd415f0`：Bayer 禁 NordAU/Kaanapali、
  TP10 输出禁 NordAU）+ 33 项描述符表；`0x5de8` 只门输入捕获。
- **建议动作**：104 若对齐 UBWC 格式校验，参考此表。

### 3.5 执行引擎（P2，本地含垫片）

- **本地结论**：`execute @0xd2c010` + `do_execute`/`do_continue`/`do_yield`/`do_execute_graph_stack`/
  `do_execute_inner`/`do_graph_env_setup`/`start_bkgrnd_workers`/`exec_{vec|mtx|elt}_worker`/
  `init_active_thread_counts` 全部落在 `graph.cpp`。语义：后台三线程 worker + 游标链表（`+0x52f8`）+
  VTCM yield 让出/重获 + 递归子图栈执行。
- **现状与优先级**：104 走**自研引擎**（kernels/ oplist_exec 按 runlist 派发），不需要逐字节对齐官方执行引擎。
  **P2**。且本地这些 worker 循环体 / wait_desc / count>0 循环仍是**垫片**，非 100% 解码。
  仅当 104 需要对齐官方 Graph 运行时语义（后台并发、VTCM yield 时机）时再参考。

---

## 4. 附：本地逆向锚点索引（104 objdump 复核用）

| 符号 | .so 地址 | 本地文件 |
|---|---|---|
| RuntimeAllocator ctor/dtor | 0xd8a880 / 0xd8ac50 / 0xd8ae70 | src/alloc/runtime_alloc.cpp |
| RuntimeAllocator::make_allocator | 0xf4e3b0 | src/vtcm/fancy_allocator.cpp |
| FancyAllocator ctor/dtor | 0xf3f2c0 / 0xf3fd30 | src/vtcm/fancy_allocator.cpp |
| map_block_reference / deserialize_blocks | 0xd8d640 / 0xd8d6e0 | src/alloc/runtime_alloc.cpp |
| Graph ctor / dtor | 0xd22c60 / 0xd263d0 | src/ir/graph.cpp |
| Graph::execute | 0xd2c010 | src/ir/graph.cpp |
| Graph::set_option_finalize | 0xd27680 | src/ir/graph.cpp |
| Graph::ubwcd_set_prepare_options | 0xd41430 | src/ir/graph.cpp |
| SFCD dump / 序列化侧 | 0x1013a90 / 0x1294570 | src/vtcm/spillfill_g4.cpp |
| SFCD 写侧 wait 打包 | 0x10210f0 | src/vtcm/spillfill_g4.cpp |
| make_xnsp_spillfill 三趟 | 0x101e0b0 / 0x101e1b0 / 0x101edd0 | （报告 §10.13） |
| construct_sfcd 本体 | 0x101eee0 | （报告 §10.13） |
| find_peak_tcm_usage | 0x1018c50 | src/vtcm/spillfill_g4.cpp |
| 详细报告 | — | audit_verify/reports/G4_spillfill_disasm.md、M36e_FancyAllocator_tail_map.md |

> 迁移时优先把本清单第 1、2 节转成 104 侧的单测断言（偏移/打包/记录头往返），
> 用本地 `spillfill_g4.cpp` / `runtime_alloc.cpp` 的输出做 golden，而不是直接替换 104 的实现。
