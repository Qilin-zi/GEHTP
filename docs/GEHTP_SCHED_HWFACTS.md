# GEHTP 调度设计原则与 V81 硬件事实表

> 生成日期 2026-09-17。来源：105 上 7 个 qcom_htp skill（`~/.claude/skills/`：
> htp-hardware-scheduling / htp-cycle-metric / qnn-htp-profiling / htp-kernel-measurement /
> qnn-native-op-flow / hmx-inline-asm / qnn-optrace-svg）的逐条评审，筛出对
> **本仓编译器（hnnx_compile）+ 执行器（WTOP/oplist_exec）** 有落点的部分。
> skill 原始数字产自 v75/8 Gen 3（GDN solve/W4A16 战役），**凡硬件相关一律以 V81 实测为准**（§1）。
> 设备纪律：本文所有 V81 数字均注明出处与获取方式；43_hwinfo 探针已入例程库，可一键复测。

---

## 1. V81 硬件事实表（52f67807 实测锚点）

| 事实 | 值 | 出处/获取方式 | 状态 |
|---|---|---|---|
| VTCM 总量 | **16 MiB**（0x1000000，base 0xff000000，整块授予） | 例 01 results：`VTCM base=ff000000 size=16777216` | ✅ 实测 |
| HVX 单元数/位宽 | 待测（`qurt_hvx_get_units()`，解码 n128=(u>>8)&0xFF） | 例 **43_hwinfo** A1 | ⏳ 探针已就绪 |
| 核心钟频 PCYCLE/µs | 待测（qtimer µs 窗 + busy spin，7 发取中位，DCVS PERF 角） | 例 43_hwinfo A3 | ⏳ 探针已就绪 |
| C15:14 ≡ qurt_get_core_pcycles | 待互证（同计数器则裸读数可直接对 QNN 口径） | 例 43_hwinfo A4 | ⏳ 探针已就绪 |
| HMX 锁语义 | 待测（NON_SHARED 进程内排他=立即 EFAILED；SHARED=多线程使能） | 例 43_hwinfo B1 | ⏳ 探针已就绪 |
| HMX 峰值 | raw K-loop **12.34 TFLOPS**；W4A16 256³ invoke 26µs=1290.6 GFLOPS | 例 14 / 例 17（PERF_REPORT §2/§3） | ✅ 实测 |
| DDR→VTCM 搬运 | wtcache pin 1MB=64µs=**16.38 GB/s** | 例 16（PERF_REPORT §2） | ✅ 实测 |
| HVX 流式带宽 | rms_norm n=1024: 0.30µs（43.9 GB/s）；mul: 102.4 GB/s | 例 15 | ✅ 实测 |
| 双 CDSP（dom3+dom4） | 并发 **2.004×**（ser 543µs → a/b 各 271µs），切分 byte-exact | 例 20 | ✅ 实测 |
| 单 CDSP 双线程 HMX | 并发/串行 ratio **0.963 ≈ 1**（单 DMA+HMX 锁线程，无线程级扩展） | 例 22 | ✅ 实测 |

复测：`cd kernels && ./examples/build_examples.sh 43`（含 A1/A3/A4/B1 四门）。
2026-09-17 注记：当日设备故障链——guest(52f67807, Android GVM) 双域 PD open 失败
（0x80000406/0x39/0x72，已知良好的 01 同样失败，排除探针本身）；整板当日被反复重启
（guest 与宿主 VM f69bec03 的 uptime 同步归零 ×2）；guest boot 期
`init.qti.write.sh /sys/kernel/boot_cdsp/boot 1` 60s 超时（boot_adsp/cvp 同病），
单独 reboot guest（宿主 vhost-device-ssr/glink_service_lrm/fastrpc-rm 均在跑）不恢复；
进一步铁证：guest /dev 从未创建 fastrpc 节点（DSP device discovery 未完成）、
guest boot 期 hab 通道协商被拒（`hab_open_listen failed -11, vcid 6a5ff001`，与宿主
dmesg `open cancelled` 对上）、探针尝试期间宿主 journal 零 SSR 活动（请求未出 guest）、
root 运行同样失败（非权限问题）——**虚拟化 SSR 通道协商层故障，需板级/hypervisor 恢复**。
注：同日更早该板另一会话曾跑通 894-op L0（见 GEHTP_DEVICE_RUNBOOK.md §6），
故障是整板反复重启后才出现的。
探针待设备恢复后跑，上表 ⏳ 项随之回填。

**对设计的直接含义（无论待测项结果如何都成立）：**
- VTCM 16MiB 是驻留池硬上限 → 编译器 `--plan-vtcm-budget` 的物理天花板（当前双池分配
  `ddr_offsets.cpp` 的 `in_vtcm` 决策）。
- HMX 峰值 ≫ wrapper 路径实测（12.34T vs 1290G ≈ 9.6×）→ 差距全在装料/卸料/调度开销，
  这正是 §2/§3 原则要收敛的方向。
- 单 CDSP 内 HMX 不可线程扩展（0.963）→ 执行器永远不要在单域内给 HMX op 开多线程；
  要扩展走双域（例 20 已证 2.004×）。

---

## 2. 执行器（oplist_exec / gehtp_runner）设计原则

编号 E1–E7。E1/E2/E6 现状已满足或半满足，写入本文是为**防退化**；E3/E4/E5/E7 是 M4/M7 的靶点。

### E1 一次派发 = 整图，永不 per-op RPC
v75 实测：196 ops 逐 op FastRPC 71ms vs 单次派发 12ms（单 op 364µs → 61µs 都嫌贵）。
**现状**：`wt_exec_run_io` 整图一把跑（runner 一次 RPC），已满足。**红线**：任何调试/插桩
改动不得引入 per-op host↔DSP 往返进热路径（optrace 逐 op 落盘仅在诊断档）。

### E2 VTCM 一次性 acquire + 全局切片
v75 教训：per-worker `HAP_compute_res_acquire` 会被资源管理器**串行化**（4 线程 scaling
1.83×→2.19× 的修复点）。**现状**：`hmx_runtime_setup` 幂等 + 主线程一次拿全量 16MiB，
worker 只拿裸指针（dc_threads.h R-D3 纪律）——已满足，保持。

### E3 标量/data-dependent scratch 禁入 VTCM（7× 反噬教训）
v75 实测：被标量代码密集访问的 scratch 挪进 VTCM 后慢 **7×**（471K→3.48M cycles）——
VTCM 为 HVX 向量/HMX/DMA 访问而生，标量 load/store 极慢。
**落点（编译器，MEMPLAN M4 期）**：`compute_ddr_offsets` 的 `in_vtcm` 决策须加
**访问类型分类**——只有「HVX 向量 / HMX / DMA 访问」的张量可标 in_vtcm；
标量访问（索引、data-dependent gather、控制流 scratch）留 DDR 靠 L2 prefetch。
当前 `in_vtcm` 无此分类，是 M4 前必须补的谓词，否则 VTCM 驻留可能越驻越慢。

### E4 DMA ping-pong 双缓冲是搬运隐藏的唯一大杠杆（17×）
v75 实测：217µs（重叠） vs 3701µs（串行）——单点最大杠杆。模式：`dma_start(next) →
compute(cur) → dma_wait()`，chunk N+1 的搬运藏在 chunk N 的计算下。
**落点**：MEMPLAN §3 M4「真 DMA runlist 算子」（把 matmul 内嵌的 cpu_to_vtcm/dc_dma_once
收编为显式 runlist 条目）应以 ping-pong 为目标形态——编译器在 runlist 里显式排
`DMA_LOAD(t+1) ∥ COMPUTE(t)` 的交错，而不是执行器隐式补。
**cache 铁律**（与既有 v81-cache-protocol-rules 一致）：DMA 源 DDR buffer 必须先
`qurt_mem_cache_clean(FLUSH)`；VTCM 作 DMA 目标无需 flush。

### E5 瓶颈层级 → cost model 必须 movement-centric
v75 量级：DDR↔VTCM I/O ~1000µs ≫ readback 120–3000µs ≫ HVX vdeal 5–126µs ≫
纯 HMX 计算 ~1µs（VTCM 驻留时 28k GFLOPS 近乎免费）。
**含义**：编译器的融合/放置/调度目标函数是**最小化 DDR 往返次数与字节量**，不是 FLOPs。
一张 op 的「成本」= 真算（HMX/HVX MAC）+ 装料（格式转换/pack）+ 卸料（DMA/readback）
三项分列（见 C4），优化优先级永远先砍搬运。

### E6 HMX 单单元串行，HVX 才谈线程
v75+V81 双证：HMX 锁进程级（SDK 语义 §1 B1 待回填），例 22 实测单域双线程 HMX
并发比 0.963 ≈ 1。**现状**：`g_hmx_spinlock`（hmx_common.c）单线程漏斗 + SHARED 锁
使能——已满足，保持；HVX op 的线程天花板 = `qurt_hvx_get_units()>>8`（43_hwinfo 回填）。

### E7 跨 op fill‖drain 重叠是 QNN supertile 的真壁垒（认知锚点，非近期目标）
v75 实测：同一 64³ conv 单跑 1970 cyc，QNN 在 32-conv 序列里 retire 间隔 ~290 cyc——
op N+1 的 fill 藏在 op N 的 drain 下。且 M-fanout 扫描证明：**字节相同 kernel 靠加大
batch 达不到**（只摊薄 prologue，不让 walk 跨 op 重叠）。
**含义**：执行器「逐 op 重放」形态的吞吐天花板 = 单 op 成本之和；要对标 QNN 序列吞吐，
需 M7 之后评估跨 op 软件流水（supertile 式）。中期不追，但**不要在不知道这条的情况下
拿我们的整图 wall 去比 QNN 的序列 retire 率**（口径陷阱，见 §4）。

---

## 3. 编译器（hnnx_compile / wtop_emit）设计原则

### C1 `num_dominant_path` 可组合 = 白捡的静态调度下界
QNN optrace 的 `num_dominant_path_cycles` = 「理想重叠后的关键依赖链」，图级值 =
链上 per-op 值之和（可组合，非黑盒）。**落点**：编译器可给每个 op 附 dominant-path
估计（实测或模型值），沿定稿序（M1 后 plan_order_ 保证拓扑）求关键链和 →
输出「若调度完美的地板」；`实测 wall − 该地板` = 可调度优化空间。这给了 M7 性能收敛
一个不依赖 QNN 的理论标尺，建议作为 manifest 可选字段输出。

### C2 HMX/HVX 分配规则（op placement 启发式）
v75 实测：HMX 只在**大矩阵**（≥512 dim）碾压（256×1024×4096 → 166× CPU）；
**窄输出/小块**被固定开销（搬运+格式转换+readback）淹没（832×128 仅 1.1× HVX）。
规则：**先问输出窄不窄**——窄输出 ⇒ readback 主导 ⇒ 直接 HVX。
落点：编译器 op lowering（wtop_ops/）选 kernel 时的 dtype 无关判据。

### C3 常量去重假依赖陷阱（前车之鉴，须主动防御）
QNN 把字节相同的全零 scratch 常量 dedup 成共享 VTCM buffer → 假 WAW 依赖 →
设计为独立的并行链被强行串行（netron/时间线都看不见，只有 op-graph 解析可见）。
**我方两防线**：① 打包层按 const id 去重仅限**只读权重**（现状即此，保持）；
② **可写 scratch/中间 buffer 永不去重合并**——即便内容全同。编译器/emit 加此类
优化时以此条为门禁。

### C4 三分类核账（真算/装料/卸料）进 cost model 与一切性能报告
QNN optrace 三分类：真算（HMX，MAC）、装料（HVX，格式转换/pack）、卸料+输入（DMA）。
v75 案例：QNN 自己的 32-conv batch 里 HMX Σbusy 只有 span 的 8.6%——装料 glue 主导。
**规则**：任何性能数字必须带分类+口径字段+shape+场景（单发 vs 序列），
禁止 lumped 单值；优化决策先看三分类占比再定方向（HVX-bound 时加 HMX∥HVX overlap 是无效杠杆，
v75 实测 mxmem 仅 6% 时 overlap 上限就是 6%）。

### C5 编码/运行时分层契约（前门 ABI 参照，现状已遵循）
scale/zero-point/bias 属于 encodings 与量化产物，**不作 runtime tensor**；bias 走 float
参数 + `--bias_bitwidth 32`；per-group W4 用 blockwise encodings 不拍平（与既有
expq-blockwise 闭合结论一致）。gehtp 现状（权重固化 blob、输入 raw 注入）与此同构，
新算子扩展时沿用。

---

## 4. 测量纪律（M7 性能收敛期强制执行）

来源：htp-cycle-metric / qnn-htp-profiling / htp-kernel-measurement，全是踩坑换来的：

1. **口径三字段，禁用自造词**：`num_dominant_path`（理想重叠下界）/ `cycles_used`
   （单元占用，**不是吞吐**）/ graph wall（makespan，终判）。跨字段比较 =  phantom gap
   第一来源（同 op 370 vs 1388 可差 3.7×）。
2. **序列吞吐用 `start_cycle` retire 间隔或 wall÷N，永不用 `cycles_used`/N**
   （流水时高估可达 5×；该坑曾连吞两个错误判决）。
3. **reps2-N 中位**，绝不取 min/单发；A/B 对比 **ACAC 同热配对**，不与历史常数比。
4. **单发 trace 只看结构**（stage 配比/相对 span），utilization% 作废；utilization 信任序：
   PMU 真值 > steady-derived（须标注）>（禁用）单发 trace%。
5. **同 field + 同 shape + 同场景**才能比；裸机 C15:14 ≈ 单发 op cycles_used，
   不可对 QNN 序列 retire 率（E7 的反面）。
6. 串行地板用 honest-tail `tail(P)=wall(P)−实测feed_Σ(P)/P`，不用常数 K 回归截距
   （feed_Σ 随 P 涨时截距虚高 3×，有实锤）。
7. 每次性能步必做「预期 DAG vs 实际编译 DAG」diff（我方 = tagged runlist 与预期
   调度序对拍；对照 QNN 时 authority = chrometrace_htp.json，不是 netron 看到的 ONNX）。

---

## 5. 工具资产（可直接复用）

| 资产 | 位置 | 用途 |
|---|---|---|
| 43_hwinfo 探针 | `kernels/examples/43_hwinfo/` | V81 硬件事实一键复测（A1/A2/A3/A4/B1） |
| HMX 指令词表 | `~/.claude/skills/hmx-inline-asm/references/hmx_unknown_inference.md` | mxmem 家族助记/字节对照（activation.ub、weight.n:2x:deep、cvt.ub=acc:sc0/sc1、mxclracc、endloop 后缀）；若需把 QNN 原生 kernel 切片字节级提为执行器 building block，配 `scripts/verify_hexagon_inline_asm.py` 整函数字节全同验收 |
| 三分类对表脚本思路 | skill htp-cycle-metric §"Report perf by the THREE QNN categories" | 与 QNN 对照实验时的报表模板 |
| optrace 解码/时间线 | skill qnn-htp-profiling Flow A/B/C | 与 QNN 同口径对比（若 M7 需要） |

## 6. 锚点索引

| 项 | 位置 |
|---|---|
| VTCM 双池分配（in_vtcm 决策本体，待补 E3 访问分类） | `compiler/src/vtcm/ddr_offsets.cpp:83-107` |
| spill 定稿 | `compiler/src/vtcm/ddr_offsets.cpp:126-147` |
| TEMPOFF 设备消费（不动） | `kernels/src/runtime/oplist_exec.c:1031-1085` |
| HMX funnel 自旋锁 | `kernels/src/runtime/hmx_common.c`（hmx_unit_acquire/release） |
| VTCM 一次性 acquire | `kernels/src/runtime/hmx_common.c`（hmx_runtime_setup，幂等） |
| 双域并发参照 | `kernels/examples/20_dualdomain/` |
| 单域双线程 HMX 0.963 实证 | `kernels/examples/22_dualcore_threads/` |
| M4 真 DMA runlist（E4 落点） | `docs/GEHTP_MEMPLAN_COMPILER_MIGRATION.md` §3 M4 |
| skill 源 | `~/.claude/skills/{htp-hardware-scheduling,htp-cycle-metric,qnn-htp-profiling,htp-kernel-measurement,qnn-native-op-flow,hmx-inline-asm,qnn-optrace-svg}/` |
