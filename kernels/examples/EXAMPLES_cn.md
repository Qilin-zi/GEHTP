# 示例导览 — 全部 33 个测试用例详解 (中文)

本文是 [TEST_CASES.md](TEST_CASES.md) (逐例判据速查) 的姊妹篇, 也是
[EXAMPLES.md](EXAMPLES.md) (英文版导览) 的中文版。[README.md](README.md) 回答
"怎么跑", TEST_CASES.md 给出每个用例的门列表, 本文则是**叙述式讲解**:
**每个例子是什么、为什么存在、在设备上实际验证了什么、一行绿色的 PASS 证明了什么。**

写作目标: 从没接触过本项目的读者, 不读源码也能把 33 个例子一个个看懂。

> **设备实测状态** (`52f67807`, V81, unsigned CDSP PD): 33 个示例全部
> 编译/签名/部署/运行通过 (`./build_examples.sh all`)。最近一次全量
> (2026-08-16) **181 门 [PASS] 0 FAIL** (32 个例子的括号格式门), 另有
> 例 15 的列式检查清单以 `--- ALL PASS ---` 收尾 (其逐项行不计入 181)。
> 性能数字见 [../PERF_REPORT.md](../PERF_REPORT.md); 单元手册见
> `../docs/api_v22_*.md` 与 `../docs/api_v23_*.md`。

---

## 目录

1. [三代示例地图](#1-三代示例地图)
2. [示例是怎么构建的](#2-示例是怎么构建的)
3. [共享骨架 (common/)](#3-共享骨架-common)
4. [这里的 "PASS" 意味着什么](#4-这里的-pass-意味着什么)
5. [33 个例子逐个讲](#5-33-例子逐个讲)
   - V2.1 算子回归 (01–15)
   - V2.2 工程单元 (16–22)
   - V2.3 工程单元 (23–33)
6. [全示例容差矩阵](#6-全示例容差矩阵)
7. [一键全量运行](#7-一键全量运行)
8. [写一个自己的示例](#8-写一个自己的示例)

---

## 1. 三代示例地图

33 个例子不是 33 个互不相关的演示, 而是一个故事的三层:

| 代 | 示例 | 层 | 回答的问题 |
|----|------|----|-----------|
| V2.1 | 01–15 | **算子层** | 库里每一个 HVX/HMX 数学算子, 输出是否与教科书定义一致? |
| V2.2 | 16–22 | **工程单元 (U1–U8)** | 从已闭合模块沉淀的系统件 (VTCM 权重缓存、W4A16 引擎、GDN 状态机、双域执行、op 列表引擎、线程) 在真实硬件上行为是否正确? |
| V2.3 | 23–33 | **工程单元 (U9–U19)** | 第二批单元 (cache fence、arena、金标框架、工人池、精度桥、树形 GDN、KV 缓存、整步执行、GEMM 路由、回退、缓冲审计) 是否站得住? |

第一代的 PASS 含义是"这个算子等于数学定义"; 第二、三代的 PASS 含义是
"这套**机制** (cache 协议、并发协议、审计契约) 在硅片上与规格逐字节一致" ——
判据通常是对独立 oracle 的 byte-exact, 而不是"自洽"。

单元 → 示例对照 (手册在 `../docs/`):

| 单元 | 名字 | 示例 | 单元 | 名字 | 示例 |
|------|------|------|------|------|------|
| U1 | wtcache (VTCM 权重缓存) | 16 | U11 | harness (金标 case 框架) | 25 |
| U2 | dcmem (arena/文件/DMA) | 17, 20, 22 | U12 | wpool (常驻工人池) | 26 |
| U3 | dcthread (线程) | 22 | U13 | pxbridge (f32↔f16↔i16) | 27 |
| U4 | w4a16 (W4A16 HMX 引擎) | 17, 21, 31 | U14 | gdntree (树形 GDN) | 28 |
| U5 | gdnsm (GDN 状态机) | 19 | U15 | kvcache (KV 槽缓存) | 29 |
| U6 | oplist (op 列表引擎) | 21, 30 | U16 | graphstep (整步执行) | 30 |
| U7 | dualdom (双域) | 20 | U17 | gemmdispatch (MatMul 路由) | 31 |
| U8 | smallm (pad-256 GEMV) | 18, 31 | U18 | rbr (回退/部分接受) | 32 |
| U9 | fence (cache 交接) | 23 | U19 | bledger (缓冲审计) | 33 |
| U10 | arena (双池分配器) | 24, 28 | | | |

---

## 2. 示例是怎么构建的

每个示例住独立子目录 (`NN_name/main.c`), 编译成**独立**的动态库
`test_NN_name.so`, 链接 `libhvxhmx_v23.so`。由
[`build_examples.sh`](build_examples.sh) 编排, 每个示例的流水线完全一致:

1. **编译** `main.c` + `common/example_util.c`, 用标准 V81 旗标
   (`-mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC`), 链接
   `-lhvxhmx_v23` 和 `-lc -ldl -lgcc` (libqurt.a 非 PIC 不可静态链;
   qurt/HAP 符号运行时由 DSP 进程解析)。
2. **SWIV 签名** (`swiv_build_utility.py`)。必须做: unsigned PD 加载器
   拒绝未签名 .so, 且**静默**拒绝含不可解析 UNDEF 符号的 .so。
3. **部署**到 `52f67807:/data/local/tmp/hvxhmx23/` (含共享 assets、
   oplist blob、金标文件)。
4. **运行**: `run_main_on_hexagon 3 test_NN_name.so` (PD 3 = cDSP;
   例 20 另用 PD 4 跑第二个域)。
5. **回读**: 示例自己写的 `NN_name.txt`; 脚本拉回 `../results/`。

跑一个或跑全部:

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh            # 全部 33 个
./build_examples.sh 02         # 只跑 02_convf16_gemm
./build_examples.sh 32_rbr     # 按全名同样可以
```

结果文件是唯一事实来源。一次好的运行以
`--- summary: N pass, 0 fail ---` 结尾, 脚本末尾横幅是
`TOTAL 181 pass 0 FAIL / ALL GREEN`。

---

## 3. 共享骨架 (common/)

33 个 `main.c` 共用同一骨架 (见
[`common/example_util.h`](common/example_util.h) 与
[`common/example_util.c`](common/example_util.c))。骨架看懂一次,
所有例子一眼可读:

1. `ex_open_result("NN_name")` — 在设备上打开结果文件作为输出通道
   (`/data/local/tmp/hvxhmx23/` 下的绝对路径; DSP 上相对路径 fopen
   会静默失败);
2. runtime 就绪 — V2.1 示例调 `hmx_runtime_setup(2MB)`; V2.2/V2.3 单元
   示例一般打开自己的单元上下文 (`wtcache_open` / `fence_*` / arena
   init / …), 内部顺带上电 HVX/HMX 并接管 VTCM;
3. `ex_fill_*` — 用**固定种子 LCG** 填 buffer (可复现, 不依赖 rand)。
   u8/i8/u16/i16/i32/f16 各有变体, 各带种子与尺度, 让数值落在受控区间
   (fp16 输入保持在 ±0.5 内, 避开 denormal 悬崖);
4. **调被测单元**;
5. **算金标** —— 或是纯标量 C 实现的数学定义 (算子示例), 或是 host
   可复现的独立 oracle (单元示例), 或是闭合模块存档的产物
   (`Y_gold_2563.raw`);
6. **逐元素比对**, 取最大误差 → `ex_check(label, err, tol)`,
   `err <= tol` 记 PASS。布尔条件必须写成 `cond ? 0 : 1` ——
   极性写反是本项目真实踩过的坑;
7. `return ex_summary()` — 有 fail 返回 1, 全 pass 返回 0。

关键是第 5 步: **金标独立于被测代码**。PASS 真正证明的是
"硬件路径 == 数学/规格", 而不是"库和自己一致"。

V2.3 的 harness 单元 (U11, 例 25) 把这套骨架固化成可复用的 case 框架,
输出带 sha 钉死; 新的金标 case 可以直接对着它写, 不必再手搓一个 main.c。

---

## 4. 这里的 "PASS" 意味着什么

每个门都是 `ex_check(label, err, tol)`:

- `err` = 实测误差 —— 数值门是逐元素最大绝对误差, 契约门是 0/1 状态码;
- `tol` = 容差。整数族 `tol = 0` (bit-exact); fp16 族 `tol = 1` (1 ULP);
  工程单元的 byte-exact 机制门 `tol = 0`;
- **门数口径**: 脚本只统计 `[PASS]` 括号行。01–14 共 41 门, 16–22 共
  42 门 (例 20 含 ser/a/b 各 1 个 `steps_dumped` 门), 23–33 共 98 门 ——
  **合计 181**。例 15 打印列式检查 (`maxrel=... tol=... PASS`),
  汇总有意不计入。

所以一行 PASS 的含义是: *在这个特定的、可复现的输入上, 在真实硅片上,
实测行为满足声明的界。* 它不是覆盖率指标, 也不是冒烟测试。

---

## 5. 33 个例子逐个讲

### 第一部分 — V2.1 算子回归 (01–15)

这批回答一个问题: **库里每个算子是否等于它的数学定义?**
全部遵循第 3 节模板: LCG 输入 → 算子 → 标量金标 → 逐元素比对。

---

#### 01_runtime_init — runtime 生命周期冒烟

**是什么。** 唯一不调任何算子的示例。它证明 runtime 层本身可用,
别的东西才敢往上叠。

**检查什么 (4 门):**

| 检查项 | PASS 判据 | 证明 |
|--------|-----------|------|
| `hmx_runtime_setup(2MB)` | 返回 0 | cDSP/fastrpc 就绪; HMX/HVX 上电; VTCM 已申请清零 |
| VTCM base 非 NULL | `get_vtcm_base() != NULL` | VTCM 真的拿到了 |
| VTCM size ≥ 申请量 | `get_vtcm_size() ≥ 2MB` | 申请被满足 (本机通常给整块 16MB) |
| 计时器单调 | 两次读非降 | HAP qtimer 可用 —— 后面所有 benchmark 依赖它 |

**为什么单独一个示例。** 任何 HMX 算子前必须 setup, 否则第一条 HMX 指令
直接 CX_FAULT。这是环境健康探针: 它挂了, 其余 32 个都跑不了,
修法几乎都在 host 侧 (重连板子、重启 adb)。

**数据准备。** 无 —— 没有 buffer。

---

#### 02_convf16_gemm — fp16 GEMM 单 tile

**是什么。** fp16 正确性的基准证明: 一次 32×32×32 fp16 矩阵乘,
跑在**真 HMX 引擎**上。

- **算子**: `hmx_convf16(act, wgt, bias, out, 32, 32, 32)`
- **数学**: `out[m,n] = bias[n] + Σ_k act[m,k]·wgt[k,n]`, HMX fp32 级
  累加器累加, `cvt.hf` 截断到 fp16
- **数据**: act/wgt 种子 7/9 scale 0.01 (值 ~±0.5, 有意避开 ±1.0 边界);
  bias 种子 11 scale 0.01
- **金标**: 每个 (m,n) 标量 fp32 累加后强转 `__fp16`
- **判据**: `maxerr ≤ 1` (1 ULP)
- **PASS 证明**: 真 HMX fp16 脉动阵列路径数值正确 —— 不是 HVX 仿真,
  不是标量回退。这是 fp16 族的基础正确性锚。

---

#### 03_convbbb_int8 — u8 × u8 → u8 GEMM

**是什么。** int8 族的基础正确性检查。

- **算子**: `hmx_convbbb(act_u8, wgt_u8, bias_i32, out_u8, 32, 32, 32)`
- **数学**: `out[m,n] = sat_u8( bias[n] + Σ_k act·wgt )` —— int32 累加 +
  u8 饱和
- **数据**: act 种子 7 range 8; wgt 种子 9 range 8; bias i32 种子 11
  (中心 200, 半宽 100)
- **金标**: 标量 int32 累加 + 手写饱和
- **判据**: `maxerr = 0` (**bit-exact**)
- **PASS 证明**: int8 路径正确。本设备 HMX int8 是 silent NOP, 库把
  int8 GEMM 切到 HVX (`vmpyacc` + `vasr_sat`)。注意整数族的 bias 是
  `int32_t`, 不是 `__fp16`。

---

#### 04_convhbh_u16 — u8 × i8 → u16 GEMM (宽动态)

**是什么。** u16 输出族的正确性检查, 一例两函数。

- **算子**: `hmx_convhbh` 与 `hmx_convhhh`, 都是 u8 × i8 → u16,
  仅 HMX 写回格式标签不同 (`:2x1` vs `:2x2`)
- **数学**: `out[m,n] = sat_u16( bias[n] + Σ_k act_u8·wgt_i8 )`
- **数据**: act u8 种子 7 range 4; **wgt int8** 种子 9 range 6; bias i32
  种子 11 (200, 100)
- **判据**: 两个函数都 `maxerr = 0` (2 门)
- **PASS 证明**: u16 输出族 (保留宽累加范围, 不钳到 u8) 正确, 且两种
  写回格式数学一致。

---

#### 05_i16_weight_convs — i16 权重族全家桶

**是什么。** 一例验 4 个 i16 权重族, 全部 `u8 act × i16 wgt`:

| 函数 | 输出 | 格式 |
|------|------|------|
| `hmx_convbcb` | u8 (饱和) | — |
| `hmx_convbnb` | u8 (饱和) | 基本型 |
| `hmx_convhch` | u16 | :2x2 写回 |
| `hmx_convhnh` | u16 | :2x1 写回 |

**判据**: 四个全 `maxerr = 0` (4 门)。

**PASS 证明**: bcb/bnb 数学等价, hch/hnh 数学等价, 全部 bit-exact。
实务提示: i16 权重带宽是 i8 的两倍 —— 模型允许就用 i8。

---

#### 06_dwconv — 深度卷积 (fp16 + u8)

**是什么。** 3×3 depthwise 卷积: 每通道独立空间卷积。

- **布局**: `act[H][W][C]` 行主序**无 padding**; `wgt[C][9]` 通道主序;
  越界样本**跳过** (clamped 不补零) —— 金标的边界 if 必须与 kernel
  完全一致
- **数据**: H=5, W=5, C=9; fp16 种子 7/9/11 scale 0.01, u8 种子 13/15/17
- **判据**: fp16 ≤ 1 ULP; u8 exact (2 门)
- **PASS 证明**: depthwise 路径 (空间卷积, 非 GEMM) 两精度都对。

---

#### 07_add — 元素级 fp16 加法 (残差)

**是什么。** ResNet 式网络的残差加法。

- **算子**: `hmx_add(a, b, bias, out, 32, 32)` —— `out = max(0, a+b+bias)`
- **判据**: Q10 定点误差 `round(|d|·1024) ≤ 1`, 即 |d| < 1/1024 ——
  比 raw ULP 更紧, 因为这条路没有深累加。

---

#### 08_divide — HVX 整除 5 变体

**是什么。** 一例验全部 5 个 HVX 除法函数, 含除零饱和。

| 函数 | 精度 | 除零 | 舍入 |
|------|------|------|------|
| `hvhx_divide_u8` | u8 | →0xFF | 截断 |
| `hvhx_floor_divide_u8` | u8 | →0xFF | floor |
| `hvhx_divide_u16` | u16 | →0xFFFF | 截断 |
| `hvhx_floor_divide_u16` | u16 | →0xFFFF | floor |
| `hvhx_divide_flat_i32` | i32 | →±INT32_MAX | **四舍五入** |

- **数据**: 每个 N=1024; 区间选择避免大量除零, 但**故意各塞 8 个
  b=0** 测饱和
- **判据**: u8/u16 exact; i32 ≤ 1 (5 门)

> ⚠️ i32 是 round-to-nearest **不是截断** —— 天真的截断金标会差 1。
> 这是历史上真实踩过的坑, 别"简化"金标。

---

#### 09_activation — HardSwish + PReLU

**是什么。** 两个 HVX 激活函数。

- `hvhx_hardswish_flat_u16` — MobileNetV3 HardSwish `x·clamp(x+3,0,6)/6`;
  u16 输入是 **int16 二补数 Q12** 的位模式; 向量路径用 1/6 ≈ 2731/16384
  近似, 容差 2 LSB
- `hvhx_prelu_u8` — 偏移二进制 u8 (零点 0x80) 域的 PReLU, slope Q7,
  容差 1

> ⚠️ HardSwish 输入是 `int16` 不是无符号 —— 金标必须按 `(int16_t)in`
> 解读。

---

#### 10_reduction — 沿 depth 归约 5 变体

**是什么。** 沿 depth 维归约: argmin/argmax、find-max、top-1、求和。
输入 flat `[hw][d]`, 每行出一个结果。

| 函数 | 输出 |
|------|------|
| `hvhx_argminmax_depth_crouton_b` | 每行 min/max + 下标 (u8) |
| `hvhx_argminmax_depth_flat_h` | 同上 (u16) |
| `hvhx_find_max_and_index_in_depth_b` | 每行 max + 下标 |
| `hvhx_top1_qu8_dLE32_cr2flt` | 每行 top-1 (值+下标) |
| `hvhx_reducesum_depth_u8` | 每行求和 |

**判据**: 五个全 exact。

> ⚠️ 名字里的 "crouton" 指**内部**处理方式; **输入仍是 flat 行主序**,
> 不要打包。

---

#### 11_lookup_unpack — 查表 + 权重解包

**是什么。** 两个 HVX 数据搬运算子。

- `hvhx_table_lookup_flat_u8` — `out[i] = table[idx[i]]`, 256 项 LUT
- `hvhx_unpack_weights` — 4-bit→8-bit 解包: `out[2i]=(in[i]>>4)&0xF`,
  `out[2i+1]=in[i]&0xF`

**判据**: 两个都逐字节 exact。这是"库的免 vgather (unsigned PD 安全)
查表策略没退化"的哨兵门。

---

#### 12_multitile_gemm — 大尺寸 fp16 GEMM

**是什么。** `hmx_convf16` 在 M/N/K > 32 时的库内多 tile 循环, 4 种
维度组合:

| run | 维度 | 压力点 |
|-----|------|--------|
| 1 | 64×32×32 | M > 32 |
| 2 | 32×32×64 | N > 32 |
| 3 | 32×64×32 | K > 32 (累加器跨 tile) |
| 4 | 64×64×64 | 全维度 > 32 |

**判据**: 四个全 ≤ 1 ULP。证明的是**公开 wrapper 的多 tile 正确性** ——
单次 `clracc` 的 K-loop 跨 tile 累加是历史上最容易 off-by-one 的地方。

---

#### 13_compat_dlsym — 老工程兼容层 dlsym

**是什么。** 运行时符号解析测试: `dlopen` V2.1 时代的
`libhvxhmx_v2.so` (由构建脚本拷入工作目录), `dlsym` 6 个符号
(4 个 v73 命名 + 2 个 v81 新几何), 逐个调用并与例 03 的标量金标比对。

**判据**: dlopen 成功 + 6 符号全 exact (7 门)。

**PASS 证明**: (a) 库能干净加载 —— 无不可解析 UNDEF (unsigned PD 会
静默拒载); (b) v73/v75/v79 兼容 wrapper 可用; (c) 老工程用
dlopen+dlsym 集成的方式继续可用。运行时必须带
`ADSP_LIBRARY_PATH`/`CDSP_LIBRARY_PATH` 指向工作目录, 否则 dlopen
目标找不到。

---

#### 14_hmx_peak_gemm — 裸 HMX K-loop 达峰值 (高级)

**是什么。** 算子层的旗舰性能示例。同一个 M=32 N=32 K=256 fp16 GEMM
跑两条路:

1. **裸 K-loop**: NK=8 个 act/wgt 32×32 slice **一次性**打包进 VTCM
   crouton, 然后手写循环 `clracc → bias → 8 条背靠背
   activation.hf/weight.hf (共享一个累加器) → cvt`。这是逼近 HMX
   硬件峰值的文档化方法。
2. **公开 `hmx_convf16` wrapper**: 同尺寸对照。

- **正确性门**: 裸 K-loop vs 标量金标 ≤ 1 ULP —— 低级路径**数值也对**,
  不只是快
- **吞吐** (52f67807 实测): 裸 K-loop ≈ **12.34 TFLOPS** (0.04 µs/call)
  vs wrapper ≈ 1.9 GFLOPS —— 差距 ~6300×, 因为 wrapper 每 tile 重新
  gather+pack, 阵列被饿死。"大 GEMM 要吞吐就预打包 + 裸 K-loop"
  这条教训, 正是例 17 W4A16 引擎的实现方式。调大 `NK` 宏可摊薄
  clracc/cvt, 逼近 ~20.4 TFLOPS 理论峰。

---

#### 15_v2_llm_ops — V2 LLM 算子层全量回归

**是什么。** V2.1 **算子族回归**: LLM 推理实际调用的 `hvhx_v2_*` 层
(60 个符号) —— rms_norm 族 / l2_norm / sqrt / sqr / sigmoid / tanh /
exp / log / scale / inverse / mul / softmax, 外加一条微型端到端管线
(manual_pack + dequant + GEMM + manual_extract、传输往返、残差输出)。

- **输出格式**: 与其他所有示例不同, 它打印**列式检查清单** ——
  `名字 maxrel=... tol=... PASS`, 一行一项, 以 `--- ALL PASS ---` 收尾。
  这些行**不计入** 181 门总数, 该示例单独记账。
- **容差**: f32 相对误差各族 0.01–0.02; 微型管线的 GEMM 段放宽到
  0.06 (反量化路径)。
- **计时行**: 9 条 `PERF` 行: rms_norm_mul n=1024 ≈ 0.30 µs
  (~44 GB/s 有效带宽), mul ≈ 0.12 µs (~102 GB/s), sigmoid ≈ 0.95 µs,
  softmax ≈ 1.52 µs。小尺寸由调用开销主导; mul 的数字已接近 HVX
  搬数带宽。
- **PASS 证明**: LLM 代码真正链接的便利层, 整个面上数值都过硬,
  不只有旗舰 kernel。

---

### 第二部分 — V2.2 工程单元 (16–22)

V2.2 保留全部 V2.1 算子 (01–15 原样跑新库), 新增从闭合模块移植的单元。
它们的 PASS 含义是"**机制**在硅片上正确", 通常是对独立 oracle 的
byte-exact。手册: `../docs/api_v22_*.md`。

---

#### 16_wtcache_pin — U1: VTCM 权重缓存 (pin + ring)

**是什么。** LLM decode 的内存策略: 权重**pin** 进 VTCM 一次
(DMA 进来, 常驻, 每步复用), 每步变化的激活走深度 4 的预取 **ring**
(DDR→VTCM 进, VTCM→DDR 出)。

**检查什么 (9 门):**

1. 三块不同大小权重 (128K/64K/96K) 分别 pin, 各自与 DDR 源
   **逐字节**校验 (3 门);
2. 三个 pin 槽地址互不相同; 槽落在声明的 pin 区内;
3. 一块 1MB 权重 pin + 校验, 顺带计时 → DDR→VTCM 带宽 (~16 GB/s:
   1MB 约 64 µs);
4. 8 块 tile 走 4+4 ring: 每块预取内容 byte-exact, drain 后每块回搬
   结果与源一致 (2 门);
5. ring 全部流量跑完后, pin 住的权重**再校验仍完好** —— ring 活动
   绝不允许踩坏 pin 区 (T6 教训)。

**值得记住的契约。** `wtcache_ring_next` 的第 3 参是**本轮**输出槽的
DDR 目标 —— ring 内部挂起 (pending) 到下一轮或 drain 才真正提交,
保证调用方先写完 DMA 才读。传**上一轮**的目标会让所有搬移错位一拍 ——
本例就是把这个真实 bug 永久钉住。

---

#### 17_w4a16_gemm — U4: W4A16 HMX 引擎 256³

**是什么。** W4A16 引擎 (4-bit 权重 × 16-bit 激活, 库里的主力 GEMM)
的永久回归锚。

**检查什么 (3 门):**

1. 标准 256×256×256 形状跑一次引擎; 输出面是 crouton16_row4, 所以先
   **解码成线性 (M,N) 矩阵** (`minv_crouton`, host `inv_crouton16.py`
   的设备镜像); 解码结果必须与 `Y_gold_2563.raw` **逐字节一致** ——
   256³ 位恒等闭合的存档 oracle: **65536/65536 元素**;
2. 同输入再跑一次, 输出 byte-exact (确定性);
3. 100 次计时 → 中位延迟 (~26 µs, 小形状约 1.29 TFLOPS) 供报告。

**金标配对陷阱。** `Y_gold_2563.raw` 对应的输入是 `act_surface.raw`
(t10 基准面)。同目录的 `act_variants/v0.raw` 是随机变体 (逐位相同仅
~5.7% —— 喂它看起来像引擎坏了, 其实是输入拿错); `Y_ref_v0.raw` 是
int8 全精度标量金标, 与 w4 引擎**没有**对应关系。t10 闭合从未拿它
做门, 本例同样不做。

---

#### 18_smallm_gemv — U8: 小 M GEMV 的 pad-256 路线

**是什么。** decode 形状的标准答案: LLM decode 每步只有 M=1..16 个
token, 但 W4A16 的 tile 结构 (`m_t=8`) 硬性要求 **M 是 256 的倍数**
(M=32/128 实测失败)。工程解法是 **pad-256**: 激活面补到 256 行
(补行填中性值 32768, 即对称量化的零点), 跑全引擎, 取真实行。

**检查什么 (5 门, 全部在解码后的线性输出上判):**

1. M=1(补) 的行 0 == M=256 全量跑的行 0 —— 补行不污染真实行;
2. M=16(补) 的行 0..15 == 全量跑的行 0..15;
3. 两个不同输入的 pad 行输出完全一致 (pad 行真的是中性的);
4. 行 0 随输入变化 (不是常数);
5. 成本门: M=1 的中位 invoke 与 M=256 同价 (实测 Δ = 0%) ——
   tile-walk-bound 性质, pad 路线对 GEMV 在**经济上**也是对的。

---

#### 19_gdn_sm — U5: GDN 循环状态机

**是什么。** GDN (Gated DeltaNet) 四族 kernel 对各自的标量 oracle,
输入用 LCG (host 可复现, 无资产传输)。对循环状态机来说, "对"的
含义是 **跑 100 步之后状态仍然对** —— 累积漂移才是头号敌人。

**检查什么 (8 门):**

1. 2 万个 f16 (含 subnormal) 的 f16→f32→f16 往返幂等 (历史事故点);
2. conv 步进 vs oracle (cos ≥ 0.9999); 块 conv 状态与逐 token 步进
   状态 byte-exact;
3. 带守卫带的 bit-exact 重跑: 状态区外围 64KB 守卫带跑完必须原封不动
   (越界写检测);
4. 100-token delta-rule 块循环 vs 逐 token oracle, 输出与终态分别
   设 cos 门;
5. solve-tri vs 回代 oracle;
6. 切分自由度: 长度 16 的块按 8+8 跑 == 整块跑。

---

#### 20_dualdomain — U7: 两个 CDSP 保护域, 一个二进制

**是什么。** **分片执行器**: `run_main_on_hexagon <dom>
test_20_dualdomain.so dd <tag> <start> <len>` (注意第 4 参是**长度**
不是结束下标 —— `dd b 8 16` 会报 range 错, 应为 `dd b 8 8`)。

**编排 (构建脚本自动做)**: 先在域 3 跑 `ser` (16 步串行基线); 然后
`a` (域 3, 步 0–8) 与 `b` (域 4, 步 8–16) **并发**。每步算完写 128KB
输出, 用设备端 sha256 哈希, 删 dump。host 侧 `analyze_dd.py` 证明
**切分等价契约**: a/b 每步哈希 == ser 对应步哈希 (门), 另有各段
`steps_dumped` 门 (3 个) 和步数齐全门。

**实测**: ser ≈ 545 µs vs a/b ≈ 272/274 µs → **~2.0× 加速**
(只报告; 调度抖动使它在 1.99–2.00 之间小幅波动)。

**PASS 证明**: 把有状态管线切到两个 PD 上跑, 与串行**逐位等价** ——
"扩并发靠加域不靠加线程"这条结论 (对比例 22) 的地基。

---

#### 21_oplist_exec — U6: blob 解析 + 执行, 带独立金标

**是什么。** op 列表引擎: host 把一段 op 序列 (PIN 权重 → MATMUL →
RMSNORM → …) 连同权重打包成 `.wtop` blob, 设备端一次 `wt_exec_run()`
执行整段, 每个 op 单独计时。

**检查什么 (6 门):**

1. **负例解析**: 5 份故意改坏的 blob 拷贝 (魔数/版本/端序标志/槽数/
   op 数) 必须各被拒绝且返回**精确的错误码** —— 错误路径也是契约;
2. 设备产出 blob 的 W3 报告行; 构建脚本将其与 **host `wt_inspect`
   工具的输出逐行 diff** (两侧同一份解析器源码 —— 任何差异都是传输
   bug);
3. `blob_w4` (单个 MATMUL) 执行; 然后 `blob_w5` 跨引擎**重新初始化**
   跑, 输出必须 byte-exact (引擎复用可靠);
4. 解码后的 MATMUL 输出与 **K2560 独立标量金标**之差 ≤ 40 LSB
   (闭合模块标称值 37 LSB);
5. RMSNORM 阶段与例内嵌的同算法标量镜像 **0 ULP** 逐位一致;
6. 全序列跑完。

**计时读法**: w4 MATMUL op ≈ 24.7 ms, w5 ≈ 46.9 ms 全路径 (权重
restage + DMA + 计算) —— 与例 17 纯 invoke ~26 µs 之间的差值,
正是 U1 wtcache 存在的理由。

---

#### 22_dualcore_threads — U3: 线程买的是正确性, 不是吞吐

**是什么。** 同域并发结论的永久回归。两个 `dc_spawn` worker
(显式 64KB 栈)、起步 barrier、从同一 arena 切出的两个引擎、一把共享
DMA 锁、一次 VTCM 旗标握手 (0→1→2)。

**检查什么 (6 门):**

1. 两个并发引擎输出与串行参考 byte-exact (2 门);
2. 三个纯 HVX 算子 (`dc_hvx_load`/`dc_norm_i16`/`dc_dot_u64`) 与主线程
   值一致 —— 跨线程确定性 (3 门);
3. 旗标握手完成。

**本例的灵魂 —— hmx_lock 交接。** HMX 锁是**持有线程**属性: 哪个线程
invoke HMX, 哪个线程就必须持锁。模板:

```
主线程:  wtcache_hmx_unlock()      ← 让锁 (open 起主线程持有)
         spawn workers
worker:  wtcache_hmx_lock() → invoke → wtcache_hmx_unlock()
主线程:  join; wtcache_hmx_lock()  ← 取回
```

不交接的后果是 **PD 直接 crash** (返回 ≈ -2147482611), 不是 FAIL 行。

**实测**: 并发/串行 ratio ≈ 0.96 —— 略**慢**, 与闭合模块结论完全一致
(单 DMA 引擎 + 全局 HMX 锁)。吞吐扩容走例 20 的域; 线程买到的是并发
**正确性**。

---

### 第三部分 — V2.3 工程单元 (23–33)

第二批: 基础设施单元 (fence / arena / harness / wpool)、数值桥
(pxbridge / gdntree)、推理状态管理 (kvcache / graphstep /
gemmdispatch)、完整性机器 (rbr / bledger)。手册: `../docs/api_v23_*.md`。
依赖: fence/arena/harness/pxbridge/rbr/bledger 独立; wpool 与 kvcache
建在 fence 上; gdntree 建在 fence+arena+pxbridge 上; graphstep 建在
wpool 上; gemmdispatch 建在例 17/18 的引擎上。

---

#### 23_fence — U9: 方向对偶的 cache fence 决策表

**是什么。** V2.2 的"cache 四铁律" (CPU→DMA 要 FLUSH; CPU 写 VTCM 要
FLUSH; DMA 写完要 INVALIDATE; 退出要 close) 收敛成**一个 API**
(`fence_handoff`), 背后是一张 (写者, 读者, 内存域) 的决策表。V2.3
所有新代码的 cache 维护都走这个单元, 不再手写 `qurt_mem_cache_clean`。

**检查什么 (7 门):**

1. **决策表**: 32 个 (写者, 读者, 域) 组合经 `fence_op_for` 解出的
   操作与内置期望表全等; "HMX 写 DDR"这个无意义格恒被拒 (`FO_INVALID`);
2. **非法组合**被参数域检查拒绝;
3. **CPU→DMA 走 DDR** (铁律①): 200 轮变长 LCG 模式, 加 fence 后
   DMA bypass 搬 DDR→VTCM, 逐字节比对;
4. **CPU→HMX→CPU 走 VTCM**: W4A16 引擎跑 20 轮 —— 同输入每轮
   bit-exact, **且**不同输入必须出不同结果 (输入敏感性, 专抓
   "缓存残留读"这种纯往返测不出来的错);
5. **CPU→HVX 走 VTCM**: memset 模式对 HVX 校验和可见 (模式敏感,
   cache 残留会露馅) 且往返复现;
6. **CPU→DMA 走 VTCM 反方向** (move-back): VTCM→DDR 逐字节一致 ——
   契约里 src FLUSH 的那一半。

**PASS 证明**: fence 表对每个合法方向对都正确编码了 cache 协议, 而且
每条 fence 在硅片上**真做了物理动作** (bypass DMA 读回是证据, 不只是
"没崩")。

---

#### 24_arena — U10: 双池对齐 arena

**是什么。** 分配器纪律单元: DDR 池 + VTCM scratch, 带对齐保证、free
与合并 —— 因为热路径不允许每次调用 `memalign`/`free` (闭合模块实测
+38 ms 分配抖动的真实 bug)。

**检查什么 (4 门):**

1. **1000 轮**随机 size/align 的 alloc/free: 每个返回指针满足请求对齐
   (128B 或 2KB), 活指针账本恒对;
2. **无泄漏、全合并**: 全部释放后 `used == 0` 且
   `largest_free == capacity` —— 无碎片残留;
3. **混合租户共存**: f32 GDN 状态与 2KB 对齐的 W4A16 面交错分配/释放
   100 轮 —— GDN 状态内容绝不被面的分配破坏 (防"一个租户的 free
   合并把别人还逻辑持有的块发出去"这类 bug);
4. **碎片上界**: 随机搅动后释放 90%, `largest_free ≥ 80%` 容量 ——
   合并真的在发生。

---

#### 25_harness — U11: 金标 case 框架

**是什么。** 元示例: 把"一个有输入、有 runner、有 oracle 的 case"
固化成带 sha 钉死记录的框架 (`harness.h`)。它用**把两个先前手工闭合
的 case 重跑一遍并要求结果一致**来证明自己。

**检查什么 (13 门):**

1. case `gdn_sm` (例 19 核心): oracle cos 门 + 框架内直跑与手写版
   逐字节一致 (首跑与复跑共 4 门);
2. case `w4a16_gold` (例 17 核心): `act_surface` → 解码 vs
   `Y_gold_2563` **65536/65536 byte-exact** (同样含复跑);
3. case `solve_tri`: 纯标量 case, 证明框架不依赖引擎也能跑;
4. **sha 确定性**: 同 case 两轮完整运行 `harn_last_sha` 完全一致 ——
   输出被逐位钉死, 将来任何回归都会翻哈希。

**PASS 证明**: 新金标 case 对着 harness 写, 能精确复现手写闭合的
结果 —— 且内容哈希稳定, 可直接当回归信号。

---

#### 26_wpool — U12: 常驻工人池

**是什么。** 并发开销的工程答案: spawn-per-op 的线程创建成本 90×+
(op82 结论), 所以 job 交给**常驻 worker**。池自带 hmx-lock 交接,
引擎 job 安全。

**检查什么 (5 门):**

1. **随机到达**: 24 个混合 job (norm/dot/hvxload) 按随机序投递 ——
   每个池执行结果与主线程串行参考逐值一致 (并发只改时序, 永不改数值);
2. **HMX 门**: 2 个引擎 job 走 unlock→job-lock→relock 交接, 输出与
   串行 invoke byte-exact —— 例 22 的模板, 产品化;
3. **spawn16**: 16 个 norm job 走池全部值精确;
4. **经济学门**: 池路径必须快过 spawn-per-op —— 实测同样 16 个 job
   **109 µs vs 484 µs (4.4×)**;
5. **压力**: 5 轮 × 12 job; `done`/`executed` 计数 == 总投递,
   每轮数值全对。

---

#### 27_pxbridge — U13: 精度桥, unipolar 契约

**是什么。** f32 / f16 / 对称 INT16 (zp = -32768) 三者之间的转换层 ——
每个量化管线都要过的边界。本例**在 4 个 scale 下**跑完整契约
(所以 20 门: 4 scale × 5 检查, 加线性、端点、批量)。

**检查什么:**

1. **f32→f16→f32 往返**: 10 万随机值, 误差在 0.5 ULP 包络内
   (含次正规邻域);
2. **INT16 对称解码**: 误差 ≤ scale/2 (半步包络), 且零 ⇔ 码 0x8000
   精确成立 —— 零有且只有一个码;
3. **负输入钳到零码** (unipolar 契约: 这座桥永不解码出负值);
4. **f16↔i16 组合桥**: 组合转换与直接解码之差 ≤ scale/2 + 半 ULP;
5. **码空间线性**: 相邻码解码差恒 == scale, **全部 65535 步**;
6. **端点钳位**: ±溢出输入落 0xFFFF/0x0000, 解码单侧有界;
7. **批量 == 标量**: 向量 API 与标量 API 逐字节一致。

---

#### 28_gdn_tree — U14: 树形 GDN, 闭式解 vs kernel

**是什么。** 树形 GDN 求值 (draft 模型的形状): 节点构成森林, 每个子
节点在自己的提交状态上做衰减。单元同时携带**三套互相独立的实现**:
串行递归 oracle、闭式解、设备 f16 kernel。

**检查什么 (8 门;** T = 8/16/32 × 3 个随机拓扑, D = 64, LCG 可复现**):**

1. **闭式解 vs 串行 oracle** (全 f32): 输出**和**逐节点提交状态
   cos = 1.0000000 —— host 数学闭合 (2 门);
2. **设备 f16 kernel vs 闭式解**: 输出 cos ≥ 0.999, 逐节点状态
   min cos ≥ 0.999 (2 门);
3. **kernel 复跑 bit-exact**;
4. **拓扑校验**: `parent[0] ≠ -1` 与 `parent[i] ≥ i` 都被拒 ——
   违序 DAG 不会被静默算错;
5. **INT16 衰减曲线**: 量化逐步衰减积 vs f32 `exp(路径和)` 参照, 保持在
   `depth·6e-4 + 1e-3` 内, 且任意深度不劣于 f16 路径 —— 深树的量化预算;
6. 状态缓冲走 U10 arena, 退出**全释放**。

---

#### 29_kvcache — U15: KV 槽缓存

**是什么。** token KV cache 的槽式管理 —— decode 循环依赖的推理状态
单元。host **影子模型**镜像每一个操作, 每个门都是与影子的字节级或
计数级比对。

**配置**: 64 槽 × 256B (K 128B + V 128B)。

**检查什么 (8 门):**

1. 非 128 倍数的槽尺寸在 init 时被**拒绝**;
2. **append 64 个位置**: 逐槽 K/V 字节与位置图和影子一致; 每次
   `lookup` 全 hit;
3. **回绕动态** (posIdsIdx): 3N 次 append 进 N 槽后, 旧位置 (<2N)
   miss、新位置 hit, **evict 计数精确**;
4. **scatter 重写隔离**: scatter 只动自己槽的字节与 posmap 项 ——
   邻槽金丝雀完好; 非法槽号被拒;
5. K 面与 V 面的 **128B 对齐**恒成立;
6. **DMA bypass 真读回**: append (含内置 fence) 后用真
   `dc_dma_once` 读一个槽的 K 面 —— 与影子逐字节一致。cache 铁律
   是被物理验证的, 不是假设;
7. **verify 重写语义**: 同槽再 scatter (目标校正) 后 read = 校正值,
   且 `lookup(pos)` 仍 hit。

> **pos ≡ slot (mod N) 同余陷阱**: 位置必须与槽号同余, 否则 lookup
> 永远 miss —— 测试位置用 `64k + slot` 形。

---

#### 30_graph_step — U16: 整步执行 vs 逐算子下发

**是什么。** graph-step 引擎: 一段 op 列表要么**融合**跑 (一次
`wt_exec_run`), 要么**分段**跑 (逐 op `run_range`), 两者的结果必须
不可区分。本例的 blob 是**在设备上**从 s256 资产直接合成的 ——
不依赖 host 打包器 —— op 序列 `NOP / PIN(wt) / MATMUL / PIN(wt) /
RMSNORM / SILU` (覆盖新 `OP_SILU_F16 = 4`)。

**检查什么 (8 门):**

1. `wt_parse` 接受合成 blob (v1, 含 SILU);
2. 融合跑完成;
3. **统计精确**: ops=6, nop=1, matmul=1, rmsnorm=1, silu=1, pin=2,
   skipped=1 —— 计数器如实讲述"跑了什么";
4. **PIN-skip 语义**: 引擎未建立时 PIN 只记账 (`skipped`); 引擎建立后
   PIN 真正 staged。首跑 (融合) → skipped=1; 二跑 (分段, 引擎已建)
   → skipped=0;
5. 分段跑完成;
6. **融合 vs 分段: 三张 temp 面逐字节恒等** —— 核心等价契约;
7. 二跑 PIN-skip 计数归零;
8. **SILU vs 标量 oracle** 在 f16 半 ULP 包络内。

融合 vs 分段总耗时 (和 per-op 表) 只报告不设门。

---

#### 31_gemm_dispatch — U17: MatMul 三路由决策

**是什么。** 所有矩阵乘的前门: 按形状在三条执行路由中选一条的路由器,
加上各路由的执行体。"kernel 单 invoke ABI 固定 M=256"这个现实就是
在这里被消化的。

**路由:**

| 路由 | 形状 | 执行体 |
|------|------|--------|
| `GR_SMALLM` | M = 1 (decode GEMV) | pad-256 + 全引擎, 取行 0 |
| `GR_DENSE_F16` | 小 M, 小 K/N | dense f16 GEMM |
| `GR_W4A16` | M = 256 / 512 (256 倍数) | W4A16 引擎, 按 256 行分块 |

**检查什么 (8 门):**

1. **决策表**: M 扫 1..600 × K/N ∈ {64, 256, 2560} —— 1800+ 个决策
   与路由规则逐点一致, 0 mismatch; 边界钉死: M=1 SMALLM,
   M=32/128 DENSE, M=256/512 W4A16;
2. **GR_W4A16 256³**: 全引擎 vs `Y_gold_2563` byte-exact
   (65536/65536) —— 例 17 的锚, 经路由器同样可达;
3. **M=512 拆块**: dispatch 层拆成 2×256 块 (kernel descriptor 硬编码
   M=256; 512 面单发高位行全错 —— V2.3 教训) —— 上半 == 金标 A,
   下半 == 金标 B, 双 byte-exact; 另有 crouton512 编解码往返恒等;
4. **GR_SMALLM**: M=1 行 0 == 全引擎行 0 == 金标行 0 (两门);
   成本门 —— M=1 中位 (1510 µs) 与 M=256 (1028 µs) 之差 < 50%,
   实测 ratio 1.47;
5. **GR_DENSE_F16**: M=32/128, K=N=64 随机 f16 vs f32 累加标量 oracle:
   cos ≥ 0.9999 且 max|Δ| ≤ 半 ULP 包络。

---

#### 32_rbr — U18: 循环状态的部分接受回退

**是什么。** 有状态引擎的投机解码安全网: 当一轮树只接受了部分草稿
步时, 引擎状态必须回滚到已接受前缀并从那里重放 —— 且 16 轮混合接受
之后, 引擎必须与诚实串行跑**逐位**相同。实现
`docs/P0-recurrent-state-rollback.md` 的验证金字塔, 用确定性 f32
模拟引擎 (每 group conv 状态 + recurrent 状态), 让 bitwise 判定有意义。

**三层 (8 门):**

- **L1 模块契约**: 快照往返把故意污染的状态逐字节恢复; SKIP 是
  **一次性**的, 且被每一个重写 in-state 的分支消费/作废 (CLEAR 分支
  曾把挂起的 SKIP 漏进下一帧 —— 本门钉死这个真实 bug); NOSNAP /
  STALE / FROZEN / 越界 rewind 全部被拒;
- **L2 轮级断言**: **缺陷指纹** (等式 1: `replay_in == phantom_out`)
  —— 一个故意有缺陷的引擎 (setup 时无条件 out→in 拷贝) 被指纹**检出**
  且偏离基线; restore 生效 (等式 2: `replay_in == snapshot_src`
  逐字节); 重放的 KV 行覆写幂等;
- **L3 端到端**: 16 轮混合接受 (含 j=0 强制全拒与全接受流) —— 引擎
  终态 + KV 行 + **两套 n_past 账本** 与非投机基线 bitwise 相等;
  计数器精确 (16 快照, 零触碰流 restore=rewind=6, skip=5)。

---

#### 33_bledger — U19: Buffer Ledger, 数据流溯源审计

**是什么。** "消费从未生产的数据立即报错"的单元: 每个缓冲行携带溯源
信息 (哪个 writer 生产的、带哪个量化 tag), ledger 在**消费时刻**就
报出经典的未初始化读 bug 家族, 而不是让垃圾静默传播。按
`docs/P1-uninitialized-read-dataflow.md` 的测试模板构建, 模拟成一条
logits 管线 (`[rows][VOC]` f32, 写入者 W_PREFILL / W_TREE / W_DECODE /
W_LATE, hash 噪声分布 + 确定性大峰, rank/argmax 判定稳定)。

**检查什么 (9 门):**

1. **契约表**: (写者/读者/状态) 组合上的完整判决定矩阵
   (OK/NEVER/CANARY/RELEASED/WRITER/QTAG);
2. **T1 冷启动**: 消费未生产行 → 立即 `NEVER`; 写入并收敛链后修复;
3. **T2 行错位**: verify = NEVER **且 E2 rank 反查能定位到错位行**;
4. **T3 canary 巡检**: canary 后读报 `CANARY` 且数据呈 0xAA 填充模式,
   加全行深 rank (区分 H2 与 H1a);
5. **T4 归还后禁读**: release 后消费 → `RELEASED`, 证据是归还前的
   canary 位样;
6. **T7 双写检测**: 未读行被**异 writer** 覆写会被计数; 读后覆写合法;
7. **T5+T6**: 新路径无写入就消费 → NEVER; qtag 读写不一致 → QTAG;
   收敛到单一配置源后通过;
8. **T8 端到端听诊**: 复刻 P1 §5 事故 —— 垃圾 draft 引擎 (d1) 配
   正确验证器 (d2), 接受率直方图崩塌到全 1、每轮 break; 修复引擎
   breaks=0, 回归锚 (首树提议 argmax == prefill 末行 argmax) 成立,
   直方图全 3;
9. **L1 timeline**: 文本时序图 (P1 §4.4) 非空且含统计。

---

## 6. 全示例容差矩阵

| 精度族 | 容差 | 涉及示例 |
|--------|------|----------|
| 整数 GEMM (bbb/hbh/bcb/bnb/hch/hnh) | **0 (bit-exact)** | 03, 04, 05, 13 |
| u8 depthwise / 归约 / 查表 / 解包 / u8·u16 除法 | **0 (bit-exact)** | 06, 08, 10, 11 |
| HVX i32 除法 (四舍五入) | ≤ 1 | 08 |
| HardSwish (Q12 近似) / PReLU | ≤ 2 LSB / ≤ 1 | 09 |
| fp16 GEMM/depthwise/add/K-loop | ≤ 1 ULP (add: < 1/1024) | 02, 06, 07, 12, 14 |
| V2 LLM 算子 (f32) | rel ≤ 0.01–0.02 (GEMM 段 0.06) | 15 |
| W4A16 vs 存档 oracle / pad-256 / 引擎复用 | **0 (byte-exact)** | 17, 18, 20, 22, 26, 31 |
| W4A16 vs K2560 独立标量金标 | ≤ 40 LSB (标称 37) | 21 |
| RMSNORM vs 同算法镜像 | **0 ULP** | 21 |
| SILU vs 标量 oracle | ≤ f16 半 ULP | 30 |
| GDN conv/delta vs oracle | cos ≥ 0.9999 | 19, 25 |
| GDN solve-tri vs oracle | cos ≥ 0.99995 | 19, 25 |
| gdntree 闭式 vs 串行 / kernel vs 闭式 | cos = 1.0 / ≥ 0.999 | 28 |
| pxbridge f16 往返 | ≤ 0.5 ULP 包络 | 27 |
| pxbridge INT16 对称 | ≤ scale/2 (半步) | 27 |
| gemm M=512 拆块 vs 独立金标 | **0 (byte-exact)** | 31 |
| dense f16 vs 标量 oracle | cos ≥ 0.9999 + ≤ 0.5 ULP | 31 |
| kvcache vs host 影子 / wpool HMX / graph_step temp | **0 (byte-exact)** | 26, 29, 30 |
| rbr 全部 (L1/L2/L3) | **0 (vs 基线 bitwise)** | 32 |
| bledger 状态机 / 计数 | **0 (精确匹配)** | 33 |

---

## 7. 一键全量运行

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh all          # 编译/签名/部署/运行全部 33 个
./build_examples.sh 20           # 单例 (数字或全名)
```

每个示例把结果写到设备, 脚本统一拉回 `../results/` (例 20 另存
ser/a/b 三段原始日志和 host 对拍文件; 例 21 追加 host vs 设备 W3
diff 门)。预期末尾横幅:

```
=== summary (results/) ===
  TOTAL                            181 pass   0 FAIL
=== ALL GREEN ===
```

看单个结果:

```bash
grep -E 'PASS|FAIL' ../results/32_rbr.txt
```

有 FAIL 时: [TEST_CASES.md](TEST_CASES.md) 有每例的完整门列表,
[../PERF_REPORT.md](../PERF_REPORT.md) §V2.3 收录了全部硬教训
(UNDEF 拒载 / skel 版本 / hmx_lock 交接 / crouton 解码 / 一次性标志 / …)。

---

## 8. 写一个自己的示例

拷一个现成的 `main.c` 当模板 ([`01_runtime_init/main.c`](01_runtime_init/main.c)
最小; V2.3 单元风格参考 [`24_arena/main.c`](24_arena/main.c))。配方:

1. `#include "hvxhmx_v23.h"` 和 `#include "example_util.h"`。
2. buffer 用 `static ... __attribute__((aligned(128)))` 全局数组 ——
   绝不用 VLA (DSP 栈很小; 128B 对齐是 HVX 载入的硬要求)。
3. runtime 就绪: 裸算子用 `hmx_runtime_setup()`; 单元代码用单元上下文
   (`wtcache_open` / arena / fence)。invoke HMX 的线程必须持锁。
4. `ex_open_result("your_name")` 开头, `return ex_summary()` 结尾 ——
   且 **PASS 和 FAIL 两条路都要** close/teardown 上下文。
5. 金标独立写 (标量循环 / host 影子模型 / 存档产物), 逐元素比对;
   容差从上面矩阵里挑。布尔条件写成 `cond ? 0 : 1`。
6. 输出如果来自 W4A16 引擎, 先解码 crouton 面 —— 绝不直接比 surface
   字节、绝不用"行偏移"。
7. 把示例加进 [`build_examples.sh`](build_examples.sh) 的 `EXAMPLES`
   数组, 然后 `./build_examples.sh your_name`。

编译/链接/签名/部署/运行流程与库自身完全一致 —— 精确的
`hexagon-clang` 命令行见 [`build_examples.sh`](build_examples.sh)。
