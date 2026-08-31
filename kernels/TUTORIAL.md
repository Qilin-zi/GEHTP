# hvxhmx_libsV2.2 新手上机完整教程
# —— 从零开始: 上电、推包、设环境变量, 跑通全部 22 个例子

> 读者对象: **从来没有上过机的新手**。你不需要懂 Hexagon DSP 的内部细节,
> 只要会打开终端、敲命令、看输出, 就能跟着这篇教程把 22 个例子在板子上全部跑起来。
>
> 每一步命令都给出**预期输出**。跑出来的结果和教程里不一样时, 先翻到
> [第 9 章 常见错误排查](#9-常见错误排查-faq), 那里收集了本项目实际踩过的每一个坑。

---

## 目录

- [第 0 章 阅读指南](#第-0-章-阅读指南)
- [第 1 章 背景知识: 我们到底在干什么](#第-1-章-背景知识-我们到底在干什么)
- [第 2 章 上机前的概念地图: 22 个例子全景](#第-2-章-上机前的概念地图-22-个例子全景)
- [第 3 章 host 环境准备](#第-3-章-host-环境准备)
- [第 4 章 板子上电与连接](#第-4-章-板子上电与连接)
- [第 5 章 第一次手动推包 (把一个例子亲手跑起来)](#第-5-章-第一次手动推包-把一个例子亲手跑起来)
- [第 6 章 一键脚本 build_examples.sh 全解](#第-6-章-一键脚本-build_examplessh-全解)
- [第 7 章 逐例精讲 01-15 (V2.1 算子回归)](#第-7-章-逐例精讲-01-15-v21-算子回归)
- [第 8 章 逐例精讲 16-22 (V2.2 新单元)](#第-8-章-逐例精讲-16-22-v22-新单元)
- [第 9 章 常见错误排查 (FAQ)](#第-9-章-常见错误排查-faq)
- [第 10 章 进阶原理 (选读)](#第-10-章-进阶原理-选读)
- [第 11 章 附录](#第-11-章-附录)

---

# 第 0 章 阅读指南

## 0.1 这篇教程会教你什么

读完并动手跟完, 你将能够:

1. **给板子上电**, 用 `adb` 连上它, 确认它处于健康可用的状态;
2. **看懂整个工程的目录结构**, 知道库、例子、文档、资产各放在哪里;
3. **手动完成一次完整的"编译 → 签名 → 推包 → 运行 → 拉回结果"**,
   明白每一条命令在干什么 (这是排错的基本功);
4. **用一键脚本**把 22 个例子全部跑一遍, 并读懂输出的每一行;
5. 逐个理解 **22 个例子分别在验证什么**、PASS/FAIL 各代表什么含义;
6. 遇到失败时, 按第 9 章的流程表**自己定位和修复**。

## 0.2 你需要准备什么

| 物品/条件 | 说明 |
|-----------|------|
| 一台 x86_64 Linux 主机 | 就是日常开发机, 本项目所有编译都在 host 上做 |
| 高通 Hexagon 评估板 (已刷好系统) | 序列号 **52f67807**。板卡型别不同批次略有差异, 以能被 adb 识别为准 |
| USB 数据线 | 连主机与板子 (调试口) |
| 主机上已装好的工具 | 见第 3 章, 逐项检查 |
| 本工程目录 | `/disk1/V81Dev/hvxhmx_libsV2.2` |

## 0.3 约定 (教程中的记号)

- `host $` 开头的行 = **在主机终端里敲的命令** (不要把 `host $` 本身敲进去);
- `device #` 开头的行 = 在板子的 shell 里敲的命令 (通过 `adb shell` 进入);
- 其余缩进块 = 命令的**预期输出**;
- `<LIB>` 指库根目录, 本文中一律是 `/disk1/V81Dev/hvxhmx_libsV2.2`,
  教程命令里直接写绝对路径, 想换路径的自己替换。

## 0.4 最重要的三条纪律 (先背下来)

1. **只使用序列号 52f67807 的板子**。其他板子 (如 f69bec03、5cd9d8c0) 即使
   `adb devices` 里显示在线, 也**绝对不要**往上面推任何东西 —— 那些板子不是
   本项目的测试环境, 推错设备轻则结果全错, 重则干扰别人正在跑的任务。
2. **所有命令先看清再回车**。尤其是带 `rm`、`adb push`、覆盖文件的命令。
3. **改库源码前先跑一遍基线**。22 例全绿是唯一可信的起点; 改完任何东西,
   用第 6 章的一键脚本回归, 别只跑一个例子就下结论。

---

# 第 1 章 背景知识: 我们到底在干什么

这一章不涉及任何操作, 只帮你建立一张正确的"世界地图"。急着上手的可以跳到
第 3 章, 但建议之后回来读一遍 —— 排错的时候这些概念会反复出现。

## 1.1 Hexagon DSP / CDSP / PD 是什么

高通 SoC 里除了 CPU (Cortex-A 系) 和 GPU (Adreno), 还有一个
**Hexagon DSP**。它有几个子域:

- **cDSP (compute DSP)**: 跑计算密集型任务的域, 我们的库就跑在这里;
- **PD (Protection Domain)**: DSP 侧的"进程"概念。我们用的是 **unsigned PD**
  (域 3 / 域 4, 即 dom3/dom4) —— 可以把它理解成 DSP 上的两个独立"沙盒进程",
  例 20 会同时在两个域里跑程序来验证"双域并发"。

CPU 上的程序通过 **FastRPC** 机制调用 DSP 上的代码: host 侧一个 loader 程序
(`run_main_on_hexagon`), 把一个 `.so` 推到 DSP 上执行, DSP 上的 `main()`
跑完把结果写回文件系统。

## 1.2 HVX 与 HMX

cDSP 上有两套向量加速硬件, 本库的名字 hvxhmx 就来自它们:

- **HVX** (Hexagon Vector eXtensions): 128 字节宽的向量 SIMD,
  擅长逐元素运算、查表、归约、搬数;
- **HMX** (Hexagon Matrix eXtensions): 矩阵脉动阵列, 擅长 GEMM
  (通用矩阵乘)。使用 HMX 前有一套固定的"上电/加锁"仪式 (见 10.2 节)。

V2.1 的例子 (01-15) 主要验证这两类硬件上的**算子数值正确性**;
V2.2 的新例子 (16-22) 验证从已闭合工程项目里沉淀下来的**工程单元**
(权重缓存、DMA、双线程、W4A16 引擎等)。

## 1.3 V81 是什么

Hexagon 处理器有版本号 (v62/v66/v68/v73/v75/v79/**v81**...)。
v81 是本项目目标芯片支持的版本。编译时 `-mv81` 决定生成什么指令集,
HVX 长度 `-mhvx-length=128B`、HMX 开关 `-mhmx` 也都是 v81 的配置。

## 1.4 VTCM: DSP 上那块珍贵的快内存

cDSP 上有一块 **128KB/核 的 L1** 和一块共享的 **VTCM** (Vector Tightly
Coupled Memory, 本设备 16MB, 地址固定 `0xff000000`)。它相当于 DSP 的
"显存": DMA 可以在 DDR 和 VTCM 之间搬数据, HMX/HVX 直接吃 VTCM 里的数据,
速度比访问 DDR 快一个量级。

V2.2 的 U1 单元 (wtcache, 例 16) 就是"怎么把权重钉 (pin) 在 VTCM 里反复用"
的工程方案。

## 1.5 SWIV 签名: 为什么 .so 推上去跑不起来

unsigned PD 有一个安全策略: **只加载签名过的动态库**。所以每个编译出来的
`.so` 都要用 `/disk1/swiv_build_utility.py` 签名, 产出 `*.signed.so`,
推到板子上的必须是签名版。签名成功的标志是输出一行
`The SWIV context is successfully generated.`

## 1.6 本工程目录结构

```
/disk1/V81Dev/hvxhmx_libsV2.2/
├── build_libs.sh              # 编译整个库 → lib/libhvxhmx_v22.so (+ 签名)
├── lib/                       # 编译产物: libhvxhmx_v22.so / .signed.so
├── src/
│   ├── runtime/  hmx/  hvx/   # V2.1 原有源码
│   ├── compat/               # v73/v75/v79 兼容层
│   └── v22/                  # V2.2 新增: 8 个工程单元的单元源 (13 个文件)
├── include/                   # 头文件 (hvxhmx.h / wtcache.h / dc_parts.h ...)
├── examples/
│   ├── build_examples.sh      # 一键: 编译+签名+推包+运行+拉结果 (01-22)
│   ├── common/example_util.c  # 所有例子共用的工具函数
│   ├── 01_runtime_init/ ... 22_dualcore_threads/   # 22 个例子, 每个一个 main.c
│   ├── README.md              # 例子索引速查
│   ├── TEST_CASES.md          # 每个用例测什么的详细说明
│   └── EXAMPLES.md            # 英文版逐例讲解
├── docs/                      # API 手册: api_v22_overview.md + 每单元一份
├── host/                      # host 侧工具 (打包/资产生成/对拍脚本)
├── assets/                    # 设备端资产 (s256 / s2560 / smallm 数据集)
├── build/                     # 编译中间产物 + host 工具 + blob
├── results/                   # 从设备拉回的每例结果 .txt (26 个文件)
└── PERF_REPORT.md             # 性能报告 (实测数字 + 基线对照)
```

记住三个位置:

| 位置 | 作用 |
|------|------|
| `lib/libhvxhmx_v22.signed.so` | 库本体, 所有例子都链它 |
| `examples/NN_名字/main.c`    | 每个例子的源码 (一个文件一个例子, 便于阅读) |
| `results/NN_名字.txt`        | 每次实跑后从设备拉回的结果 |

## 1.7 一次"跑例子"的完整链路 (先睹为快)

后面第 5 章会一步步做, 这里先给全景图。所谓"跑一个例子"是:

```
host: main.c ──hexagon-clang──▶ test_NN.so ──SWIV 签名──▶ test_NN.so.signed
                                                                  │ adb push
device: /data/local/tmp/hvxhmx22/test_NN.so  ◀───────────────────┘
        + libhvxhmx_v22.so (库, 部署一次)
        + librun_main_on_hexagon_skel.so (加载器骨架, 部署一次)
        + assets/ (数据, 部署一次)
device: ./run_main_on_hexagon 3 test_NN.so     ← FastRPC 把 .so 拉到 CDSP 执行
        DSP 上 main() 跑完 → 把 [PASS]/[FAIL] 写进 NN_名字.txt
host:   adb shell cat .../NN_名字.txt  (或脚本自动拉回 results/)
```

---

# 第 2 章 上机前的概念地图: 22 个例子全景

不用背, 跑之前扫一眼, 知道每个例子大概属于哪一族即可。
"门" = 判定 PASS/FAIL 的检查项数量 (2026-08-15 实测 83 门全绿)。

| # | 例子 | 一句话 | 族 | 门 |
|---|------|--------|-----|----|
| 01 | runtime_init | 环境探针: HMX 上电+拿 VTCM+计时器 | 冒烟 | 4 |
| 02 | convf16_gemm | fp16 矩阵乘 32³, 1 ULP | GEMM | 1 |
| 03 | convbbb_int8 | u8×u8→u8 矩阵乘, exact | GEMM | 1 |
| 04 | convhbh_u16 | u8×i8→u16 宽动态输出, exact | GEMM | 2 |
| 05 | i16_weight_convs | i16 权重族 4 兄弟, exact | GEMM | 4 |
| 06 | dwconv | 深度卷积 fp16+u8 | 卷积 | 2 |
| 07 | add | 逐元素 fp16 加+ReLU | 逐元素 | 1 |
| 08 | divide | HVX 整除 5 变体+除零饱和 | 逐元素 | 5 |
| 09 | activation | HardSwish + PReLU 定点 | 激活 | 2 |
| 10 | reduction | 沿 depth 归约 5 变体 | 归约 | 5 |
| 11 | lookup_unpack | 查表 + 4bit 解包 | 搬数 | 2 |
| 12 | multitile_gemm | fp16 大尺寸多 tile | GEMM | 4 |
| 13 | compat_dlsym | dlopen 老库 + dlsym 6 符号 | 兼容 | 7 |
| 14 | hmx_peak_gemm | 裸 K-loop 打峰值 12.34 TFLOPS | 性能 | 1 |
| 15 | v2_llm_ops | LLM 常用算子 48 项 (norm/sigmoid/...) | LLM | 48* |
| 16 | wtcache_pin | VTCM pin+ring 权重缓存 | V2.2 工程 | 9 |
| 17 | w4a16_gemm | W4A16 引擎 256³ 位恒等金标 | V2.2 工程 | 3 |
| 18 | smallm_gemv | 小 M 补到 256 的 GEMV 路线 | V2.2 工程 | 5 |
| 19 | gdn_sm | 门控 DeltaNet 状态机 4 kernel 族 | V2.2 工程 | 8 |
| 20 | dualdomain | dom3/dom4 双域并发切分等价 | V2.2 工程 | 2+3* |
| 21 | oplist_exec | op 列表引擎 (blob 端到端) | V2.2 工程 | 6 |
| 22 | dualcore_threads | 双线程 HMX 并发数值恒等 | V2.2 工程 | 6 |

分水岭: **01-15 = 算子层正确性** (单个数学函数对不对);
**16-22 = 工程层正确性+性能** (缓存/DMA/并发/引擎这些"系统集成件"对不对)。

> 门数口径说明: 脚本汇总的 **83 门** 只统计 `[PASS]` 括号格式行
> (01-14 共 41 门, 16-22 共 42 门, 其中例 20 含 ser/a/b 三段各 1 门
> steps_dumped)。例 15 的 48 项是列式输出 (`maxrel/tol PASS` 格式),
> 真实判定项但不计入 83 —— 所以全库真实判定项合计 **131 项**。

---

# 第 3 章 host 环境准备

所有编译、签名、推包都在 host 上做。这一章逐项检查工具是否就位。
**每一节都有"自检命令"和"预期输出"**, 全部通过才进入第 4 章。

## 3.1 检查工程与编译器

本项目使用 Qualcomm Hexagon SDK 自带的 clang 交叉编译器。先确认路径存在:

```bash
host $ ls /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
预期输出: /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang

host $ /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang --version | head -2
预期输出 (版本号示意):
Hexagon Clang version ...
Target: hexagon
```

> 说明: 脚本里用环境变量 `HEXAGON_SDK_ROOT` 允许覆盖, 未设置时默认取上面这个
> 路径。如果你的 SDK 装在别处, 在 ~/.bashrc 里加
> `export HEXAGON_SDK_ROOT=/你的/SDK/路径` 即可, 其余教程命令不用改。

## 3.2 检查 adb (Android Debug Bridge)

adb 是主机与板子之间唯一的通道: 推文件、执行 shell 命令都靠它。

```bash
host $ which adb
预期输出: /usr/bin/adb

host $ adb version | head -2
预期输出:
Android Debug Bridge version 1.0.41
Version 28.0.2-debian
```

版本号不必完全一致, 1.0.41 系列即可。

## 3.3 检查 SWIV 签名工具

```bash
host $ ls -la /disk1/swiv_build_utility.py
预期输出: -rwx... ... /disk1/swiv_build_utility.py

host $ python3 --version
预期输出: Python 3.8+ 均可
```

用法 (后面会用到): `python3 /disk1/swiv_build_utility.py -i 输入.so -o 输出.so.signed`

## 3.4 检查工程完整性

```bash
host $ cd /disk1/V81Dev/hvxhmx_libsV2.2
host $ ls
预期输出 (至少包含):
build_libs.sh  lib  src  include  examples  docs  host  assets  build  results  PERF_REPORT.md

host $ ls lib/
预期输出: libhvxhmx_v22.so  libhvxhmx_v22.signed.so  ...
```

如果 `lib/` 里没有 `.signed.so`, 说明库还没编译过 (或者被清掉了),
先跑一次:

```bash
host $ ./build_libs.sh
预期末尾输出: 库编译成功 + 签名行 "The SWIV context is successfully generated."
产物: lib/libhvxhmx_v22.so 和 lib/libhvxhmx_v22.signed.so
```

## 3.5 host 侧环境变量一览 (可选)

脚本对所有变量都有默认值, **不设也能跑**。需要定制时才设置:

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `HEXAGON_SDK_ROOT` | /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0 | SDK 根目录 |
| `DEVICE` | 52f67807 | 目标板序列号 (**永远不要改成别的板子**) |
| `SWIV_TOOL` | /disk1/swiv_build_utility.py | 签名工具路径 |

临时指定 (只对当次命令生效):

```bash
host $ DEVICE=52f67807 ./examples/build_examples.sh 01
```

## 3.6 环境自检清单 (勾完再往下走)

- [ ] hexagon-clang --version 有输出, Target 是 hexagon
- [ ] adb version 正常
- [ ] /disk1/swiv_build_utility.py 存在
- [ ] 工程目录完整, lib/ 下有 libhvxhmx_v22.signed.so (没有就先跑 ./build_libs.sh)

---

# 第 4 章 板子上电与连接

## 4.1 上电步骤

不同批次板卡的电源形态略有差异 (DC 适配器 / Type-C 供电), 通用流程如下:

1. **接调试线**: 用 USB 线把板子的调试口 (通常是标着 USB/DEBUG 的 Type-C)
   连到主机;
2. **上电**: 接上电源, 按电源键 (如果有) 或直接由 USB 供电启动;
3. **等待开机**: 首次上电约 30~60 秒进系统。期间板载指示灯会经历
   闪烁→常亮的变化;
4. **确认枚举**: 主机上 `lsusb` 应能看到 Qualcomm 的设备出现。

如果板子之前一直通着电, 直接从第 4 步开始。

## 4.2 确认设备在线 (第一次 adb 接触)

```bash
host $ adb devices
预期输出:
List of devices attached
52f67807\tdevice
```

三种常见状态及含义:

| 显示 | 含义 | 处理 |
|------|------|------|
| `52f67807  device` | 正常在线 | 可以继续 |
| `52f67807  unauthorized` | 未授权调试 | 看板子屏幕点允许; 无屏板卡重启 adb: `adb kill-server; adb start-server` |
| 列表为空 | 没连上 | 检查线/上电; `adb wait-for-device` 挂着等, 出现即就绪 |

**再次强调纪律**: 如果列表里还出现了 `f69bec03`、`5cd9d8c0` 等其他序列号,
说明接了别的板子 —— 后续所有命令都必须带 `-s 52f67807` (本项目脚本已内置),
且不要对其他设备做任何操作。

## 4.3 设备侧初检

```bash
host $ adb -s 52f67807 shell "echo ok; uname -a; ls /data/local/tmp | head -5"
预期输出:
ok
Linux localhost ... #1 SMP ... aarch64 Android
...(若干目录/文件名)
```

能看到输出就说明板子的 shell 可用。再检查本项目依赖的**设备端公共资源**
(部署一次即可, 一般已在位):

```bash
host $ adb -s 52f67807 shell "ls -la /data/local/tmp/hvxhmx_libs/ | grep -E 'skel|run_main|libhvxhmx_v2'"
预期输出 (关键三行, 日期/大小可能不同):
-rwxrwxrwx ... 30976 ... librun_main_on_hexagon_skel.so     ← 加载器骨架 (必须是 2026-08-10 之后的新版!)
-rwxrwxrwx ... ...... run_main_on_hexagon                    ← 加载器本体
-rwxrwxrwx ... ...... libhvxhmx_v2.so                       ← V2.1 库 (例 13 需要)
```

> **skel 版本坑 (历史上真实发生过)**: `/data/local/tmp` **顶层**也有一份
> 老的 `librun_main_on_hexagon_skel.so` (2022 年版), 用它跑会报
> `0x80000406` 错误。**必须**用 `hvxhmx_libs/` 目录下 2026-08-10 的新版
> (支持 dom3/dom4 unsigned PD)。V2.2 脚本会自动从正确位置拷贝, 见 6.3 节。

## 4.4 (可选) 顺手清一次 adb 环境

如果之前有人在同一台主机上用过 adb 且状态混乱:

```bash
host $ adb kill-server && adb start-server && adb devices
```

## 4.5 上电自检清单

- [ ] `adb devices` 显示 52f67807 且状态是 device
- [ ] `adb -s 52f67807 shell echo ok` 输出 ok
- [ ] `/data/local/tmp/hvxhmx_libs/` 下有新版 skel + run_main_on_hexagon + libhvxhmx_v2.so

---

# 第 5 章 第一次手动推包 (把一个例子亲手跑起来)

一键脚本很方便, 但**至少要手动走一遍完整流程** —— 以后脚本哪一步出问题,
你才知道该看哪里。本章用最简单的 01_runtime_init 当小白鼠。

## 5.1 第 0 步: 在设备上建好工作目录

```bash
host $ adb -s 52f67807 shell "mkdir -p /data/local/tmp/hvxhmx22"
```

设备目录约定 (后面所有例子都住这里):

```
/data/local/tmp/hvxhmx22/
├── libhvxhmx_v22.so                     ← V2.2 库 (一次)
├── librun_main_on_hexagon_skel.so       ← 加载器骨架 (一次, 从 hvxhmx_libs 拷新版)
├── run_main_on_hexagon                  ← 加载器本体 (一次, 顶层拷贝即可)
├── test_NN_xxx.so                       ← 每个例子的签名产物
├── assets/                              ← 数据集 (s256/s2560/smallm)
├── blob_w4.wtop / blob_w5.wtop          ← 例 21 的 op 列表 blob
├── rms_w.f16.raw / Y_gold.raw           ← 例 21 的权重/金标
└── NN_xxx.txt                           ← 每例在设备上写的结果文件
```

## 5.2 第 1 步: 部署公共件 (库 + 加载器)

```bash
host $ cd /disk1/V81Dev/hvxhmx_libsV2.2
host $ adb -s 52f67807 push lib/libhvxhmx_v22.signed.so /data/local/tmp/hvxhmx22/libhvxhmx_v22.so
预期输出: ...file pushed... (速率数字随意)

host $ adb -s 52f67807 shell "cp /data/local/tmp/hvxhmx_libs/librun_main_on_hexagon_skel.so /data/local/tmp/hvxhmx22/ && cp /data/local/tmp/run_main_on_hexagon /data/local/tmp/hvxhmx22/ && chmod 755 /data/local/tmp/hvxhmx22/*"
```

> 为什么 skel 要从 hvxhmx_libs 拷而不能从顶层拷? 见 4.3 节的版本坑。
> `cp ... || true` 式的兜底在脚本里有, 手动做时确认 cp 没报错即可。

## 5.3 第 2 步: 编译例子

一条命令 (这就是脚本里 `run_one()` 干的事, 逐个参数解释):

```bash
host $ SDK=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0
host $ HT=$SDK/tools/HEXAGON_Tools/19.0.07/Tools
host $ CC=$HT/bin/hexagon-clang

host $ $CC \
    -mv81 -O2 -mhvx -mhvx-length=128B -mhmx \
    -shared -fPIC -std=gnu11 -Wall \
    -I$SDK/incs -I$SDK/incs/stddef \
    -I$SDK/rtos/qurt/computev81/include \
    -I$SDK/rtos/qurt/computev81/include/qurt \
    -I/disk1/V81Dev/hvxhmx_libsV2.2/include \
    -I/disk1/V81Dev/hvxhmx_libsV2.2/examples/common \
    -o build/test_01_runtime_init.so \
    examples/01_runtime_init/main.c examples/common/example_util.c \
    -L/disk1/V81Dev/hvxhmx_libsV2.2/lib -lhvxhmx_v22 \
    -lc -ldl -lgcc
```

参数速查:

| 参数 | 意思 |
|------|------|
| `-mv81` | 生成 Hexagon v81 指令集 |
| `-mhvx -mhvx-length=128B` | 允许用 128 字节 HVX 向量指令 |
| `-mhmx` | 允许用 HMX 矩阵指令 |
| `-shared -fPIC` | 编成动态库 (DSP 侧按 .so 加载) |
| `-Wall` | 打开全部常规警告 (警告要认真看) |
| `-I...` | 头文件搜索路径: SDK 头 + 本库 include + 例子公共头 |
| `-L... -lhvxhmx_v22` | 链接本库 |
| `-lc -ldl -lgcc` | 只链这三个; **不要 -lqurt** (libqurt.a 不是位置无关代码, 静态链会报 R_HEX_32_6_X 重定位错误; qurt 符号由 DSP 进程在运行时解析) |

预期输出: 只有一条 "unused-command-line-argument" 警告 (-L 在纯编译时无用), 无错误。
编译成功后 `ls -la build/test_01_runtime_init.so` 应看到 ~10KB 的文件。

> **编译失败必须停下来修**。历史上出现过"编译报错被忽略, 推上去跑的是旧 .so"
> 的事故 —— 脚本里专门为此加了硬检查 (见 6.4 节)。

## 5.4 第 3 步: SWIV 签名

```bash
host $ python3 /disk1/swiv_build_utility.py -i build/test_01_runtime_init.so -o build/test_01_runtime_init.so.signed
预期输出:
The SWIV context is successfully generated.
```

看到这行才算签名成功。顺手做一个安全检查 (vgather 指令必须是 0):

```bash
host $ $HT/Tools/bin/hexagon-llvm-objdump -d build/test_01_runtime_init.so | grep -ci vgather
预期输出: 0
```

> vgather (向量 gather 指令) 在 unsigned PD 上是禁用指令, 出现即拒载。
> 本工程所有 .so 都保持 vgather=0。

## 5.5 第 4 步: 推包

```bash
host $ adb -s 52f67807 push build/test_01_runtime_init.so.signed /data/local/tmp/hvxhmx22/test_01_runtime_init.so

host $ adb -s 52f67807 shell "echo 'FARF=0xFFFFFFFF' > /data/local/tmp/hvxhmx22/test_01_runtime_init.so.farf"
```

第二行是可选的**日志开关** (FARF 是 DSP 侧的打印框架):
`<库名>.so.farf` 文件写掩码后, 库内部的调试日志会打到 logcat。
平时不开 (保持安静), 排错时打开。

## 5.6 第 5 步: 运行

在设备上用加载器把 .so 送到 CDSP 的 PD 3 执行:

```bash
host $ adb -s 52f67807 shell "cd /data/local/tmp/hvxhmx22 && \
    ADSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22 \
    CDSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22 \
    ./run_main_on_hexagon 3 test_01_runtime_init.so"
预期输出:
Successfully called main() on Hexagon DSP and received return value of 0.
```

这行命令的每个部分:

| 片段 | 意思 |
|------|------|
| `cd /data/local/tmp/hvxhmx22` | 必须在工作目录里跑 (资产/库都在这) |
| `ADSP_LIBRARY_PATH=...` / `CDSP_LIBRARY_PATH=...` | 告诉 DSP 动态链接器去哪个目录找 .so —— **不设的话 dlopen 找不到库, 例 13 会挂** |
| `./run_main_on_hexagon` | host 侧加载器, 通过 FastRPC 把 .so 拉到 DSP |
| `3` | 目标域号。3 = CDSP PD3 (默认); 例 20 会用到 4 (dom4) |
| `test_01_runtime_init.so` | 要执行的库文件名 |

返回值 0 = main() 正常跑完; 返回 1 = 例子内部有 FAIL (看结果文件);
负数/崩溃码 = 运行时炸了 (去第 9 章)。

## 5.7 第 6 步: 查看结果

例子把结果写在**设备上**的工作目录里, 文件名 = 例子目录名:

```bash
host $ adb -s 52f67807 shell "cat /data/local/tmp/hvxhmx22/01_runtime_init.txt"
预期输出:
=== 01_runtime_init ===
[PASS] hmx_runtime_setup                    err=0 tol=0
VTCM base=ff000000 size=16777216 (0x1000000) ctx_id=1946539192
[PASS] VTCM base non-NULL                   err=0 tol=0
[PASS] VTCM size >= request                 err=0 tol=0
[PASS] hmx_perf_now_us monotonic            err=0 tol=0
teardown done
--- summary: 4 pass, 0 fail ---
```

## 5.8 读懂结果文件: ex_check 的语义

所有例子共用一套判定输出格式 (common/example_util.c):

```
[PASS] 标签名                     err=数值 tol=数值
[FAIL] 标签名                     err=数值 tol=数值
--- summary: N pass, M fail ---
```

- `err` = 实测误差 (整数族就是最大绝对误差, 工程门是 0/1 状态码);
- `tol` = 容差;
- **判定规则: `err <= tol` 即 PASS** (不是"err 是真值"! 写自己的例子时
  布尔条件要写成 `cond ? 0 : 1` 的形式, 否则极性是反的 —— 这是我们真实踩过的坑);
- 最后一行 summary 是整个例子的结论; 加载器的 return value 也会变 1。

## 5.9 手动流程小结 (以后排错就按这个顺序)

```
编译(5.3) → 签名(5.4) → 推包(5.5) → 运行(5.6) → 看结果(5.7)
   │            │           │            │
   clang 报错    SWIV 行没出   push 失败     "returned 1" → 看结果文件哪个 FAIL
                             设备不在线     负数/崩溃     → 第 9 章
```

---

# 第 6 章 一键脚本 build_examples.sh 全解

手动流程懂了以后, 日常一律用脚本。位置:
`/disk1/V81Dev/hvxhmx_libsV2.2/examples/build_examples.sh`

## 6.1 用法

```bash
host $ cd /disk1/V81Dev/hvxhmx_libsV2.2
host $ ./examples/build_examples.sh          # 跑全部 22 个 (约 4 分钟)
host $ ./examples/build_examples.sh 17       # 只跑 17 号 (数字或全名均可)
host $ ./examples/build_examples.sh 20_dualdomain
```

## 6.2 脚本按顺序干的八件事

1. **检查库**: `lib/libhvxhmx_v22.signed.so` 不存在就自动先跑 `build_libs.sh`;
2. **host 工具**: 编译 `host/pack_oplist` (例 21 的 blob 打包器) 和
   `host/wt_inspect` (W3 报告对拍工具);
3. **打包 blob**: 生成 `build/blobs/blob_w4.wtop`、`blob_w5.wtop`
   (例 21 的 op 列表, 内含权重);
4. **部署**: 建 `/data/local/tmp/hvxhmx22`, 推库 + assets + blob + 金标,
   从 hvxhmx_libs 拷新版 skel, 拷 V2.1 库 (例 13 要 dlopen 它),
   预建 `dd_out` 目录 (例 20 的 dump 目录, 见 9.8 节);
5. **逐例**: 编译 → SWIV 签名 → push → 运行 → 拉回 `results/NN.txt`;
6. **例 20 特殊编排**: 先 ser (dom3 跑 16 步), 再 dom3/dom4 **并发**各跑 8 步,
   最后 host 用 `analyze_dd.py` 对拍每步 sha256 是否与 ser 一致;
7. **例 21 特殊后处理**: 设备产出的 W3 报告行与 host 侧 `wt_inspect` 输出
   **逐行 diff**, 一致才算额外一门 PASS;
8. **汇总**: 统计 results/ 下所有 PASS/FAIL, 打出 TOTAL 和 ALL GREEN。

## 6.3 部署阶段的设备端布局 (脚本自动做, 供核对)

```
/data/local/tmp/hvxhmx22/
├── libhvxhmx_v22.so            ← lib/libhvxhmx_v22.signed.so
├── libhvxhmx_v2.so             ← 从 hvxhmx_libs 拷 (例 13 dlopen)
├── librun_main_on_hexagon_skel.so ← 从 hvxhmx_libs 拷 (新版, 支持 dom3/4)
├── run_main_on_hexagon
├── assets/s256/                ← 256³ 引擎资产 + Y_gold_2563.raw 位恒等金标
├── assets/s2560/               ← K2560 金标资产
├── assets/smallm/              ← 例 18 的 pad 面
├── blob_w4.wtop / blob_w5.wtop / rms_w.f16.raw / Y_gold.raw
└── dd_out/                     ← 例 20 dump 目录 (host 预建)
```

## 6.4 全量运行的预期末尾输出

```
=== summary (results/) ===
  TOTAL                            83 pass   0 FAIL
=== ALL GREEN ===
=== done ===
```

如果有 FAIL, 汇总会列出坏例和门数, 例如 `FAILURES: 18_smallm_gemv(3)`,
然后去 `results/18_smallm_gemv.txt` 看 [FAIL] 行, 再翻第 8 章对应小节。

## 6.5 结果文件去哪了

全部拉回 host 的 `results/` 目录 (26 个文件):

```
results/
├── 01_runtime_init.txt ... 22_dualcore_threads.txt   # 22 个例子
├── 20_dualdomain_{ser,a,b}.txt                       # 例 20 三段原始日志 (sha 证据)
├── 20_dualdomain.txt                                 # 例 20 host 对拍门
└── w3_device.txt                                     # 例 21 设备 W3 行 (对拍用)
```

## 6.6 常用单例复跑姿势

改了一个例子的 main.c 之后, 只需:

```bash
host $ ./examples/build_examples.sh 18     # 重编译+签名+推+跑+拉结果, 一条龙
```

改了**库源码** (src/v22/) 之后:

```bash
host $ ./build_libs.sh v22                 # 增量重编库 + 重新签名
host $ ./examples/build_examples.sh        # 全量回归 (必须全量!)
```

---

# 第 7 章 逐例精讲 01-15 (V2.1 算子回归)

这一批例子回答一个问题: **"库里的每一个数学算子, 输出和教科书定义一致吗?"**

它们全部遵循同一个模板 (见 common/example_util.c):

1. 用**固定种子**的 LCG 伪随机数填输入 (不依赖 rand(), 每次跑数据一样);
2. 调被测的 HVX/HMX kernel;
3. 用**纯 C 标量循环**算一遍 golden (不用任何库函数);
4. 逐元素对比, 取最大误差, `err <= tol` 即 PASS。

所以这一批的 PASS 含义非常硬: **硬件向量路径 == 数学定义**, 不是"自洽"。

运行方式都一样 (以 02 为例):

```bash
host $ ./examples/build_examples.sh 02
```

---

## 7.1 例 01_runtime_init —— 环境健康探针 (先跑它!)

**干什么**: 什么都不算, 只验证"上电三件套":
HMX runtime 能 setup、VTCM 拿得到、计时器在走。

**为什么它是 01 号**: 任何 HMX kernel 之前必须 `hmx_runtime_setup()`
(否则硬件直接 CX_FAULT)。如果这个例子挂了, 后面 21 个全没戏,
先修环境 (通常重启板子 + 重连 adb)。

**涉及 API**: `hmx_runtime_setup / hmx_runtime_teardown`,
`hmx_get_vtcm_base/size`, `hmx_perf_now_us`。

**预期输出**:

```
=== 01_runtime_init ===
[PASS] hmx_runtime_setup                    err=0 tol=0
VTCM base=ff000000 size=16777216 (0x1000000) ctx_id=1946539192
[PASS] VTCM base non-NULL                   err=0 tol=0
[PASS] VTCM size >= request                 err=0 tol=0
[PASS] hmx_perf_now_us monotonic            err=0 tol=0
teardown done
--- summary: 4 pass, 0 fail ---
```

**逐行解读**:
- `VTCM base=ff000000 size=16777216`: 本设备 VTCM 固定映射在 0xff000000,
  整块 16MB (十六进制 0x1000000)。ctx_id 是本次会话编号, 每次不同, 不用管;
- `monotonic`: 计时器两次采样单调递增 —— 后面所有性能数字都靠它, 先验明正身;
- `teardown done`: 资源正常释放 (没释放的话跑多了会耗尽会话)。

**常见失败**: setup 返回非 0 → FastRPC/CDSP 没就绪, 板子重启 + `adb kill-server`。

---

## 7.2 例 02_convf16_gemm —— fp16 矩阵乘 (fp16 族之根)

**干什么**: `hmx_convf16` 做 32×32×32 的 fp16 矩阵乘
`out[m,n] = bias[n] + Σ_k act[m,k]·wgt[k,n]`, 与标量 fp32 累加强转 fp16 的
golden 比, 容差 1 ULP。

**为什么存在**: fp16 走的是 **HMX 脉动阵列** (真硬件矩阵单元),
不是软件仿真。这 1 ULP 证明了 HMX fp16 路径数值正确 ——
后面的多 tile (12)、峰值 (14) 都建立在它之上。

**预期输出**:

```
=== 02_convf16_gemm ===
[PASS] hmx_convf16 32x32x32                 err=0 tol=1
--- summary: 1 pass, 0 fail ---
```

**解读**: err=0 表示 1024 个输出全部和 golden 逐位一致 (容差还给了 1 的余量)。

**小知识**: 数据故意取 ~[-0.5,0.5] (scale 0.01), 避开 fp16 的
denormal/边界值, 让对比聚焦在常规范围。

---

## 7.3 例 03_convbbb_int8 —— u8×u8→u8 (int8 族之根)

**干什么**: `hmx_convbbb` 做 u8 激活 × u8 权重 → u8 输出,
int32 累加 + u8 饱和。**容差 0 (逐元素 exact)**。

**为什么能 exact**: 这族在本设备上走的是 **HVX `vmpyacc`+`vasr_sat` 路径**
(整数运算是确定性的, 没有舍入)。有趣的历史: 本设备的 HMX 对 int8 是
silent NOP (算了不出结果), 库因此把 int8 族全部切到了 HVX —— 这个例子
就是那件事的回归锚。

**注意**: int8 族的 bias 是 `int32_t` (和 fp16 族的 `__fp16` bias 不同型)。

**预期输出**:

```
=== 03_convbbb_int8 ===
[PASS] hmx_convbbb 32x32x32 u8xu8->u8       err=0 tol=0
--- summary: 1 pass, 0 fail ---
```

---

## 7.4 例 04_convhbh_u16 —— 宽动态 u16 输出

**干什么**: `hmx_convhbh` 与 `hmx_convhhh` (u8 激活 × i8 权重 → **u16** 输出),
两个函数数学相同、只有写回格式不同 (:2x1 vs :2x2 打包)。容差 0。

**为什么存在**: u16 输出族保留宽动态范围 (不饱和到 u8),
是量化推理里中间层的常见形态。两个写回格式都要验证 —— 历史上出过
"打包位序搞反"的事故。

**预期输出**:

```
=== 04_convhbh_u16 ===
[PASS] hmx_convhbh u8xi8->u16               err=0 tol=0
[PASS] hmx_convhhh u8xi8->u16               err=0 tol=0
--- summary: 2 pass, 0 fail ---
```

---

## 7.5 例 05_i16_weight_convs —— i16 权重族全家桶

**干什么**: 一口气验 4 个 `u8 × i16` 族的函数:

| 函数 | 输出 |
|------|------|
| `hmx_convbcb` | u8 (饱和) |
| `hmx_convbnb` | u8 (基本型) |
| `hmx_convhch` | u16 (:2x2 写回) |
| `hmx_convhnh` | u16 (:2x1 写回) |

bcb/bnb 数学等价、hch/hnh 数学等价 —— 所以 4 门其实测了 2 组数学 × 2 种写回。

**性能提示**: i16 权重带宽是 i8 的两倍, 速度慢一半 (~11-16µs vs 6µs),
能用 i8 就别用 i16。

**预期输出**:

```
=== 05_i16_weight_convs ===
[PASS] hmx_convbcb  u8xi16->u8              err=0 tol=0
[PASS] hmx_convbnb  u8xi16->u8              err=0 tol=0
[PASS] hmx_convhch  u8xi16->u16             err=0 tol=0
[PASS] hmx_convhnh  u8xi16->u16             err=0 tol=0
--- summary: 4 pass, 0 fail ---
```

---

## 7.6 例 06_dwconv —— 深度可分离卷积

**干什么**: 3×3 depthwise 卷积, 每个通道独立做空间卷积,
fp16 (容差 1) 和 u8 (exact) 各一门。H=5, W=5, C=9 的小图。

**和 02/03 的区别**: GEMM 是"乘加折叠", depthwise 是"空间滑窗",
边界样本**跳过**而不是补零 (clamped 语义) —— golden 的边界 if 和 kernel
必须一致, 这是这类 kernel 最容易翻车的地方。

**预期输出**:

```
=== 06_dwconv ===
[PASS] hmx_dwconvf16 3x3 fp16               err=0 tol=1
[PASS] hmx_dwconvbbb 3x3 u8                 err=0 tol=0
--- summary: 2 pass, 0 fail ---
```

---

## 7.7 例 07_add —— 逐元素加 + ReLU (残差连接)

**干什么**: `hmx_add(a, b, bias, out, 32, 32)`,
`out = max(0, a + b + bias)` —— 残差网络的粘合剂。容差换算成 Q10 定点 ≤1
(即 |误差| < 1/1024)。

**预期输出**:

```
=== 07_add ===
[PASS] hmx_add fp16 (relu)                  err=0 tol=1
--- summary: 1 pass, 0 fail ---
```

---

## 7.8 例 08_divide —— HVX 整除五兄弟 + 除零

**干什么**: 5 个除法函数一起验, 重点是**舍入语义**和**除零饱和**:

| 函数 | 精度 | 除零 | 舍入 |
|------|------|------|------|
| `hvhx_divide_u8` | u8 | →0xFF | 截断 |
| `hvhx_floor_divide_u8` | u8 | →0xFF | floor |
| `hvhx_divide_u16` | u16 | →0xFFFF | 截断 |
| `hvhx_floor_divide_u16` | u16 | →0xFFFF | floor |
| `hvhx_divide_flat_i32` | i32 | →±INT32_MAX | **四舍五入** |

**大坑提示**: `i32` 版是 round-to-nearest, **不是 C 的 `/` 截断**!
golden 必须按四舍五入算, 容差给到 1。输入里**故意塞了 8 个 b=0** 测饱和。

**预期输出**:

```
=== 08_divide ===
[PASS] hvhx_divide_u8                       err=0 tol=0
[PASS] hvhx_floor_divide_u8                 err=0 tol=0
[PASS] hvhx_divide_u16                      err=0 tol=0
[PASS] hvhx_floor_divide_u16                err=0 tol=0
[PASS] hvhx_divide_flat_i32 (round)         err=1 tol=1
--- summary: 5 pass, 0 fail ---
```

---

## 7.9 例 09_activation —— HardSwish 与 PReLU (定点近似)

**干什么**: 两个 HVX 激活:
- `hvhx_hardswish_flat_u16`: MobileNetV3 的 `x·clamp(x+3,0,6)/6`,
  输入 u16 按 **int16 二补数 Q12** 解读; 向量路径用 2731/16384≈1/6 近似,
  容差 2 LSB;
- `hvhx_prelu_u8`: PReLU, u8 偏移二进制零点 0x80, slope Q7, 容差 1。

**大坑提示**: u16 输入**不是无符号数**, 是 int16 的位模式 ——
golden 里必须 `(int16_t)in` 解读。

**预期输出**:

```
=== 09_activation ===
[PASS] hvhx_hardswish_flat_u16 (Q12)        err=2 tol=2
[PASS] hvhx_prelu_u8 (slope=128)            err=0 tol=1
--- summary: 2 pass, 0 fail ---
```

---

## 7.10 例 10_reduction —— 沿 depth 归约五变体

**干什么**: 输入 flat `[8 行][32 depth]`, 每行归约出一个结果:

| 函数 | 输出 |
|------|------|
| `hvhx_argminmax_depth_crouton_b` | 行 min/max + 各自下标 |
| `hvhx_argminmax_depth_flat_h` | 同上 (u16 版) |
| `hvhx_find_max_and_index_in_depth_b` | 行 max + 下标 |
| `hvhx_top1_qu8_dLE32_cr2flt` | top-1 (值+下标) |
| `hvhx_reducesum_depth_u8` | 行求和 |

全部 exact。**命名提示**: 名字里的 "crouton" 指内部处理方式,
输入仍是普通 flat 行主序, 别被吓到。

**预期输出**:

```
=== 10_reduction ===
[PASS] hvhx_argminmax_depth_crouton_b       err=0 tol=0
[PASS] hvhx_argminmax_depth_flat_h          err=0 tol=0
[PASS] hvhx_find_max_in_depth_b             err=0 tol=0
[PASS] hvhx_top1_qu8_dLE32_cr2flt           err=0 tol=0
[PASS] hvhx_reducesum_depth_u8              err=0 tol=0
--- summary: 5 pass, 0 fail ---
```

---

## 7.11 例 11_lookup_unpack —— 查表与解包 (纯搬数)

**干什么**: 两个数据搬运算子:
- `hvhx_table_lookup_flat_u8`: `out[i]=table[in[i]]`, 256 项 LUT, exact;
- `hvhx_unpack_weights`: 4-bit → 8-bit 解包, 高 nibble 在前
  (`out[2i]=(in[i]>>4)&0xF`, `out[2i+1]=in[i]&0xF`), exact。

**为什么单独测**: HVX 的 gather (查表) 类指令在 unsigned PD 有兼容性
风险 (库全部规避 vgather, 用 valign/shuffle 组合实现), 这门是"规避策略
没退化"的哨兵。

**预期输出**:

```
=== 11_lookup_unpack ===
[PASS] hvhx_table_lookup_flat_u8            err=0 tol=0
[PASS] hvhx_unpack_weights (4bit→8bit)    err=0 tol=0
--- summary: 2 pass, 0 fail ---
```

---

## 7.12 例 12_multitile_gemm —— 尺寸超过一个 tile 之后

**干什么**: `hmx_convf16` 硬件 tile 是 32×32。当 M/N/K 超过 32 时,
库内部要多 tile 循环 + **K 方向跨 tile 累加** (acc 只清零一次)。
本例验 4 种组合: 64×32×32 / 32×32×64 / 32×64×32 / 64×64×64, 容差 1 ULP。

**为什么重要**: 单 tile 对了不代表拼接对 —— K 累加路径是历史上
出过 off-by-one 的地方。

**预期输出**:

```
=== 12_multitile_gemm ===
[PASS] hmx_convf16 64x32x32 (M>32)          err=0 tol=1
[PASS] hmx_convf16 32x32x64 (N>32)          err=0 tol=1
[PASS] hmx_convf16 32x64x32 (K>32)          err=0 tol=1
[PASS] hmx_convf16 64x64x64 (all>32)        err=0 tol=1
--- summary: 4 pass, 0 fail ---
```

---

## 7.13 例 13_compat_dlsym —— 老工程还能 dlopen 我们吗

**干什么**: 在 DSP 上 `dlopen("libhvxhmx_v2.so")` (V2.1 库),
然后 `dlsym` 解析 6 个老/新符号并调用比对:
4 个 `hmx_v73_*` 兼容符号 + 2 个 v81 新几何符号。

**为什么存在**: 很多已交付工程用 dlopen+dlsym 的方式集成老版本库。
这一门证明: (a) 库没有 UNDEF 符号能被正常加载; (b) 老符号的兼容行为没坏。

**部署依赖**: V2.2 脚本会把设备 hvxhmx_libs 目录里的 V2.1 库拷到
工作目录 (libhvxhmx_v2.so)。**没有这个文件, 本例直接 FATAL** —— 见 9.6 节。

**预期输出**:

```
=== 13_compat_dlsym ===
[PASS] dlopen libhvxhmx_v2.so               err=0 tol=0
[PASS] hmx_v73_convbbb1x1_stride1           err=0 tol=0
[PASS] hmx_v73_convbbb_stride2              err=0 tol=0
[PASS] hmx_v73_convbbb1x1deep_stride1       err=0 tol=0
[PASS] hmx_v73_convbbb_dilate_stride1       err=0 tol=0
[PASS] hmx_convbbb1x1_stride1               err=0 tol=0
[PASS] hmx_convbbbNx1_stride2               err=0 tol=0
resolved 6/6 symbols
--- summary: 7 pass, 0 fail ---
```

---

## 7.14 例 14_hmx_peak_gemm —— 裸 K-loop 峰值 (性能旗舰)

**干什么**: 同一个 M=32 N=32 K=256 的 fp16 GEMM 跑两条路:
(a) **裸 K-loop** —— 手工把 8 个 32×32 slice 预打包进 VTCM,
背靠背 8 条 activation.hf/weight.hf 共享累加器, 中途不搬数;
(b) 公开 wrapper `hmx_convf16` 对照。

**实测数字**:

```
raw K-loop  M=32 N=32 K=256 : 0.04 us/call  12.34 TFLOPS
hmx_convf16 M=32 N=32 K=256 : 268.8 us/call  0.0020 TFLOPS
speedup raw/wrapper = 6325x
```

**怎么读这个 6325×**: 不是 wrapper "有 bug", 而是它为了易用性每次
调用都重新 gather+pack 数据, HMX 被 CPU 搬数饿死。教训写在库的性能文档里:
**大 GEMM 要吞吐就必须预打包 + 裸 K-loop** (V2.2 的 W4A16 引擎正是这么做的,
见例 17)。正确性门是裸 K-loop vs golden ≤1 ULP。

**预期输出**:

```
=== 14_hmx_peak_gemm ===
[PASS] raw K-loop vs golden (fp16)          err=0 tol=1
raw K-loop  M=32 N=32 K=256 : 0.04 us/call  12.34 TFLOPS
hmx_convf16 M=32 N=32 K=256 : 268.8 us/call  0.0020 TFLOPS
speedup raw/wrapper = 6325x  (wrapper 受每 tile gather+pack 限制)
--- summary: 1 pass, 0 fail ---
```

---

## 7.15 例 15_v2_llm_ops —— LLM 常用算子 48 连测

**干什么**: V2 层的 LLM 算子全家桶: rms_norm 族 / l2_norm / sqrt / sqr /
sigmoid / tanh / exp / log / scale / inverse / mul / softmax 等 48 项,
f32 精度, 相对误差容差 0.01~0.02。混着 4 项 PERF 计时行。

**输出长什么样** (截选):

```
=== 15_v2_llm_ops (V2 LLM 算子设备验证) ===
runtime up: VTCM=ff000000
rms_norm_mul_f32 n=1024                      maxrel=0.000000 tol=0.0100 PASS
rms_norm_f32 n=1024                          maxrel=0.000000 tol=0.0100 PASS
...
PERF rms_norm_mul n=1024: 0.3 us  (43.89 GB/s eff: 3x4KB r+w)
PERF mul n=1024: 0.12 us (102.40 GB/s)
PERF sigmoid n=1024: 0.95 us
PERF softmax n=1024: 1.52 us
--- summary: 48 pass, 0 fail ---
```

**注意**: 这个例子的 PASS 行格式和别家不同 (没有 [PASS] 前缀, 用
`maxrel/tol PASS` 列式) —— 汇总脚本按两种格式都统计, 不用慌。

**性能数字怎么读**: rms_norm 0.3µs 处理 1024 元素 (读 x3 写 x1, 4KB×3),
有效带宽 ~44 GB/s —— 对小数据量这是"调用开销主导"的正常水平;
mul 0.12µs → 102 GB/s 才接近 HVX 搬数带宽。

---

# 第 8 章 逐例精讲 16-22 (V2.2 新单元)

这 7 个例子验证从已闭合工程项目 (wtcache_pin / dualcore / htpw4a16 /
gdnsm / oplist / dualdomain 等模块) 沉淀进库的**工程单元**。
它们的 PASS 含义不再是"某个算子数值对", 而是"一整套系统机制
(缓存/搬运/并发/引擎) 在真实硬件上行为正确"。

单元编号对照 (docs/api_v22_overview.md):

| 单元 | 名字 | 例 |
|------|------|----|
| U1 | wtcache (VTCM 权重缓存) | 16 |
| U2 | dcmem (arena/文件/DMA) | 17, 20, 22 (作基础设施) |
| U3 | dcthread (双线程) | 22 |
| U4 | w4a16 (W4A16 HMX 引擎) | 17, 21 |
| U5 | gdnsm (GDN 状态机) | 19 |
| U6 | oplist (op 列表引擎) | 21 |
| U7 | dualdom (双域 worker) | 20 |
| U8 | smallm (小 M pad-256 用法) | 18 |

---

## 8.1 例 16_wtcache_pin —— 把权重钉在 VTCM 里 (U1)

**背景**: LLM decode 每步都要读同一份权重。从 DDR 读太慢,
wtcache 的方案是: 一次性 **pin** (DMA 搬入 + 常驻) 到 VTCM,
之后所有步骤直接吃 VTCM; 激活这类"每步都变"的数据则用 **ring**
(深度 4 的在途预取环) 流水搬运。

**干什么** (9 门):

1. 三块不同大小的权重 (128K/64K/96K) 分别 pin 进 VTCM,
   逐字节回读校验 (**pin_bitexact** ×3);
2. 三个 pin 槽地址互不重叠 (**pin_slots_distinct**);
3. 槽地址落在 pin 区范围内 (**pin_ptr_stable_session**);
4. 一整块 1MB 权重 pin + 校验 (**pin_1MB_verify**), 顺带测带宽;
5. 8 块 16KB 激活走 4+4 ring: 预取内容正确 (**ring_prefetch_bitexact**),
   输出搬回 DDR 完好 (**ring_moveback_bitexact**);
6. ring 流量跑完后, pin 住的权重仍然完好 (**pin_survives_ring**)
   —— 防"预取把权重区踩了"这类灾难。

**怎么跑**:

```bash
host $ ./examples/build_examples.sh 16
```

**预期输出**:

```
=== 16_wtcache_pin ===
vtcm=ff000000 size=16777216 pin_cap=4194304
[PASS] pin_bitexact                         err=0 tol=0
  pin0 131072B -> ff000000 verify=0
[PASS] pin_bitexact                         err=0 tol=0
  pin1 65536B -> ff020000 verify=0
[PASS] pin_bitexact                         err=0 tol=0
  pin2 98304B -> ff030000 verify=0
[PASS] pin_slots_distinct                   err=0 tol=0
[PASS] pin_ptr_stable_session               err=0 tol=0
[PASS] pin_1MB_verify                       err=0 tol=0
  pin 1MB in 64 us -> 16.38 GB/s
[PASS] ring_prefetch_bitexact               err=0 tol=0
[PASS] ring_moveback_bitexact               err=0 tol=0
  ring stats inflight=0,0 peak_vtcm=4325376
[PASS] pin_survives_ring                    err=0 tol=0
--- summary: 9 pass, 0 fail ---
```

**逐行解读**:
- `pin_cap=4194304`: 本会话开 wtcache 时声明的 pin 容量 4MB
  (VTCM 16MB 的 1/4, 其余留给 ring/引擎面);
- `pin0 131072B -> ff000000 verify=0`: 权重落在 VTCM 基址, verify 返回 0=OK;
- `pin 1MB in 64 us -> 16.38 GB/s`: 单次 DMA 大块搬运的有效带宽;
- `ring stats peak_vtcm=4325376`: ring 峰值占用 ~4.1MB。

**踩过的坑 (写成例子的由来)**:
- ring move_back 的**契约**是"当轮给本轮输出槽的目标地址, ring 内部
  pending 到下一轮或 drain 才真正 submit" —— 传错一拍就全错位;
- CPU 写完 VTCM 必须先 FLUSH, DMA 才能读到 (cache 铁律, 见 10.4 节)。

---

## 8.2 例 17_w4a16_gemm —— W4A16 量化 GEMM 引擎 (U4 核心门)

**背景**: W4A16 = 4-bit 权重 × 16-bit 激活, LLM 权重压缩的主流形态。
库里的引擎 (`dc_w4_*`) 是从 t10/htpw4a16 工程一路闭合过来的生产路径,
在 256³ 形状上曾经拿到 **65536/65536 位恒等** (对 QNN 金标)。
本例就是把这个"位恒等"作为回归锚, 永久钉在例程集里。

**干什么** (3 门):

1. 用 `act_surface.raw` (金标对应的**基准激活面**) 跑一次引擎,
   输出面**先解码再比** (见下), vs `Y_gold_2563.raw`
   要求 **65536/65536 逐位一致**;
2. 同输入再跑一次, 两次输出 byte-exact (**确定性**);
3. 100 次计时取中位 → GFLOPS (性能报告)。

**关键概念: 输出是 crouton 面, 必须解码再比**
引擎的输入/输出都是 **crouton16_row4 布局** (HMX 硬件要求的交织格式,
行在面上不是连续存放的!)。所以"第 m 行输出"在面上的物理位置是散的,
例程先用 `minv_crouton()` 把面还原成线性 (M,N) 矩阵再比对。
**直接拿面字节做 memcmp/按行偏移取数据 = 全错** —— 这是本教程
最重要的一条军规 (我们自己在例 17/18 上真实翻过车)。

**输入配对铁律**: 金标 `Y_gold_2563.raw` 对应的激活是 **`act_surface.raw`**。
`act_variants/v0.raw` 是随机变体, 喂它对金标只有 ~5.7% 逐位相同,
会被误判为引擎坏了。同目录的 `Y_ref_v0.raw` 是 v0 的 int8 全精度
标量金标, 和 w4 引擎**没有**对应关系, 不是任何判据。

**预期输出**:

```
=== 17_w4a16_gemm ===
  vs Y_gold_2563: 65536/65536 exact
[PASS] vs_gold2563_byteexact                err=0 tol=0
[PASS] rerun_byteexact                      err=0 tol=0
invoke med 26 us (25.0 us 最小), 256^3 W4A16 = 1290.56 GFLOPS
[PASS] invoke_sanity_us                     err=0 tol=0
--- summary: 3 pass, 0 fail ---
```

**数字怎么读**: 256³ = 2×256³/26µs ≈ 1.29 TFLOPS —— 这是"小形状 +
表回填开销"下的引擎吞吐; 真正的摊薄大形状数字见例 21 (K2560) 和
PERF_REPORT.md。

---

## 8.3 例 18_smallm_gemv —— decode 场景: M=1 的矩阵向量乘怎么办 (U8)

**背景**: LLM decode 步每层只有 1~16 个 token (M=1/16), 但 W4A16 引擎的
m_t=8 tile 结构硬性要求 **M 是 256 的倍数** (M=32/128 实测 FAIL)。
工程解法: **pad-256** —— 把激活面补到 256 行 (补的行填 32768, 即
对称量化的中性点/零), 输出取真实行即可。

**干什么** (5 门, 全部在解码后的线性输出上判):

1. `p1_row0_eq_full_row0`: M=1(pad) 的输出行 0 == M=256 全量跑的行 0
   —— 证明"补行不污染真实行";
2. `p16_rows0_15_eq_full`: M=16(pad) 的行 0..15 == 全量跑;
3. `pad_rows_invariant`: 两个不同输入 (v0/v1) 的 pad 行输出完全一致
   —— pad 行确实是中性的;
4. `row0_tracks_input`: 行 0 随输入变化 (不是恒等常数);
5. `cost_M_invariant`: M=1 与 M=256 的 invoke **同价** (tile-walk bound,
   走满整个 tile 与 M 无关) —— 这是 pad 路线"划算"的经济学证明。

**预期输出**:

```
=== 18_smallm_gemv ===
[PASS] p1_row0_eq_full_row0                 err=0 tol=0
[PASS] p16_rows0_15_eq_full                 err=0 tol=0
[PASS] pad_rows_invariant                   err=0 tol=0
[PASS] row0_tracks_input                    err=0 tol=0
M=256 med 26 us vs M=1(pad) med 26 us (delta 0.00%)
[PASS] cost_M_invariant                     err=0 tol=0
--- summary: 5 pass, 0 fail ---
```

**资产**: `assets/smallm/act_p1_v0.raw` 等由 `host/gen_smallm_assets.py`
从 s256 解码 → 取真实行 → pad → 重打包生成, 自带 round-trip 自检。

---

## 8.4 例 19_gdn_sm —— 门控 DeltaNet 状态机 (U5)

**背景**: GDN (Gated DeltaNet) 是线性注意力一族的状态空间模型,
decode 是一个**带状态的循环**: 每步 `S ← α·S + Δ`, 新状态依赖旧状态。
它的正确性不是"一步算对", 而是"跑 100 步后状态仍然对" —— 累积误差
才是这类模型的头号杀手。

**干什么** (8 门, 输入全部用 host 可复现的 LCG, 不需要设备外资产):

1. `f16_roundtrip_idempotent`: 2 万个 f16 (含 denormal) 做
   f16→f32→f16 往返, 要求幂等 —— 这是历史事故点 (denormal 处理);
2. `conv_step_cos` / `conv_block_state_bytes`: conv 步进 vs 标量 oracle
   (cos ≥ 0.9999), 块状态 byte-exact;
3. `conv_state_bitexact_guarded`: 状态区之外放 64KB 守卫带,
   跑完必须原封不动 —— 防 kernel 越界写 (又是真实事故);
4. `delta_100tok_cos` / `delta_state_cos`: delta kernel 跑 100 token,
   输出与状态分别对 oracle (cos ≥ 0.9999);
5. `solve_tri_cos`: 三角 solve kernel;
6. `chunk_split_16_8_8`: 长度 16 的块切成 8+8 跑, 结果与整块一致
   —— 切分不变性。

**预期输出**:

```
=== 19_gdn_sm ===
[PASS] f16_roundtrip_idempotent             err=0 tol=0
[PASS] conv_step_cos                        err=0 tol=0
[PASS] conv_block_state_bytes               err=0 tol=0
[PASS] conv_state_bitexact_guarded          err=0 tol=0
[PASS] delta_100tok_cos                     err=0 tol=0
[PASS] delta_state_cos                      err=0 tol=0
[PASS] solve_tri_cos                        err=0 tol=0
[PASS] chunk_split_16_8_8                   err=0 tol=0
--- summary: 8 pass, 0 fail ---
```

---

## 8.5 例 20_dualdomain —— 两个 DSP 域并发跑 (U7)

**背景**: CDSP 有 dom3/dom4 两个 unsigned PD, 各自独立调度。
问题: 把 16 步任务切成 a[0,8)+b[8,16) 放两域并发,
**每一步的数值结果必须和串行一模一样** (切分等价), 时间上拿到 ~2×。

**它不是一个普通例子**: 本例的 .so 是"分片执行器", 要跑**三次**:

```
./run_main_on_hexagon 3 test_20_dualdomain.so dd ser 0 16   ← dom3 串行基线
./run_main_on_hexagon 3 test_20_dualdomain.so dd a 0  8  \  ← dom3 前 8 步
./run_main_on_hexagon 4 test_20_dualdomain.so dd b 8  8  /  ← dom4 后 8 步 (并发!)
```

参数含义: `dd`=选择器, `tag`=ser/a/b, `start`=起始步, `len`=步数
(**第 4 个参数是"长度"不是"结束"** —— 传 `8 16` 会报 range 错,
又一个真实踩过的坑)。

**host 对拍** (analyze_dd.py 自动做, 2 门):
- `dd_split_equiv`: a 的 8 个 sha256 + b 的 8 个 == ser 对应步的 sha256
  —— 每步输出 128KB 全量哈希, 逐字节等价的铁证;
- `dd_steps_all`: 三段合计 32 条 sha 记录齐全 (防止"空转也 PASS"的空门)。

**预期输出** (results/20_dualdomain.txt):

```
=== 20_dualdomain_host (host 对拍) ===
[PASS] dd_split_equiv                       err=0 tol=0
[PASS] dd_steps_all                         err=0 tol=0
dd speedup ser=543us a=271us b=271us → 2.004x (报告, 不设门)
```

加速比只报告不设门 (调度抖动会小幅波动)。原始三段日志在
`results/20_dualdomain_{ser,a,b}.txt`, 里面是每步的 sha256 证据链。

---

## 8.6 例 21_oplist_exec —— op 列表引擎: 一份 blob 跑一整段子图 (U6)

**背景**: 真实模型一步要跑几十个 op。oplist 引擎把一段计算
(PIN 权重 → MATMUL → RMSNORM → ...) 编译成一个 **blob** (`.wtop` 文件,
host 侧 `pack_oplist` 打包, 内含权重数据), 设备上 `wt_exec_run()` 一次执行
整个列表, 还能给每个 op 单独计时。

**干什么** (6 门):

1. `negatives_rejected`: 5 个**故意构造的坏 blob** (魔数错/长度越界/...)
   必须被拒绝且返回**正确的错误码** —— 错误路径也是契约;
2. w4 blob 的 MATMUL (K2560): 解码输出 vs 独立标量金标 `Y_gold.raw`,
   **≤40 LSB** (K2560 形状的历史闭合值是 37 LSB);
3. `matmul_reinit_byteexact`: 引擎复用 (重传输入再跑) byte-exact;
4. `rmsnorm_bitexact`: RMSNORM vs host 镜像逐位一致 (0 ULP);
5. `w3_host_device_lines_identical`: 设备产出的 W3 报告行与 host 侧
   `wt_inspect` 工具的输出**逐行一致** —— blob 解析器两套实现
   (设备 C / host C) 不许有分歧;
6. `suite_complete`: 全序列跑完。

**预期输出** (节选):

```
=== 21_oplist_exec ===
[PASS] negatives_rejected                   err=0 tol=0
{"t":"W3","verdict":"PASS"}
w4 op_us matmul=27063
[PASS] matmul_reinit_byteexact              err=0 tol=0
[PASS] matmul_vs_gold_lsb                   err=0 tol=0
[PASS] rmsnorm_bitexact                     err=0 tol=0
op_us matmul=45351 rmsnorm=68132
[PASS] suite_complete                       err=0 tol=0
[PASS] w3_host_device_lines_identical err=0 tol=0
--- summary: 6 pass, 0 fail ---
```

**数字怎么读**: `op_us matmul=45351` = w5 blob 的 MATMUL op 全路径 45.4ms
(含 3.2MB 权重 restage + DMA + 计算); w4 blob 是 27.1ms。
单 invoke 计算本身只要 ~1ms 级 —— 差值全是搬运, 这正是 U1 缓存存在的意义。

---

## 8.7 例 22_dualcore_threads —— 同域双线程并发 (U3)

**背景**: 单域内开两个线程分头算, 能不能提速? 答案 (模块 4C 闭合结论):
**基本不能** —— 单 DMA 引擎 + 全局 HMX 锁, 两个线程会串行排队。
但"不提速"不等于"不能并发": 双线程**必须做到数值恒等**,
否则它在任何并发架构里都是隐患。本例把这个结论变成永久回归。

**干什么**:

1. 主线程先**串行**跑两个引擎 (同引擎同输入顺序) 拿参考输出;
2. spawn 两个 worker 线程 (qurt barrier 同步起步), 各自引擎并发跑
   `DMA → dc_w4_invoke → read_out`, 共享一个 DMA submit 锁;
3. 判定 (6 门):
   - 并发输出 == 串行输出 (两引擎各一门, byte-exact);
   - hvx/norm/dot 三个纯 HVX 算子跨线程结果 == 主线程参考 (确定性 ×3);
   - 旗标握手完成 (VTCM 内 f: 0→1→2 的跨线程信号);
4. 报告 serial vs concurrent 耗时比 (不设门)。

**hmx_lock 交接 (本例的灵魂, 必须懂)**:
HMX 锁是**持有线程属性** —— 哪个线程要 invoke HMX, 哪个线程就必须持有锁。
模板:

```
主线程: wtcache_hmx_unlock()   ← 把锁让出来 (wtcache_open 时主线程默认持有)
spawn workers
worker:  wtcache_hmx_lock()    ← 自己拿锁
         dc_w4_invoke(...)     ← 只有持锁线程能碰 HMX
         wtcache_hmx_unlock()
主线程: wtcache_hmx_lock()     ← join 后取回
```

不给 worker 加锁的后果是 **PD 直接 crash** (返回 -2147482611 一类的负数),
不是 FAIL —— 这是模块 C P3 实验换来的铁律。

**预期输出**:

```
=== 22_dualcore_threads ===
[PASS] concurrent_e0_byteexact              err=0 tol=0
[PASS] concurrent_e1_byteexact              err=0 tol=0
[PASS] hvx_deterministic                    err=0 tol=0
[PASS] norm_deterministic                   err=0 tol=0
[PASS] dot_deterministic                    err=0 tol=0
[PASS] flag_handshake_done                  err=0 tol=0
serial 1684 us | concurrent 1748 us | ratio 0.963 (HMX 锁+单 DMA 引擎, 预期 ~1, 只报告不设门)
inner_us a=1709 b=840
--- summary: 6 pass, 0 fail ---
```

**ratio 0.963 怎么读**: 并发比串行还慢 4% —— 在预期内 (锁竞争 + 排队)。
想真提速: 走例 20 的**双域** (跨 PD) 路线。

---

# 第 9 章 常见错误排查 (FAQ)

按"你看到的现象"组织。每条都给出: 现象 → 原因 → 处理。
带 ⭐ 的是本项目**真实发生过**的事故, 不是理论可能。

## 9.1 adb devices 里没有设备

```
现象: List of devices attached (后面空)
```

- 检查 USB 线 (要数据线, 不是纯充电线);
- 确认板子上电完成 (指示灯稳定);
- `adb kill-server && adb start-server && adb devices`;
- 还不行: `adb wait-for-device` 挂着, 重启板子电源, 出现即恢复。

## 9.2 状态是 unauthorized / offline

```
adb kill-server && adb start-server
adb -s 52f67807 shell echo ok     # 能回 ok 就继续
```

## 9.3 ⭐ 加载器报 "returned 1" 且没有结果文件

**现象**: `run_main_on_hexagon` 输出 `Error: Main on Hexagon DSP returned 1`,
或 `!! no result file`, 结果文件根本没生成。

**头号嫌疑: 库有 UNDEF 符号, dlopen 静默失败**。
unsigned PD 的动态链接器看到 .so 里有解析不了的符号, 直接拒载,
而且**不打印任何错误** —— 你只能看到 main "returned 1"。

诊断 (host 上):

```bash
host $ HT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
host $ $HT/Tools/bin/hexagon-nm -D lib/libhvxhmx_v22.so | grep ' U '
预期输出: 只剩 qurt/HAP 系运行时符号 (由 DSP 进程提供)
如果出现 dma_desc_init / dma_desc_submit 一类 → src/v22/dma_utils.c 没编进去
```

修复: `./build_libs.sh` 重编库 (确认 src/v22/dma_utils.c 在编译清单里),
重新部署库, 再跑例子。

**真实事故**: V2.2 首次部署时 dma_utils.c 没拷进 src/v22/, 22 个例子全灭,
现象就是这一条 —— 用 nm 一查便知。

## 9.4 ⭐ 报错 0x80000406

**现象**: run_main_on_hexagon 直接吐十六进制错误码。

**原因: skel 版本太老**。`/data/local/tmp` 顶层有一份 2022 年的
`librun_main_on_hexagon_skel.so`, 不支持 dom3/dom4 unsigned PD。

修复: 永远用 `/data/local/tmp/hvxhmx_libs/` 下 2026-08-10 之后的版本:

```bash
host $ adb -s 52f67807 shell "cp /data/local/tmp/hvxhmx_libs/librun_main_on_hexagon_skel.so /data/local/tmp/hvxhmx22/"
```

V2.2 的 build_examples.sh 已自动做这一步; 手动推包时别拷错源。

## 9.5 ⭐ PD crash (返回巨大负数, 如 -2147482611)

**现象**: 例子没跑完就崩, return value 是负数。

**头号嫌疑 (例 22 场景): 工作线程没拿 hmx_lock 就 invoke HMX**。
HMX 锁是持有线程属性。修复 = 按 8.7 节的交接模板加锁。

其他可能: VTCM 越界写 (用例 19 的守卫带门就是防它)、DMA 描述符野指针
(submit 前先 wait, 见 10.5)。定位手段: 二分注释掉计算段, 先确认是哪个
invoke 崩的。

## 9.6 ⭐ 例 13 FATAL: dlopen FAIL

**现象**:

```
FATAL: dlopen FAIL: ...libhvxhmx_v2.so...
```

两个原因按序查:

1. **文件没部署**: `adb shell ls /data/local/tmp/hvxhmx22/libhvxhmx_v2.so`
   —— 没有就从设备 hvxhmx_libs 目录拷 (脚本已自动做);
2. **环境变量没设**: 运行时必须带
   `ADSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22 CDSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22`
   (脚本已自动设; 手动跑别漏)。

## 9.7 ⭐ 例 17 的 vs_gold2563_byteexact FAIL (一致性 < 65536)

按可能性排查:

1. **激活喂错** (最常见): 必须用 `act_surface.raw`, 不是 `act_variants/v0.raw`;
2. 金标文件没部署: 检查 `assets/s256/Y_gold_2563.raw` (131072 字节) 在设备上;
3. 库被人改过 → `./build_libs.sh v22` 重编后全量回归;
4. 输出解码逻辑被改坏 (minv_crouton 与 host inv_crouton16.py 必须逐位一致)。

顺带说明: 如果看到有人拿 `Y_ref_v0.raw` 当判据, 那是误用 ——
它是 int8 全精度标量金标, 与 w4 引擎无对应关系 (见 8.2 节)。

## 9.8 ⭐ 例 20 报 "dd_run FAIL: dump 0"

**原因**: dump 目录 `dd_out` 不存在。DSP 上 **`system()` 没有 shell**,
`mkdir -p` 静默失败, fopen 建不了目录 → dump 出 0 个文件。

修复: host 侧预建目录 (脚本已自动做):

```bash
host $ adb -s 52f67807 shell "mkdir -p /data/local/tmp/hvxhmx22/dd_out"
```

**教训泛化**: 任何设备端要写文件的例子, 目录都必须 host 预建。

## 9.9 例 20 报 "range 8+16 > 16"

参数语义搞错: 第 4 个参数是**长度**不是结束下标。b 段应传 `dd b 8 8`
(start=8, len=8), 不是 `dd b 8 16`。

## 9.10 编译期错误速查

| 报错 | 原因 | 处理 |
|------|------|------|
| `R_HEX_32_6_X relocation` | 链了 `-lqurt` (libqurt.a 非 PIC) | 去掉 -lqurt, 只留 `-lc -ldl -lgcc` |
| `unknown target` / 找不到 clang | SDK 路径不对 | 核对 3.1 节路径或设 HEXAGON_SDK_ROOT |
| 头文件找不到 qurt.h | -I 少了 computev81 路径 | 照抄 5.3 节的完整命令 |
| 警告 `unused-command-line-argument: -L...` | 正常现象 (纯编译阶段 -L 无用) | 忽略 |

## 9.11 签名后还是拒载

- 确认推的是 `*.so.signed` 改名后的文件, 不是未签名的中间产物;
- `hexagon-llvm-objdump -d xxx.so | grep -ci vgather` 必须 = 0
  (vgather 是 unsigned PD 禁用指令);
- SWIV 输出必须有 `The SWIV context is successfully generated.`。

## 9.12 "同名 .so 行为没变" (CDSP 库缓存)

改了库/例子重推, 跑出来的结果却像旧的 —— CDSP 有库缓存。
处理: 换文件名 (比如 test_xx2.so), 或重启板子。
脚本每次 push 都覆盖同名文件, 一般没问题; 手动试验时容易遇到。

## 9.13 结果里 PASS 但 perf 数字离谱

- 计时粒度: HAP 计时器 ~µs 级, 太小的操作 (亚 µs) 中位数会吸附在
  粒度整数上 (比如反复出现 26µs) —— 正常现象, 对比时看数量级;
- 设备后台有负载 (别人在跑东西) → 数字整体漂移, 复测。

## 9.14 排错流程总图

```
跑挂了
 ├─ 加载器没起来? ──→ 9.3 (UNDEF/库) / 9.4 (skel) / 9.11 (签名)
 ├─ 崩了 (负返回值)? → 9.5 (hmx_lock / 越界)
 ├─ 某例 FAIL? ─────→ 翻第 7/8 章该例的"常见失败"小节
 │                     例 13 → 9.6; 例 17 → 9.7; 例 20 → 9.8/9.9
 └─ 编译/推包就报错? → 9.10 / 9.1 / 9.2
```

---

# 第 10 章 进阶原理 (选读)

## 10.1 例子为什么都是一个 main.c 一个 .so

DSP 侧每个 .so 被加载后在 PD 里执行一次 main()。把每个用例做成独立 .so:
隔离性好 (一个例子崩了不影响别的)、部署粒度小 (改哪个推哪个)、
判读简单 (一个文件一个结果)。代价是每次加载有 ~百 ms 开销 ——
对测试场景完全可接受。

## 10.2 HMX 使用的"仪式"全景

从裸机到能跑一次 HMX GEMM, 完整链路是:

1. `hmx_runtime_setup(size)` (V2.1 路径) 或 `wtcache_open()` (V2.2 路径):
   上电 HVX/HMX + 拿 VTCM + 拿 hmx_lock;
2. 数据进 VTCM (CPU memcpy + FLUSH, 或 DMA);
3. HMX 面 (激活/权重/bias/表) 必须 **2KB 对齐**;
4. invoke (持锁线程!);
5. 读输出: INVALIDATE + memcpy 出 DDR;
6. 退出 `wtcache_close()` / `teardown()` —— **PASS 和 FAIL 两条路都要关**
   (不关的话会话资源泄漏, 跑几十次后 setup 开始失败)。

## 10.3 crouton16_row4 布局到底是什么

HMX 的 crouton 格式把 (M,K) 矩阵按"物理槽位对"重排:

```
for phase in 0..8:
  for kt in 0..K/32:
    for m32_group in 0..M/32:
      for row_pair in 0..2:        # 行 m 与 m+1 交织
        for col in 该 32 列:
          依次放 [row0,col], [row0+1,col]
```

效果: 逻辑上相邻两行的同列元素在物理面上背靠背。HMX 喂数按这个顺序最省
带宽, 但代价是**人不能直接读** —— 消费必须逆布局。host 侧参考实现
`host/inv_crouton16.py`, 设备侧镜像 `minv_crouton()` (例 17/18/21 各有一份,
必须逐位一致)。属性速记:

- 面大小 = M×K×2 字节 (和线性一样, 只是重排);
- 行偏移、列偏移在面上**都没有意义**;
- 权重面还有自己的另一套打包 (q8_0 nibble), 由 host 打包器负责。

## 10.4 cache 四铁律 (V2.2 全库通用)

CPU (DSP 核)、DMA 引擎、HMX 三家看内存的方式不同, 共享数据必须显式
维护 cache 一致性:

| 铁律 | 场景 | 操作 |
|------|------|------|
| ① CPU 写 DDR → DMA 读 | 激活/权重送 DMA 前 | `qurt_mem_cache_clean(..., FLUSH, DCACHE)` |
| ② CPU 写 VTCM → DMA/HMX 读 | 输出槽写完回搬前 | 同上 FLUSH |
| ③ DMA 写 → CPU 读 | 搬回的输出读之前 | `qurt_mem_cache_clean(..., INVALIDATE, DCACHE)` |
| ④ 会话结束 | 无论 PASS/FAIL | `wtcache_close()` / teardown |

漏掉任何一条的表现极其迷惑: 有时对有时错 (取决于 cache 残留),
这也是"上机前反复静态检查"纪律的由来 —— 这类 bug 在 desk 上看不出来,
一上机就是玄学。

## 10.5 UserDMA 三大坑 (源自 wtcache 闭合时的实证)

1. submit 前必须先 wait 空闲 —— 否则描述符链上挂野指针;
2. VTCM → DDR 方向: FLUSH 源 + bypass=1;
3. submit 前 FLUSH_INVALIDATE 目的端。

例 16 的 ring 就是把这些规则封装好的成品 —— 直接用 ring,
别自己裸写 DMA (除非你想体验一遍这些坑)。

## 10.6 为什么例 20 双域能 2× 而例 22 双线程不能

- 双域 (dom3/dom4): 两个**独立 PD**, 各自的 DMA 引擎/HMX/调度都独立
  → 真并行 (实测 2.004×);
- 双线程 (同域): DMA submit 全局锁 + HMX 全局锁 → 两线程在锁上排队
  (实测 ratio 0.963)。
- 推论: 想并发扩容, 加域不加线程。

## 10.7 从例子到自己的工程

1. 拷 `examples/01_runtime_init/main.c` 当模板;
2. 数据用 `static ... __attribute__((aligned(128)))` 全局数组
   (DSP 栈很小, 别放栈上; 128 对齐是 HVX 向量载入的硬要求);
3. 判定用 `ex_check(label, err, tol)` —— 记住 err≤tol 才 PASS,
   布尔条件写成 `cond ? 0 : 1`;
4. 金标尽量独立标量实现 (参考第 7 章模板), 别"用库自己验自己";
5. 涉及 HMX 引擎输出时, 先过 `minv_crouton` 再比对;
6. 退出路径无条件的 wtcache_close/teardown。

---

# 第 11 章 附录

## 附录 A: 全部 22 例判据总表 (83 门)

| # | 例子 | 门数 | 关键判据 |
|---|------|------|----------|
| 01 | runtime_init | 4 | setup=0 / VTCM 非 NULL / size≥2MB / 计时单调 |
| 02 | convf16_gemm | 1 | fp16 32³ ≤1 ULP |
| 03 | convbbb_int8 | 1 | u8×u8→u8 exact |
| 04 | convhbh_u16 | 2 | u8×i8→u16 exact ×2 格式 |
| 05 | i16_weight_convs | 4 | bcb/bnb/hch/hnh exact |
| 06 | dwconv | 2 | 3×3 dw fp16 ≤1 / u8 exact |
| 07 | add | 1 | fp16 add+relu Q10 ≤1 |
| 08 | divide | 5 | 5 除法变体 (i32 round ≤1, 余 exact) |
| 09 | activation | 2 | hardswish ≤2 LSB / prelu ≤1 |
| 10 | reduction | 5 | 5 归约 exact |
| 11 | lookup_unpack | 2 | LUT / 4bit 解包 exact |
| 12 | multitile_gemm | 4 | 4 维度组合 ≤1 ULP |
| 13 | compat_dlsym | 7 | dlopen + 6 符号 exact |
| 14 | hmx_peak_gemm | 1 | 裸 K-loop ≤1 ULP (另报 12.34 TFLOPS) |
| 15 | v2_llm_ops | 48* | LLM 算子 maxrel≤tol (列式, 不计入 83) |
| 16 | wtcache_pin | 9 | pin/ring bit-exact + 1MB 16.38 GB/s |
| 17 | w4a16_gemm | 3 | 256³ 位恒等 65536/65536 + 复跑 + 计时 |
| 18 | smallm_gemv | 5 | pad 行0/0..15 == full + pad 不变 + M 同价 |
| 19 | gdn_sm | 8 | f16 幂等/cos 门/状态 byte/64KB 守卫/切分 |
| 20 | dualdomain | 2+3* | 切分 sha 等价 + 32 步齐 + 各段 steps_dumped (另报 2.004×) |
| 21 | oplist_exec | 6 | 负例错误码/≤40 LSB/0 ULP/W3 逐行/复用 |
| 22 | dualcore_threads | 6 | 并发==串行 + 跨线程确定 + 握手 |

## 附录 B: 命令速查卡

```bash
# ---------- 一次性 ----------
adb devices                                        # 确认 52f67807 在线
./build_libs.sh                                    # 编库 (+签名)
./examples/build_examples.sh                       # 全量 01-22

# ---------- 日常 ----------
./examples/build_examples.sh 18                    # 单例 (数字/全名)
./build_libs.sh v22                                # 改库源后增量重编
grep -E 'PASS|FAIL' results/18_smallm_gemv.txt     # 看单例结果

# ---------- 手动链路 (排错用) ----------
SDK=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0
CC=$SDK/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
$CC -mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC -std=gnu11 -Wall \
    -I$SDK/incs -I$SDK/incs/stddef \
    -I$SDK/rtos/qurt/computev81/include -I$SDK/rtos/qurt/computev81/include/qurt \
    -Iinclude -Iexamples/common \
    -o build/test_NN.so examples/NN_xxx/main.c examples/common/example_util.c \
    -Llib -lhvxhmx_v22 -lc -ldl -lgcc
python3 /disk1/swiv_build_utility.py -i build/test_NN.so -o build/test_NN.so.signed
adb -s 52f67807 push build/test_NN.so.signed /data/local/tmp/hvxhmx22/test_NN.so
adb -s 52f67807 shell "cd /data/local/tmp/hvxhmx22 && \
    ADSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22 \
    CDSP_LIBRARY_PATH=/data/local/tmp/hvxhmx22 \
    ./run_main_on_hexagon 3 test_NN.so"
adb -s 52f67807 shell cat /data/local/tmp/hvxhmx22/NN_xxx.txt

# ---------- 例 20 手动三段 ----------
./run_main_on_hexagon 3 test_20_dualdomain.so dd ser 0 16
./run_main_on_hexagon 3 test_20_dualdomain.so dd a 0 8 &   # dom3
./run_main_on_hexagon 4 test_20_dualdomain.so dd b 8 8 &   # dom4 并发
python3 host/analyze_dd.py results/

# ---------- 诊断 ----------
HT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
$HT/Tools/bin/hexagon-nm -D lib/libhvxhmx_v22.so | grep ' U '     # UNDEF 检查
$HT/Tools/bin/hexagon-llvm-objdump -d build/test_NN.so | grep -ci vgather  # 应为 0
adb -s 52f67807 shell "cat /data/local/tmp/hvxhmx22/test_NN.so.farf"      # 日志开关
```

## 附录 C: 术语表

| 术语 | 解释 |
|------|------|
| CDSP | compute DSP, 计算域, 本库运行处 |
| PD (dom3/dom4) | DSP 保护域, unsigned PD 是无签名权限的沙盒进程 |
| FastRPC | CPU ↔ DSP 的远程调用机制 |
| HVX | 128 字节宽向量 SIMD |
| HMX | 矩阵脉动阵列 (GEMM 专用) |
| VTCM | DSP 紧耦合快内存, 本设备 16MB @ 0xff000000 |
| hmx_lock | HMX 全局锁, 持有线程属性 |
| pin | 权重常驻 VTCM (wtcache U1) |
| ring | 激活流水的在途预取环 (深度 4) |
| crouton16_row4 | HMX 数据交织布局, 消费需逆布局解码 |
| W4A16 | 4-bit 权重 × 16-bit 激活 |
| pad-256 | 小 M 补到 256 行 (中性值 32768) 的 GEMV 路线 |
| oplist blob (.wtop) | op 列表+权重打包文件, 设备端 wt_exec_run 执行 |
| SWIV | unsigned PD 动态库签名机制 |
| skel | librun_main_on_hexagon_skel.so, FastRPC 加载骨架 |
| ex_check | 例子判定宏, err≤tol 即 PASS |
| 金标 / oracle | 独立实现的期望输出 (标量 C 或工程闭合存档) |
| LSB | 最低有效位, 整数量化误差单位 |
| ULP | 浮点最后位单位 |

## 附录 D: 相关文档导航

| 想了解 | 看 |
|--------|-----|
| 22 例判据详解 | examples/TEST_CASES.md |
| 英文逐例讲解 | examples/EXAMPLES.md |
| 单元 API 手册 | docs/api_v22_overview.md (总览) 及 api_v22_*.md 每单元一份 |
| 实测性能与基线对照 | PERF_REPORT.md |
| 每个例子测什么的原始记录 | results/*.txt |
| V2.1 算子层速查 | examples/README.md 上半部分 |

## 附录 E: 完整一天上机流程 (打印贴墙版)

```
1. 插线/上电 → adb devices 看到 52f67807
2. cd /disk1/V81Dev/hvxhmx_libsV2.2
3. ls lib/libhvxhmx_v22.signed.so    (没有 → ./build_libs.sh)
4. ./examples/build_examples.sh      (~4 分钟, 末尾必须 ALL GREEN 83 pass)
5. 有 FAIL → results/ 找对应文件 → 本教程第 7/8 章该例 + 第 9 章 FAQ
6. 改代码 → 单例复跑 (build_examples.sh NN) → 收工前再全量一次
7. 下班前: 确认没有遗留进程 (adb shell 里没有挂着的 run_main)
```

---

*(完) 教程版本: 2026-08-15, 对应库 libhvxhmx_v22.so (184,624 B),
基线: 22 例 83 门 ALL GREEN。*
