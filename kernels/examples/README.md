# hvxhmx_libsV2.3 — 设备可运行示例 (01-33)

每个子目录是一个独立、自包含的示例: 准备数据 → 初始化 runtime → 调用 kernel → 与标量 golden 对比 →
把 PASS/FAIL 写到设备上的 `/data/local/tmp/hvxhmx23/<name>.txt`。

## 一键运行

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh          # 编译/签名/部署/运行全部 33 个示例
./build_examples.sh 02       # 只跑 02_convf16_gemm
./build_examples.sh all      # 全部
```

脚本会: 编译 `main.c` + `common/example_util.c` → 链接 `-lhvxhmx_v23` → SWIV 签名 →
push 到 `52f67807:/data/local/tmp/hvxhmx23/` → `run_main_on_hexagon 3` 跑 → 拉回 result。

设备默认 `52f67807` (仅此一台)。验收状态 (2026-08-16): 33/33 全绿, 181 门 PASS 0 FAIL (+例 15 内嵌 49 项)。

## 示例索引

| # | 目录 | 教学点 | 涉及 API |
|---|------|--------|----------|
| 01 | [01_runtime_init](01_runtime_init/) | HMX runtime 生命周期最小示例 (任何 HMX kernel 前必做) | `hmx_runtime_setup/teardown`, `get_vtcm_*`, `perf_now_us` |
| 02 | [02_convf16_gemm](02_convf16_gemm/) | fp16 GEMM 单 tile (32×32×32), 1 ULP 容差 | `hmx_convf16` |
| 03 | [03_convbbb_int8](03_convbbb_int8/) | int8 GEMM (u8×u8→u8), 走 HVX 路径, exact | `hmx_convbbb1x1_stride1`, `hmx_convbbb_stride2` |
| 04 | [04_convhbh_u16](04_convhbh_u16/) | u16 输出族 (u8×u8→u16), exact | `hmx_convhbh` |
| 05 | [05_i16_weight_convs](05_i16_weight_convs/) | i16 权重族 (bcb/bnb/hch/hnh), exact | `hmx_convbcb`, `hmx_convbnb`, `hmx_convhch`, `hmx_convhnh` |
| 06 | [06_dwconv](06_dwconv/) | depthwise 卷积 (fp16 + int8) | `hmx_dwconvf16`, `hmx_dwconvbbb` |
| 07 | [07_add](07_add/) | 元素级 fp16 加法 | `hmx_add` |
| 08 | [08_divide](08_divide/) | HVX 除法 (i32/u16/u8, floor 变体) | `hvhx_divide_flat_i32`, `hvhx_divide_u16/u8`, `floor_divide_*` |
| 09 | [09_activation](09_activation/) | hardswish + prelu 激活 | `hvhx_hardswish_flat_u16`, `hvhx_prelu_u8` |
| 10 | [10_reduction](10_reduction/) | 沿 depth 归约 (argminmax/find_max/top1/sum) | `hvhx_argminmax_*`, `find_max_*`, `top1_*`, `reducesum_*` |
| 11 | [11_lookup_unpack](11_lookup_unpack/) | 查表 + 权重解包 | `hvhx_table_lookup_flat_u8`, `hvhx_unpack_custom_weights` |
| 12 | [12_multitile_gemm](12_multitile_gemm/) | 大尺寸 fp16 GEMM (M/N/K > 32 多 tile) | `hmx_convf16` (多维度组合) |
| 13 | [13_compat_dlsym](13_compat_dlsym/) | v73/v75/v79 兼容层 dlsym + 调用 | `dlsym` + `hmx_v73_*` (兼容符号) |
| 14 | [14_hmx_peak_gemm](14_hmx_peak_gemm/) | 裸 HMX K-loop 达峰值 (~12 TFLOPS) vs 公开 wrapper | `hmx_phase0_gemm_fp16_core` + 裸 asm K-loop (低级路径) |
| 15 | [15_v2_llm_ops](15_v2_llm_ops/) | V2.1 LLM 算子层全量回归 | `hvhx_v2_*` 60 符号 |
| 16-22 | V2.2 单元 | wtcache / w4a16 / smallm / gdn_sm / dualdomain / oplist / dualcore | 见 [TEST_CASES.md](TEST_CASES.md) V2.2 节 |
| 23-33 | V2.3 单元 | fence / arena / harness / wpool / pxbridge / gdn_tree / kvcache / graph_step / gemm_dispatch / rbr / bledger | 见 [TEST_CASES.md](TEST_CASES.md) V2.3 节 + `../docs/api_v23_*.md` |

## 每个用例测了什么

[TEST_CASES.md](TEST_CASES.md) 给出每个示例的**详细逐例说明**: 测什么、数据怎么准备、golden 怎么算、容差多少、PASS 证明什么. 本索引表只是速查.

叙事式导览 (33 例逐个讲"是什么/为什么/PASS 证明什么"): 英文版 [EXAMPLES.md](EXAMPLES.md), 中文版 [EXAMPLES_cn.md](EXAMPLES_cn.md).

## 容差矩阵 (PASS 判据)

| 精度族 | 容差 | 说明 |
|--------|------|------|
| int8 (bbb/bbh/bcb/bnb/...) | 0 | exact (走 HVX int8 GEMM, 逐元素) |
| u16 输出 (hbh/hhh/hch/hnh) | 0 | exact (走 HVX int8 GEMM, 输出 u16) |
| fp16 (convf16/dwconvf16/add) | ≤1 ULP | HMX fp16 systolic, 截断到 qf16 |
| runtime/timer | 0 | setup 返回 0, VTCM 非空, 计时单调 |

## 共享工具 (common/)

- [common/example_util.h](common/example_util.h) / [.c](common/example_util.c) —
  `ex_open_result(name)` 打开结果文件, `ex_log/fill_u8/i8/u16/i16/i32/f16` 初始化 buffer (固定种子 LCG, 可复现),
  `ex_check(label, err, tol)` 记 PASS/FAIL, `ex_summary()` 汇总返回。

## 编写自己的示例

复制 [01_runtime_init/main.c](01_runtime_init/main.c) 或任一示例作模板。要点:

1. `#include "hvxhmx.h"` + `#include "example_util.h"`
2. 数据用 `static ... __attribute__((aligned(128)))` 全局 buffer (禁 VLA, DSP 栈小)
3. HMX kernel 前必须 `hmx_runtime_setup(2*1024*1024)`; HVX-only kernel 也建议 setup
4. `ex_open_result("你的示例名")` 开头, `return ex_summary()` 结尾
5. 标量 golden 用普通 C 嵌套循环, 与 kernel 输出逐元素对比

编译命令 (照搬 build_examples.sh):

```bash
hexagon-clang -mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC -std=gnu11 -Wall \
    -I/disk1/V81Dev/hvxhmx_libs/include -I/disk1/V81Dev/hvxhmx_libs/examples/common \
    -o build/test_xxx.so your_dir/main.c common/example_util.c \
    -L/disk1/V81Dev/hvxhmx_libs/lib -lhvxhmx \
    -L$SDK/rtos/qurt/computev81/lib -lqurt -lc -ldl -lgcc
python3 /disk1/V81Dev/swiv_build_utility.py -i build/test_xxx.so -o build/test_xxx.so.signed
adb -s 52f67807 push build/test_xxx.so.signed /data/local/tmp/hvxhmx_libs/test_xxx.so
adb -s 52f67807 shell "cd /data/local/tmp/hvxhmx_libs && ./run_main_on_hexagon 3 test_xxx.so"
adb -s 52f67807 shell "cat /data/local/tmp/hvxhmx_libs/你的示例名.txt"
```

## 故障排查

- **setup FAIL** — CDSP/fastrpc 未就绪。GVM 是 host 管理的, guest 重启后需 host 侧重连 (见 USERGUIDE 故障排查节)。
- **dlopen FAIL (示例 13)** — `ADSP_LIBRARY_PATH`/`CDSP_LIBRARY_PATH` 未指向部署目录, build_examples.sh 已设。
- **CDSP 库缓存** — 同名 .so 改了行为但跑出旧结果: 换部署路径或换文件名 (见 docs/data_layout.md)。
- **FUSA PD 拒载** — `.so` 见 UNDEF 符号即拒; 确认 SWIV 签名成功且 vgather=0
  (`hexagon-llvm-objdump -d test_xxx.so | grep -ci vgather` 应为 0)。


---

# V2.2 增补 (examples 16-22)

- 库换 `libhvxhmx_v22.so`, 设备目录换 `/data/local/tmp/hvxhmx23/` (与 V2.1 并存互不干扰)。
- **链接不要 `-lqurt`** (libqurt.a 非 PIC, R_HEX_32_6_X; qurt 符号运行时由 CDSP 进程解析):

```bash
hexagon-clang -mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC -std=gnu11 -Wall     $INC_HEX -I../include -I./common -o build/test_xx.so XX/main.c common/example_util.c     -L../lib -lhvxhmx_v22 -lc -ldl -lgcc
```

- 脚本自动准备并推送: `assets/{s256,smallm}` + host 打包 `blob_w4/w5.wtop` + `rms_w.f16.raw`
  + `Y_gold.raw` (K2560 金标)。
- 例 20 是分片执行器: `./run_main_on_hexagon <dom> test_20_dualdomain.so dd <tag> <start> <len>`;
  脚本编排 ser → dom3/dom4 并发, host `analyze_dd.py` 对拍切分等价。
- 结果拉回 `../results/`, 末尾汇总 PASS/FAIL; 全部实测数字进 `../PERF_REPORT.md`。
- 单元 API 手册: `../docs/api_v22_*.md` (总览 `api_v22_overview.md`)。
