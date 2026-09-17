# 编码规范

本库源码遵循的约定. 改动库代码时遵守, 保证风格一致 + 行为可验证.

## 1. 文件头 banner

每个 `.c` / `.h` 顶部统一格式:

```c
/*
 * <file> — <一句话职责>
 * =====================================================================
 * <所属模块: runtime / hmx / hvx / compat>
 * <关键约束 (若有): 编译特例 / 设备限制 / 行为不变式>
 */
```

不写逆向期叙述 (反汇编 ground truth / Phase-N 探索 / silent NOP 诊断). 这些背景属于
`hvxhmx_kernel/SUMMARY.md`, 不属于库源码.

## 2. 命名

| 前缀 | 含义 | 例 |
|------|------|----|
| `hmx_*` | HMX 族函数 (走 HMX/HVX GEMM 路径) | `hmx_convf16` |
| `hvhx_*` | HVX 族函数 (走 HVX SIMD) | `hvhx_divide_u8` |
| `HVHX_*` | 公共宏 / 常量 | `HVHX_VEC_ELEM_U8` |
| `hmx_v{73,75,79}_*` | 兼容层 wrapper | `hmx_v73_convbbb1x1_stride1` |
| `HMX_*` | HMX 内部宏 (intrinsic 封装) | `HMX_LOAD_TILES_FP16` |

**公开 API 符号名不可改** — 它们是 .so 契约 (与 libQnnHtpV81.so 对齐, dlsym 依赖).

## 3. 函数注释 (公共 API)

用 Doxygen 风格 (头文件内):

```c
/**
 * @brief 一句话功能.
 * @param[in]  act   激活 [M][K], 128B 对齐.
 * @param[out] out   输出 [M][N], 128B 对齐.
 * @pre  K % 32 == 0;  hmx_runtime_setup() 已调.
 * @return 无 (或错误码说明).
 * @note 关键行为 (饱和/除零/容差).
 */
```

参数方向、对齐、尺寸约束、数学公式、除零/饱和行为必写.

## 4. include 组织

- 公开头 (`include/`): 只 include 公开头. `extern "C"` + include guard 全覆盖.
- 内部头 (`hmx_common.h` / `hmx_fields.h` / `q6_intrinsics.h`): 库内部用, 不进公共
  umbrella (`hvxhmx.h`).
- `.c` include 顺序: 自身头 → 相关公共头 → SDK 系统头 (`<hexagon_types.h>` 等).

调用方只 `#include "hvxhmx.h"`.

## 5. 修饰符

- `const` + `__restrict__` 一致用于只读输入指针
- `static` 用于文件局部 helper
- 公开函数不加 `static`

## 6. 行为不变式 (硬约束)

改库代码必须遵守:

1. **算法 / intrinsic 序列 / ABI 不变**. "梳理"只动注释 / banner / include 组织,
   不动数值路径.
2. **回归门**: 改后 `./verify_libs.sh` 必须仍 20/20 + 39/25 PASS + vgather=0.
3. **编译特例保留**:
   - `hvx_int8gemm.c` → `-O1` (软件流水线 stale-vector bug)
   - `INT8_FILES` 空 (本设备 int8 全走 HVX)
4. **诊断 core 隔离**: `phase1_*` / `crouton_ex` / `diag` / `pseudoint8` 标注为
   "实验/诊断, 非稳定 API", 不鼓励生产用.

## 7. 验证 (改后必做)

```bash
./build_libs.sh                    # 重编
./verify_libs.sh                   # 设备回归
hexagon-llvm-objdump -d lib/libhvxhmx.so | grep -ci vgather   # = 0
```

改动涉及核心循环时, 抽检 `objdump` 前后差异应只有寄存器分配/行号, 无指令语义差异.

## 8. 诊断输出

- 保留 `hmx_common.c` 里 `FARF(ALWAYS, ...)` 的设备状态日志 (power_on / setup 成败) —
  那是必要的运行时可见性.
- 删孤立 debug `printf` / `FARF` (kernel 内部).
