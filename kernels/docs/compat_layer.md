# 兼容层 (v73 / v75 / v79) — 附录

声明: [`src/compat/hmx_v73_compat.c`](../src/compat/hmx_v73_compat.c) 等.

库附带 131 个薄 dispatch wrapper, 对齐老版本 HMX 算子的符号名. **内部全部尾调对应 v81
族函数** (数学相同, 仅 HMX 编码效率差异), 所以行为由构造保证正确.

## 何时用

- 你的代码 / 模型已 hardcode 老版本符号名 (如 `hmx_v73_convbbb1x1_stride1`), 不想改调用点.
- 从老 SDK 迁移, 需要 dlopen 老 .so 里同名的符号.

不用就忽略它们 — v81 通用族 ([api_hmx_gemm.md](api_hmx_gemm.md)) 是推荐 API.

## 符号清单

### v73 (119 个, `hmx_v73_*`)
全族覆盖: add, convbbb(8+stride), convbcb(2), convbnb(6+stride), convbnh(1),
convf16(10), convhbh(10+stride+Nx), convhch(3+stride), convhhh(6+stride+Nx),
convhnh(7+stride+Nx), dwconvbbb(1).

代表 (完整列表 `hexagon-nm -D lib/libhvxhmx.so | grep hmx_v73_`):
```
hmx_v73_add
hmx_v73_convbbb1x1_stride1
hmx_v73_convbbb1x1deep_stride1_sparsity
hmx_v73_convbnb_stride1
hmx_v73_convbnh1x1_stride1
hmx_v73_convf16_5x5_stride1
hmx_v73_convhbh1x1_lp_stride1              /* lp = low-power loop 优化 */
hmx_v73_convhch_5x5_stride1
hmx_v73_convhhh_NxN_stride1
hmx_v73_convhnh1x1deep_lp_stride1
hmx_v73_dwconvbbb1x1_stride1
```

### v75 (6 个, `hmx_v75_*`)
```
hmx_v75_convbbh1x1_stride1
hmx_v75_convhbh1x1_stride1
hmx_v75_convhbh1x1deep_stride1
hmx_v75_convhbh_dilate_stride1
hmx_v75_convhbh_stride1_aligned
hmx_v75_convhbh_stride2
```

### v79 (6 个, `hmx_v79_*`)
```
hmx_v79_convbnb_stride1
hmx_v79_convbnb_stride1_aligned
hmx_v79_convbnb_stride1_aligned_sparsity
hmx_v79_convbnb_stride1_sparsity
hmx_v79_convbnb_stride2
hmx_v79_convbnb_stride2_sparsity
```

## dlsym 用法

```c
#include <dlfcn.h>

void *h = dlopen("libhvxhmx.so", RTLD_NOW | RTLD_LOCAL);
typedef void (*fn_t)(const uint8_t*, const uint8_t*, const int32_t*,
                     uint8_t*, uint32_t, uint32_t, uint32_t);
fn_t f = (fn_t)dlsym(h, "hmx_v73_convbbb1x1_stride1");
if (!f) { /* 符号缺失 */ }
f(act, wgt, bias, out, M, K, N);
```

完整 dlsym + 功能抽样验证见 [examples/13_compat_dlsym](../examples/13_compat_dlsym/main.c)
和权威测试 `hvxhmx_kernel/test/test_compat_sym.c` (39/39 符号 + 25/25 功能 PASS).

## 命名约定

| 族 | 维度前缀 | 单词前缀 |
|----|----------|----------|
| convbbb / convhbh / convhnh | **无下划线** (`convbbb1x1`) | 有下划线 (`convbbb_stride1`) |
| convf16 | **全有下划线** (`convf16_1x1_stride1`) | 有下划线 |

`lp` 变体 (`convhbh1x1_lp_stride1`): 数据类型同非 lp (act u8 + wgt i8), "lp" 仅 loop 优化.

## v85 stub

17 个 v85 符号明确 **`return -1` 不实现** (本设备无 v85 HMX). 不要调用.
