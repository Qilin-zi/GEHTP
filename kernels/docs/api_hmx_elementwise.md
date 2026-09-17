# HMX Elementwise Add API 参考

声明: [`include/hmx_kernels.h`](../include/hmx_kernels.h) (Element-wise 段)

残差连接 (ResNet) 用的逐元素加.

## hmx_add (fp16)

```c
void hmx_add(const __fp16 *a, const __fp16 *b,
             const __fp16 *bias, __fp16 *out,
             uint32_t M, uint32_t N);
```

```
out[m,n] = max(0, a[m,n] + b[m,n] + bias[n])     (含 ReLU)
```

| 参数 | 含义 |
|------|------|
| `a`, `b` | 输入 [M][N], 各 128B 对齐 |
| `bias` | per-column 偏置 [N] |
| `out` | 输出 [M][N] |
| `M, N` | 行/列 |

fp16 路径: add 必须经 Vqf16 (HVX 半精度加) 再 narrow. 容差 ≤1 ULP.

example: [examples/07_add](../examples/07_add/main.c)
