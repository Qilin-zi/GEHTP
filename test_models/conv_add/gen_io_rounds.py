#!/usr/bin/env python3
"""conv_add 多组输入 + fp16 纪律金标生成 (GEHTP 阶段 10, Level 1 输入轮换)

产物(每轮 i=0..2):
  in{i}.f16.raw   随机输入 NCHW f16 (seed 100+i, 与模型权重无关)
  gold{i}.f16.raw fp16 纪律金标: Y=conv_f32(Wq,X)→f16; Z=(Yf32+X)→f16 → NCHW
与设备执行算法一致: conv/add 均 f32 累加, op 边界一次 f16 舍入(1ULP 对拍基准)。
权重与 conv_ref 与 gen_conv_add.py 同源(OIHW → [kh,kw,ci,co])。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, C, H, W = 1, 32, 32, 32
KH = KW = 3
PAD, STRIDE = 1, 1

Wt = np.fromfile(os.path.join(HERE, "W.f32.raw"), dtype=np.float32).reshape(C, C, KH, KW)
Bt = np.fromfile(os.path.join(HERE, "B.f32.raw"), dtype=np.float32)
# 设备槽里是 params.bin 的 f16 舍入权重(converter f32→f16)→ 拓宽回 f32
Wt = Wt.astype(np.float16).astype(np.float32)
Bt = Bt.astype(np.float16).astype(np.float32)
Wq = Wt.transpose(2, 3, 1, 0).reshape(KH * KW * C, C)  # [kh,kw,ci,co] 序 K×N


def conv_ref_f16(x_nchw):
    """NCHW f32 输入 → NCHW f16 纪律输出

    与设备 exec_conv2d/exec_add 完全同算法同累加序:
      acc = bias[c] (f32); for k in 0..K-1: acc += A[r,k]*W[k,c] (串行 f32)
      → f16 存储; add: f32(a+b) → f16。逐元素串行, 保 1ULP 对拍基准。
    """
    # NCHW → NHWC
    x_nhwc = x_nchw.reshape(N, C, H, W).transpose(0, 2, 3, 1)
    xp = np.pad(x_nhwc, ((0, 0), (PAD, PAD), (PAD, PAD), (0, 0)))
    K = KH * KW * C
    cols = np.zeros((N, H, W, K), dtype=np.float32)
    for r in range(KH):
        for s in range(KW):
            cols[:, :, :, (r * KW + s) * C:(r * KW + s + 1) * C] = \
                xp[:, r:r + H, s:s + W, :]
    A = cols.reshape(-1, K)
    Wf = Wq.astype(np.float32)
    Bf = Bt.astype(np.float32)
    M = A.shape[0]
    y = np.zeros((M, C), dtype=np.float32)
    for r in range(M):
        for c in range(C):
            acc = Bf[c]
            for k in range(K):
                acc += A[r, k] * Wf[k, c]   # 设备同序串行 f32
            y[r, c] = acc
    y = y.reshape(N, H, W, C).astype(np.float16).astype(np.float32)  # conv 边界 f16
    z = np.empty_like(y, dtype=np.float32)
    xf = x_nhwc.astype(np.float32)
    for i in range(y.size):                                        # add 同序(逐元素)
        z.flat[i] = np.float32(np.float32(y.flat[i]) + xf.flat[i])
    z = z.astype(np.float16).astype(np.float32)                    # add 边界 f16
    return z.transpose(0, 3, 1, 2).reshape(-1)                     # → NCHW


for i in range(3):
    rng = np.random.default_rng(100 + i)
    X = rng.standard_normal((N, C, H, W)).astype(np.float32)
    X.astype(np.float16).tofile(os.path.join(HERE, f"in{i}.f16.raw"))
    # 设备看到的是 f16 舍入后的输入(槽内 f16)→ golden 同步舍入
    X = X.astype(np.float16).astype(np.float32)
    Z = conv_ref_f16(X)
    Z.astype(np.float16).tofile(os.path.join(HERE, f"gold{i}.f16.raw"))
    print(f"round {i}: in{i}.f16.raw + gold{i}.f16.raw (f16)")
print("done")
