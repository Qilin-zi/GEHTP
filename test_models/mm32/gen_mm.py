#!/usr/bin/env python3
"""最小单 MatMul 契约模型: Y = X @ W  (W const → FullyConnected)
隔离目标: 权重加载(转置方向) + FC 标量/HMX matmul。dims 由 argv 给。
用法: python3 gen_mm.py [M] [K] [N]
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
TAG = f"m{M}_k{K}_n{N}"

RNG = np.random.default_rng(7)

# W 非对称 (确保转置错误可被检出): W[k,n] = 0.1*sin + 行/列扰动
W = (0.15 * RNG.standard_normal((K, N))).astype(np.float32)
W = (W - np.mean(W)).astype(np.float32)  # 去均值, 数值更敏感

Wt = helper.make_tensor("W", TensorProto.FLOAT, [K, N], W.tobytes(), raw=True)
node = helper.make_node("MatMul", ["X", "W"], ["Y"])
graph = helper.make_graph(
    [node], "mm32",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
    initializer=[Wt],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, f"{HERE}/mm_{TAG}.onnx")

# ---- 金标 (f16 纪律, 对齐 gen_transformer) ----
def f16(a):
    return a.astype(np.float16).astype(np.float32)

X = RNG.standard_normal((M, K)).astype(np.float32)
Xf16 = X.astype(np.float16)
Xf16.tofile(f"{HERE}/mm_{TAG}.in.f16.raw")
Wf = f16(W)
Y = f16(Xf16.astype(np.float32) @ Wf)
Y.astype(np.float16).tofile(f"{HERE}/mm_{TAG}.gold.f16.raw")
print(f"[OK] mm_{TAG}.onnx  M={M} K={K} N={N}")
print(f"     gold Y 范围 [{Y.min():.4f}, {Y.max():.4f}] mean {Y.mean():.4f}")
