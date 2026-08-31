#!/usr/bin/env python3
"""conv2d+add 契约模型生成器 (GEHTP 阶段 1)

模型: X(1,32,32,32) -> Conv(3x3, 32->32, s1, same pad, 带 bias) -> Y;  Z = Y + X
ONNX 标准 NCHW 语义(shape 对称)。残差双激活 Add: 避免 converter BN-folding 把
常量 Add 折进 conv bias,保证图中 "Conv2d" 与 "Eltwise_Binary" 两个独立节点都存在。
K = 32*9 = 288 = 32*9,满足 hmx_convf16 K%32==0 约束。

产物:
  conv_add.onnx            模型
  X.f32.raw                NCHW 行主序 fp32 输入
  W.f32.raw / B.f32.raw    OIHW 权重 / 偏置 (与 net.json 权重交叉核对)
  Z_ort.f32.raw            onnxruntime fp32 参考输出 (golden #1)
  Z_ref_f16.f32.raw        fp16 纪律 host 参考 (golden #3, 供设备 1ULP 对拍)
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

N, C, H, W = 1, 32, 32, 32
KH = KW = 3
PAD, STRIDE = 1, 1

X = RNG.standard_normal((N, H, W, C)).astype(np.float32)   # NHWC
Wt = RNG.standard_normal((C, C, KH, KW)).astype(np.float32)  # OIHW
B = RNG.standard_normal((C,)).astype(np.float32)


def make_init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)


nodes = [
    helper.make_node("Conv", ["X", "W", "B"], ["Y"], name="conv1",
                     pads=[PAD, PAD, PAD, PAD], strides=[STRIDE, STRIDE]),
    helper.make_node("Add", ["Y", "X"], ["Z"], name="add1"),
]
graph = helper.make_graph(
    nodes, "conv_add",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, H, W, C])],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [N, H, W, C])],
    [make_init("W", Wt), make_init("B", B)],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(HERE, "conv_add.onnx"))
print("saved conv_add.onnx")


def conv_ref_f32(x, w, b, pad=1, stride=1):
    """NCHW x (ONNX 标准语义), OIHW w -> NCHW out (f32, 直接卷积参考)"""
    n, c, h, wc = x.shape
    o, i, kh, kw = w.shape
    assert c == i
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wc + 2 * pad - kw) // stride + 1
    cols = np.zeros((n, ho, wo, c * kh * kw), dtype=np.float32)
    for r in range(kh):
        for s in range(kw):
            cols[:, :, :, (r * kw + s) * c:(r * kw + s + 1) * c] = \
                xp[:, :, r:r + ho * stride:stride, s:s + wo * stride:stride].transpose(0, 2, 3, 1)
    # K 序 = [r,s,ci] (tap 序, 与设备 GEMM 契约一致): W OIHW -> (kh,kw,ci,co) -> KxN
    wm = w.transpose(2, 3, 1, 0).reshape(c * kh * kw, o)
    out = cols.reshape(-1, c * kh * kw) @ wm + b
    return out.reshape(n, ho, wo, o).transpose(0, 3, 1, 2)


# golden #3: fp16 纪律 (f32 累加, op 边界一次 f16 舍入)
Y = conv_ref_f32(X, Wt, B, PAD, STRIDE)
Yf16 = Y.astype(np.float16).astype(np.float32)
Z = (Yf16 + X).astype(np.float16).astype(np.float32)

for name, arr in [("X", X), ("W", Wt), ("B", B), ("Z_ort", None)]:
    pass

X.tofile(os.path.join(HERE, "X.f32.raw"))
Wt.tofile(os.path.join(HERE, "W.f32.raw"))
B.tofile(os.path.join(HERE, "B.f32.raw"))
Z.tofile(os.path.join(HERE, "Z_ref_f16.f32.raw"))

# golden #1: onnxruntime fp32
import onnxruntime as ort
sess = ort.InferenceSession(os.path.join(HERE, "conv_add.onnx"), providers=["CPUExecutionProvider"])
Z_ort = sess.run(["Z"], {"X": X})[0]
Z_ort.tofile(os.path.join(HERE, "Z_ort.f32.raw"))

# 自检: ORT fp32 与直接参考一致(1e-4 相对), fp16 纪律参考与 ORT 差在 fp16 舍入内
# 注意模型输出 Z = Y + X (残差), 参考须含 +X
Z_ref32 = conv_ref_f32(X, Wt, B, PAD, STRIDE) + X
assert np.allclose(Z_ort, Z_ref32, rtol=1e-4, atol=1e-5), "ORT mismatch"
diff = np.abs(Z - Z_ort).max() / np.abs(Z_ort).max()
print(f"saved raws; Z shape={Z.shape}; fp16-vs-ort maxrel={diff:.3e}")
assert diff < 1e-3, "fp16 reference too far from ORT"
