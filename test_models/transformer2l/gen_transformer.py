#!/usr/bin/env python3
"""2 层单头 transformer 契约模型 (GEHTP 先跑通)

模型: X(L,D) -> 层0[ RMSNorm -> attn(QKV投影,QK^T,scale,softmax,PV,Wo,残差)
                    -> RMSNorm -> FFN(up, SiLU, down, 残差) ]
                -> 层1(同) -> RMSNorm -> Z

只用已验证算子:
  MatMul        投影/QK^T/PV  (OP_MATMUL_F16)
  RMSNorm 分解  Pow→ReduceMean→Add(eps)→Sqrt→Div→Mul  (fuses to RmsNorm)
  Softmax       axis=-1                                   (OP_SOFTMAX_F16)
  Transpose     K^T                                       (OP_TRANSPOSE_GEN_F16)
  Sigmoid+Mul   SiLU 门控
  Add           残差
  Mul           scale

产物:
  transformer2l.onnx  模型 (opset 13)
  in0.f16.raw         f16 输入 X [L,D]
  gold0.f16.raw       f16 纪律金标 (f32 累加, op 边界一次 f16 舍入, 权重 f16 舍入)
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

L, D, FF = 8, 32, 64      # seq, hidden, ffn
HEAD = D                 # 单头
EPS = 1e-6
SCALE = 1.0 / np.sqrt(float(HEAD))


def make_init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)


nodes = []
inits = []


def add_init(name, arr):
    inits.append(make_init(name, np.asarray(arr, dtype=np.float32)))
    return name


def w(name, shape):
    # Xavier 正态 (fan_in = shape[0]), 保持激活 ~O(1), 防 softmax exp 溢出
    a = RNG.standard_normal(shape).astype(np.float32) * (1.0 / np.sqrt(float(shape[0])))
    add_init(name, a)
    return a


def gamma(name):
    g = RNG.uniform(0.9, 1.1, (D,)).astype(np.float32)
    add_init(name, g)
    return g


# ---- 常量 ----
add_init("two", np.float32(2.0))
add_init("eps", np.float32(EPS))
add_init("scale", np.float32(SCALE))

# ---- 权重 ----
Wq   = [w(f"Wq{i}",   (D, HEAD)) for i in range(2)]
Wk   = [w(f"Wk{i}",   (D, HEAD)) for i in range(2)]
Wv   = [w(f"Wv{i}",   (D, HEAD)) for i in range(2)]
Wo   = [w(f"Wo{i}",   (HEAD, D)) for i in range(2)]
g1   = [gamma(f"g1_{i}")       for i in range(2)]
g2   = [gamma(f"g2_{i}")       for i in range(2)]
Wup  = [w(f"Wup{i}",  (D, FF))   for i in range(2)]
Wdn  = [w(f"Wdn{i}",  (FF, D))   for i in range(2)]
gfin = gamma("g_final")


def rmsnorm(x_name, gamma_name, tag):
    """分解 RMSNorm: Pow→ReduceMean→Add(eps)→Sqrt→Div→Mul"""
    x2   = f"{tag}_x2"
    mean = f"{tag}_mean"
    ae   = f"{tag}_ae"
    sq   = f"{tag}_sq"
    dv   = f"{tag}_dv"
    out  = f"{tag}_out"
    nodes.append(helper.make_node("Pow",        [x_name, "two"],      [x2]))
    nodes.append(helper.make_node("ReduceMean", [x2],                 [mean], axes=[-1], keepdims=1))
    nodes.append(helper.make_node("Add",        [mean, "eps"],        [ae]))
    nodes.append(helper.make_node("Sqrt",       [ae],                 [sq]))
    nodes.append(helper.make_node("Div",        [x_name, sq],         [dv]))
    nodes.append(helper.make_node("Mul",        [dv, gamma_name],     [out]))
    return out


cur = "X"
for i in range(2):
    # ---- RMSNorm1 + attention ----
    r1 = rmsnorm(cur, f"g1_{i}", f"l{i}_r1")
    q = f"q{i}"; k = f"k{i}"; v = f"v{i}"
    nodes.append(helper.make_node("MatMul", [r1, f"Wq{i}"], [q]))
    nodes.append(helper.make_node("MatMul", [r1, f"Wk{i}"], [k]))
    nodes.append(helper.make_node("MatMul", [r1, f"Wv{i}"], [v]))
    kt = f"kt{i}"
    nodes.append(helper.make_node("Transpose", [k], [kt], perm=[1, 0]))
    sc = f"sc{i}"
    nodes.append(helper.make_node("MatMul", [q, kt], [sc]))
    scs = f"scs{i}"
    nodes.append(helper.make_node("Mul", [sc, "scale"], [scs]))
    at = f"at{i}"
    nodes.append(helper.make_node("Softmax", [scs], [at], axis=-1))
    cx = f"cx{i}"
    nodes.append(helper.make_node("MatMul", [at, v], [cx]))
    o = f"o{i}"
    nodes.append(helper.make_node("MatMul", [cx, f"Wo{i}"], [o]))
    res1 = f"res1_{i}"
    nodes.append(helper.make_node("Add", [cur, o], [res1]))
    # ---- RMSNorm2 + FFN (SiLU) ----
    r2 = rmsnorm(res1, f"g2_{i}", f"l{i}_r2")
    up = f"up{i}"
    nodes.append(helper.make_node("MatMul", [r2, f"Wup{i}"], [up]))
    sig = f"sig{i}"
    nodes.append(helper.make_node("Sigmoid", [up], [sig]))
    act = f"act{i}"
    nodes.append(helper.make_node("Mul", [up, sig], [act]))
    dn = f"dn{i}"
    nodes.append(helper.make_node("MatMul", [act, f"Wdn{i}"], [dn]))
    cur = f"res2_{i}"
    nodes.append(helper.make_node("Add", [res1, dn], [cur]))

Z = rmsnorm(cur, "g_final", "final_rms")

graph = helper.make_graph(
    nodes, "transformer2l",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [L, D])],
    [helper.make_tensor_value_info(Z,  TensorProto.FLOAT, [L, D])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(HERE, "transformer2l.onnx"))
print(f"[OK] transformer2l.onnx: {len(nodes)} nodes, {len(inits)} initializers")


# ================= 金标 (f16 纪律) =================
def f16(a):
    return a.astype(np.float16).astype(np.float32)


# 设备槽里是 converter f32→f16 舍入后的权重 → 金标同步舍入
Wq_f = [f16(Wq[i]) for i in range(2)]
Wk_f = [f16(Wk[i]) for i in range(2)]
Wv_f = [f16(Wv[i]) for i in range(2)]
Wo_f = [f16(Wo[i]) for i in range(2)]
g1_f = [f16(g1[i]) for i in range(2)]
g2_f = [f16(g2[i]) for i in range(2)]
Wup_f = [f16(Wup[i]) for i in range(2)]
Wdn_f = [f16(Wdn[i]) for i in range(2)]
gfin_f = f16(gfin)


def rms_np(x, g):
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    r = x / np.sqrt(ms + EPS)
    return f16(r * g)


def softmax_np(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return f16(e / e.sum(axis=-1, keepdims=True))


def mm(a, w):
    return f16(np.matmul(a.astype(np.float32), w.astype(np.float32)))


def run_ref(x):
    x = x.astype(np.float32)
    for i in range(2):
        r1 = rms_np(x, g1_f[i])
        q = mm(r1, Wq_f[i]); k = mm(r1, Wk_f[i]); v = mm(r1, Wv_f[i])
        s = mm(q, k.T)
        s = f16(s * SCALE)
        a = softmax_np(s)
        c = mm(a, v)
        o = mm(c, Wo_f[i])
        x = f16(x + o)
        r2 = rms_np(x, g2_f[i])
        up = mm(r2, Wup_f[i])
        act = f16(up * (1.0 / (1.0 + np.exp(-up))))   # SiLU
        dn = mm(act, Wdn_f[i])
        x = f16(x + dn)
    return rms_np(x, gfin_f)


X = RNG.standard_normal((L, D)).astype(np.float32)
Xf16 = X.astype(np.float16)
Xf16.tofile(os.path.join(HERE, "in0.f16.raw"))
Zref = run_ref(Xf16.astype(np.float32))
Zref.astype(np.float16).tofile(os.path.join(HERE, "gold0.f16.raw"))
print(f"[OK] in0.f16.raw {Xf16.shape} / gold0.f16.raw {Zref.shape}")
print(f"     gold 输出范围 [{Zref.min():.4f}, {Zref.max():.4f}]")
