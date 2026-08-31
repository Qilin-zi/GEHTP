# src/ops/ — op 库

对应真实 `*.cc` (200+ op 文件)。op 注册 + 参数解析 + 指令选择 + host execute。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `ops.cpp` | `op_package_ops_opts_registration.cc` | **154 op 注册**, generic_construct (提 op_data->params, 选 wrapper), TypicalOp::cost/serialize_internal/**execute** (host reference, 10+ op 数值), select_wrapper (HMX/HVX/Scalar) |
| `hlx_hmx.cpp` | `hlx_*.cc` / `hmx.cc` | 17 个 HLX/HMX 扩展 op (Abs/Ceiling/ConvWeights/Elementwise/Exp/Floor/Gelu/Hadamard/LinearClip/Prelu/QElementwise/ResizeBilinear/RotaryPosEmbd/Round/Softmax + Hmx) |
| `quantize.cpp` | `quantize.cc` | Quantizer: compute_params / quantize_int8 / dequantize 往返 |
| `weights.cpp` | `compose_weights.cc` / `scatter_conv_weight.cc` | WeightProcessor: compose / convert / scatter (多 NSP) / wtshare / pickle |

## 指令选择 (select_wrapper)

```
HMX_Matrix     Conv/MatMul/Dense + Int8/Int16 + v73+
HMX_MatrixInt4 同上 + Int4/UInt4
HVX_Vector     elementwise/激活 + 量化 dtype
HVX_Scalar     浮点后备
```

## host execute (TypicalOp::execute)

按 op_type_name 分派, Float32 参考:
- Relu/ConvActivations: max(0,x)
- Add/Sub/Mul/Div: 逐元素
- Sigmoid/Tanh/Exp/Neg
- Softmax: 数值稳定 (减 max)
- Conv/MatMul: 简化直通 (非真实卷积)
