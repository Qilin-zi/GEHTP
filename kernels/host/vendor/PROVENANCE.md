# Vendored: weight_pack (来自 revlibHtpPrepare)

拷贝自 `/4090disk2/Qwen35dev/revlibHtpPrepare/`，**拷贝件在本目录只读不改**（修改需先在本文件登记）；原目录零改动。

| 文件 | 源路径 | sha256 (拷贝时) | 用途 |
|---|---|---|---|
| `weight_pack.h` | `include/weight_pack.h` | `eaf93142…10ec94` | ggml Q4_0/Q8_0 → tile-major crouton 打包 API + crate 载体 + conv FP16 打包 |
| `weight_pack.cc` | `src/weight_pack.cc` | `f0b9ef1f…dccb5c4` | 实现 (纯标量 C++, 无 HVX/Q6, host 可编译) |
| `weight_layout_pack.py` | `tools/weight_layout_pack.py` | `48d3d7af…050372a` | QNN conv golden 参考实现 (190/190 验证), 只作参考不改 |
| `test_weight_pack.orig.cc` | `test/test_weight_pack.cc` | `1b2e576f…249b04` | 原测试 (依赖 revlib 图机器, 不在本树编译, 仅参考); 本树用 `tests/` 独立重写 |

来源血统 (上游注释): ggml-hexagon `ggml-hexagon.cpp:478-529 (q4_0), 701-747 (q8_0)`; crouton pair-interleave 出自 `hvxhmx_libs hmx_crouton.h`; conv 布局 RE 自 InceptionV3 golden。

## 本树新增 (开发在这里)

```
host/gguf_reader.{h,cc}     GGUF v3 最小读取器 (header/tensor table/Q4_0 blocks)
host/repack_sweep.cc        模型逐 tensor sweep → crate blob + manifest.json (sha)
host/requant_w4a16.cc       Route B: Q4_0 → int4[-7,7] 对称 + folded bias (重量化, 非 permutation)
host/verify_roundtrip.cc    每 tensor unpack(pack(W))==W bit-exact
tests/                      独立闭合测试 (不依赖 vendor 的原 test 框架)
```

## 两条路线的区分 (重要, 别混)

| | Route A (本 vendor 件) | Route B (需新写) |
|---|---|---|
| 布局 | tile-major crouton 576B/tile (Q4_0, 含 F16 scale) | `pack_w4_kblock32_nmajor_k4_lohi XOR 0x88` |
| 变换性质 | **纯重排** (permutation, 可 bit-exact round-trip) | **重量化** (Q4_0 per-32 F16 scale → 对称 int4[-7,7], 不可逆) |
| 消费方 | hvxhmx_libsV2 fp16 tiled GEMM (dequant 路径) | htpw4a16_v81 native W4A16 HMX kernel (已闭合) |
| 验收 | round-trip bit-exact | dequant 误差有界 + 喂已闭合 kernel 对 scalar gold |
