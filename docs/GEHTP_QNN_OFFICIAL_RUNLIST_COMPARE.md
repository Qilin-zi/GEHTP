# GEHTP vs QNN 官方 SDK 编译产物 runlist 对比 — Qwen3-0.6B (2026-09-18)

> 方法: **同一前端** (gehtp compile 第 1-2 步产出的 qairt-converter + dlc_repair 后 DLC)
> 分别进两个 prepare 后端 — 官方 `libQnnHtp.so` (qnn-context-binary-generator, sa8797)
> vs GEHTP `hnnx_compile` (逆向重实现)。runlist 提取:
> 官方 = `--save_backend_op_mapping` 的 `qwen3_06b_bottom_mapping.json` (35146 op 全单);
> GEHTP = `qwen3_06b.wtop.manifest.json` 的 `opcodes` 序列 (1682 op)。
> 资产: `test_assets/qwen3_06b/official/` (ctx-bin 1.19GB + 两份 mapping json)。

---

## 1. 总量对比

| 维度 | GEHTP (hnnx_compile→wtop) | 官方 (libQnnHtpPrepare) |
|---|---|---|
| runlist 长度 | **1682 op** | **35146 op** (×20.9) |
| 粒度 | 节点级 (与源图 1684 节点 ~1:1) | **tile 级** (HMX 32 列 tile 分解 + VTCM 编舞) |
| blob 体积 | 1.343 GB | 1.195 GB (ctx-bin) |
| 权重形态 | f16 平铺 (slot 池, DDR 直读) | f16 + `pack_fp16_pkweights` 重排 (896 处) |
| 搬运编舞 (mapping 统计) | temps 全 DDR 池 | **DRAM read 1.19GB (权重一遍); VTCM read 2.14GB / write 1.45GB** |
| 引擎分配 | 当前标量 C (+W1 HVX env 门新接线, 默认关; W4A16 走 HMX 仅 Q4 变体) | **uses_hmx 8646 / uses_hvx 13247 / dma 7868 / null_exec 2198** |

## 2. 类型级展开率 (源图节点 → runlist op)

| 源类型 (源图数) | GEHTP | 官方 | 官方展开机理 |
|---|---|---|---|
| FullyConnected 197 | 253 MATMUL_F16 (含 attention 56) | **11972** (×60.8) | N/32 tile × (weights_to_vtcm + ConvLayer) + Concat |
| MatMul (attn) 56 | (同上) | **4930** (×88) | Slice_contig+pack+ConvLayer, 按头/列 tile |
| EltwiseBinary 392 | 252 BINARY + 140 ADD + 280 BROADCAST | **11742** (×30) | 按 64 元素 tile + SlicePad 胶 |
| RmsNorm 113 | 113 RMSNORM2 | **451** (×4) | rmsnorm_fp16.tcm 行分块 + 部分转 ConvLayer |
| Softmax 28 | 28 SOFTMAX | **448** (×16) | **每头一个** (16 头 × 28 层) |
| EltwiseNeuron(silu) 28 | 并入 UNARY(sub12) | **1344 Swish_fp16** (×48) | 按 3072/64=48 tile |
| EltwiseUnary 56 | 308 UNARY | **112 Neg** (×2) | 仅 ×2 tile |
| Transpose 140 | 140 TRANSPOSE_GEN | **5209** (×37) | 转 ConvLayer/Transpose_impl + 布局胶 |
| Reshape 506 | 0 (全为免费元数据) | **56+3248 Reshape/SlicePad** | 多数也消元 (506→56 真 op) |
| StridedSlice 112 | 112 | **224** (×2) | |
| Concat 56 | 56 CONCAT | (并入各分组胶) | |

## 3. 结构性差异 (调度形态)

1. **官方把全层 K 转置类布局 op 前提到队首**: runlist 开头连续 28 个
   `q::Concat | /Transpose_{1,6,11,…,136}` — 28 层同一类布局变换**整批前置**
   (局部性/流水考虑); GEHTP 严格拓扑序逐层交织 (RMSNORM2→MATMUL_F16→UNARY→…)。
2. **显式 VTCM 编舞**: 官方 runlist 内含 `ConvLayer.opt.weights_to_vtcm` ×7864
   (dma 标记) + `zero_to_vtcm` + `pack_fp16_pkweights_dynamic` — 权重/打包/置零
   全部作为 runlist 一等公民 op; GEHTP f16 blob 无 staging op (xop 体内 DDR 直读;
   VTCM staging 仅 W4A16 PIN/SPILL/FILL 机制, 本 blob 未用)。
3. **布局协议 op**: `ForceFormat_Crouton` ×1794 (HMX 表面布局) / `ForceFormat_Flat`
   ×448 — 官方把布局转换显式化; GEHTP 全 flat + 标量 TRANSPOSE_GEN。
4. **tile 切片胶**: `SlicePad_shape_inplace` ×3200 + `Slice_contig.tcm` ×896 —
   大 op 切 tile 的切片全在 VTCM/in-place 完成; GEHTP STRIDED_SLICE 整 op 标量。
5. **null_exec ×2198**: 官方 ~6% op 是纯元数据 (免费); GEHTP 的 Reshape/copy_sem
   同样在 emit 期消元 (本 blob 无独立 RESHAPE op)。
6. **softmax 按头拆**: 官方每头独立 (16×28=448), 与 HVX flash-attn 结构对齐;
   GEHTP 每层一个 rows=512 的整 op。

## 4. 对 GEHTP 的启示 (按 GEHTP_SCHED_HWFACTS §2/§3 原则落点)

- **E4/E5 印证**: 官方 = movement-centric 极致形态 (权重 DRAM 一遍 + 激活 VTCM 循环,
  计算 op 全部 tile 化贴着 HMX/HVX) — GEHTP MEMPLAN M4 真 DMA runlist + VTCM 驻留
  是同一方向, 当前 f16 blob 的 DDR 直读是最大性能缺口 (也解释了标量引擎分钟级耗时)。
- **可吸收**: ① 布局 op 整批前置 (编译器 pass, 零运行时成本); ② weights_to_vtcm/
  pack 显式 staging op 概念 → GEHTP f16 GEMM 的 HMX 接线 (W2) 应带 PIN 式驻留而非
  per-call repack; ③ softmax 按头拆可与 HVX flash_attn 原语直接对接;
  ④ null_exec 元数据 op 消元已对齐 (GEHTP 同做)。
- **粒度哲学差异**: GEHTP 保留大 op 给 xop 体自由度 (GDN/长链算子整体内核化路线),
  官方全 tile 化换调度自由。GEHTP W2 HMX GEMM 落地时应选 **op 内 tile 循环**
  (保留 runlist 1682 短单) 而非官方式 35146 展开 — 一次派发整图的原则 (E1) 两侧同,
  差别只在 runlist 长度, 不影响 RPC 次数。

## 5. 复现命令

```bash
# 官方 (复用 gehtp compile 的 DLC, sa8797 离线缓存)
QAIRT=/4090disk2/QCtools/qairt_2.48.40.260702
LD_LIBRARY_PATH=$QAIRT/lib/x86_64-linux-clang \
$QAIRT/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --dlc_path qwen3_06b.dlc --backend $QAIRT/lib/x86_64-linux-clang/libQnnHtp.so \
  --htp_socs sa8797 --binary_file out.serialized.bin \
  --save_backend_op_mapping --output_dir /tmp/qnn_official
# GEHTP runlist: test_assets/qwen3_06b/qwen3_06b.wtop.manifest.json 的 opcodes 字段
# (注意: manifest 的 op_ids 是 QNN IR 节点 id 序列, 不是 opcode)
```
