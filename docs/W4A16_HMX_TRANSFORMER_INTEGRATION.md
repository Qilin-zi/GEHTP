# W4A16 HMX → Transformer 管线集成任务书

状态: 2026-09-21。HMX kernel 已在 110 板验证 (256³ 位恒等 65536/65536, 1290 GFLOPS)。
本任务书 = 把 HMX 接进 0.8B transformer 的 W4A16 GEMM 路径。相关提交 a284e63 (host 标量参考)。

## 已完成的基线

- 例 17_w4a16_gemm 上板: `vs_gold2563_byteexact / rerun_byteexact / invoke_sanity` 3 门全绿。
  资产 `/data/local/tmp/hvxhmx23/assets/s256/`, 主机源 `kernels/assets/s256/`。
- Host 标量 W4A16 参考 (tile 解码+f32 GEMM) bit-exact, 129 个 W4A16 op 全过,
  输出全有限 (cos 0.127 vs memplan9 基线 — 即数值分歧与 W4A16 无关, 见 §5)。
- 发射器 GEHTP_TILE=1 tile 路由 + 显式供给槽 9 参 arity (bias/atbl/otbl 槽 id 进 args)。

## kernel 精确契约 (闭包权威来源)

来源: `/4090disk2/htpw4a16_v81/closure_host/m2_pack_surfaces.py` +
`qcom_htp_link/example/handwritten_hmx_matmul/prepare_owned_inputs.py`

- **权重**: int8 [-7,7] → `nib=(w+8)&0xF` → `pack_w4_kblock32_nmajor_k4_lohi`:
  kb(外) × N32 × kg(4) × n×kr(4) lohi → K*N/2 字节 → **XOR 0x88**。
- **folded bias**: `pack_native_a16_bias` → (N/32)*512 字节 (scale 折叠, u16 饱和铁律)。
- **激活**: u16 [M,K] a16 域 (反量化 (q+offset)×scale; encoding min=-8/7 max=1.0,
  ÷7 域常量 −18 在 kernel 内) → `pack_a16_crouton16_row4_surface`:
  row4_phase(8) 外 × kt × m32_group × row_pair(2), 相邻两行成对。
- **atbl/otbl**: 8×(K/32) 个 u32 字节偏移 (表 stride 0x800/条目), 驱动重写为绝对指针。
- **约束**: m/k/n %32==0; M 按 256 行批量 (m_total_minus_step=8); wtcache 独占 VTCM。
- **驱动**: `dc_parts.c` carve (act/out/wt/bias/atbl/otbl/mask/extra, 2KB 对齐) +
  `w4a16_invoke` (表重写+FLUSH+裸 kernel); 权重/偏置 UDMA 搬 VTCM (dma_to_vtcm)。

## 集成步骤 (按序)

### ① 发射器: 640B tile → kernel 格式 (编译期全可做)
- `wtop_ops.hpp` 的 `repack_q4_0_tiles` 换成 kernel 格式:
  gguf Q4_0 反量化 → 每 32×32 块量化到 int8[-7,7] (scale=max|v|/7, 行向或块向)
  → k4-lohi + XOR 0x88 → 槽 = K*N/2 字节; scale 折叠进 folded_bias 槽。
- 生成的 bias/atbl/otbl 槽: bias=(N/32)*512 (折叠 scale), atbl=8*(K/32)*4 偏移表,
  otbl=8*(N/32)*4 偏移表 — 偏移公式按闭包 m2_pack_surfaces 的 COMPACT_STRIDE 0x800。
- 校验: host 标量参考 (dc_w4_invoke 桩) 改为按新格式解码 → 与 python 独立解码 bit-exact
  (现 tile 解码已在 host_stubs.c, 改格式同步改)。

### ② 设备侧: act f16 → u16 a16 域 + crouton 打包 (新 dc 函数)
- 运行时新函数 (dc_parts.c 或 oplist_exec.c): f16 面 → u16 量化 (encoding 常量
  对齐闭包 _a16_encoding) → crouton16_row4 面 (闭包 pack 公式) → atbl 偏移表生成
  (编译期槽已给模板? 否 — atbl 偏移 = 纯公式, 设备侧或发射器都可生成; 驱动只重写指针)。
- 输出: u16 面 → 反量化 f16 (otbl 面 + inv_crouton)。

### ③ M pad 256 + N 分块
- M=32 → 零行填充到 256 (act 面 pad, 输出取前 32 行)。
- lm_head N=248320: 权重 tile 159MB > 8MB VTCM → N 分块 (每块 N≤4096),
  每块: 权重块 UDMA 入 VTCM → invoke → 输出块出; 复用 htp-hardware-scheduling 的
  DMA 双缓冲套路。

### ④ 单 GEMM 对拍门
- 先用闭包资产 (s256) 验证新打包器产出的面 == 闭包资产 (byte-exact)。
- 再随机 [M,K,N] (256/512 基) 对拍 host 标量参考 vs 设备 kernel 输出。

### ⑤ 全模型上板
- GEHTP_TILE=1 重编 blob (权重新格式) → host 数值门 → 110 板全模型 → judge_logits。

## 挂账项 (§5)

- **全模型数值 cos≈0.06-0.11 vs golden (所有路径共有, 非 W4A16 特有)**。
  已修 12+ 真 bug (末轴/池耗尽/gguf match 表偏移/Softplus/Pad/kw/SELECT 广播/
  双 sigmoid/int32 输入/bias 槽/tile 路由/malloc 截断 — 见 git log 0e61587 前)。
  剩余疑点: conv 因果核与 gguf 权重序对齐 (params.bin 布局未对上)、
  权重 [K,N]vs[N,K] 布局 (转置实验 cos 0.094 未决)、attn_iter 递归链细节。
- **设备 PD 堆**: 3.3GB (反量化) blob read fail; 1.67GB 可跑; 新格式权重 = K*N/2
  更小, 2.68GB tile 待测; 长期需流式/映射权重。

## 关键文件

- kernel: kernels/src/hmx/w4a16_driver_dc.c, w4a16_v81deep_conv1x1_kernel.inc
- 驱动/执行: kernels/src/runtime/dc_parts.c (carve/invoke), oplist_exec.c exec_matmul
- host 标量参考: kernels/host/host_stubs.c (dc_w4_carve/invoke 真数学)
- 发射器: compiler/tools/wtop_ops/op_matmul.cpp (w4 路由+供给槽), wtop_ops.hpp (repack)
- 闭包资产: /4090disk2/htpw4a16_v81/ (closure_host/*.py = 格式权威)
- 板测: kernels/examples/17_w4a16_gemm/main.c
