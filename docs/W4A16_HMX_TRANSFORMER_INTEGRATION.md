# W4A16 HMX → Transformer 管线集成任务书

状态: 2026-09-21。HMX kernel 已在 110 板验证 (256³ 位恒等 65536/65536, 1290 GFLOPS)。
① 发射器 kernel 格式已完成 (129 W4A16 op, blob 2.63GB, 闭包 byte-exact)。
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

### ① 发射器: 640B tile → kernel 格式 (编译期全可做) — 已完成 (2026-09-21)
- `wtop_ops.hpp` 的 `repack_q4_0_tiles` 换成 `pack_w4a16_kernel`:
  gguf Q4_0 反量化 → **每列** int8[-7,7] (scale=max|col|/7, 全零列→1)
  → k4-lohi + XOR 0x88 → 槽 = K*N/2 字节; 列 scale f16 → 独立槽 (N*2)。
  (列向而非块向: 块向 scale 无法从 GEMM 提因子, 列向可在出面反量化按列乘回 —
  见 ②; 这也与 W4A16 业界 per-channel 惯例一致。)
- bias=(N/32)*512 折叠 bias 槽 + atbl=8*(K/32)*4 / otbl=8*(N/32)*4 真实
  0x800-stride 偏移表槽 (闭包 m2_pack_surfaces COMPACT_STRIDE; 驱动运行时重写
  绝对指针)。
- **预收集遍坑**: wtop_emit 的通用权重槽收集遍先按 f16 建槽并缓存, 后续发射器
  全部撞缓存 → W4A16 129 op 全落 MATMUL_F16 (实锤)。修法 = OpW4Registrar 注册表:
  kernel-格式消费方由发射器自身注册, 预收集遍查同一张表 (单一真相源, 无硬编码)。
- **tie 权重双格式**: 嵌入表被 GATHER 与 GEMM 共享 → kernel-格式独立缓存
  (w4_wslots) 与 f16 缓存并存, 各消费方取各格式。
- 校验 (全绿): wt/bias 槽 vs 闭包 pack byte-exact (129/129); scale vs gguf 真值
  cos=1.0; host 参考全有限自洽; 每 GEMM vs 旧 tile 路径 cos 0.978 (列向量化
  噪声, 预期)。
- **解码双坑** (host 标量参考): ① 存储字节 = ((w+8)&0xF)^0x88 = w 的 4-bit 补码
  本身, 解码直接补码, 再 XOR = 双重变换 (w≥0 错 w-8, cos -0.58); ② scale 槽
  已含 /7, 解码勿再除 (多除 = 幅度 7× 错)。

### ② 设备侧: act f16 → u16 a16 域 + crouton 打包 (新 dc 函数) — 已完成 (2026-09-21)
- 新平台无关文件 `kernels/src/runtime/w4a16_quant.c` (host 单测 + 设备 lib 同源):
  `w4a16_act_scale` = max|a| (全零→1); `w4a16_quant_f16` = round-half-away
  (a/A_s·32767)+32768 纯 C 实现 (libc roundf 设备运行时未证; 钳位边界 v>-32767→
  65535 / v<-32768→0); `w4a16_pack/unpack_crouton` (闭包 crouton16_row4 精确正逆);
  `w4a16_dequant_out/_crouton` = A_s·S[n]·(q-32768)/32767 (S=列 scale 槽, 已含 /7)。
- 新入口 `dc_w4_run` (exec_matmul 全链调用; 例程旧 dc_w4_invoke 契约不变):
  - 设备 (dc_parts.c): 量化+crouton 融合写 VTCM 面 (**M→256 零行 pad=32768 —
    a16 域 q=0 是 real=-1.0, 零行必须填零值点!**) → FLUSH → invoke → INVALIDATE
    → crouton 序直读反量化 (行≥m 丢弃, 无中间缓冲)。
  - host (host_stubs.c): 同签名纯 f32 数学 (act 不量化; 设备 a16/>>8 噪声由
    ④ 容差门承担) — 全模型 7946240 元素 vs 旧参考 byte-exact。
- exec_matmul: carve 按 M pad 后 (kernel M=256 硬约束, m_total_minus_step=8);
  act/out 不再走 DMA 包装 (CPU 写 out_ddr → FLUSH; 旧 DMA-out 的 INVALIDATE
  会丢 CPU 写)。**M pad 256 由 ② 收编, ③ 只剩 lm_head N 分块。**

### ③ N 分块 (lm_head)
- lm_head N=248320: 权重槽 127MB > 8MB VTCM → N 分块 (每块 N≤4096),
  每块: 权重块 UDMA 入 VTCM → invoke → 输出块出; 复用 htp-hardware-scheduling 的
  DMA 双缓冲套路。其余 GEMM 最大组合 (qkv: act 512KB + wt 3MB + out 3MB +
  bias 96KB ≈ 6.6MB) 已可单次装入。

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
