# T012 全量匹配表 + W4 现状盘点 — US2 机制裁决（2026-10-09）

## 盘点结论（代码实查 + 数据回放，非推测）

spec 对 US2 的前提「`--gguf/--match` 通道完整、仅未启用」**不成立**。实查两条 W4 路线的真实状态：

### 路线 A：Q4 tile 存储 + 设备 dequant → f16 HMX GEMM（PROVENANCE.md Route A）
- 消费链：emit `repack_q4_0_tiles`(wtop_ops.hpp) 产 640B tile（payload 576 + 64B pad）→
  设备 `hvhx_v2_dequant_tiled_to_fp16`（ggml 移植 HVX 反量化）直写 crouton wgt 面 →
  现役 `exec_matmul_f16_hmx`（OP_MATMUL_F16/SPLIT，0.8B lm_head 51.7× 就是它）。
- 原缺陷（已修）：旧 emit repack 是「per-k 重量化 + 转置布局」自发格式；设备侧无 Q4 槽
  消费分支。本期修复 = 重写 repack 为 nibble 置换 + exec_matmul_f16/exec_gather 增 Q4 分支。
- 性质：权重域 f16 2 B/elem → 0.625 B/elem（640/32/32）= **FC/MM 全 Q4 后 ~403MB +
  兜底 f16 ~1MB < 0.6GB（SC-003 ✅；embed 走 Gather Q4 行级 dequant，f16 兜底会 509MB
  破预算，故 Gather 是 Q4 消费者第 1 类）**；HMX 算力仍 f16 率（无 4-bit 算力加速）；
  权重 DDR 流量 3.2× 降。
- 精度：权重 = GGUF Q4_0 原量化（nibble 置换零损），计算 f16 → FR-004 的 q4_0 golden 口径门正确。
- 性能顺带收益：HVX dequant 一次调用替代 exec_matmul_f16_hmx 原 wgt 标量 pack 循环
  （~4s → ~0.15s 级），US2 对 wall 净负。

### 路线 B：t10 原生 W4A16 HMX kernel（OP_MATMUL_W4A16, exec_matmul + dc_w4_*）
- 设备侧已闭合（s2560 oracle cos 0.99999988；003 T014/T015 合成 blob 板验证 w4v3=7654a8f6）。
- **四重缺口**（实查）：
  1. 权重 = 512B/tile 紧凑格式 + folded_bias/act_table/out_table 三张供给槽（exec_matmul 按槽长扫找），
     打包器 = `requant_w4a16.cc`（PROVENANCE.md「本树新增」清单）**从未写、git 无痕**；
     原 t10 oracle 目录 `/disk1/DFlashSpecCppRunner/htpw4a16_v81/` 已随盘清理消失。
  2. act 需 u16 对称面（32768 偏置、compact_stride 2048 tile-major，数值回放实证非 row-major f16）。
  3. out 为 crouton u16 面（exec 有还原函数，但仅 t10 路径接线）。
  4. M 必须 256 倍数（api_v22_w4a16.md 硬约束；v17 prefill M=32 → 需 pad-256 面）。
  5. 重量化数值方案（per-block scale 如何折进 u16 act/int4[-7,7]//7/bias_fold）是开放设计题，非 sprint 尺寸。
- 数值回放取证：packed_weight.raw（K·N/2=3276800B）两种 nibble 布局 + per-n (s,b) 线性拟合
  对设备输出 dev_w4.f16.raw 均无相关（resid ~31k LSB）→ 紧凑格式不是无 scale 的裸 nibble，
  佐证 folded_bias/act 面共同携带量化参数，无法从在树代码推出。

## 裁决

US2 按**路线 A** 实施（sprint 尺寸、SC-003/FR-003/FR-004 全可达）；路线 B 记录为后续立项
（需先重建 requant 打包器 + 量化方案设计，收益 = 真 4-bit HMX 吞吐 ~3.2 TFLOPS 级）。
spec AC-2.1 的「判定走 OP_MATMUL_W4A16 HMX」机械式表述按本裁决修订为
「命中权重 Q4_0 tile 入池（640B/tile），OP_MATMUL_F16/SPLIT 设备侧 Q4 槽 dequant 消费」。

## nibble 面序仲裁（2026-10-09 实测，定案）

对拍 `token_embd.weight` 行 0：GGUF Q4_0 反量化 vs HF F32 `model.embed_tokens.weight[0]`：
- **B 平面序**（qs[j] lo=元素 j(0..15)，hi=元素 j+16）：**cos 0.9961，maxerr 0.0056**（= Q4 量化噪声量级）→ **本 GGUF 是 B**
- A 对偶序（qs[j] lo=元素 2j，hi=元素 2j+1）：cos 0.0066 = 乱码
- 结论：旧 emit 注释与 vendored `unpack_tiled_q4_0` 的 B 假设才是对的；初版新代码按 A 写是错的，已改。

tile 内统一 **A 配对**（HVX dequant task 的逐字节对偶解读）：`quants byte[i*32 + n] =
v(2i) | v(2i+1)<<4`（i = k/2）。GGUF 源是 B → repack 做 **B→A nibble 置换**（纯置换、
量化值零损、非纯字节置换）。三处消费端全部 A 序自洽，实证：
- vendored `unpack_tiled_q4_0`（tile→GGUF 块）按 `byte[cp*32+row] = tile_quants[row][2cp..2cp+1]`
  读 tile、按 B 重建 qs —— 是新 repack 的逐字节逆变换（round-trip 闭环）。
- HVX task `dequantize_tiled_weight_to_fp16_task_q4_0`：lo/hi nibble 分离 + `vshuff(-1)`
  逐对交织 + `vshuff(-2)` scale 双列 [d0,d0,d1,d1]，输出 = crouton（core_dot 直喂
  HMX_LOAD_MPY_DEEP_F16，与 exec_matmul_f16_hmx mxmem:deep 同指令同布局）。
- `exec_gather` 行级 dequant：`byte[(k/2)*32+n]`，lo=k 偶/hi=k 奇。

## lm_head = embed 转置 tie（wtlink 规则依据）

`onnx__MatMul_208884` 权重存储 [1024, 248320] == `E.T` 逐元素（抽样 3000 点 diff=0.0）；
GGUF 无 `output.weight`（llama.cpp tied 约定同证）→ wtlink 把 lm_head 也映射到
`token_embd.weight`。匹配表 303 节点双向完备。

## 契约（路线 A 权重槽）

- 槽判定：`len == (K/32)*(N_full/32)*640`（0.625 B/elem；与 f16 2.0、W4A16 紧凑 0.5 互斥）
  且 K%32==0、N_full%32==0。
- tile（ct,kt）@ `(ct*n_k + kt)*640`：quants A 配对 `byte[i*32 + n] = v(2i)|v(2i+1)<<4`
  （B→A 置换见上）；scales = GGUF block d f16 原样 @ `tile+512 + 2n`，尾 64B pad 0
  （对齐 `hvhx_v2_weight_aligned_tile_size(Q4_0)=640`，保 HVX 128B 读不越界）。
- 设备消费：
  - `exec_matmul_f16_hmx` wgt pack：`hvhx_v2_dequant_tiled_to_fp16(w_q4 + (n0/32)*n_dot*640,
    wgt_cr, N, K, HVHX_V2_WT_Q4_0)` 整块直写 crouton（SPLIT chunk 基址平移，tile 槽
    ct 主序/kt 次序连续）。
  - `exec_gather`（embed 表）：行级标量 dequant，32×K 元素/pass，标量足够。
- emit 准入门 `q4_consumer_kind`：0=否 / 1=Gather / 2=对齐非 batched FC/MatMul。
  kind 2 miss = FR-003 硬错误；Gather 需要 kind 1（SC-003 预算，见上）。
- emit 兜底（T014）：Q4 但形状未对齐 → f16 反量化槽（reason=shape_misalign）；
  对齐但消费者不合规（非 FC/MM/Gather / batched）→ f16（reason=consumer_gate，spec
  三枚举的扩展）；F32 源 → f16（reason=none）。Q4 tile 槽被 kind 0 消费者复用 = 编译期硬错误。
- 设备 Q4 槽 + 标量路径 = 硬错误（同 003 T008 W4A16 无标量路的纪律）；host 执行器无 HMX，
  Q4 blob 属设备专属。

## T012 交付物

- `scripts/gguf_wtlink.py --gguf /disk2/aiDAPTIV/Qwen35-0.8B.gguf --net-json <v2 net.json>` 全量 match.tsv（双向完备 fail-fast 已内建；输出 `.scratch_007/match_map.{json,tsv}` 303 节点）。
- emit 编译期源预检（FR-003）：--gguf/--match 单独给、GGUF 不可读、TSV 零条目、matmul 权重 miss（w4_missing）→ 全部硬错误非静默 f16。
- T013 域报告：`<out>.w4report.tsv` sidecar + stdout w4 占比摘要（domain/reason/size）。

## 测试覆盖（test_wtop_emit §12，宿主 gcc-9.4 build_host 全绿 + 全量 ctest 48/48）

- §12a 直连单测：repack 输出总长 4×640、尾 64B pad 0、scale d f16 原样 @ tile+512+2r、
  vendored 逆回环 nibble 零损（tile→B 平面 qs 重建 == 源块）、gather A 配对解码 == B 平面
  参考、dequant_q4_0_f16 == B 平面参考反量化（f16 位级，取值全正常域）。
- §12b 端到端（FC 图 + 合成裸 Q4 块 GGUF + match TSV，grouping 手工赋值）：
  - 对齐 K64/N96 → w4 槽 len==(K/32)*(N/32)*640=3840、count=K*N、blob tile 字节 ==
    repack 输出逐字节、sidecar 记 `test_w\ttest_w\tw4\tnone`；
  - N=48 未对齐 → f16 槽 len==K*N*2=6144、sidecar reason=shape_misalign；
  - FR-003 kind2 miss（`--gguf` 空文件可读即可，miss 在 bounds 检查前命中）→ 退出码 4
    且 stderr 点名 absent_w。
- 数值对拍（真实 GGUF，`.scratch_007/verify_nibble_perm.py`）：token_embd 行 0-7 ×
  每 128 块抽 1，gather 语义 8192 点失配 0、vendored 逆回环 4096 点失配 0。
