# GEHTP Qwen3-0.6B 最小 LLM 线 — 会话交接 (2026-09-17)

> 目标: 最基础香草 LLM (LLaMA 级) 经 GEHTP 编译上板, HVX/HMX 应用尽用。
> 状态: **导出/编译/host 门全绿; 设备门待板恢复**; HVX W1 接线已落地 (默认关)。

---

## 1. 选型与设计

- 模型: `/4090disk2/Qwen3-0.6B` (28 层 dense, hidden 1024, 16Q/8KV, dh=128, vocab 151936,
  QK-RMSNorm + RoPE θ=1e6 + SwiGLU, tie embeddings)。比 0.8B 混合 GDN 严格更简单:
  无 ScatterNd/CumSum/Conv1D_SSM — 天然绕开 A3 暗雷。
- **折叠 embedding**: 图输入 = `inputs_embeds` f16 [1,32,1024] (host 侧查表),
  绕开引擎 f16-only 边界 + int Gather/Cast 雷区。position_ids/因果掩码/RoPE cos/sin
  全部烘焙为常量缓冲 → **单 f16 输入单 logits 输出**, 全图落在引擎已验证边界内。
- seq=32 对齐 0.8B 先例; 输出全 logits [1,32,151936]。

## 2. 产物与门

| 资产 | 位置 |
|---|---|
| 导出脚本 (纯 torch 手写前向, 无 transformers 版本依赖) | `scripts/export_qwen3_06b_onnx.py` |
| ONNX (外链权重, qairt 兼容) | `test_models/qwen3_06b/model.onnx` + sidecars |
| golden (4 prompts embed f16 raw + HF fp32 logits + manifest) | `test_models/qwen3_06b/cpu_golden/` |
| 设备 blob + manifest + 中间产物 | `test_assets/qwen3_06b/qwen3_06b.wtop` (1.34GB/1682op/649槽) |
| 设备门一键脚本 (含 skel md5 门卫 + 通道探针) | `scripts/run_qwen3_06b_device.sh` |
| judge (cos≥0.9999 + top1 32/32×4) | `scripts/judge_qwen3_06b.py` |
| 官方 QNN SDK 对比产物 + runlist 对比报告 | `test_assets/qwen3_06b/official/` + `docs/GEHTP_QNN_OFFICIAL_RUNLIST_COMPARE.md` |

**门状态**:
- 导出自检: torch 手写模块 ≡ HF fp32 (cos=1.0, top1 100%) + ORT ≡ torch ✅
- 编译: `gehtp compile` 全链过 (qairt-converter→dlc_repair→to-json→hnnx_compile→wtop_emit) ✅
- **host 门 4/4**: host_run vs HF golden — cos≥0.99999998, top1 32/32×4 全中 ✅
- **设备门 4/4 (110/d0f1784, 2026-09-21)**: 全标量执行 cos≥0.99999628, top1 32/32×4 全中 ✅
  (p0 0.99999715 / p1 0.99999737 / p2 0.99999758 / p3 0.99999628; max_valdiff≤0.144)

**设备数值坑 (已修, commit 254ffbd)**: hexagon libm `expf` 在溢出区 (|x|≳88) 返回垃圾
常数 — 实测 gate∈[-155,-90] 时 σ 恒为 0.370117 (经 gate·silu 放大到 logits maxdiff
~4000)。op 级 bisect (host 执行器 wt_host_exec 与设备逐 op 对拍) 锁定唯一发散点 =
末层 MLP sigmoid。修复 = UNARY case 8 加 ±80 钳制 (f16 输出在该域已饱和, 位级等价)。
另见: 偶发 `rc=1511 unary ref fail` (temp 空引用, 旧 lib 上两次, 修复版 4 连跑未复发 —
疑 DSP 池 bump 竞态, 复发时用 wt_host_exec + 设备 optrace 对拍)。

op 分布 (net.json 1684 节点, 11 型): Reshape 506 / EltwiseBinary 392 / FC 197 /
Transpose 140 / RmsNorm 113 / StridedSlice 112 / Unary 56 / Concat 56 / MatMul 56 /
Softmax 28 / Neuron(silu) 28。FC 形状 M=32,K∈{1024,3072},N∈{1024,3072,151936} — 全 32 倍数, HMX 形状契合。

## 3. HVX/HMX 应用尽用接线 (W1 已落地, 默认关)

**新增内核族** `kernels/src/hvx/hvx_f16face.c` (头 `include/hvxhmx_v2_f16face.h`):
f16 直收直发 HVX 包装 — `gehtp_hvx_mul_f16` (原生) / `gehtp_hvx_unary_f16` /
`gehtp_hvx_silu_f16` / `gehtp_hvx_softmax_f16` / `gehtp_hvx_rmsnorm_mul_f16` /
`gehtp_hvx_cvt_f16↔f32`。数学=f32 中间路径; scratch 调用者给 (零 malloc);
任意地址安全 (前导/主体/尾三段, 镜像 hvhx_v2_add_f16 结构)。

**执行器接线** `kernels/src/runtime/oplist_exec.c` (全 env 门默认关, 标量回落):
| env 门 | 算子族 | 内核 |
|---|---|---|
| GEHTP_HVX_ADD=1 (既有) | binary add (+opcode ADD) | hvhx_v2_add_f16 |
| GEHTP_HVX_MUL=1 | binary mul | gehtp_hvx_mul_f16 |
| GEHTP_HVX_UNARY=1 | unary neg/exp/sqrt/rsqrt/log/sigmoid/tanh/gelu/silu | v2 f32 族 |
| GEHTP_HVX_SILU=1 | Neuron silu | hvhx_v2_silu_f32 |
| GEHTP_HVX_SOFTMAX=1 | softmax (n%32==0) | hvhx_v2_softmax_f32 |
| GEHTP_HVX_RMSNORM=1 | rmsnorm2 | hvhx_v2_rms_norm_mul_f32_rows |

构建验证: `kernels/build_libs.sh` 绿, 7 符号全出, **vgather=0 铁律保持**。
注: softmax/rmsnorm HVX 版全程 f32 (标量体中间有 f16 落盘) — 更接近金标, A/B 判据=对 golden 值差不劣化。

**A/B 验收协议 (板恢复后)**: 默认跑 (全标量) 出基线 → 逐族开门重跑 →
judge 判据不降 + runner 日志/optrace 对照 wall-µs。

**W2 待做 (HMX 侧)**:
- **FC/MatMul f16 → HMX** (`hvhx_v2_hmx_gemm_dot_fp16` 在库): 形状全 32 倍数契合;
  需权重列 interleave (hmx-utils.h static inline) + VTCM/arena 协同 (参照 exec_matmul
  W4A16 的 wtcache/carve 路径); lm_head N=151936 须 N 向分块 (建议 chunk=4096 列,
  scratch ~8MB DDR 静态)。naive 版先求对, 编译期 tile-major 固化是性能正解 (MEMPLAN 线)。
- **Q4_0 变体 = 零新代码的 HMX 路径**: `--gguf .../Qwen_Qwen3-0.6B-Q4_0.gguf` 重编译即走
  既有 MATMUL_W4A16 HMX 引擎; 需为本模型写 match map (GGUF blk.N.* ↔ 本手写模块参数名
  l{N}_self_attn_q_proj 等, 机械映射; 参照 /disk2/match_map.tsv 格式 + gguf_wtlink.py)。

## 4. 设备事故与恢复 (2026-09-17, 全链实证)

**根因 (双层)**: ① **skel 层 (已修)**: `hvxhmx23/librun_main_on_hexagon_skel.so` 被 09-16 的
推送换成**陈旧版** (md5 d0bfbc76…, 与仓库 ship/ 旧拷贝同源); 正确版 = 2026-08-10
`hvxhmx_libs/` 版 (md5 **380f3cbf**…, RUNBOOK §2 早有记载)。陈旧 skel 使 PD open
拒载 0x80000406 (AEE_EUNABLETOLOAD), 且**失败 open 会把 fastrpc 通道楔死**
(glink 线程归零, 后续全 0x39)。
② **虚拟化 SSR/hab 协商层 (板级, 未修)**: 另一会话取证 — guest boot 期 hab 通道协商被拒
(hab_open_listen failed -11)、部分 boot 里 /dev/fastrpc-* 节点不创建、boot_adsp/cdsp/cvp
60s 超时 — 该层时好时坏 (同日另一会话曾跑通 894-op L0; 本会话 18:48 boot 也曾通道健康),
**整板重启 ×3 均未稳定恢复, 需物理断电/宿主管控层介入**。

**已修**: ① 设备 skel 已换正确版 (chmod 755 齐); ② 仓库 `ship/` skel 已更新为 380f3cbf
版 (本树; **4090 树 ship/ 仍陈旧, patch 回合须同步** — 若 4090 侧 setup 再推旧 skel 会再毒化)。

**仍阻塞**: 通道楔死需**整板重启**才能清 (guest 单独重启不够 — vhost 后端不重新协商);
PVM (f69bec03) 从 adb shell 全锁死 (reboot/systemctl/setenforce 均 Access denied,
SELinux 锁定不可 permissive)。**需要: 物理断电冷启动, 或其他会话以其渠道重启整板。**
板回后: `scripts/run_qwen3_06b_device.sh` 一把过 (内置 skel md5 门卫 + 通道探针 +
setenforce 自愈 + 4 prompts run + judge)。

**附加发现**: GVM 的 SELinux 有看门狗会自动翻回 Enforcing (跑前须复查);
GVM 启动比 PVM 晚 ~22min (vmm-boot-lcm 设计如此, 勿误判为重启失败);
boot_adsp/cdsp/cvp 60s 超时在本 build 上**即使通道健康也存在** (良性, 非故障签名)。

## 5. 复跑清单 (110 板, 2026-09-21 实战版)

```bash
# 战场已迁 110/d0f1784 (52f67807 楔死)。上板前必读 board-run-preflight-checklist 记忆。
cd /disk1/GEHTP
kernels/build_libs.sh                     # 重建 lib (含 sigmoid 修复 + W1)
# runner 变体身份制 (共享板 job.txt 撞车防护):
#   hexagon-clang ... -DJOB_PATH="/data/local/tmp/hvxhmx23/gehtp_q6b/job.txt" \
#     examples/42_gehtp_runner/main.c examples/common/example_util.c ... → SWIV 签
# 跑: 等板空闲 → 推 lib+runner+blob+input+job → nohup run_main_on_hexagon 3 runner → 拉回 judge
# 坑: 并行会话会覆盖板上 lib (多次实测) — launch 前 md5 对账; 提交后的重建都含修复,
#     所以只要板 lib 带 80.0f 钳制 (objdump 查 42a00000 ×2) 即安全。
# HVX A/B: 设备侧 env 开门 (runner 环境注入) 后重跑 + judge 不降
```
