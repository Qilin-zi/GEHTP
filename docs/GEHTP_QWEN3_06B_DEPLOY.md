# Qwen3-0.6B Q4_0 部署任务

## 状态: host 门闭合 / 设备门待板上

### 编译产物

| 文件 | 大小 | ops | GGUF hits |
|------|------|-----|-----------|
| `test_models/qwen3_06b/qwen3_06b_q4.wtop` | 1.34GB | 1682 | 194 (Q4_0) |

### Host 门 (PASS)

- `wt_host_exec` 4 提示词 vs Q4 golden: cos=0.999996-0.999997, top1=100%
- vs HF fp32: cos=0.951-0.965, top1=78-97% (量化域差距, 3 层 Q4_1 回落 f16)
- wall time ~1:54/prompt

### 0.6B Q4 管道关键实现

#### 1. GGUF wtlink (194 节点 ↔ GGUF 双向映射)

- `scripts/wtlink_qwen3_06b.py`: gguf_wtlink FC 节点 ↔ GGUF tensor 双向 matcher
- 排除 Q4_1 tensors (blk.0/1/2.ffn_down) → fallback f16 pool
- 产出 `match_map.json` 供编译器和 golden 共用

#### 2. Q4 golden 参考

- `scripts/golden_qwen3_06b_q4.py`: dequantize Q4_0 GGUF weights → forward pass
- 输出 `q4_logits_fp32.npy` [4,32,151936] (4 prompts, 32 tokens, 151936 vocab)
- vs HF fp32 cos=0.959, top1=90.6%

#### 3. Q4 发射器 & compile

- `goetp compile --gguf Q4_0`: wtlink 194 hit → 发射器 blockwise dequant
- 1682 ops, 1.67GB Q4_0 blob
- Q4_1 自动回退 f16 (发射器 Q4_1 unpacker 未实现)

#### 4. Host judge

- `scripts/judge_qwen3_06b.py`: cosine similarity + top1 对比工具
- 支持 vs Q4 golden + vs HF fp32 双轨 judge

### 设备门 (待板上)

- 110 板 d0f1784 冷启动恢复后执行
- 脚本: `scripts/run_qwen3_06b_device.sh`
- skel md5 校验门 (380f3cbf) + capability probe + 占用检查
- 预期: cos≥0.99999, top1=100% ×4

### 坑记录

1. Q4_1 不支持: 3 层 ffn_down (blk.0/1/2) 用 Q4_1, 发射器无 unpacker → f16 回退
2. activation reuse: lm_head 151136→61 chunks, prep 一次 invoke 61 次 (dc_w4_run_prep/invoke/fini)
3. 4090 ship/ skel 仍为未签版 (d0bfbc76) — 须同步换 380f3cbf 否则再毒化板子

### 关联文档

- [[gehtp-qwen3-06b-q4-host-gate]] — host 门详情
- [[gehtp-qwen3-06b-line]] — Qwen3-0.6B 最小 LLM 线概览
- [[gehtp-board-recovery-skel]] — 板恢复规程
- [[gehtp-qwen3-accelerator]] — DFlash 能否加速 Qwen3 架构分析
