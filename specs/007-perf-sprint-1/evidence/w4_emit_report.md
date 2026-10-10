# T015: 0.8B Q4 blob 编译报告（2026-10-09）

## 配方（v4 基线仅追加 --gguf/--match）

```
wtop_emit --bin /disk1/gehtp/.scratch_003_rebuild/qwen35_08b_v4.tagged.bin   # v4 正典 bin（compile md5 稳定）
          --input-f16 ids_from_golden.i32.raw,mask_ones.f16.raw,pos_arange.i32.raw  # .scratch_a3/qwen35_08b_v2.wtop.work/
          --out   .scratch_007/qwen35_08b_q4_007.wtop   # + .manifest.json
          --gguf  /disk2/aiDAPTIV/Qwen35-0.8B.gguf
          --match .scratch_007/match_map.tsv            # T012 全量表 303 节点
```
（宿主 gcc-9.4 build_host/wtop_emit，同源代码 commit 1930574+src_ref 修正+dedup）

## 产物与体量

| 指标 | f16 基线 (qwen35_08b_v4_007) | Q4 (qwen35_08b_q4_007) | 倍率 |
|---|---|---|---|
| blob bytes | 2,041,268,992 | 499,665,664 (476.5MB) | **4.08×↓** |
| 权重域 | ~1.95GB f16 | w4 469,565,440 B (447.8MB) + f16 2,287,168 B (2.2MB) | — |
| ops | 14188 | **14188（逐 op 组分完全一致**，gehtp_prof 对拍：opcode 直方图 18 类全同） | 1× |
| slots | 1942 | 1941 | −1（tie 去重共享） |
| gguf 命中 | — | hits=382 / miss=339（miss=小 const 走 const_pool f16，by-design） | — |

**SC-003 权重预算：447.8+2.2 = 450.0MB ≤ 614.4MB (0.6GB) ✅**（f16 embed 兜底口径 509MB 破预算，
故 Gather 行级 Q4 消费为必要路径——契约见 us2_route_recon.md）。

## const_data 折半核验（T015 验收项）

- Q4 tile 域 0.625 B/elem（640/32/32）vs f16 2.0 B/elem = 3.2×；总权重域实测 1.95GB→450MB = 4.33×↓
  （超额部分来自 tie 去重：token_embd.weight 在图里有 Gather(embed)+MatMul(lm_head) 两个 const，
  单 GGUF 张量身份去重后 tile 只发一份 158,924,800 B，省 151.5MB）。
- w4 张量 151 个（reason=none）；f16 兜底 36 个（全 shape_misalign：in_proj_a/b 等 N 非 32 倍小投影，
  合计 2.2MB）；source_missing=0、consumer_gate=0；FR-003 硬门未触发（matmul 权重全命中）。
- blob md5（复现锚）：2975ec2cc1c5…

## 数值语义链（FR-004 口径）

设备 logits 判分将切 `cpu_golden/q4_0` 金标（T016）：权重 = GGUF Q4_0 原量化（nibble 置换零损，
无二次重量化），计算 f16 —— 与 q4_0 golden（GGUF 同权重 f16 反量化推理）同口径；
hf 口径 cos 不可与 q4_0 口径直比（双列登记见 data-model.md）。
