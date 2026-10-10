# T016 三族 census digest — US1+US2 落地后重跑 (1010)

**源**：`.scratch_008/t15A.op_ts.bin`（T015 Run A, fault=0 DMA 真路径, ver=4, 14188 op）
+ `.scratch_008/qwen35_08b_008t12.wtop.manifest.json`（op_names 交叉标注）。
**口径**：`python3 scripts/gehtp_prof.py --wtop … --op-ts … --csv …`，wall=13.04s
（exec 13.02s + 缝隙 0.1%）。**spec.md 表内 7-8s 投影作废**——US2 收口后实测 wall
13.04s、三族合计 **7.246s = 55.6% wall**，是 US3 的真实起点。

## 三族新盘子（US3 靶）

| family | n | dur | %wall | out 字节 | 有效带宽 |
|---|---|---|---|---|---|
| STRIDED_SLICE_F16 | 2373 | 3.060s | 23.5% | 62.6MB | 20.5MB/s |
| UNARY_F16 | 2909 | 2.903s | 22.3% | 78.3MB | 27.0MB/s |
| TRANSPOSE_GEN_F16 | 175 | 1.283s | 9.8% | 21.7MB | 16.9MB/s |
| **合计** | **5457** | **7.246s** | **55.6%** | **162.6MB** | **~22MB/s** |

**核心机理**：三族全是标量引擎非线性/跨轴访问，有效带宽 ~20MB/s（峰值 59.9GB/s 的
**0.03%**）。成本不在带宽（162MB 搬运量 vs 7.25s = 微不足道），而在**访问形态**。
→ 两条消解路径恰好对应用户故事里的两个任务：能引用消除的走 T017（字节零搬运），
不能的走 T018（HVX 线性化把 ~20MB/s 抬到 GB/s 量级）。

## UNARY 可折叠分解（精确, 由 manifest op_names 交叉）

| 子型 | n | dur | 处置 |
|---|---|---|---|
| **Reshape** | 2684 | **2.594s** | **纯恒等拷贝 → T017 引用消除（可折叠）** |
| Eltwise_Unary | 122 | 0.183s | 真激活, 物理 |
| ElementWiseNeuron | 102 | 0.126s | 真神经元, 物理 |
| Cast | 1 | 0.0004s | 类型转换, 物理 |
| 物理残余 | 225 | **0.309s** | T018 HVX 候选（量小） |

**UNARY 92.3% 是 Reshape**（2684/2909），折叠后该族只剩 0.309s。这是三族里最干净、
可精确预告的一笔：**−2.59s**。

## STRIDED_SLICE 结构（stride 分解 = T017 第一步，1010 终判）

编译器侧 `dump_view_census` 补 contiguous-run 计数（`GEHTP_DUMP_VIEWS` emit 期，
`pass_view_fold.cpp`）。折叠判据 = **单一连续字节段**（设备 temp 模型 {offset,size}
可表示）：全轴 stride==1 且部分轴至多一个且非最内轴（最内轴部分→行间洞→非连续）。
实测（v4 全图 2373 op）：

| 类 | n | elems | 字节 | 处置 |
|---|---|---|---|---|
| ss_contig_full（恒等） | 6 | 6,144 | 12KB | T017 折叠（恒等） |
| ss_contig_sub（偏移视图） | **37** | **4,718,688** | 9.4MB | **T017 折叠（offset 视图）** |
| ss_strided（跨步） | **2330** | **26,561,184** | 53.1MB | **T018 HVX（不可折叠）** |

**关键修正（推翻 digest 旧版「若 128KB×30 连续则门可达」的乐观）**：contiguous
子切片只有 **37 个**（正是那 30×128KB 主力 + 7 个中块，~0.196s），**可折叠但量小**；
**2330 个碎片切片（2.86s）全是 strided（跨步/多部分轴/最内轴部分），不可折叠**，
占该族 93% wall 且必走 T018 HVX。即：STRIDED_SLICE 的折叠收益上界仅 ~0.2s，剩余
~2.86s 是 HVX 硬目标。

## TRANSPOSE 结构

out_b 分布：**128KB × 90 = 0.716s + 256KB × 36 = 0.542s**（126 op = 1.258s，占该族
98%）。Transpose 恒改字节序 → **不可折叠，全归 T018 HVX**。有效带宽 16.9MB/s，HVX
线性化（007 P1 broadcast 三形态范式）预估 20–50×，1.283s → ~0.05s 量级。

## 门投影（三族合计 ≤2s, SC-005 前半）—— 终判版

| 处置 | 预估 |
|---|---|
| T017 折叠 Reshape | −2.59s（UNARY → 0.31s） |
| T017 折叠连续切片 | −0.20s（STRIDED_SLICE → 2.86s 全 strided） |
| T018 HVX strided 切片 | 2.86s → ~0.15s（20×） |
| T018 HVX TRANSPOSE | 1.28s → ~0.06s |
| T018 HVX UNARY 物理残余 | 0.31s → ~0.02s |
| **T017+T018 后三族合计** | **~0.54s**（≤2s 门可达） |

**诚实结论（FR-006，不硬凑门）**：T017 折叠只能吃 Reshape（2.59s）+ 37 个连续切片
（0.2s），合计 ~2.8s。**≤2s 门达成的硬前提是 T018 把 2330 个 strided 切片（2.86s，
占三族残余 55%）HVX 化**——它才是三族里最大的单一物理搬运块，且不可引用消除。
Transpose（1.28s）+ UNARY 物理（0.31s）为辅。若 T018 只做 Transpose+UNARY 而不做
strided 切片，三族仍剩 ~3.2s，门不可达，T019 如实指认。

## T017/T018 入口

- T017 第一步（已收口）：编译器侧 contiguous-run 计数落 `pass_view_fold.cpp`
  `dump_view_census`（`GEHTP_DUMP_VIEWS`），终判见上表。
- T017 第二步：Reshape（2684 op/2.59s）+ 37 个连续切片（offset 视图）→ emit 期
  引用消除（temp 别名 + 生命周期传播 + release_at 别名防 double-free，op 消失，
  opt-stats 计数，位级一致）。
- T018：2330 个 strided 切片（2.86s）+ Transpose（126 op/1.26s）+ UNARY 物理残余
  （0.31s）→ HVX 化（007 P1 broadcast 三形态范式；strided 切片需 gather 形态）。
