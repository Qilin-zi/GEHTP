# GEHTP 分支拓扑与合并记录

> 建立日期：2026-09-24。记录 5 个分支的拓扑关系、一次深度 diff 的结论、
> 以及「代码合一」中两处易被误读为「刻意不要」的裁定纠正。
> 目的：让后来者不再在 `main`（8-31 老基线）上重新发明已经存在的东西，
> 也不再误读 integ/unified 里「暂不支持/弃用」的注释。

## 1. 分支拓扑（按 HEAD 时间）

| 分支 | 时间 | 作者 | 定位 |
|---|---|---|---|
| `main` | 08-31 | Qilin-zi | M2 老基线 |
| `gehtp-review` | 09-22 | dumppool | code review 修复 + opt pass + wtop_ops |
| `r3-merge-line` | 09-22 | dumppool | = gehtp-review + 一行 .gitignore（同线） |
| **`integ/unified`** | 09-23 | Qilin-zi | **最全基线**：gehtp-review 全量 + 设备链(dc_dma/runtime_alloc/spillfill_g4/oplist_exec 重写/host_run/W4A16/qwen3_06b) |
| `task/prof-wp` | 09-23 | Qilin-zi | 从更早分叉的 OpenAI serving 特性分支（裁掉 W4A16/qwen3_06b/op_pad/.specify，未合入 dc_dma 修复） |

**结论：当前应基于 `integ/unified` 工作。** `task/prof-wp` 是发散特性分支，其 serving 栈需移植回 integ/unified（已完成，见 §3）。

## 2. 深度 diff 结论（2026-09-24，113 个 M 文件逐一比对）

- `integ/unified` 是 `main` / `gehtp-review` / `r3-merge-line` 的**完整超集**（0 个功能性缺口）。
- `integ/unified` vs `task/prof-wp`：**分叉**，2 处真缺口 + 2 处「易误读的裁定」（见 §4）。

## 3. 已完成的合并（本记录建立时）

| commit | 内容 |
|---|---|
| `90d6506` | 移植 serving 栈（23 文件，device 执行是桩/协议未对齐，已知问题见 docs/GEHTP_OPENAI_SERVING.md） |
| `edcf055` | 合入 task/prof-wp 的 `ptr_in_vtcm` 回绕修复 + 设备逐-op dump 钩子 `wt_exec_set_hook` |

## 4. 两处「刻意分叉」的纠正（易误读）

### 4.1 DMA runlist / VTCM 驻留（`0x4000|temp` 预铺 act）—— **是要做的，不是不要**

`oplist_exec.c` 里 `exec_matmul` 的「暂不支持」注释，**不是拒绝功能**，是拒绝
task/prof-wp 那条**有 bug 的实现**：

> prof-wp 旧路径用 `dc_w4_invoke` 直跑，但 ① 预铺的是原始 f16 act，而 e.act 面契约是
> 量化+crouton 后数据；② dc_w4_invoke 不写回 out（结果滞留 VTCM e.out，out_t temp 恒为
> 陈旧值 = 静默数值腐败）。正确的集成须走 `dc_w4_run_prep/invoke` 链 + 出面写回，
> **待 P5 完成后开放**。

- **DMA runlist / VTCM 驻留是 M4/M5 目标**（roadmap「真 DMA runlist」、MEMPLAN 关键路径 E→B→C）。
- W4A16 matmul **现在就能跑**（走 `dc_w4_run_prep/invoke` 链，oplist_exec.c:542-586）。
- 缺的是「显式 DMA 预铺 act」这一格，**待正确集成**，不是放弃。

### 4.2 ScatterNd block>1（12 参 vs 14 参）—— 是格式选型，不是丢能力

`oplist_parse.h:140` 的「合并裁定」采 12 参版，是因为 14 参版多的 `block>1` 写块能力
**无发射器产出该形态，留存即死码 + 重定义编译错**。block>1 将来真有需要，按 12 参扩即可。

## 5. 待办主线（按 roadmap M4/M5）

- P5 真 DMA runlist：`0x4000|temp` 预铺 act 走 `dc_w4_run_prep/invoke` + 出面写回（正确集成）。
- SFCD spill/fill 写侧（MEMPLAN「最大单一缺口」，只读侧 388 行 vs oracle 779 行）。
- RuntimeAllocator 跨组复用（`map_block_reference`/`deserialize_blocks` 仅声明）。
