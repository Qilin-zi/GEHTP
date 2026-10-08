# US3a (T014-T017) 证据 — fc_bias_fold 扩展 MatMul 链头 + 发射门/硬报错

日期: 2026-10-08 · 分支 task/prof-wp · 设备 d0f1784 (私有名栈)

## 结论

SC-004 无回退门 **通过**。0.8B 真图 0 次 MatMul 折叠(经 net.json 扫描验证为**正确结果**:
全部 36 处 MatMul→EB 对的另一侧均非 const), 发射产物与已过门基线 005_final **字节级相同**。

## T014 单元测试 ([15] 段, test_opt_passes)

| 用例 | 图 | 期望 | 结果 |
|---|---|---|---|
| [15a] 非对齐正例 | MatMul[3,100]×[100,64] + Add(bias const) | Add dead / bias→inputs[2] / has_bias=1 标记 | ok |
| [15b] 32 对齐反例 | 64×64×64 | 不折 (HMX 门: 设备 v1 仅标量路径应用 bias) | ok |
| [15c] batched 反例 | BMM[2,3,100]@[2,100,64] | 不折 (设备硬错防御) | ok |
| [15d] 数值门 | execute_host 折/不折/手算 ref 三方对拍 | 折/不折**位级一致**; vs ref <1e-5 相对 | ok |

ctest 47/47 (含 [15] 段全绿)。

## T015/T016 实现

- `pass_fc_bias.cpp`: 链头接受 `"MatMul"`; 形状/标志经 `extract_matmul_extra` **同源**推导
  (serialized_extra 在 pass 相位为空, 不能直接读); batched `continue`; HMX 32 对齐门不变。
- `graph_prepare.{hpp,cpp}`: extract_matmul_extra 移出匿名 ns (external linkage + 前置声明),
  写 `ExtraMatMul.has_bias`(+24, op_data scalar 同口径; 旧 24B bin 无此字段=0)。
- `op_matmul.cpp` 发射门: MatMul+has_bias 标记 → OP_MATMUL_F16_BIAS(36); 旧 bin(24B) 强制
  has_bias=false; **32 对齐 MatMul+has_bias 硬报错**(违禁组合, 对齐 W4A16+bias 硬错风格);
  ExtraMatMul 读取 min-size memcpy (28B/24B 双兼容)。
- `oplist_parse.h`: OP_MATMUL_F16_BIAS=36 + WT_ARITY 8 (fcbias 血统, 随本提交首次入库)。

## T017 集成门 (0.8B v2 真图)

- `--opt-stats`: fc_bias_fold **18/18** (全部 FC 链头; **MatMul 0 折** — 42 个 spec 候选实为
  activity-activity linear_attn 加法, `(0,False,'MatMul')` 类对 36 处全非 const 另侧)。
- .bin +560B = ExtraMatMul 24→28B × ~140 个 MatMul (结构体增长, 唯一差异源)。
- wtop_emit: ops **14152** / 字节数 / **md5 a322d06a… 与已过门基线 005_final 完全一致**。

## 设备无回退 (SC-004, 私有名全栈)

栈: libhvxhmx_005.so(SWIV 签名) + gehtp_runner_005.so(-DGEHTP_JOB_PATH/-DRESULT_NAME 私有)
+ job_005.txt + **in_005.f16.raw**(输入也私有化, 见事故) 。

| 指标 | 005_final 基线 (10-08 晨) | 005_us3a (本次) | 判 |
|---|---|---|---|
| ops | 14152 | 14152 | = |
| blob md5 | a322d06a… | a322d06a… (同字节) | = |
| exec | 31.49s | 31.43s | 持平 |
| wall | 35.04s | 35.99s | 设备 IO 慢 12% 环境噪声 |
| opcode36 次数 | 18 | 18 (avg 11.9ms) | = |
| 全零行指纹 | [0-15,18] | [0-15,18] | = (SC-006 既有) |
| NaN | 0 | 0 | 无新 NaN |
| 双跑 bitdiff | — | 37.5% | 设备非确定常态(与既往观测一致) |
| r16 logits top5 | — | 全中, 首值逐项相同 | 无数值回退 |

## 设备共享态事故记录 (跨会话协调)

1. **2GB blob read 失败 ×1**: 与别会话并发占 PD 内存同时发生, 复跑即过 (非 runner 缺陷;
   小 blob 同栈全通, read_file 16MB 分块/malloc 逻辑完好)。
2. **input.f16.raw 被并发会话改写**: 一次跑成 `in 4` (读到 4B 残file), 一次 128B (他方
   i32 ids) — 输出指纹完全不同且**无任何告警**。教训: **私有名隔离必须覆盖 input 通道**,
   仅 lib/runner/job 私有不够。job_005.txt 的 input 键指 in_005.f16.raw (64B f16 ids p0,
   = .scratch_eqfix/ids_p0_f16.raw, 与设备 ids_p0.i32.raw 同 token 序)。
3. 共享 libhvxhmx_v23.so / gehtp_runner.so 11:29/11:35 被别会话替换 (md5 不符) — 私有名栈
   规避 (gehtp-p5-device-gate 已录, 本次补 input 一层)。

## 提交边界 (T023 精确暂存)

- 全文件: pass_fc_bias.cpp / graph_prepare.{hpp,cpp} / op_extra.hpp / test_opt_passes.cpp /
  tasks.md / 本证据。
- **部分暂存**: op_matmul.cpp (只入 has_bias 五段; 003 T015 W4A16 act 规范化与
  cur_engine 注解段留在工作树) ; oplist_parse.h (只入 opcode36+arity 两 hunk; 003 的
  VER_3/carve/annot/wt_annot_pack 留在工作树)。wtop_ops.hpp 整体不入 (assign_engine 是
  003 未提交新函数, 我在其内的 3 行 min-size 修订随 003 走)。
- 暂存版 op_matmul.cpp 经 g++ -fsyntax-only 验证。
