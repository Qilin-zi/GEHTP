# GEHTP 并行任务书 — 四模型上板 + 通用性 + 调度/内存可靠性战役

> 生成日期 2026-09-17。基线树：**/4090disk2/GEHTP（唯一基线，用户拍板）**，HEAD=8936c01 + VL 线在途未提交改动。
> 本文由 104 机会话写成（/4090disk2 对本机只读，文档暂存本树 docs/，分发时拷入 4090 树 docs/）。
> 本文是任务分发总账：每条任务含范围/文件所有权/依赖/完成门/验证命令，会话认领后在 §7 登记表签名。

---

## 0. 战役目标（用户三条优先级，按序）

1. **P0 模型上板正确**：qwen3.5-0.8B、qwen3.5-4B 编译上板**正确**；MiniCPM5-2B、Qwen3-VL-4B（文本塔）编译上板。
2. **P1 通用正确性**：工具通用、消除硬编码、消除静默兜底（假绿）。
3. **P2 核心可靠性**：调度 / 内存规划 / 分配 / 最终 runlist 正确可靠（内存规划收编 M1/M2 已落地，续 M3+）。

统一判据纪律（PORTAL §3）：f16 短链 ≤1 ULP 位判；长链（81+ op）**值差**判据（L3 基准 max_valdiff=0.000488）；logits 用 `test_assets/qwen35_08b/judge_logits.py`。

## 1. 事实底座（全部经代码/日志实证，2026-09-17）

### 1.1 四模型 op 清单（op_inventory.py 实测）

| 模型 | 节点 | op 型 | 输入 | 输出 | 超出现覆盖 | 当前死因 |
|---|---|---|---|---|---|---|
| qwen35_0.8b | 15129 | 20 | input_ids/position_ids/attention_mask [1,32] | logits [1,32,248320] | 无（基线） | L0 GDN 903-op：rank 三修复（16af35d+c5d4c2a）已落待重跑 |
| qwen35_4b | 20249 | 21 | 同上 [1,32] | logits [1,32,248320] | **Tile ×48** | ①Transpose perm 属性丢失（`/model/rotary_emb/Transpose`，perm 是 tensor_params 且 dlc-to-json 丢属性）②Tile 无 handler |
| qwen3vl_4b | 2183 | 13 | inputs_embeds [1,32,2560] | logits [1,32,151936] | 无（子集） | 同上根因：`transpose perm 轴数 10485760>5`（读错字段） |
| minicpm5_2b | 2492 | 12 | input_ids [1,16] | logits [1,16,130560] | 无（子集） | 全模型未走通；旧树 split stageA(1270 op)/stageB 绕过，动因待查明 |

**关键结论：缺口不是 op 型数量（仅 Tile 一个新型），是 handler 配置健壮性**——同型 op 换形态即死（perm 张量化/属性丢失/秩>4/形状对齐）。

### 1.2 暗雷（数值正确性）

- **ScatterNd 假语义**：0.8B 1154 处、4B 1538 处，emit 走 `op_copy_sem` 恒等拷贝（`compiler/tools/wtop_ops/op_copy_sem.cpp:22`），host execute 有真语义（`ops.cpp:888`）→ host/设备分叉风险，全模型数值门必须收口。
- **host execute 静默直通**：154 注册 op 仅 ~55 有语义，其余直通第一输入 → 对拍假绿源。
- **RoPE 无 op**：靠 converter 分解成 Gather/Slice/Mul/Add；perm 丢失时 VL 线在途改动做形状贪心推断（`op_transpose.cpp` 未提交 diff）——贪心推断**必须配 host 对拍门**，推断错误不可静默。

### 1.3 在途占用（VL 线未提交，23 文件 +869/-135）

`compiler/`: op_extra.hpp、qnn_ir_loader.cpp、graph_prepare.cpp、ops.cpp、pass_const_fold.cpp、host_run.cpp、wtop_emit.cpp、**op_transpose.cpp**、wtop_ops.hpp
`kernels/`: 42_gehtp_runner/main.c(+175)、oplist_exec.c(+94)、oplist_parse.{c,h}、oplist_exec.h、hvx_binary.c(+99)、hvxhmx_v2_binary.h、hvxhmx_types.h、example_util.c、build_examples.sh
`scripts/gehtp`(+28)

判读：VL 线在做**多输入注入 + binary 扩展 + transpose perm 兜底**。A1/B3 与其重叠——**不另起炉灶，先交接**。

### 1.4 设备与资产

- 设备 52f67807 本机 adb 可达（另有一块 f69bec03，PORTAL 铁律仍只认 52f67807）。设备单块 → 所有设备段走 §2.3 队列。
- 回归资产：`test_assets/l3/`（81 op 注意力层 wtop+in+gold+manifest，判据 max_valdiff≈0.000488）；conv_add 在 `test_models/conv_add`；L0 资产由例41线（disk2-ca）收口后迁入 `test_assets/l0/`。
- minicpm5_2b 资产：`test_models/minicpm5_2b/`（全模型 net.json/tagged.bin/dlc）+ `minicpm5_2b_split/`（stageA.wtop 1270 op 已产）。
- golden：qwen35_08b `cpu_golden/`（HF+Q4_0 双 golden + 文本 top1）；qwen3vl_4b `golden_logits.f32.npy` + `in_embeds.f16.raw`；minicpm5_2b `gold.f16.raw` + `in.ids32.raw`。

## 2. 协调协议（所有认领人必读）

### 2.1 文件所有权与错峰

- 动工前：`git status --porcelain` + 查 §7 登记表 + 看目标文件 mtime。已被占用的文件（§1.3）除非是该文件认领人，否则不碰。
- 每任务**独立分支或工作树**、**独立 commit**、门绿才回合主干。commit message 带任务号（如 `A2: Tile op handler ...`）。
- 接口冻结点（改动须先在 §7 表登记+通报全窗口）：
  - **opcode 编号**：`kernels/include/oplist_parse.h` 是唯一真源，新 opcode **只追加不插中间**；同步改 `arity_of`/`arg_is_slot`/`g_op_exec_table`。
  - **Emitter 接口**：`compiler/tools/wtop_ops/wtop_ops.hpp`；新 handler = 新 `op_*.cpp` 文件 + 自注册宏，**不改公共头**除非必要。
  - **blob 格式**：16B 头/slot 表/op 记录/TEMPOFF/权重区；任何格式变动升 ver 并在 PORTAL 登记。

### 2.2 完成门通用件（每个任务都要跑）

1. host：`cmake --build compiler/build_linux` 零告警 + `ctest` 44/44（或当时全量）。
2. 编译回归：conv_add 默认 + spill 变体 blob 与基线**逐字节全同**（除非任务本就改格式，则需明示）。
3. 设备回归（改了 emit/引擎/库时必跑）：`scripts/gehtp_quickstart.sh` → `=== SAMPLE ALL GREEN ===`；`gehtp run test_assets/l3/layer_3.wtop` 值差 ≤0.001。
4. 判据/工具入仓：新验证脚本放 test_assets/ 或 scripts/，不放 /tmp（一日两清）。

### 2.3 设备队列协议

- 设备段一律先 `adb -s 52f67807 shell "ps | grep run_main_on_hexagon"`，有主就等。
- 在 §7 登记表「设备时段」列填预计占用窗口；长任务（>10min）拆段，段间释放。
- adb 掉线/CMA 耗尽按 PORTAL §4 处理，不自行 reboot。

## 3. WS-A — emit 覆盖墙（P0，所有模型线的前置）

> wtop_ops/ 每算子一文件 + 自注册（29af37d），天然并行：每任务新增/修改自己的 op_*.cpp。
> 公共依赖：§1.3 在途改动先落地（VL 线提交或交接）。

### A1 Transpose perm 兜底收口 【与 VL 线交接，不另起炉灶】
- 范围：VL 线在途 op_transpose.cpp 的 perm 形状贪心推断收口 + 回归。
- 必答：①4B `/model/rotary_emb/Transpose` 与 VL `perm 轴数 10485760` 是否同一根因（dlc→json 丢属性 vs tensor_params 误读）？②贪心推断在 qwen35 rotary（[1,64,16]→[1,16,64] 类）之外的形状上是否收敛？
- 门：4B/VL compile 过 Transpose 点；host_run 逐 op 对拍该 op cos=1.0（推断错不可静默过）；conv_add/L3 回归绿。
- 证据锚点：`compiler/tools/wtop_ops/op_transpose.cpp`（在途 diff）、`test_models/qwen35_4b/compile.log:59`、`test_models/qwen3vl_4b/compile.log:54`。

### A2 Tile op 三件套（4B 唯一新 op 型，48 处）
- 范围：net.json Tile 语义（scalar_params 有 `_multiples` 后缀表，95cba62 已扩）→ host execute 语义（ops.cpp）+ emit handler（新 `op_tile.cpp`）+ 设备 opcode（OP_TILE_F16 追加编号）+ xop 执行体。
- 门：单 op host vs 设备对拍 ≤1ULP（多种 multiples/秩组合）；4B compile 全通。
- 接口：opcode 编号按 §2.1 登记；设备执行体可先做标量正确版（性能归 WS-B）。

### A3 ScatterNd 真语义（数值正确性最大暗雷；0.8B 1154 处 / 4B 1538 处）
- 范围：①逐站点审计 0.8B/4B net.json 的 ScatterNd 形态（reduction 键值、indices/updates 形状），分类：哪些恒等安全、哪些必须真 scatter；②emit 真 ScatterNd（新 opcode 或分解）+ 设备执行体；③host execute 语义对齐（ops.cpp:888 已有一部分）。
- 门：审计报告入 docs/（形态分类+每类处置）；含非平凡 ScatterNd 的子图 host vs 设备对拍值差门；0.8B 全模型 compile 后 runlist 中无恒等拷贝冒充（静态断言脚本入仓）。
- 注意：L3 81/81 是在恒等拷贝下过的——需先回答 L3 的 ScatterNd 为何恒等安全，否则 A3 落地后 L3 判据可能变化（预期更好，但须重测）。

### A4 秩>4/秩不齐 编译期规范化
- 范围：loader/emit 两侧对 rank-5 张量折前导 size-1 轴（c5d4c2a 的 strided_slice 修法 + 16af35d 的 transpose 修法推广为统一规范化 pass 或 loader 钩子）；设备 TRANSPOSE_GEN rank≤4 限制在编译期保证不触达。
- 门：构造 rank-5/6 合成图 + L0 903-op blob 全通；PORTAL 排错表删除"rank 5"条目改"不可能"。

### A5 编译预检门禁（全图静态扫描，秒级报缺）
- 范围：`gehtp compile` 第 0 步插预检：net.json op 型 ⊆ wtop_ops 注册表 + op 配置白名单（conv 仅 sh=sw=1/group=1/same-pad 等已知窄门）逐 op 校验；缺失即列全清单退出（不再跑到 70% 才死）。op_inventory.py --expect 机制接入。
- 门：四模型预检报告入仓（各模型缺什么一目了然）；引入一个未知 op 的合成 net.json 必须秒级拒。

### A6 host execute 静默直通治理
- 范围：ops.cpp execute 分派末尾 else 直通分支改显式报错（或环境变量门控的 warn+计数）；对拍脚本认 warn 数≠0 即假绿报警。
- 门：0.8B/L3 host 链全绿且无直通 warn；ctest 全过。

## 4. WS-B — 执行引擎（kernels/，与 WS-A 文件零重叠；opcode 扩展走 §2.1 登记）

### B1 快速件接线（标量→向量，性能主杠杆）
- 范围：xop 执行体内调库函数替换标量循环：matmul_f16→HMX（裸 K-loop 12.34 TFLOPS 件）、rmsnorm2/softmax/silu/unary/binary→HVX V2.1 件（43.9-102 GB/s）；每 op 保持判据不变。
- 门：L3 层 wtop 执行时间基线前后对比入 PERF_REPORT；conv_add/L3 判据不变全绿；optrace 逐 op 耗时表前后对照入仓。
- 注意 hmx_lock 线程属性纪律（PERF_REPORT §5.3）与热路径禁 per-op memalign（gemmdispatch 坑 2——rmsnorm 执行体现有 per-op malloc 必须先治）。

### B2 SPILL/FILL DMA 快路径
- 范围：oplist_exec 的 OP_SPILL/OP_FILL 标量 memcpy → UserDMA（dma_utils 已在库；cache 四铁律走 U9 fence 决策表）。
- 门：spill 变体 byte-exact 不变 + 大 spill（≥1MB）耗时下降入报告。

### B3 多 EXT_IN 注入【先查 VL 线在途是否已覆盖，覆盖则本任务改评审+补门】
- 范围：引擎 `g_ext_in` 单指针→指针数组；emit 多 EXT_IN 标记；`gehtp run --input a,b,c` 逗号多文件；manifest 多 input_slot。
- 门：双运行时输入合成图（add+conv，x0/x1 都变）设备对拍 byte-exact；0.8B 三输入（ids/pos/mask 全运行时注入）run 通。

### B4（延伸，非本次必交）KV_APPEND/KV_GATHER 执行体 + ARGMAX 发射
- decode 前置。xop_kv_unimpl（oplist_exec.c:1276）→ 真体（U15 kvcache 单元接线）；wtop_emit 发射点。本次战役不闭环 decode，但执行体先备。

## 5. WS-C — 模型线（每条：compile 通 → host 对拍 → 设备 run → golden 门）

> 设备段按 §2.3 排队。判据：logits vs golden——qwen35 系 judge_logits.py + 文本 top1（golden_qwen35.py）；VL 对 golden_logits.f32.npy cos；minicpm 对 gold.f16.raw 值差。

### C1 qwen35_0.8B 全模型 prefill 【可立即启动，不依赖 WS-A】
- 步骤：①L0 GDN 收口（与例41线/disk2-ca 交接：c5d4c2a 后重跑 903-op blob，资产迁 test_assets/l0/）；②全模型 `gehtp compile`（三输入中 ids 运行时注入，pos/mask 经 --input-f16 固化 seq=32）；③host_run 逐 op 对拍（host execute 语义真源，A6 落地后无静默直通）；④设备 run → judge_logits 门 + HF golden 文本 top1。
- 已知边界：全模型 15129 op / 权重 f16 ~1.6GB blob；WT_MAX_OPS 65536 容量够；留意 CMA。
- 门：logits 值差/cos 达标 + top1 4 提示全中。

### C2 qwen35_4B 【依赖 A1+A2】
- 步骤：compile（`--external-weights` 路线，8936c01 已备；GGUF 链接 gguf_wtlink.py——E1 先修其 sys.path 死路径）→ host 抽层对拍（全模型 host 参考链太长，按层抽样：首/中/尾各 1 GDN 层 + 1 注意力层）→ 设备 run。
- 风险：f16 全权重 ~8GB——先核设备 DDR/CMA 预算；不够则 `--external-weights` + 权重分段或量化（超出本战役则如实挂账）。
- 门：logits vs HF golden cos + top1。

### C3 qwen3vl_4b（文本塔）【依赖 A1】
- 步骤：compile（inputs_embeds 单输入，in_embeds.f16.raw 已有）→ 设备 run → vs golden_logits.f32.npy cos 门。
- 注意：ViT 不在范围（导出脚本已丢弃 visual）；mrope 已烘焙为图内常量。
- 门：cos≥0.9999（f16 链长 2183 op，判据可先跑后校准并记录实测值）。

### C4 minicpm5_2b 【依赖 A5 预检报告先行】
- 步骤：①查明旧树 split stageA/B 动因（stageA.wtop 1270 op 已产——是全模型 compile 失败还是 blob/内存超限？查 .wtop.work 残留与会话记录）；②A5 预检全模型 net.json；③全模型单 blob compile（目标：不做 split）；④设备 run vs gold.f16.raw（in.ids32.raw [1,16] 注入）。
- 门：全模型单 blob 设备输出值差达标；若必须 split，书面记录硬件限制证据。

## 6. WS-D — 调度/内存/runlist 可靠性 + WS-E 通用性工程化

### D1 内存规划收编 M3+（续 GEHTP_MEMPLAN_COMPILER_MIGRATION.md）
- 范围：M3（emit 侧 shadow 旗标删除、旧 bin 回退路径到期评估）→ 该计划 §3 后续阶段；cp_solver 默认化评估（HNNX_VTCM_ALLOCATOR=cp* 与贪心对比报告）。
- 门：迁移计划各阶段门照旧（byte-exact 回归）；cp_solver 对比报告入 docs/。

### D2 runlist 三方一致性验证器
- 范围：tagged.bin（TAG_PLAN_ORDER+TAG_MEM_PLAN）↔ blob（emit 发射序+TEMPOFF）↔ 设备 optrace（实际执行序+逐 op 耗时）三方自动对拍脚本：顺序一致性、每 op 输入输出 temp 偏移落在 plan 内、生命期不重叠断言。
- 门：conv_add/L3/0.8B 三图全一致；引入一个故意错位（手工改 blob 偏移）必须被检出。

### D3 temp 生命期冲突静态检查器
- 范围：对 blob 做静态分析：temp id 复用时生命期区间不重叠（66bae03 修复类 bug 的系统门禁）、TEMPOFF 偏移+size 不越池界、slot 引用合法。做成 `wt_inspect` 扩展或独立 host 工具。
- 门：四模型 blob 全过；合成冲突 blob 必检出。

### D4 cost model 实测标定
- 范围：用 optrace 逐 op 实测耗时回归 cost_model.cpp 的每元素常数表（Conv=9/MatMul=4…）；产出标定报告与拟合误差。
- 门：四模型各层 op 预测 vs 实测误差表入 docs/；误差>2× 的 op 型列清单（后续迭代）。

### E1 环境自检与路径治理
- 范围：gehtp 五件套路径（SDK/PY/HEX/SWIV/设备）全部环境变量化+默认值；新增 `gehtp doctor`（逐项检查存在性+版本+设备在线）；修死路径：gguf_wtlink.py:24 sys.path、device_run.sh:17、conv_add/gen_all.sh:7-8；PROVENANCE 路径刷新。
- 门：干净 shell（无 ANDROID_SERIAL 等预设）下 `gehtp doctor` 全绿；删掉任一依赖 doctor 必报。

### E2 正门 bug 与诊断
- 范围：`--input-f16 a,b` 逗号 realpath 即死修复（gehtp compile 的 in_f16 realpath 行）；compile 各阶段 `| tail -1` 吞诊断改失败时自动吐全 log；.work 中间产物加 `--keep-work` 开关+默认成功即清（4B 一次 35GB）。
- 门：逗号多输入 compile 通；人为制造第 3 步失败能看到完整错误。

### E3 通用 HF 导出器
- 范围：一个 `export_hf_onnx.py` 参数化架构（qwen35-GDN / qwen3vl-text / minicpm / 通用 fallback），取代 export_qwen35_onnx.py / export_qwen35_4b_onnx.py / export_qwen3vl_4b_onnx.py 三份 95% 复制脚本；minicpm 导出直接走它（E3 是 C4 的前置输入）。
- 门：三架构导出 ONNX 与现存专用脚本产物 ORT 自检同级（cos 值入报告）；新模型（任选一个小 HF 模型）零新代码导出。

### E4 CI smoke 与文档
- 范围：host 侧 CI（无设备）：编译器 ctest + quickstart 的 host 段 + op_inventory 预检回归；设备段保留人工门禁手册化。README/PORTAL/PROVENANCE 三件套刷新到 8936c01 实况（README 里程碑、PORTAL 矩阵、PROVENANCE 修改登记）。
- 门：CI 配置入仓 + 一次全绿记录；文档与代码无矛盾（抽查 10 处锚点）。

## 7. 认领登记表（会话认领后填写；改动接口前必须先在此登记）

| 任务 | 认领会话 | 状态 | 占用文件 | 设备时段 | 完成门证据(commit/log) |
|---|---|---|---|---|---|
| A1 | （建议 VL 线本人） | | op_transpose.cpp | 否 | |
| A2 | | | op_tile.cpp(新)/oplist_parse.h/ops.cpp | 设备段 | |
| A3 | 105 会话 (branch gehtp-08b) | **进行中** (①审计) | op_copy_sem.cpp/ops.cpp/oplist_parse.h | ②③时才需要 | |
| A4 | | | qnn_ir_loader.cpp/wtop_ops.hpp | 否 | |
| A5 | | | scripts/gehtp/op_inventory.py | 否 | |
| A6 | 105 会话 (branch gehtp-08b) | **闭合** 98e5e1d | ops.cpp/graph_prepare.cpp | 否 | L0/L3 零直通 + bogus 三模式 + ctest 44/44; 新增 factory-null 暗洞同治 |
| B1 | 105 会话 (branch gehtp-08b) | 排队 (G3 后集成; default-off 纪律) | oplist_exec.c (独立 xop 文件+表登记) | 短槽 op 级 | |
| B2 | | | oplist_exec.c | 设备段 | |
| B3 | | （先查 VL 在途） | oplist_exec.c/42_runner/gehtp | 设备段 | |
| B4 | 105 会话 (branch gehtp-08b) | G3 后 | oplist_exec.c/wtop_emit.cpp | 否 | |
| C1 | 105 会话 (branch gehtp-08b) | **进行中** (P0-P3; 与例41线/disk2-ca 交接) | test_models/qwen35_08b/, test_assets/l0/(新) | 设备段长 | T0+G0 闭合 (04ea082; conv_add 与 4090 VL 构建逐字节全同) |
| C1b (新行) | 105 会话 | 进行中 | gdn_kern.c/gdn_sm.h wrapper + ex19 harness | 短槽 | gdn_kern 0.8B 参数化 (d_inner=2048/h=16/d=128), 目标 ex19 8/8 |
| opcode 预登记 | 105 会话 | 通报 | oplist_parse.h | — | OP_SCATTER_ND(A3②备)/OP_GDN_STEP/OP_QGEMV/OP_GQA_FAD 只追加, 与 A2 OP_TILE_F16 错峰 |
| C2 | | | test_models/qwen35_4b/ | 设备段长 | |
| C3 | | | test_models/qwen3vl_4b/ | 设备段 | |
| C4 | 104 本会话 (branch gehtp-08b, 2026-09-17) | **进行中** | test_models/minicpm5_2b*/ scripts/judge_minicpm.py | 设备段(编译完后) | 预检通过(12 型全覆盖); 撞 id 根因见 §8 |
| D1 | | | graph_prepare.cpp/wtop_emit.cpp | 否 | |
| D2 | | | scripts/ 新增 | 设备段(只读 optrace) | |
| D3 | | | kernels/host/ | 否 | |
| D4 | | | cost_model.cpp | 设备段(只读 optrace) | |
| E1 | | | scripts/gehtp 等 | 否 | |
| E2 | | | scripts/gehtp | 否 | |
| E3 | | | scripts/export_hf_onnx.py(新) | 否 | |
| E4 | | | .github 或等效/docs | 否 | |

**并行启动序**：
- 立即并行：A5、A6、C1、D2、D3、D4、E1、E2、E3、E4（互不依赖，文件无交集）。
- VL 在途落地后：A1（交接）、A4、B3。
- A1 后：C3；A1+A2 后：C2；A5 后：C4 预检起步。
- A2/A3 的 opcode 登记与 B 线引擎改动错峰（同一文件 oplist_exec.c/oplist_parse.h）。
- B1/B2 与 A2/A3 的设备执行体改动同文件（oplist_exec.c）→ 按 §2.1 错峰，建议 B 线先行，A 线新 op 执行体随后追加（不同函数，冲突小）。

**总验收门（战役结束判据）**：四模型 `gehtp compile` 全通 + 设备 `gehtp run` 输出达各自判据 + conv_add/L3/L0 回归全绿 + D2 三方一致性四图全过 + `gehtp doctor` 干净环境全绿。

---

## 8. 战役日志（追加区，新进展置顶）

### 2026-09-17 C4 线 (104 会话) 中段发现

1. **loader 撞 id 根因（四模型共享）已双向定案**：tensor_params(perm/axes) 与权重 const 共用 JSON id 编号空间，`append_const_node` 撞 id **静默返 0**（不覆盖不后移），旧码不查返回值且把占用者 name_tag 改写为 perm 名 → Transpose.inputs[1] 接到权重 const。minicpm 11 处、4B/VL 同款（"perm 轴数 0xA00000/0xC00000>5" 实为权重字节数/4）、0.8B 也藏 1 处（layers.0/linear_attn/Transpose_5_perm id=155 撞 onnx::MatMul_188646 [1024,2048]——C1 线全模型编译必经此雷）。VL 在途（已入 T0=04ea082）含同款根治（"0.8B 全图 88 处撞号"实锤 + op156 SIGSEGV 实锤），本线独立复证。emit 侧 perm 形状贪心推断（VL, op_transpose.cpp）为缺属性兜底，两层互补。
2. **minicpm split 动因定案**：split stageA/B（1249/1252 节点，均 54-63MB 小 blob 走外部权重）正是全模型 compile 撞 perm-collision 的绕行；根因修复后全模型单 blob 路线成立，C4 按单 blob 推进。
3. **test_e2e 基线回归**（8936c01 即挂，非本次引入）：`in_bc` 广播阅读器对无形状调用方恒读元素 0 → Add 断言失败。已修（8e1d78a，空 in_defs 回退线性直读），ctest 44/44 复绿。
4. **EXT_IN int32 槽位尺寸瑕疵**（未修，E 线登记）：emit 输入槽按 elems×2(f16) 建，int32 ids（elems×4）尺寸不符；运行时 g_ext_in 指针覆盖使其不影响正确性，但槽 len 语义错误，B3 多输入改造时一并收口。
5. **host_run 输入加载实为 raw 字节直读**（无 f32 转换），int32 ids 位模式可直接过 Gather——E2 可把旗标名 `--input-f32` 改 `--input-raw` 消除误导。
6. minicpm 预检结论：12 op 型全在注册表；rank>4 张量 252 个全可折前导 1（且仅触及 Reshape/Eltwise_Binary，broadcast 的 fold4 右对齐已覆盖该形态）；Transpose 无 rank>4；Neuron 6=SIGMOID、Binary 13=MUL/0=ADD 映射齐。

### 2026-09-17 C4 线 (续) — WTOP v2 登记 + 设备事故通报

7. **WTOP v2 格式登记（接口冻结点 §2.1）**：slot 线记录 16B→24B {len,count,offset_lo,offset_hi,addr,reserved}，offset 升 u64。动因：minicpm f16 全模型 blob 5.15GB 撞 v1 u32 偏移 4GB 天花板（emit 自校验 max_slot_end=4309975296 实锤；旧树 split 即此限制的历史绕行）。emit 仅在权重区（内联/外置）>0xFFFFFF00 时写 v2，小模型继续 v1（conv_add blob 逐字节回归已验）；`--force-v2` 供小模型走 v2 通路做回归。改动：oplist_parse.h/.c（parse 双版本）、wtop_emit.cpp（编码）、wt_slot.offset 内存侧 u64。引擎/上游工具全指针算术自动兼容。附带修复：manifest op_names 多引号 bug（T0 引入，quickstart 判据脚本 JSON 解析实锤）。
8. **设备事故（2026-09-17 ~17:55）**：52f67807 被重启（uptime 4min 实锤，非本线操作），重启后 **SELinux 回弹 Enforcing** → FastRPC Capability API 失败 + unsigned PD dom3 拒绝（0x72）。重启前同二进制同流程全绿（16:30/17:50）。设备文件（装载器/skel/库/runner）均幸存。**恢复需要 root 权限操作（setenforce 0 或等价 sepolicy 处置），本线无 adb root 权限（策略拦截），设备段挂起等待。** 教训入坑表：设备门禁应加 `getenforce` 预检。
9. minicpm 全模型 compile 实证：hnnx_compile 32min/32.7GB RSS 通过（2492 op）；tagged.bin 10.07GB（修复后比修复前多 2.2GB = 撞 id 被静默丢弃的权重全数回收——**修复前任何全模型 tagged.bin 都在静默丢权重**）。

### 2026-09-17 C4 线 (续二) — host 门全绿 + 设备套件就位(等楔死恢复)

10. **C4 host 全链闭合**: hnnx_compile ✓(2492 op, 10.07GB, 32min) → wtop_emit v2 ✓(641 slots/2532 ops/5.15GB, host 三方校验) → **host_run 全模型参考链 vs gold.f16.raw: PASS — cos=0.9999887, top1 16/16, max|d|=0.048(相对 0.25%)**(判据 scripts/judge_minicpm.py)。剩唯一门=设备 run。
11. **设备套件已就位(身份制合规)**: 设备 gehtp/ 内有 minicpm5_2b.wtop(5153758080B 尺寸核验)+ in_c4.ids32.raw + **job_c4.txt + gehtp_runner_c4.so**(独立 job 路径/二进制名, 不共享 job.txt; vgather=0)。恢复后一条命令: `adb -s 52f67807 shell "cd /data/local/tmp/hvxhmx23 && ADSP_LIBRARY_PATH=. CDSP_LIBRARY_PATH=. ./run_main_on_hexagon 3 gehtp_runner_c4.so"` → 拉回 out_c4.f16.raw → `python3 scripts/judge_minicpm.py --out out_c4.f16.raw --gold .../gold.f16.raw`。
12. **ABI 注意(v2 结构 widen)**: wt_slot.offset u32→u64 使 wt_blob 内存布局变化; **旧 ABI runner(如板上 gehtp_runner_prof.so @09-17 早)+ 新 libhvxhmx_v23.so = 读错位**。板上标准 lib 已是 HEAD(v2)构建; 各位面 runner 必须随 HEAD 重编(默认=旧行为条款的例外, 特此通报)。
13. **CDSP 楔死定罪(板方问题, 与本线流程无关)**: 失败在装载器 `remote_handle_control(DSPRPC_GET_DSP_INFO)`(payload 未加载即败); 未碰过的 8-17 老 payload 同败; 内核日志 `boot_cdsp` 服务 60s 超时 exit 1(CDSP 本次没起来); SELinux 已是 Permissive(排除); 槽位 _a 未切(排除)。与 [[v81-board-shared-session-traps]] 的 0x39 楔死同族 → **恢复=板主/物理上电, 勿自重启**。
14. VL-4B 编译探测(A1 证据, 顺带): 用 T0(含 VL 撞 id 修复)重编 qwen3vl_4b 中(旧 tagged.bin 系修复前产物, 静默丢权重不可用)。
