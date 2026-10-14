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
| C4 | 104 本会话 (branch gehtp-08b, 2026-09-17) | **设备门** | test_models/minicpm5_2b*/ scripts/judge_minicpm.py | d0f1784 经 110, 2026-09-20 排队 (~30min: 推 5.15GB+2532op run); 只碰 *_c4* 文件 | host 门全绿(790a198 cos=0.9999887 top1 16/16); 撞 id 根因见 §8 |
| PROF-G2 | 105 本会话 (branch task/prof-wp) | **排队** (C1/C4 后) | oplist_exec.c/42_gehtp_runner/scripts/gehtp_prof.py (W-P2/W-P1, 未提交) | d0f1784 经 110, 2026-09-20, 短槽 ~5min (conv_add 2s run + 拉回 op_ts/device.txt) | G-P1 host 全绿 (合成件+真 v2 blob); 上机后=conv_add 零退化 + op_ts 真数据报表 |
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

### 2026-10-14 0.6B decode 线 — GEMV t1 内核 ILP 修复: step_us 655→118.2ms (5.5×) + M=1 GEMV 上 HMX 路线裁决 NO-GO (85cf28d)

**归因**: 0.6B (Qwen3-0.6B) decode 经 gehtp_proxy 常驻 runner (f3003eb) 稳定 655ms/步 (1.53 tok/s)。op_ts 分解: MATMUL_F16 ×253 = 530.7ms = **91.5% wall** (lm_head 143.8ms + 252 层 FC 386.9ms), 全 HVX 2.0-2.2GB/s (标定 UserDMA 59.9GB/s), Σdma_us=0 — 无任何 staging/tiling 决策参与。

**三臂裁决 (例 50_q6b_gemv_ab, d0f1784, 引擎 ts 实锤 + 确定性整数金标 byte-exact 全门绿)**: "VTCM/tiling 规划" (= 编译期 crouton 布局 + HMX + 线性 DMA 驻场, K=1024/N=2048/W=4MB 同数据同形状) 对 M=1 GEMV 族 **NO-GO**:
- A M=1 HVX 现状 2596µs (1.50GB/s) ‖ B M=32 HMX 运行时散 pack 35047µs = 0.07× ‖ C M=32 HMX 预打包 crouton+DMA (w_pp=flags bit3) 4664µs = 0.56×。
- C 已把 pack 消掉 (B→C 7.5×) 但 HMX MMA ~1.75µs/pair ≈ 37GFLOP/s 是发射率墙 (与 0.8B lm_head 37.6 同墙), M=32 单行 tile 喂不饱 HMX — 是发射率墙不是搬运墙。w_pp 机制本身正常, 大 M matmul 仍可用。

**真杠杆落地 (85cf28d, 库 19c988d1, runner 1759b60d, 位级不变)**: t1 gemv 内核 4 列独立累加器交错发射 + hex_l2fetch 前瞻 2 组 (`oplist_exec.c` exec_matmul_f16_gemv); 位级不变量 = 每列 rp 仍 k 升序同一 mpyacc 序 + 标量尾段原样。
- 单 op: 2596→1021µs (2.54×, 3.83GB/s)。
- **0.6B E2E decode: step_us 655.0→118.2ms 均值 (p50 118.6/min 115.3/max 119.9) = 8.46 tok/s, token 10/10 位级 == 金标** [13,576,6722,315,15344,374,21718,13,576,6722]; 全图 1.2GB 权重流/步 ÷ 118ms ≈ 10GB/s 聚合 = vmemu-DRAM 实测上限 (ex49 标定) — **带宽已到顶, 下一步空间在字节减半 (W4A16 供给侧)**。
- 0.8B 各线共享此内核且设备 v23 lib 即此版, 重跑自动受益 (位级不变量保证数值门零回归)。

**坑**: ①重建 session 前必须 `rm req.bin resp.bin` — idle 自退后残留的 shutdown 帧会毒死新 boot 的 runner (首请求即优雅退出); ②单 op 小 N 摊不平启停 (3.83GB/s) vs 全图聚合 10GB/s, 勿用单 op 带宽外推全图。

### 2026-10-08 003 线 — blob-v3 契约收口(引擎指派/显式回落/carve+TEMPOFF) + 0.8B 全新管线无补丁复现

**blob ver=3 契约落定** (`oplist_parse.h` / `wtop_emit.cpp`):
- 24B slot 记录恒定; 引擎注解 footer = n_ops×u32 (`engine` bit0-1 ∈{0,1,2} | `fb_allowed` bit2 | `fb_engine` bit3-4); 可选 VTCM carve footer `[magic 0x43524156][count][count×20B {op_index, vtcm_offset, size, residency_start, residency_end}]`, 生命期窗口按发射边界解析。
- emit 组装段三重不变量**硬错误**: ①carve 生命期重叠⇒空间必不交 ("carve alias" 退出) ②footprint>vtcm_cap 退出 ③carve/TEMPOFF 双源 VTCM 布局集合不一致退出。ctest 负例锁定: `test_wtop_emit.cpp` §10 conv 图预算化后腐蚀 TAG_MEM_PLAN 全 entries 同偏移→wtop_emit 退出≠0 且错误指向 carve 不变量。
- 运行期回落裁决统一入口 `engine_fallback_or_die`: fb_engine=scalar 且允许→dev=1 留痕按标量执行; fb 非 scalar 或禁回落=die (dev=2), 绝不静默换单元; op_ts ver3 三态落盘, `gehtp_prof.py` 不变量机验。

**0.8B v4 全新管线验收** (当前树 hnnx_compile `--no-pass fc_bias_fold` + wtop_emit 三输入, 无补丁无硬编码, d0f1784):
- 14188 op 注解全合法 (scalar 10999 / HVX 2869 / HMX 320; fb 恒允许→scalar — 数值敏感禁回落留 US3 细化)。
- 设备偏差报告: **13165 按计划 / 1023 允许回落 / 0 中止**, 不变量 PASS (回落全为 broadcast HVX→scalar, 该 op 无 HVX 内核, 留痕非静默)。
- 精度 cos 0.998642 / top1 30/32 = **同库 (a42720a5) v17 口径逐位一致** (Δ4e-6 已由 004 裁决为 175 个 M<32 matmul 标量舍入, 无回归); wall 35.31s — 同代 v17 实测 38.94s 快 3.6s; 33.93s 为 09-28 旧库锚, 当前库任何 blob 未复现 (v3 逐 op 裁决/记录开销 ~1.4s 为 SC-002 证据的代价, 优化留后续)。
- **坑 (PORTAL 已登记)**: 烘焙 mask 必须走 **f16 槽面文件** (`mask_ones.f16.raw`) — 图输入节点 loader 口径全 Float32, `.i32.raw` 后缀强保 i32 原文→设备 f16 视图把 i32 1 读成两个 +0.0→`EQ(0,0)=1`→mask 全 -inf (v3f 同症状异根因); pos (i32 arange) 则必须 `.i32.raw`, 否则被 f32→f16 当 denormal 洗成 0。

### 2026-09-28 C1 线 (eqfix 会话) — 0.8B 设备语义偏差收敛: Gather axis 空 op_data + SELECT 三输入广播缺双根因, logits cos 0.15→0.988

**动机**: 接上条 — v3 修复全 -inf 后设备 logits cos 仅 0.14~0.22 (top1 近乎全错), 语义级偏差二分。

**根因 ① — Gather axis 读空 op_data (v4 修复)**:
- tagged.bin deserialize **不填 `OpDef::op_data`**(`serialize_opdef` 只写 extra 尾随段 → `serialized_extra`), op_gather.cpp 直读 `od->op_data` 恒得空 → axis 默认 0 → GDN gather (真轴 3, compact 后 2) 全错位, op222 cos 0.0103。
- 修复: 与 op_concat/op_softmax/op_cumsum 同源, 改读 `serialized_extra`(ExtraAxis, 20B)。v4 blob 校验: 1192 gathers 轴分布正确 (3×axis=0, 55×axis=2, 1134×axis=3)。
- **坑表登记: tagged.bin 路径下所有标量参数必须走 serialized_extra + extra_info_registry 提取器; op_data 仅在 loader 编译期路径填充。**

**根因 ② — SELECT(Eltwise_Ternary) 三输入广播缺 (v5 修复, 设备全 nan 的直接来源)**:
- 设备 `exec_binary` SELECT(subtype=8) 原实现: loop 界 = true 分支元素数 `an`, 只支持 cond/假值的**标量**广播。GDN 门控链 (op id=219): cond `[1,1,64,64]` 4096 元素(16 头广播到 65536) + true 分支**标量** 1 元素 + false 65536 → 设备只写 1 个元素, 其余 65535 读旧 temp 垃圾 → op106 Unary 全 nan → Gather tbl 全 nan (op109 nan 根因)。
- host ops.cpp 同款 op 是 `in_bc()` 三输入 numpy 广播 — emit 侧从未物化。
- 修复双侧:
  - emit `op_eltwise_ternary.cpp`: 与 op_eltwise_binary 同款, 任一输入元素数 < 输出即 OP_BROADCAST_F16 物化为全尺寸 temp (cond 4096→65536, true 1→65536), SELECT 保持纯元素语义。
  - 设备 `exec_binary` SELECT: loop 界改 `max(cn,an,fn)` + true 分支标量广播 (`tv[an<=1?0:i]`) — 防御标量 true 直通路径。
- v5 blob: n_ops 16297→16443 (+146 broadcast 物化), GDN gate SELECT 实锤带 cond/true 双物化。

**设备验证 (d0f1784, v5, hook 阶梯对拍)**:
| 锚点 | op | cos |
|---|---|---|
| op109 MatMul | id=215 | 1.000000 |
| op110 Binary | id=216 | 1.000000 |
| op113 SELECT | id=219 | 0.610667 (514/65536 元素错位, 下述) |
| op114 Unary | id=221 | 0.999999 |
| op117 **Gather** | id=222 | **1.000000** (修复前 nan) |
| op125/137 ScatterNd | id=243/266 | 0.999999 |
| op128 Gather | id=245 | 1.000000 |
- **logits (P0): 修复前 cos 0.153 top1 1/32 → 修复后 cos 0.988039 top1 26/32, nan=0 inf=0** — 与 host 链同 prompt 水平 (cos 0.9987 / top1 31/32) 同量级, C1 设备链语义偏差收敛到 host 链残差口径内。

**挂账**: ①op113 SELECT 残差 514/65536 元素 (512 处 host=0 而 dev=in2 直通, 2 处反之) — 模式 = cond 全 1 列(head 0 对角线)被当 false; 下游 op114 cos 0.9999 显示影响甚微, 疑似 host/device cond 在边界列的 dump 口径差, 未收口。②op122 Reduce(id=237) 16 元素 dev 全 0 vs host 非零 (cos=0) — 同族小锚点, 下游 op125 ScatterNd cos 0.9999 未传播, 挂账。③C1 总门 cos≥0.9999+top1 4/4 仍未达 (host 链自身亦然), 门收口另行。④build_linux 有 root 所有残留 (09-22 容器误 build) 致 4 个 test 目标目录陈旧 — 非本树代码问题, /tmp 干净副本全量构建 ctest 47/47 绿; build_linux 需 root 清理后重 cmake。

**清理**: 全部临时调试代码已删 (op_gather.cpp GEHTP_DEBUG_GATHER×4 块, wtop_ops.hpp [wslot]/[i32slot]×4, op_eltwise_binary.cpp [bdiag]); ctest 47/47 绿 (/tmp 干净副本)。

### 2026-09-22 C1 线 (eqfix 会话) — 全模型设备 run 三重门连环爆破: emit 静默兜底 ×2 + MAX_TEMPS 物理上限

**动机**: CONVERGE Phase 1.2("broadcast opcode 补齐")复盘。结论先行: ①broadcast 执行体本树 09-01 已具(de237a1), 09-18"停点"系板上陈旧 lib 假象; ②真实阻塞 = emit 两处静默兜底毒化数值; ③**本树 lib 的 MAX_TEMPS=256 物理上限使全模型从未在本树跑通过**——09-19 22:41 optrace 全过(/tmp 已清)是平行树 lib 的功劳, 非本树能力。

**实锤链**:
1. 0.8B 设备存量输出(output_08b/dev_out, /tmp/gehtp_110 遗迹)经判分: **7,946,240 元素全 -inf**——执行链 rc=0 但数值全死。
2. blob 取证(.scratch_a3 v2): idx23 `BINARY [0x8001,7,5,32,sub=0]` = **EQUAL(operation=3) 被 qnn_binary_to_sub 静默兜底成 ADD**(全图唯一 Equal = pad 掩码 Equal(attention_mask, Const_17)); 另 **SOFTPLUS(Neuron 7) ×18 被发成 UNARY sub=0xFFFFFFFF**, exec_unary default 静默直通(GDN 门控全错)。optrace 实锤 `code=11 sub=4294967295`。
3. v2 blob 三个烘焙输入槽**全零**(编译未给 --input-f16, PORTAL "不给则全零")——mask=0 + ADD 兜底恰好不点火(ADD(0,0)=0), 全 -inf 另有来源(SOFTPLUS 直通/scatter_nd/Pad 候选); **EQ 修复后烘焙必须给真值(mask=ones/pos=arange), 否则 EQ(0,0)=1=全掩码=必全 -inf**。
4. **MAX_TEMPS=256 物理上限实锤**: v3 复跑死 op 2617(1-based), err=`broadcast ref fail`; blob temp id 至 1403+(输出 temp 1401), 而本树 lib `#define MAX_TEMPS 256`(09-01 起)——`ref_ptr/temp_get/wt_exec_temp_last_bytes` 对 id≥256 全返 NULL/0, **即使跑完全程也会写出 0 字节输出**。升 4096(对齐 WT_MAX_SLOTS)后同一 blob 顺利越过停点跑至 16297/16297。

**修复**(全部已设备验证):
- emit 映射: `qnn_binary_to_sub` case 3→4(EQ), `qnn_neuron_to_sub` case 7→13(SOFTPLUS); **静默兜底改硬错误**(op_eltwise_binary/op_unary 遇未知 operation 即报错退出, A5 预检精神最小落地)。
- 设备执行体: exec_binary case 4(EQ, host ops.cpp 同款 0/1), exec_unary case 13(SOFTPLUS 稳定化公式与 host 逐字同款)。
- 契约登记: oplist_parse.h subtype 注释(BINARY 4=EQ, UNARY 13=SOFTPLUS); subtype 扩展不动 opcode/arity, 旧 blob 行为不变(旧 blob 的 Equal 仍是错误 ADD 语义, 需重编)。
- **MAX_TEMPS 256→4096**(oplist_exec.c + WT_EXEC_MAX_TEMPS 同步, 例 38-41 guard 语义不变)。
- 测试: test_wtop_emit +6 断言(EQ/SOFTPLUS subtype 契约 + 未知 op 硬错误 ×2 + 建图 name_tag 坑), ctest 47/47 绿。
- 构建环境登记: 宿主机 g++-9 无法编译(他方 __builtin_bit_cast, 已由 aae0e09 修 C++17), **编译器一律容器 `gehtp-toolchain:24.04` (GCC 13)**: `docker run --rm -u $(id -u):$(id -g) -v /disk1:/disk1 -v /disk1/toolchains:/opt/toolchains:ro -w /disk1/gehtp/compiler gehtp-toolchain:24.04 bash -lc "cmake -S . -B build_<tag> && cmake --build build_<tag> -j"`; scripts/gehtp 已修 source env.sh(五件套死路径收敛)。
- v3 blob: v2 tagged.bin 直接重发射(免 2-4h hnnx_compile), f16 权重域(GGUF 不在本机), 2.04GB, EQ sub=4 ×1 + SOFTPLUS sub=13 ×18 + 烘焙 ids/mask=ones/pos=arange 逐项槽位校验。

**设备门 (d0f1784, lib=本树构建含 eqfix+MAX_TEMPS 4096)**: 复跑 v3 ×4 prompts → judge 结果见下条。

**2026-09-23 设备复跑日志（eqfix 会话）**:
- v3 blob 已在 d0f1784，尺寸一致，设备侧未提供可选外置 `qwen35_08b_v3.weights.bin`；runner 正确回退到 blob 内嵌 f16 权重（不是失败原因）。
- `qwen35_08b_v3.device.txt` 实锤：`wt_parse ok`、`wt_exec_run_io rc=0`、`ops=16297`、`out_temp=1401`；总耗时约 275.7 s，说明 MAX_TEMPS=4096 后已完整跑过全图。
- 之前 P2/P3 复跑因多个 run 并发共享同一设备 `job.txt`/`output.f16.raw` 通道而互相覆盖，退出码不具判读性；后续改为串行，每次完成 pull 后立即 judge。

**2026-09-23 4-prompt 判定矩阵（串行完成，同 lib 同 blob）**:
| prompt | nan/inf | cos | top1 | host 链同 prompt cos/top1 |
|---|---|---|---|---|
| P0 | 0/0 | 0.153144 | 1/32 | 0.9987 / 31/32 (p1_logits_0_fixed) |
| P1 | 0/0 | -0.067401 | 0/32 | 0.9986 / 30/32 |
| P2 | 0/0 | 0.220459 | 0/32 | 0.9995 / 32/32 |
| P3 | 0/0 | 0.142622 | 4/32 | 0.9988 / 26/32 |
- **eqfix 疗效确认**：修复前设备输出 7,946,240 元素全 -inf（死输出）；修复后四 prompt 全部有限值（nan=0/inf=0、logits 分布合理 ±17）——全 -inf 毒化已根除，剩下的是**语义级偏差**（cos 0.14~0.22，top1 近乎全错）。
- 输出结构：设备各行 top1 偏向小 token id（16/314/88…），行间 cos 0.71（golden 0.39）= 部分塌缩；row0 cos 0.24 / row31 cos 0.41，信号存在但整体偏。
- 分叉定位工具落地：**exec_hook_dump 双平台化**（原 hostsim 限定 GEHTP_HOOK → 加 `wt_exec_set_hook` + runner job.txt `hook` 键 + `gehtp run --hook "idx:temp,..."`，设备产物 /data/local/tmp/hrt/gehtp/hook_<idx>_<temp>.f16.raw 自动拉回 prof/）。锚点 = embedding(op8→temp0) + 全部 79 个 RmsNorm 输出（80 点 spec 762 字符），host 对照 = .scratch_a3/dump_fixed（P0 输入全图 host_id dump，实锤覆盖 80/80 锚点）。

**2026-09-23 06:xx 复现性实锤（eqfix 会话，lib 全量重建后）**: 用正确 `HEXAGON_SDK_ROOT=/disk1/toolchains/...`（build_libs.sh 默认 /local/mnt 死路径，env.sh 的 TC 不被它读——坑表补登）全量重编+重签 libhvxhmx_v23 并 gehtp setup 重推 d0f1784，串行复跑 p0~p3 → judge 数值**逐位复现**上表凌晨矩阵（P0 0.153144/1·32, P1 -0.067401/0·32, P2 0.220459/0·32, P3 0.142622/4·32）。跨 lib 重建、跨 4.5h、cos/top1 完全一致 = **偏差是确定性语义错误，排除竞态/随机硬件噪声**，hook 对拍二分具有可重复判读基础。产物 .scratch_eqfix/out_v3_rerun_p{0..3}.f16.raw。
- 行错位排除（新增判读）：设备 32 行 logits 与 golden 32 行两两 cos 最优匹配行号散乱（P0 最优行 ∈ {17,31,2,19,6}，P2 ∈ {0,2,20,19}），无 clean permutation，最优 cos 0.2~0.71 —— **不是 seq 维行移位/转置**，是逐行内部数值部分塌缩（行间自相关 0.71 vs golden 0.39），与 scatter_nd/GDN 门控污染假说相容。

**挂账**: ①设备/host 分叉点定位中（hook dump 对拍链：hook_<idx> ↔ manifest.op_ids[idx] ↔ host_id_<oid>，M3c 机制复用）；首要嫌疑 scatter_nd×1154 首实跑/Pad 尾部垃圾/Cast; ②C1 总门 cos≥0.9999+top1 4/4 未达(host 链自身 0.9987/26~31 亦未达，门收口另行); ③oplist_exec.c 已被 58cc915 合并提交（eqfix+P5 OP_DMA 修订），ptr_in_vtcm 重复定义编译错误由对方会话 01:49 修复；dump 改动（exec_hook_dump 双平台化）未提交，commit 需与活跃会话错峰; ④v2 blob(EQ 错误语义+烘焙全零)及其设备输出全部作废, test_assets/qwen35_08b/qwen35_08b.wtop 同源需按 v3 流程重编替换。

### 2026-09-17 C1 线 (105 会话) — VL Gather 回归定罪 + A3 判决 + A3② 落地 + 设备楔死旁证


1. **VL 在途 Gather 通用化重写 = host 链回归源（bisect 三组对照定案）**：compact 剥前导 size-1 维后 axis 未同步左移。L0 GDN tbl `[1,16,1,64,64]` axis=3（converter 按原始 rank 写）→ 压缩 `[16,1,64,64]` 后真轴=2，VL 用 axis=3 取末维 → 全层错位。现行链 L0 host vs HF 锚点 cos=**0.53**（pre-VL=1.0 / VL minus Gather=1.0 / 4090 VL 二进制=0.53，三方对照）。**修复 = `axis -= (原始rank - 压缩rank)`**（105 提交 ef5869e），修复链 cos=1.0、vs gold_ncf 0.999999。**104 基线树 host_run 对全模型 GDN 层输出当前不可信（66 gather/层 × 18 层同型），修复须回灌。**
2. **A3 判决（数值实锤，docs/A3_COPY_SEM_AUDIT.md）**：ScatterNd 恒等拷贝 → L0 cos **0.102 死刑**（GEHTP_HOST_*_IDENTITY 仿真旗标已入 ops.cpp）；Pad 恒等 L0 无害（尾部垃圾挂账）；Cast 待全模型；**"L3 恒等安全"是伪命题——L3 含 0 个 ScatterNd/Pad/Cast**，从未考验该路径。0.8B 全部 1245 copy-sem 站点（除 Reshape）均可达 logits，无死端。
3. **A3② 已落地**（c3f6e7f+52bf1d3）：OP_SCATTER_ND_F16=28 三件套（emit op_scatter_nd.cpp 自注册 / parse arity+arg_is_slot / exec_scatter_nd 坐标块写与 host 同口径）。L0 重编 blob 含真 scatter ×64。设备门待 DSP 恢复。
4. **设备楔死独立旁证**（与 C4 §8.13 同判）：0x39 全域失败=CDSP (gunyah VM) 本次未启动——remoteproc 未注册、boot_adsp/boot_cdsp 60s 超时 exit 1、SELinux 已 Permissive、两版 skel（380f3cb 正典/d0bfbc third_party）同败、adb root 同败。**非 skel/装载器问题，恢复=板主物理上电**。另：build_examples.sh 会推 third_party skel 覆盖设备正典件（md5 不同），已记坑。
5. **P1 host 全模型**：修复后链跑通但 logits **全 NaN**（m5 时代则是 cos≈0 无 NaN——全模型 host 链从未正确过）；dump+看门狗二分 NaN 首现 op 进行中。L3 注意力层 host vs HF 锚点 cos=0.966（mask/cos/sin 布线已核），残差集中 channel 0 区（RMSNorm 约定候选），精确定位续查。
6. **NaN 根因已定罪（输入序乌龙，非编译器 bug）**：dump+看门狗抓到首个 NaN op=4634（注意力 Softmax），逐头 row 0 全 -inf。链路：pad 掩码列 0 -inf（Equal(arange[0]=0, 0)=true）+ 因果基 row 0 只剩 col 0 → row 0 全 -inf → NaN。**真因 = host_run 三输入顺序：图 Input 按张量 id 升序 = input_ids(1), attention_mask(2), position_ids(3)——正确喂法 `ids,mask,pos`**，我按直觉 ids,pos,mask 喂反（m5 时代 cos≈0 同族：他们直接喂了三份 prompt-ids）。修正后全模型 host run 在跑。**坑表登记：host_run 多输入顺序 = 图张量 id 序，不是命名直觉序；喂错不报错只出 NaN/垃圾**。
8. **L3 0.966 亦破案（输入布局，host 链无辜）**：cos/sin 文件是 [1,32,64] nfc，图声明 [1,64,32] ncf——转置喂入后 **L3 host vs HF 锚点 cos=1.0**（vs 设备 gold 0.999999）。隐藏态同法（ncf 声明）。**host 链对两类层均 HF 忠实（L0/L3 均 0.999999），"host 有 RMSNorm 约定 bug" 排除**；逐 op 锚点对照显示 RoPE/Softmax/门控全链无失真。坑表：host_run 喂数据必须按**图声明布局**（ncf 族），文件布局不一致时静默错。
7. **P3 资产预备**：09-16 的 1.67GB 全模型 blob 含恒等 scatter（1154 站）已废——A3② 重编（真 scatter ×1154 + Gather 修复编译器）进行中。设备 run 待 DSP 恢复。
9. **opcode 28 ABI 分裂警报（双 C1 树各自实现）**：/disk1/GEHTP（平行 C1 会话）与 task/prof-wp（105 会话）都加了 OP_SCATTER_ND_F16=28 但 **arg 布局不同**（他们 [data,idx,upd,out,n_idx,K,dims] arity 11 vs 本方 [data,idx,upd,out,n_out,rank,dims,K,n_idx,block] arity 14）——**blob 与 lib 必须同树配套，跨树混跑静默出垃圾**。设备共享 libhvxhmx_v23.so 标准名被多会话反复覆盖（md5 对拍实锤），身份制（独立 runner 名+job 通道）在合一前是铁律。合一方向：以 task/prof-wp（含 A6/审计/C1b/PORTAL 全套超集）为 C1 家，对方树转只读参考（其 E2/RUNBOOK/judge 修复已移植完）。
10. **skel 解毒完成（task/prof-wp）**：third_party ship/ skel 已换正典 380f3cbf（d0bfbc 陈旧版=0x80000406 毒源，失败 open 级联楔死 fastrpc 通道——本次 0x39 全域失败的完整链条：他方推陈旧 skel → 通道楔死 → CDSP VM 重启未起）。**4090 基线树 ship/ 仍陈旧，patch 回合必须带上 skel 修正，否则 104 侧 setup 再推旧件再毒化**。GVM SELinux 有看门狗自动翻回 Enforcing，设备段跑前必查 getenforce。

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

### 2026-09-17 C4 线 (续三) — VL-4B 编译全通 (A1 修复有效性实证)

15. **qwen3vl_4b 用 T0 编译器 (含撞 id 修复 + perm 兜底) 全链编译通过**: hnnx_compile 2183 op → tagged.bin 16.09GB → wtop_emit **v2 blob 8.42GB** (664 slots/2178 ops, WTOP OK)。此前死因 `perm 轴数 10485760>5` 消除——A1 根因修复(loader 撞 id 双向防护)对 VL 线有效,C3 剩余=设备 run vs golden_logits(同 CDSP 楔死阻塞)。注意: VL-4B 编译期曾现双进程同写一文件插曲(后台任务复活叠 nohup 重发), 教训=大模型编译产物落盘前核对 PID 唯一性。
16. v2 格式第二实证: 8.42GB blob host 自校验通过(继 minicpm 5.15GB 后)。

### 2026-09-20 PROF 线 — G-P2 上机窗口排队登记 (105 本会话)

17. **PROF 战役 G-P2 登记**: d0f1784 经 110, 短槽 ~5min (conv_add 2s run + 拉回 op_ts.bin/device.txt)。身份制合规: gehtp_runner_prof.so + job_prof.txt 独立通道 (不碰 job.txt)。lib 换装窗口: 备份设备标准 lib (f35e06d6) → 我方 d8fc9502 (同 HEAD v2 ABI 超集: PROF+arity14; 对在册 runner/blob 兼容) → 跑完恢复。**勘误 #12**: 本线 runner 已 09-18 随 v2 ABI 重编 (两态 trace 版), 非 #12 所指 09-17 早旧件。排队序: C1 (在跑 qwen35_08b_scatter) → C4 (minicpm 30min) → PROF (本行)。
18. **skel 层①字节级定案 (顺带, 见 §8 前文 09-17 事故)**: 好坏 skel 符号表/strings 全同, 唯一差=SWIV 签名段 (56B); 未签件补签 = 已知好件 380f3cbf 逐字节一致 (确定性签名)。治理已入 task/prof-wp scripts/gehtp cmd_setup [0/5] (ship 缺签→现场补签→推后读回校验; 缺件→源码重建) + device_run.sh 拷链门卫。新板 skel 已 380f3cbf ✓ 勿动。

### 2026-09-24 EVAL 线 (本会话) — qwen35_08b_v3 精度+性能评估（复跑确认）

1. **设备 lib 被覆盖事故 + 恢复**: d0f1784 标准 lib 被它会话换成异树构建 (fc79cfb4, arity-12 scatter ABI), v3 blob wt_parse 即拒 (opc=28 na=14 arity=12 rc=-11); `gehtp setup` 回本树签名 lib (325ec2d9) 后恢复正常。再次实锤 #9 ABI 分裂坑 + 身份制必要性。
2. **复跑确认（第 3 次逐位复现）**: `gehtp run qwen35_08b_v3.wtop --input ids_p0.i32.raw` rc 0, 输出与 09-23 rerun **逐字节一致**; wall 279.31s (上次 279.64s, Δ0.1%), lm_head 224.03s (上次 224.13s)。精度/性能均为确定性结论, 非噪声。
3. **精度判决 (vs HF fp32 golden, judge_logits.py)**: P0 cos=0.153144/top1=1/32, P1 cos=-0.067401/0, P2 cos=0.220459/0, P3 cos=0.142622/4 — **全 4 prompt CHECK (FAIL)**, 维持 eqfix 挂账① (语义级偏差, hook 对拍分叉定位在途, 首要嫌疑 scatter_nd/Pad 尾垃圾/Cast)。
4. **性能拆解 (口径=gehtp runner per-op wall µs + graph wall, 对应 QNN per-op cycles/graph wall 字段; 场景=seq32 prefill 整网串行 runlist, 16297 ops; 非 num_dominant_path, 无 HMX/HVX 单元拆分)**: graph wall **279.3s** (read 3.6 + exec 275.1 + 写回 0.03)。真算-MAC 93.2% (MATMUL_F16 256.1s + CONV1D_SSM 0.4s), 装料-prep 6.8% (STRIDED_SLICE 6.7 + BROADCAST 6.1 + UNARY 3.7 + TRANSPOSE_GEN 1.3 + 杂 0.9s), spill/fill/DMA=0。**单一 lm_head (op#16296, [32,1024]×[1024,248320]) = 224.0s = 80% wall, 有效 0.073 GFLOPS** — 唯一数量级杠杆 (疑似走了朴素 HVX 路径); 其余 325 个 matmul 合计 32.0s (avg 98.5ms)。产物 .scratch_eqfix/out_v3_eval_p0.f16.raw。

### 2026-10-10 004 线 (decode-on-device) — 特性全收口: 0.8B 真自回归上线 (d0f1784)

1. **两阶段战果**: P1 (T001-T014) = M<32 GEMV 快路径 (M=1 无标量回落, seq1 wall 36.7→17.7s 28.5×) + seq=1 blob + tier1 金标 (decode==prefill 首位 cos 0.999960/top1 一致) + deploy loop 复现 (v17 prefill cos 0.998642/30-32 无回归)。P2 (T015-T023) = 命名状态表 (42 槽 20.02MiB DDR) + KV/SSM 六新 op + decode 图手术发射 (surgery_decode_graph.py, host 双路径 cos=1.000000) + DSP 常驻文件队列 runner + OpenAI serve 端到端 + tier2 金标。
2. **tier2 金标 (R1 解除)**: HF Qwen3_5 DynamicCache 实证携带 GDN/SSM 状态 (增量 vs 全序列 cos≥0.999999) → 增量贪心即真 decode 语义。设备 vs HF: random prompt 16/16 token 一致 (worst cos 0.999991)、真实文本 12/12 (0.999994; "The capital of France is"→" Paris.\nThe capital of France is Paris.\nThe" 与 serve E2E/直驱三链同串)。
3. **各阶段 wall (d0f1784 同批口径)**: prefill seq32 单发 exec 34.1s (1.07s/tok); persistent decode 步设备 1.10s (trace 关; 开=2.66s 勿混) / 含 adb 文件队列 1.28s / serve 端到端 1.15s; blob 装载 ~20s (一次性, 模型注册 21s 懒启动)。**单步 decode ≪ prefill 34s = 31× 达门; per-token 持平是诚实注记** (本阶段赢在权重一次装载+状态常驻+流式; per-op 优化空间留给 006/008 线)。
4. **架构定论 (qurt 无 socket)**: persistent 语义 = run_main_on_hexagon 单次 FastRPC 内 main() 文件请求循环 (req/resp 4B BE len+JSON+二进制体); 原型 socket 设计在 DSP 目标编译不可能。三根因史: ① optrace 默认开 2.66s/步; ② 4B BE 前缀 0x00 致 strstr 空 C 串; ③ **非对齐输入指针进 wt_exec_run_io 杀 DSP 进程** (req+4+hdr_len 对齐随 JSON 头长漂移 → 128B 对齐静态缓冲拷贝修复)。
5. **运维铁律 (decode 线新增)**: kill AP 侧 run_main 留幽灵 DSP 会话 (数十秒 reaper 回收期, 新实例 -2147482611) — 停服务恒先发 shutdown op; `ps -A` NAME 是 15 字符 comm, 判活必须 `pgrep -f`; serve 端 `input_length` 必须显式注册 (=MAX_POS 64, decode manifest input_elems=1 是单步输入非上下文)。
6. **commits**: T015-T017 状态表+op (b6ac302) / T018 图发射 (55aeb8d+2789fd9) / T019 persistent 文件队列 (dc95064) / T020 serve 接线 (fb47ca9) / T021 tier2 (0555f6f) / T022-T023 (c1731e4)。金标资产: cpu_golden/hf/tier2_0.* + text_tier2.*; host 资产 /tmp/t18d/decode.wtop (md5 910d2945 与设备一致)。
