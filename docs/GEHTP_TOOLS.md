# GEHTP 技术文档 — 作用、工具链与工具用法手册

> 生成 2026-10-09。本文是 GEHTP 的**总览 + 工具手册**：§1-§3 回答"GEHTP 是什么、怎么搭起来"，
> §4 起按分类给出**仓库内全部工具**及每个工具的用法（命令、旗标、输入输出、注意事项）。
> 姊妹文档：[PORTAL](GEHTP_PORTAL.md)（正门 CLI 速查）、[ARCHITECTURE](GEHTP_ARCHITECTURE.md)（内部工作原理）、
> [FORENSICS](GEHTP_FORENSICS.md)（取证操作手册）、[DEVICE_RUNBOOK](GEHTP_DEVICE_RUNBOOK.md)（上板交接）。

---

## 1. GEHTP 是什么

**GEHTP = 通用 HTP 图编译器 + V81 设备执行工具链**。目标是一套通用的 compiler + runtime：

```
ONNX → qairt-converter → net.json (QNN IR)
  → compiler/ (B线图编译器): qnn_ir_loader → OpDef IR → 前端优化 pass
      → 依赖/寿命分析 → ST-Cut 调度 → 单算子 tiling → FancyAllocator/cp_solver 内存规划
      → 序列化 (tagged-record runlist)
  → wtop_emit: 读编译期内存计划照抄 → .wtop blob (单文件: 权重+runlist+槽位+静态偏移表)
  → V81 板自研引擎 (kernels/): oplist_parse → oplist_exec 按 runlist 派发 HMX/HVX/标量 kernel
```

三个核心主张：

1. **不依赖官方 qnn-net-run**——设备侧执行器（`gehtp_runner.so`）是自研通用 runner，同一份 .so 执行任意编译产出的 blob，不随模型改动。
2. **任意模型、零适配、不硬编码**——编译器覆盖 20+ 型 opcode 家谱（0.8B 全模型 16297 op 实测），新 op 才需要扩编译器。
3. **内存编译期定死、运行期照抄**——依赖/寿命/DDR 偏移/VTCM 驻留全在编译期规划进 blob（TAG_MEM_PLAN），设备侧零 malloc。

编译器本体（`compiler/`）是对 Qualcomm `libQnnHtp.so` / `libHtpPrepare.so`（QNN SDK 2.48.40.260702）的**逆向工程 + C++ 重实现**（REQNN 线，代码内 293+ 处标注反汇编来源地址）；设备 kernel 库（`kernels/`）源自 V81Dev/hvxhmx_libsV2.3。谱系详见 [PROVENANCE.md](PROVENANCE.md)。

### 1.1 五大件架构

```
 ┌─ host（编译期）──────────────────┐            ┌─ device（运行期）────────────────┐
 │ ① 图解析 + 前端优化 pass          │            │ ③ 图执行器                        │
 │   QNN IR → OpDef 图 → 规范化      │            │   blob → wt_parse → 逐 op 分发     │
 │ ② 图编译：调度 + 内存规划 + 发射  │  .wtop     │   （函数指针表）→ 引擎门选路径     │
 │   依赖/寿命 → TAG_MEM_PLAN        │   blob     │   → DDR/VTCM 搬移 → 输出 raw f16  │
 │   → 序列化 → wtop_emit            │ ─────────▶ │                                  │
 └──────────────────────────────────┘  (唯一契约) └──────────────────────────────────┘

 ④ profiling：op_ts.bin / optrace.txt（设备产出）→ gehtp_prof.py（host 分析）
 ⑤ 逐层 dump：GEHTP_HOOK（设备张量）↔ host_run（host 参考）→ 逐 op 对拍
```

外围（不属于本体）：`scripts/gehtp` CLI 壳（①的输入准备 + adb 收发）、001 serving（③之上的 OpenAI 兼容服务层）。

### 1.2 唯一契约：.wtop blob 三件套

`gehtp compile` 一次产出三件：

| 产物 | 内容 |
|---|---|
| `<name>.wtop` | 设备 blob：`"WTOP"` magic + 槽表（v3 每条 24B）+ op 表 + 128B 对齐权重区 + v3 引擎注解 footer（可选 carve footer） |
| `<name>.wtop.manifest.json` | `input_slot` / `input_elems` / `output_temp` / `n_slots` / `n_ops`；v2 起另有 `op_ids`/`opcodes`/`op_names`（blob op 序号 ↔ 图 op_id 映射钥匙） |
| `<name>.wtop.work/` | net.json、tagged.bin 等中间产物（排查用） |

关键概念：**槽（slot）= 编译期常量**（权重/bias/固化输入，占 blob 体积）；**temp = 运行期中间值**（占内存池，由 op 表 args 隐式定义）。特殊槽标记：`0xFFFFFFFF`=EXT_IN 主输入、`0xFFFFFFFE`=EXT_OUT、`0xFFFFFFFD`=TEMPOFF 静态偏移表、`0xFFFFFFFC`=EXT_WGT 外部权重。

### 1.3 能力边界与验证矩阵（实测）

| 已验证 | 内容 |
|---|---|
| conv_add × 3 输入 + spill 变体 | byte-exact 32768/32768 × 3 |
| L3 注意力层（81 op / 45MB blob） | max_valdiff=0.000488（值差判据达标） |
| Qwen3.5-0.8B 全模型 | 16297 op → 1.67GB blob（GGUF Q4_0 外重）；cos 0.9986 级、top1 29/32 |
| seq=1 decode（004 线） | wall 36.7→17.7s，双门 0.999987 |
| 007 性能冲刺后 wall | ≈24.24s（007 US1 broadcast HVX，ab73491） |

边界：**单运行时主输入、单输出、f16 语义**；其余图输入编译期固化（双动态输入不支持，扩展已设计待排期）；编译器 opcode 覆盖之外的 op 报 `opcode N unhandled`。

---

## 2. 仓库布局

| 目录 | 内容 | 来源 |
|---|---|---|
| `compiler/` | ①② + ⑤的 host 半（REQNN 字节级重实现：前端+IR+调度+内存规划+序列化）+ 三工具 + 45 个 ctest | REQNNFRAME/REQNN |
| `kernels/` | ③ + ④⑤的设备半（oplist_parse/exec 引擎 + HMX/HVX kernel 库）+ 33 例单元测试 + host 辅助工具 | V81Dev/hvxhmx_libsV2.3 |
| `scripts/` | 正门 CLI 壳 + prof/判据/导出/金标/replay/审计 + serving 栈 | 新建 + vtcm_engine_probe |
| `specs/` | 001-009 feature 规格（见 §15） | 新建 |
| `docs/` | 本文档 + PORTAL/ARCHITECTURE/FORENSICS/RUNBOOK 等 | 新建 |
| `test_models/` | conv_add / qwen35_08b 等模型与转换产物 | 新建 |
| `test_assets/` | 门户回归资产（l3/ 等；`/tmp` 一日两清，长期资产一律入此） | 设备实测 |
| `tests/` | serving 栈 pytest（contract/integration/unit + test_persistent_runtime） | 新建 |
| `env.sh` | 五件套工具链路径收敛（见 §3.1） | 新建 |

---

## 3. 环境准备

### 3.1 env.sh — 五件套路径收敛

`source env.sh`（`scripts/gehtp` 已内嵌 source）。自动检测容器内 `/opt/toolchains` 或宿主机 `/disk1/toolchains`：

| 变量 | 指向 | 用途 |
|---|---|---|
| `SDK` | qairt_2.48.40.260702 | 前端 qairt-converter / qairt-dlc-to-json |
| `PY` | python3（容器内 system py3.12） | QAIRT Py312 扩展 + numpy |
| `HEX` / `HT` | Hexagon SDK 6.6.0.0 / Tools 19.0.07 | kernels 交叉编译 |
| `SWIV` | swiv_build_utility.py | .so SWIV 签名 |
| `Q` | qairt_2.40.1.251119 | qnn-model-lib-generator |
| `NDK` | android-ndk-r26c | aarch64 android 交叉编译 |
| `DEVROOT` / `GDIR` | `/data/local/tmp/hvxhmx23`（+ `/gehtp`） | 设备侧工作目录 |

### 3.2 构建

```bash
# 编译器（host, Linux）
cmake -S compiler -B compiler/build_linux -G Ninja && cmake --build compiler/build_linux
# 内核库（设备, V81; 需 Hexagon SDK + SWIV）
cd kernels && ./build_libs.sh            # 全量 → lib/libhvxhmx_v23.so (+ 签名版)
./build_libs.sh one <base>               # 只重编 <base>.c + 链接（例: one fence）
# 编译器测试（45 个 ctest）
cd compiler/build_linux && ctest
```

### 3.3 设备准备（一次性）

```bash
scripts/gehtp setup [--device <serial>]
```

做四件事：skel SWIV 签名治理 → 编译/签名内核库 → 交叉编译通用 runner（vgather=0 铁律检查）→ 推送四件套到设备 `/data/local/tmp/hvxhmx23/`（装载器 `run_main_on_hexagon` + skel + `libhvxhmx_v23.so` + `gehtp_runner.so`）。详见 §5.1。

**设备纪律**：脚本默认 serial `52f67807`（历史正门口径；当前主设备以 [ARCHITECTURE](GEHTP_ARCHITECTURE.md) 基线锚 `d0f1784` 为准，`--device` / `ANDROID_SERIAL` 可覆盖）。跑前先查占用：`adb shell "ps | grep run_main_on_hexagon"`，有主就等。

### 3.4 Python 环境

| 任务 | 解释器 |
|---|---|
| serving 栈 + pytest | `/disk1/miniforge3/envs/llmserv/bin/python`（fastapi/uvicorn/tokenizers/httpx 已装） |
| 模型导出 / golden / replay | `/home/speech/miniforge3/bin/python3`（torch/transformers/onnxruntime/safetensors） |
| prof / 判据 / 审计类脚本 | 任意带 numpy 的 python3 |
| pip 安装 | 走清华镜像 |

---

## 4. 工具总索引

| 类 | 工具 | 一句话 |
|---|---|---|
| **正门** | `scripts/gehtp` | compile / run / setup 三子命令的用户入口 |
| | `gehtp_quickstart.sh` | conv_add 全链演示，期望 `=== SAMPLE ALL GREEN ===` |
| **编译器** | `hnnx_compile` | QNN IR → tagged runlist（编译主工具） |
| | `wtop_emit` | tagged.bin → .wtop blob + manifest |
| | `host_run` | host 参考执行器（f32 精确语义，逐 op dump） |
| **流水线脚本** | `conv_add_pipeline.sh` | conv2d+add host 全链 / 上板一键 |
| | `run_host_ci.sh` | host 侧 CI 门（构建+ctest+判据正负例） |
| | `device_run.sh` | 通用 .so 上板模板（构建→签名→跑→拉回） |
| | `dlc_repair.sh` | qairt-converter DLC 零 CRC 修复 |
| **性能/取证** | `gehtp_prof.py` | op_ts.bin → roofline 三分账/top-N/chrome-trace 报表 |
| | `check_op_ts.py` | op_ts.bin × blob 交叉硬校验门 |
| | `cmp_hook.py` / `cmp_l0.py` | 设备 hook dump vs host dump 逐锚点对拍 |
| **精度门** | `check_device_golden.py` | 设备输出 vs golden（cos/top1/max\|d|）CI 门 |
| | `check_3way.py` | 两份 f16 raw 三判据逐位判分器 |
| | `judge_logits.py` | 0.8B prefill/decode 双门 |
| | `judge_minicpm.py` | minicpm5_2b 判据门 |
| **模型资产** | `export_qwen35_onnx.py` | Qwen3.5 prefill ONNX 导出（transformers≥4.57） |
| | `export_qwen35_4b_onnx.py` | 同上，走 QPM modeling（4B/旧路径） |
| | `export_qwen3vl_4b_onnx.py` | Qwen3-VL-4B 纯文本 prefill 导出 |
| | `export_layer_onnx.py` | 单层 ONNX 导出 + 层 golden |
| | `golden_qwen35.py` | HF/GGUF 双 golden 生成 |
| | `golden_f16_weights.py` | f16 权重域 golden |
| | `gguf_wtlink.py` | net.json 权重节点 ↔ GGUF 张量匹配表 |
| | `replay_attn_layer3.py` / `replay_gdn_layer0.py` | HF 手动复刻单层全锚点（黄金参考） |
| **审计/清单** | `op_inventory.py` | net.json 算子类型清单统计 |
| | `audit_copy_sem_sites.py` | copy-sem 恒等站点静态审计（入仓门禁） |
| **serving** | `gehtp_server.py` + 9 个库模块 | OpenAI 兼容推理服务（001） |
| **设备侧** | `build_libs.sh` / `build_examples.sh` | 内核库 / 33 例构建编排 |
| | 例 `42_gehtp_runner` | 通用 WTOP runner（门户执行端） |
| | 例 `42_gehtp_persistent` | 常驻 Unix-socket runtime（serving 端） |
| | `kernels/host/*` | wt_inspect / wt_dump_ops / check_offsets / pack_* 等 host 辅助 |
| **库（被 import）** | `gehtp_prof`、`golden_qwen35`、serving 库群（`model_registry`/`persistent_runtime`/`device_runtime`/`sessions`/`tokenizer`/`sampler`/`metrics`/`errors`/`logging_config`） | 双重身份或纯库 |

---

## 5. 正门 CLI：`scripts/gehtp`

```bash
scripts/gehtp compile model.onnx -o model.wtop        # host: ONNX → 设备 blob
scripts/gehtp run model.wtop --input in.f16.raw --output out.f16.raw
scripts/gehtp setup [--device 52f67807]               # 一次性设备准备
```

### 5.1 `gehtp setup`

流程：① skel 治理——树内 `third_party/run_main_on_hexagon/ship/` 预编译件为真源，缺则源码重建（`make V=hexagon_ReleaseG_toolv19_v81`）；无 SWIV 段则现场补签（补签后 md5 期望 `380f3cbf`）；② 内核库 `build_libs.sh`；③ 交叉编译 runner：`hexagon-clang -mv81 -O2 -mhvx -mhmx -mhvx-length=128B` + SWIV 签名 + **vgather 计数必须为 0**（V81 errata，违反即 die）；④ 推送并读回校验 skel SWIV 段（无签 skel 的失败 open 会楔死全板 fastrpc 通道，2026-09-17 事故）。

### 5.2 `gehtp compile <model.onnx> -o <out.wtop> [旗标]`

内部五步：`qairt-converter`（ONNX→DLC）→ `dlc_repair.sh` + `qairt-dlc-to-json`（→net.json）→ `hnnx_compile --format tagged` → `wtop_emit` → 完成。

| 旗标 | 作用 |
|---|---|
| `--input-f16 a[,b,c...]` | 输入数据：**首文件 = 主输入烘焙兜底**；后续文件 = 其余图输入固化值（不给则全零） |
| `--net-json X --weights-bin Y` | **直通模式**：复用已有 QNN IR，跳过 converter（大模型必备；symlink 不拷贝） |
| `--gguf G --match T` | GGUF 外部权重直读 + Q4_0 tile-major repack（0.8B blob 从 f32 池 2.55GB 降到 ~1.67GB；未命中走 f16 兜底） |
| `--ddr-budget N` | DDR temp 池预算（字节）；超预算自动插 SPILL/FILL 算子（编译期内存规划） |
| `--vtcm-budget N` | VTCM 驻留池预算（双池分配） |
| `--external-weights` | 路线 B：权重外置为 `<name>.weights.bin`，blob 只留描述符 |

全模型直通示例（0.8B，16297 op → 1.67GB）：

```bash
scripts/gehtp compile \
    --net-json test_models/qwen35_08b/conv/model_net.json \
    --weights-bin test_models/qwen35_08b/conv/model.params.bin \
    --gguf /disk2/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf --match /disk2/match_map.tsv \
    -o test_assets/qwen35_08b/qwen35_08b.wtop
```

**输入规则**：主输入 = net.json 输入列表首个（slot0，EXT_IN），`gehtp run --input` 运行时注入；**其余图输入编译期固化**，改了要重编译。烘焙 dtype 陷阱：各输入按消费语义定 dtype——`ids.i32.raw`（i32）、`mask_ones.f16.raw`（f16 槽面）、`pos_arange.i32.raw`（i32，走 f32→f16 会被 denormal 洗成 0）。emit 侧 `GEHTP_DEBUG_INPUTS=1` 逐槽探针核对。

### 5.3 `gehtp run <blob.wtop> --input <in> --output <out> [旗标]`

流程：读 manifest `output_temp` → 推 blob/输入/job.txt（大 blob 设备同名同尺寸即跳过重推）→ 板上 `run_main_on_hexagon 3 gehtp_runner.so` → 拉回输出 + prof 三件套。

| 旗标 | 作用 |
|---|---|
| `--device <serial>` | 设备号（缺省 `ANDROID_SERIAL` 或 52f67807） |
| `--trace` | 逐 op optrace.txt 落盘行（**调试取证用，性能测量勿开**——append 累积、落盘干扰时序） |
| `--hook "idx:temp,..."` | 精度排查：逐命中 op dump 张量（见 §7.4） |

自动拉回到 `<output目录>/prof/`：`<名>.device.txt`（runner 判定日志）、`<名>.op_ts.bin`（逐 op 时间戳）、trace 模式追加 `<名>.optrace.txt`、hook 模式追加 `hook_<idx>_<temp>.f16.raw`。run 前自动清设备侧旧输出/旧 dump，拉到的保证是本次的。

**输出规则**：输出 = manifest `output_temp` 指定的 temp，f16 raw 字节流。对拍纪律：f16 短链判 ≤1 ULP（conv_add 实测逐字节全同）；**长链（81+ op）用值差判据不用位差**（f16 舍入积累在小值区放大位差）。

---

## 6. 编译器三工具（`compiler/tools/`，构建于 `compiler/build_linux/`）

### 6.1 `hnnx_compile` — QNN IR → tagged runlist

```
hnnx_compile --net-json <net.json> [--weights-bin <params.bin>] --output <out.bin> [--format tagged]
```

| 旗标 | 默认 | 说明 |
|---|---|---|
| `--net-json` / `--cpp` | — | QNN IR 两源（二选一；`--cpp` 必配 `--weights-bin`） |
| `--weights-bin` | — | TAR（W.raw/b.raw）或 qairt 2.48 params.bin（自动识别转 TAR） |
| `--output` | `output.bin` | 输出 |
| `--format` | `qnn` | `qnn`=路径A金样重放 \| `tagged`=**我方 runlist（产品路径）** |
| `--ddr-budget <n>` | 0 | DDR 池字节（0=不规划；喂 TAG_MEM_PLAN） |
| `--plan-vtcm-budget <n>` | auto | VTCM 驻留池（auto = 16MB − HMX scratch 峰值 − 余量；0=关） |
| `--vtcm-budget <n>` | 0 | 旧口径 VTCM 预算（0=默认 8MB；小值强制 spill） |
| `--no-pass <名>` / `--enable-pass <名>` | — | pass 开关（可重复；`--enable-pass fusion_*` 开融合） |
| `--static-input <名>=<文件>` | — | 图输入导入期物化为 const（f16/f32 raw；主输入与 .i32.raw 不支持） |
| `--opt-stats` | off | 打印各 pass 匹配/折叠统计 |
| `--ddr-offsets <tsv>` | — | 导出静态 DDR 偏移表 TSV（`check_offsets` 消费） |
| `--graph-name` / `--verbose` | — | 图名 / 详细输出 |

### 6.2 `wtop_emit` — tagged.bin → .wtop blob

```
wtop_emit --bin <tagged.bin> [--input-f16 <f16.raw,...>] --out <blob.wtop>
          [--manifest <json>] [--gguf <g> --match <tsv>]
          [--ext-weights [--weights-bin <params.bin>]]
          [--dma-runlist] [--allow-missing-inputs] [--list-ops]
```

| 旗标 | 说明 |
|---|---|
| `--bin` | 输入 tagged.bin（必填） |
| `--input-f16 a[,b,...]` | 主输入=slot0 EXT_IN；其余图输入烘焙为槽（f32 文件转 f16；`.i32.raw` 保持 int32 原文）；文件数>图输入数=硬错 |
| `--out` / `--manifest` | 输出 blob / manifest |
| `--gguf` + `--match` | GGUF 权重供给（两者须同给；match TSV 列=name,type,file_offset,nbytes,dims） |
| `--ext-weights` [+ `--weights-bin`] | 权重外置：blob 描述符 + `<name>.weights.bin` |
| `--dma-runlist` | 溢出搬移发射真 OP_DMA（而非 SPILL/FILL） |
| `--allow-missing-inputs` | 缺文件的烘焙槽退回零填充（默认硬错，防静默零） |
| `--force-v2` | 已并入 v3（恒 24B slot） |
| `--list-ops` | 列出 `wtop::op_registry()` 已注册算子名并退出 |

逐算子发射器在 `compiler/tools/wtop_ops/`（18 个 `op_*.cpp`，注册制，与设备侧 `g_op_exec_table` 同构）。环境变量：`GEHTP_DEBUG_INPUTS`（逐输入槽分配诊断）、`GEHTP_DEBUG_ENGINE`（逐 op 引擎选择及原因）。

### 6.3 `host_run` — host 参考执行器（对拍链枢纽）

```
host_run --net-json <net.json> [--weights-bin W] [--bin <tagged.bin>]
         [--input-f32 a[,b,...]] --out <out.f32.raw> [--no-pass/--enable-pass <名>]
```

按 **f32 精确语义**跑同一张图，逐 op 落 `host_id_<op_id>.f32.raw`（默认 `/tmp/`，`GEHTP_DUMPDIR` 改目录，`GEHTP_SEQDUMP=1` 追加按执行序编号版）。`--input-f32` 按图输入声明序逗号喂入（元素宽自适应：8B=i64、4B=f32、2B=f16）。

**host_run 输入铁律**：① 多输入顺序 = **图张量 id 升序**，不是命名直觉序（0.8B 喂法 `ids,mask,pos`；喂反 = 全 NaN 无报错）；② 布局 = 图声明布局，资产不一致先转置（不转 = RoPE 全烂的假绿）。

### 6.4 构建与测试

`compiler/CMakeLists.txt`：主库 `htp_core`（STATIC，59 cpp）+ 三工具；测试 45 个 ctest 用例（`test_ir`/`test_e2e`/`test_wtop_emit`/`test_static_input`/`test_vtcm_auto` 等，其中三个跑真工具二进制的集成测试）。`cmake -S compiler -B compiler/build_linux -G Ninja && cmake --build compiler/build_linux`，`cd compiler/build_linux && ctest`。

---

## 7. 性能与取证工具（④⑤）

### 7.1 `gehtp_prof.py` — 整网性能归因报表

```bash
scripts/gehtp_prof.py --wtop model.wtop --op-ts prof/model.op_ts.bin \
    [--manifest ...] [--device-txt prof/model.device.txt] [--optrace prof/model.optrace.txt] \
    [--top 20] [--trace-json prof/model.trace.json] [--csv prof/model.csv] \
    [--peak-flops F] [--peak-bw B] [--strict]
```

- `--op-ts` 优先（有 start）；`--optrace` 回退源。`--trace-json` 拖 Perfetto 看逐 op 时间线。
- 输出五章：总览（wall−Σdur=缝隙）→ **三分账**（算/搬/缝——先看哪桶大再定优化方向）→ 标定行 → top-N（engine 列见 scalar 即回落现场）→ per-opcode 聚合。
- 标定常数（V81HexSim 器件标定）：`PEAK_FLOPS_F16=12.34T`、`DRAM 59.9GB/s`、`VTCM 109GB/s`/线程、小 op 阈值 20µs。
- 纪律：top-20 有幸存者偏差，改引擎门前先逐 op 拆 engine 字段；`op_ts.bin` 只在 rc==0 时写。
- 双重身份：被 `check_op_ts.py`、`cmp_l0.py` import（提供 `parse_wtop`/`parse_op_ts`）。

### 7.2 `check_op_ts.py` — op_ts 交叉校验门

```bash
scripts/check_op_ts.py --wtop <blob.wtop> --op-ts <op_ts.bin> [--manifest ...] [--no-monotonic]
```

对 op_ts.bin × blob 做长度一致、字段域（engine/planned_engine∈{0,1,2}、deviation∈{0,1,2}）、时间单调性、T020 不变量（dev==0 ⟺ planned==actual）四类硬校验，违例 exit 1。

### 7.3 optrace（`gehtp run --trace`）

逐 op 文本轨迹（含 temp pool 偏移列），设备 `/data/local/tmp/hrt/gehtp/optrace.txt`，自动拉回 `prof/`。调试用；性能测量勿开。

### 7.4 hook 张量 dump + 对拍

```bash
# 设备侧：跑时下钩子，落 hook_<idx>_<temp>.f16.raw
scripts/gehtp run model.wtop --input in.f16.raw --output out.f16.raw --hook "320:7,321:2"
# host 侧（host 构建）：同 spec 走环境变量，落 /tmp/hook_*.f16.raw
GEHTP_HOOK="320:7,321:2" ./wt_host_exec <blob> ...

# 对拍：设备 f16 vs host f32（cos + max|d|）
scripts/cmp_hook.py "320:7,321:2" --manifest <m.json> --host-dir /tmp --dev-dir /tmp/prof
scripts/cmp_l0.py "105:3" --wtop <b.wtop> --manifest <m.json> --host-dir /tmp --dev-dir /tmp/prof
```

- op 序号口径 = blob op 表下标 = optrace 行号；temp 号 = op args 的输出 temp 字段。blob 序号 i ↔ 图 op_id = `manifest.op_ids[i]`。
- `cmp_hook.py`：OK（cos≥0.999）/warn（≥0.99）/BAD/SIZE 四级，报首个 BAD 定位发散；锚点须取该节点最后写终态 temp 的 op。
- `cmp_l0.py`：进阶版——锚点 temp 非 op 自身输出时自动沿 blob args 找 last-writer 的 host dump。
- **伪影警示**：hook dump 时机是"该 op 执行完"，但 temp 复用——可疑 dump 先确认没被后续 op 重写。
- **host 构建陷阱**：host 下原生 HVX 入口是空函数，跑 wt_host_exec 必须五门全关：`GEHTP_HVX_ADD=0 GEHTP_HVX_MUL=0 GEHTP_HVX_SILU=0 GEHTP_HVX_SOFTMAX=0 GEHTP_HVX_RMSNORM=0`。

### 7.5 精度二分链（"输出垃圾/精度崩"的标准打法）

```
① HF 黄金   replay_attn_layer3.py / replay_gdn_layer0.py（锚点 npy）
     ↕ cos/max|d|
② host 参考  host_run 逐 op 落 host_id_<op_id>.f32.raw
     ↕ blob序号i ↔ op_ids[i]
③ 设备实测   --hook 拉回 hook_<idx>_<temp>.f16.raw
     ↓
④ 终判      judge_logits.py（门 = 不回归基线）
```

一刀切技巧：怀疑设备/编码问题时先在 host 跑同一 blob——host 复现 = 编译/语义问题；host 正常 = 设备侧问题。

---

## 8. 精度门与判据脚本

### 8.1 `check_device_golden.py` — 通用数值门

```bash
scripts/check_device_golden.py --device out.f16.raw [--device-dtype f16] \
    --gold gold.f32.raw [--gold-dtype f32] [--gold-pattern 'gold_{i}.raw' --prompts 4] \
    --cos-min 0.9986 [--top1-min 29/32] [--rows 0]
```

NaN/Inf 直接判坏；不达标 exit 1（供 CI 串联；`run_host_ci.sh` 冒烟即用它）。**0.8B 共享基线口径 = cos 门 0.998646 不回归**（`judge_logits.py` 的 0.9999 是期望值不是验收门，勿混用）。

### 8.2 `check_3way.py` — f16 三判据逐位判分器

```bash
scripts/check_3way.py ref.f16.raw act.f16.raw [--cos-min C] [--top 4] [--json out.json]
```

三判据（任一满足即 ok）：同号位距≤2 \| \|d\|≤2·ulp_f16(ref) \| \|d\|≤2⁻¹⁴（denormal 噪声地板 6.1e-5）。硬门 fail3==0 且 nans==0；字节数须相等且为偶数。

### 8.3 `judge_logits.py` — 0.8B 双门

```bash
scripts/judge_logits.py <dev_out.f16.raw> [prompt_idx] [--row N] [--device-prefill prefill.f16.raw]
```

- 无 `--row` = prefill 整面门（对 HF golden `test_models/qwen35_08b/cpu_golden/hf/logits_{i}.f32.raw`）；`--row N` = decode 门（P2 KV 后用 31/末位，负数从尾数）；`--device-prefill` = 设备自洽主门（cos≥0.9999）。
- 常量：VOCAB=248320、SEQ=32、COS_BASELINE=0.99864、TOP1_BASELINE=29/32。

### 8.4 `judge_minicpm.py` — minicpm5_2b 门

```bash
scripts/judge_minicpm.py --out dev.raw --gold gold.f16.raw [--shape 1,16,130560]
```

按字节数嗅探 dtype；硬门 cos≥0.9999、top1 16/16；max\|d\| 参考 L3 基准 0.000488（仅记录）。

---

## 9. 模型资产线（导出 / 金标 / 匹配 / replay）

均需 torch 环境（§3.4）。所有导出固定种子 42、ORT 自检门 cos>0.99999。

### 9.1 ONNX 导出

| 工具 | 用法 | 要点 |
|---|---|---|
| `export_qwen35_onnx.py` | `--model-dir D --seq 32 --out m.onnx [--opset 17]` | transformers≥4.57 内建 Qwen3_5；猴补丁 causal mask（禁 `torch.triu`——Trilu 丢 k 属性）；Trilu 降级为 Where 链；`GEHTP_EXPORT_RAW_ONLY=1` 跳过后处理 |
| `export_qwen35_4b_onnx.py` | 同上 | 走 QPM modeling（`QPM_ROOT=/disk2/Qwen35dev/qwen3.5_4b_base/example1`）；强制 torch fallback |
| `export_qwen3vl_4b_onnx.py` | 同上 | Qwen3-VL-4B；单输入 `inputs_embeds [1,seq,2560]` f16；rope cos/sin 图内预烘；同目录产 `in_embeds.f16.raw` + `golden_logits.f32.npy` |
| `export_layer_onnx.py` | `--model-dir D --layer N --seq 32 --out <目录>` | 单层导出 + 层输入/输出 golden raw（全注意力层另出 mask/cos/sin/pos_ids） |

### 9.2 金标生成

```bash
# 双 golden（HF f32 与 GGUF Q4_0 反量化，互拍验证同源）
scripts/golden_qwen35.py --model-dir D --out-dir G [--seq 32] [--n-prompts 4] [--mode both] [--text "..." ]
# 切行模式（无需模型环境）
scripts/golden_qwen35.py --cut-row -1 ... 
# f16 权重域 golden（权重先 f32→f16→f32 往返）
scripts/golden_f16_weights.py --model-dir D --out G
```

产物 `<out>/{hf,q4_0}/input_{i}.i64.raw + logits_{i}.f32.raw`。内含 GGUF 读取/反量化工具函数（被 `gguf_wtlink.py` import）。GGUF 契约：v3 only、Q4_0 块 18B、RMSNorm 权重烤入要减 1、`ssm_a=-A`→`A_log`。

### 9.3 `gguf_wtlink.py` — 权重匹配表

```bash
scripts/gguf_wtlink.py --net-json <net.json> --gguf <model.gguf> --out match_map.json   # 同名 .tsv 一并产出
```

按 qwen35 角色表把 net.json 权重节点映射到 GGUF 张量名，形状交叉验证 + 双向完备性，不完备 exit 1。tsv 供 `gehtp compile --gguf --match` 消费。

### 9.4 replay 黄金参考

```bash
scripts/replay_attn_layer3.py --model-dir D --layer-dir <export_layer 产物> --out <目录>
scripts/replay_gdn_layer0.py  --model-dir D --layer-in layer_0_in.f32.raw --out <目录>
```

按 HF modeling 公式手算单层全锚点（~20/~30 个 npy），与 `host_run` 的 `host_id_*.f32.raw` 逐锚点对拍，裁决"编译器 host 语义 vs HF 真语义"。分别硬编码 layer 3 全注意力 / layer 0 GDN 的头数、head_dim、partial-RoPE 维度等常量。

---

## 10. 审计 / 清单

```bash
# net.json 算子类型清单（兼容 list/dict 两种 schema）
scripts/op_inventory.py --net-json <net.json> [--out op_inventory.json] [--expect expected.txt]
# copy-sem 恒等站点审计（ScatterNd/Pad/Cast 恒等=丢语义，仅死端安全；存在 REVIEW 站点 exit 1 = 入仓门禁）
scripts/audit_copy_sem_sites.py [net.json] [--report docs/A3_COPY_SEM_AUDIT.md]
```

---

## 11. Serving 栈（specs/001，OpenAI 兼容）

```bash
# 起服务（llmserv 环境）
cd /disk1/gehtp
/disk1/miniforge3/envs/llmserv/bin/python scripts/gehtp_server.py --host 0.0.0.0 --port 8000 [--log-level info]
# 测试
/disk1/miniforge3/envs/llmserv/bin/python -m pytest tests/ -q
```

- **端点**：`POST /v1/chat/completions`（SSE 流式+非流式）、`GET /v1/models`、`POST /v1/models/register`（202 异步）、`POST /v1/models/{id}/unload`、`GET /metrics`、`GET /health`。
- **模块分工**：`gehtp_server.py`（FastAPI 主体，并发 4 / 队列 16，超限 429）+ 库群——`model_registry.py`（模型注册状态机）、`persistent_runtime.py`（设备 Unix-socket 常驻客户端，协议=4B 大端长度+JSON）、`device_runtime.py`（runtime 抽象，原型返回占位 token 0）、`sessions.py`（会话/KV 元数据）、`tokenizer.py`（tokenizers 库）、`sampler.py`（greedy/top-p/top-k/temperature）、`metrics.py`、`errors.py`、`logging_config.py`。
- **设备端**：例 `42_gehtp_persistent` 监听 `/data/local/tmp/gehtp/runtime.sock`，最多 4 模型槽位，持久化 wt_blob 免重复解析（需手动启动）。
- 已知限制（原型期）：同模型并发共享一个 device slot；runtime 返回 int=已采样 token、返回 list[float]=logits 由 host 采样。详见 [GEHTP_OPENAI_SERVING.md](GEHTP_OPENAI_SERVING.md)。

---

## 12. 设备侧（kernels/）

### 12.1 `build_libs.sh`

全量编译 `src/{runtime,hmx,hvx,compat}/*.c` → `lib/libhvxhmx_v23.so`（+SWIV 签名版）。文件级例外：`hvx_int8gemm.c` 用 -O1（编译器软件流水线 bug）、`dma_utils.c` 只读、工程单元 -Wextra。`one <base>` 单文件重编模式（注意：`build one` 只重编单 .o——若 setup 检测到 .signed.so 会跳过重建，陈旧库会引发 opcode arity 类设备故障，改完源码请手动全量或删签名件）。

### 12.2 `build_examples.sh` — 33 例单元测试

```bash
cd kernels/examples && ./build_examples.sh            # 全部（33 例 181 门）
./build_examples.sh 02                                # 只跑 02_convf16_gemm
./build_examples.sh 32_rbr                            # 按全名
```

三代谱系：01-15 算子回归（V2.1）、16-22 工程单元 U1-U8（wtcache/W4A16/oplist/dualdom…）、23-33 工程单元 U9-U19（fence/arena/kvcache/gemmdispatch/rbr…）；34 号以后为后续新增——GEHTP 线的关键例：`37_conv2d_add`（conv_add 端到端）、`40/41_qwen_layer*`（层资产）、`42_gehtp_runner`（门户通用 runner）、`42_gehtp_persistent`（常驻 runtime）、`43_gdn_sm_08b`、`44_us1_split`、`46_hmx_bias`。叙述式导览见 `examples/EXAMPLES_cn.md`，判据速查 `TEST_CASES.md`，API 手册 `kernels/docs/api_*.md`（37 份）。

### 12.3 例 42 `42_gehtp_runner` — 通用 WTOP runner（门户执行端）

读 `job.txt`（定值路径）执行 `wt_parse → wt_exec_run_io → wt_exec_temp_last_bytes` 定长写出。

**job.txt 键**：`blob <路径>` / `input <路径>` / `output <路径>` / `out_temp <N>` / `weights_size <N>`（外置权重时）/ `trace 1`（逐 op 轨迹）/ `hook <idx:temp,...>`（张量 dump）。

**产物路径**（设备 `/data/local/tmp/hrt/gehtp/`）：`output.f16.raw`（结果）、`42_gehtp_runner.txt`（判定日志）、`op_ts.bin`（"WTS1" 24B 头 + 逐 op 记录，rc==0 才写）、`optrace.txt`、`hook_<idx>_<temp>.f16.raw`。

### 12.4 执行器内部（oplist_parse / oplist_exec）

- 分发 = 函数指针表 `g_op_exec_table[opcode]`；加 op = 表加一项 + 一个 `exec_*`，不动主循环。
- **引擎门 = shape 对齐谓词**（核心 `%32==0`）：matmul f16 → HMX 当 M/K/N 均 %32==0 否则标量；conv2d 同理；add → HVX 当 n≥64（`GEHTP_HVX_ADD`）；rmsnorm HVX 门（`GEHTP_HVX_RMSNORM`）；W4A16 恒 HMX。
- VTCM 双 acquire 互斥（`hmx_runtime_setup` vs `wtcache_open`），003 线正把它改为编译期 carve 定界。
- blob ver=3 契约：引擎注解 footer（engine/fb_allowed/fb_engine）+ 可选 carve footer；carve 重叠/超预算/双源不一致 = emit 期硬错误；运行期计划引擎不可用 = die 绝不静默换单元。
- **算子输入契约单点注册表（011）**：每 opcode 的 arity + 每 arg 的 `name`/`kind`/`slot_check` 唯一事实源 = `kernels/include/op_spec.inc` 的 `wt_op_spec[]`（`oplist_parse.h` include 之）；`arity_of()`/`arg_is_slot()` 改读该表，旧 `#define WT_ARITY_*` 宏与两张 switch 已删除。`compiler/tests/test_wtop_emit.cpp` 的 `test_op_spec_registry` 永久守卫（新增 opcode 漏填 name/kind 构建期即暴露）。迁移为 byte-exact：转录经 38 opcode×16 arg 逐值 parity 对拍，设备例 21 host↔device W3 逐行一致。

### 12.5 `kernels/host/` host 辅助工具

| 工具 | 用途 |
|---|---|
| `wt_inspect.c` | blob 检视 |
| `wt_dump_ops(.c)` | dump blob op 表 |
| `check_offsets.c` | 校验 `hnnx_compile --ddr-offsets` 导出的静态偏移 TSV |
| `pack_oplist.cc` / `pack_matmul_split.cc` / `pack_hmx_bias.py` / `weight_layout_pack.py` / `inv_crouton16.py` | op 列表/权重布局打包工具 |
| `analyze_dd.py` / `gen_smallm_assets.py` | DMA 分析 / smallm 资产生成 |
| `gtest_v23.c` / `test_gemv_f16.c` / `test_kv_ssm_ops.c` / `c1b_gdn_08b_host.c` | host 侧测试/参考实现 |
| `wt_host_exec`（host 构建线，FORENSICS §3 引用） | 同一 blob host 整链复现（一刀切设备/语义二分；五门全关铁律见 §7.4） |

### 12.6 010 静态内存规划契约（pin_cap / 权重 pin / carve 复用）

specs/010 把「内存决策编译期定死、执行器退化为查表」的最后一公里收口，新增三条 blob/执行器契约（编译器 `compiler/` + 执行器 `kernels/` 双向）：

**① pin_cap 编译期计算与下发（US4，T003-T007）** — 替换执行器写死的 4096。

- `MemPlan`（`graph_prepare.hpp`）新增 `pin_cap: u64`（2048 对齐字节；`0` = 不 pin → 执行器回退 4096 旧口径）。
- `vtcm_auto_scan` 真算 `pin_cap = align2048(weight_sum)`（非 W4A16 权重 `const_data_size` 求和，共享权重重复计 = 保守上界）；当 `pin_cap + scratch_max + conv_head + 512KB ≥ 16MB` 时回退 `0`（权重塞不进，流式）。
- 序列化格式两处加宽：
  - `TAG_MEM_PLAN` 头部 5→6 u32：`[ver=1][ddr_cap][vtcm_cap][bump_reserve][pin_cap][n_entries]`（entries 每条 20B，偏移 +20 起）。
  - **TEMPOFF 槽头 3→4 u32**：`[cap][reserve][pin_cap][n]`（entries 移到 `+16`，每条 12B `{temp_id,offset,size}`）。旧 3-u32（12B）头被 16B 头取代。
- 执行器 `wt_exec_load_tempoff` 读 pin_cap（`+8`）、n（`+12`）；`wtcache_pin_arg() = pin_cap ? pin_cap : 4096` 替换两处 `wtcache_open(wc, 4096)`。**注意**：旧版 blob（无 TEMPOFF 或 12B 头）会被新内核拒 `tempoff slot short`——重编须用新编译器再生 blob。

**② 权重 pin/restage 时序静态化（US1，T008-T013）** — 消除每 MATMUL 无条件 `cpu_to_vtcm` + PIN/MATMUL 双搬（D2）。

- 编译器 `op_matmul.cpp`：W4A16 权重槽首次消费前发一条 `OP_PIN[w_raw]`（`em.w4_pinned` 去重集）→ planned restage = OP_PIN 条数 = 去重权重槽数。
- `wtop_emit.cpp`：`arg_takes_bare_slot` 登记 OP_PIN 裸槽，防 `pp_recs` GC 剥权重槽。
- 执行器：`xop_pin` 真搬一次并登记 `wt_resident_slot`/`bias_resident_slot`；`exec_matmul` 驻留 slot 命中即跳过 `cpu_to_vtcm`（字节恒等：同源同址跳过拷贝结果不变）；re-carve（形状/live 变化或 f16/conv HMX 抢 VTCM）置空驻留标记强制重搬。
- `wt_exec_stats.restage`（`oplist_exec.h`）记录实际 DDR→VTCM 换入次数，`wt_w3_report` 可与编译期规划值对账（SC-003）。

**③ op_annot bit5 = carve 复用标志（US5，T002）** — 编译期预判逐 op carve 是否可复用上一 carve。

- `oplist_parse.h` 新增 `wt_annot_set_reuse(u32*,int)` / `wt_annot_reuse(u32)`（对称 `wt_annot_pack`）。
- v3 引擎注解 footer 位布局：bit0-1 engine / bit2 fallback_allowed / bit3-4 fallback_engine / **bit5 carve_reuse** / bit6-31 保留。
- emit 侧 `wt_annot_pack` 后置位；exec 侧读 bit5 决定跳过 re-carve。

**未收口（重 scope 留待后续）**：US2（W4A16 aux 三槽标注，供给侧编译器从不产 bias/atbl/otbl）、US3 剩余（conv2d f16 + Q4_0 运行时 crouton pack；f16 matmul 已由 008 T012 `kFlagWgtPP` 收口）。0.8B E2E 不走 W4A16/conv2d 路径，故 010 的 E2E 相关价值已被 007/008 吸收（见 specs/010 范围发现）。

---

## 13. 环境变量速查

### host（compiler / wtop_emit / host_run）

| 变量 | 作用 |
|---|---|
| `GEHTP_DUMPDIR` | host_run 每 op dump 目录（默认 /tmp/） |
| `GEHTP_SEQDUMP` | 追加按执行序编号的 dump |
| `GEHTP_OPTRACE` | execute_host 每 op 打印轨迹行 |
| `GEHTP_STOP_SEQ=N` | 执行到 seq==N 早停（NaN 溯源） |
| `GEHTP_EXDIAG` | 执行诊断打印 |
| `GEHTP_NAMEMAP` | 输出 grouping↔op_id 映射文件 |
| `GEHTP_DEBUG_INPUTS` | wtop_emit 输入槽分配诊断 |
| `GEHTP_DEBUG_ENGINE` | 逐 op 引擎选择及原因 |
| `GEHTP_DUMP_VIEWS` | pass 前后 view-census |
| `HNNX_VTCM_ALLOCATOR` | 分配器选择（fancy / cp / cp-paging / cp-remat…） |
| `HNNX_CP_DEBUG` / `HNNX_CP_TRACE` / `HNNX_CP_NODE_LIMIT` / `HNNX_CP_TIME_LIMIT_MS` / `HNNX_CP_COST_MODEL` | CP solver 调试/限制/代价模型 |
| `GEHTP_HOST_PASSTHRU` | host 无真语义 op 处置：warn（默认）/off/error |
| `GEHTP_HOST_{PAD,SCATTER,CAST}_IDENTITY` / `GEHTP_HOST_COPY_SEM_IDENTITY` | host 恒等仿真开关（数值判决用） |
| `GEHTP_EXPORT_RAW_ONLY` | 导出脚本跳过后处理与 ORT 自检 |

### 设备 / host 构建运行时

| 变量 | 作用 |
|---|---|
| `GEHTP_HOOK` | hook spec（host 构建；设备侧走 job.txt `hook` 键） |
| `GEHTP_HVX_ADD` / `GEHTP_HVX_RMSNORM` | HVX 引擎门开关（host 构建必须置 0，连同 MUL/SILU/SOFTMAX 五门） |
| `ANDROID_SERIAL` | `gehtp` 缺省设备号 |

---

## 14. 纪律与排错

**设备纪律**：跑前查 `run_main_on_hexagon` 占用；adb 列表突然空 = host adb server 抽风，`adb kill-server` 即愈；CMA 耗尽 / `failed to get map da` → 停下通知用户勿自重启；`0x39 unsigned PD 拒绝` = CDSP 未启动（板方恢复）；runner .so 必须 chmod 755。

**判据纪律**：f16 短链 ≤1 ULP；长链用值差；0.8B 精度门 = 不回归 cos 0.998646；blob 序列化非确定性 → 禁字节对拍 blob，用 opt-stats/host_run 数值/golden 判。

**二进制产物纪律**：禁手搓解析 blob/op_ts——用 `gehtp_prof.py`（或其 `parse_wtop`/`parse_op_ts` 库函数）、`wt_inspect`、`wt_dump_ops`。

**常见症状速查**（详表见 [PORTAL §5](GEHTP_PORTAL.md)）：

| 症状 | 首查 |
|---|---|
| 设备输出全 -inf / 全同值 | 烘焙输入：其余图输入未给 `--input-f16` 固化全零 → mask 全 -inf；或槽 dtype 喂错 |
| cos 崩 / 部分 op 全 nan | 标量参数静默默认 0（axis 类）；三输入 op 广播未物化 |
| `no output on device` | runner 日志 `[FAIL]` 行 |
| 性能悬崖 | prof top-N 的 engine 列见 scalar = 回落现场 |

---

## 15. 文档与规格地图

| 文档 | 内容 |
|---|---|
| [PORTAL](GEHTP_PORTAL.md) | 正门 CLI 速查、输入输出规则、排错表、已验证矩阵 |
| [ARCHITECTURE](GEHTP_ARCHITECTURE.md) | 五大件内部原理 + blob 布局实解 + file:line 索引 |
| [FORENSICS](GEHTP_FORENSICS.md) | prof/hook/对拍三条取证链操作手册 |
| [DEVICE_RUNBOOK](GEHTP_DEVICE_RUNBOOK.md) | 上板交接：签名/skel/权限全坑 |
| [OPENAI_SERVING](GEHTP_OPENAI_SERVING.md) | serving 边界情况与已知限制 |
| [ROADMAP](GEHTP_ROADMAP.md) | 三条主线工作计划（v1.1） |
| [PROVENANCE](PROVENANCE.md) | 代码谱系与拷贝细节 |
| MEMPLAN_* / BACKPORT / CONVERGE / PARALLEL_TASKBOOK / A3_COPY_SEM_AUDIT | 内存规划迁移/收编/并行任务簿/审计报告 |
| `specs/001-010` | 001-serving ✓、002-vtcm-arena（US2 退役入 003）、003-编译器驱动调度、004-decode、005-图恒等 pass ✓、006-激活融合（NO-GO 归档）、007-broadcast HVX ✓、008-vtcm 驻留 DMA、009-GDN 块融合、010-静态内存规划收口（pin_cap/权重 pin/carve 复用，见 §12.6） |
| `kernels/examples/{EXAMPLES_cn,EXAMPLES,TEST_CASES}.md` | 33 例导览与判据（34+ 后增例见 §12.2） |
| `kernels/docs/api_*.md`（37 份） | runtime/HMX/HVX/v22/v23 单元 API 手册 |

---

*活文档：工具增删/旗标变更请同步本文对应小节。*
