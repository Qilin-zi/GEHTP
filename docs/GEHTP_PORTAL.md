# GEHTP 正门 — gehtp CLI 用法 (2026-09-15)

两条命令的模型上板旅程：**`gehtp compile` 编译，`gehtp run` 上板跑**。
面向任意被编译器覆盖的 ONNX 模型，运行时零 malloc，设备仅 52f67807。

```bash
scripts/gehtp compile model.onnx -o model.wtop        # host: ONNX → 设备 blob
scripts/gehtp run model.wtop --input in.f16.raw --output out.f16.raw
```

快速上手（可跑 sample，含设备互斥检查 + golden 对拍）：

```bash
scripts/gehtp_quickstart.sh          # conv_add 全链, 期望 === SAMPLE ALL GREEN ===
```

---

## 1. 三个子命令

### `gehtp setup [--device 52f67807]` — 一次性设备准备

构建/检查内核库 → 交叉编译通用 runner（例 42,`vgather=0` 铁律）→ SWIV 签名 →
推送 `libhvxhmx_v23.so` + `gehtp_runner.so` 到 `/data/local/tmp/hvxhmx23`。
库源码变更后需重跑（脚本检测到签名库缺失才重建；源码改了请手动 `./build_libs.sh`）。

### `gehtp compile <model.onnx> -o <out.wtop> [旗标]`

内部五步：qairt-converter → dlc_repair → qairt-dlc-to-json → hnnx_compile(tagged runlist)→ wtop_emit(blob+manifest)。

| 旗标 | 作用 |
|---|---|
| `--input-f16 a[,b,c...]` | 输入数据：首文件 = 主输入烘焙兜底数据；**后续文件 = 其余图输入固化值**（不给则全零） |
| `--net-json X --weights-bin Y` | **直通模式**：复用已有 QNN IR 产物，跳过 qairt-converter（大模型必备；symlink 引用不拷贝） |
| `--gguf G --match T` | GGUF 外部权重直读 + Q4_0 tile-major repack（全模型 blob 从 f32 池 2.55GB 降到 ~1.67GB；未命中走 f16 兜底，数值正确） |
| `--ddr-budget N` | DDR temp 池预算（字节）。超预算张量 → runlist 自动插 SPILL/FILL 算子（编译期内存规划，第7步阶段二） |
| `--vtcm-budget N` | VTCM 驻留池预算（双池分配，b5df521 已设备验证） |

全模型示例（0.8B,16297 op → 1.67GB blob）：

```bash
gehtp compile --net-json test_models/qwen35_08b/conv/model_net.json \
    --weights-bin test_models/qwen35_08b/conv/model.params.bin \
    --gguf /disk2/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf --match /disk2/match_map.tsv \
    -o test_assets/qwen35_08b/qwen35_08b.wtop
```

产物三件套：
- `out.wtop` — 设备 blob（权重 + runlist + 槽位 + TEMPOFF 静态偏移表，单文件）
- `out.wtop.manifest.json` — input_slot / input_elems / **output_temp** / n_slots / n_ops
- `out.wtop.work/` — net.json、tagged.bin 等中间产物（排查用）

### `gehtp run <blob.wtop> --input <in.f16.raw> --output <out.f16.raw> [--device ...]`

读 manifest 拿 `output_temp` → push blob/输入/job.txt → 板上通用 runner
（`wt_parse → wt_exec_run_io → wt_exec_temp_last_bytes` 定长写出）→ 拉回输出。
**run 前自动删设备侧旧输出**（失败跑不会误拉上一轮结果）。

---

## 2. 输入规则（重要）

| 图输入 | 处理 | 运行时能变？ |
|---|---|---|
| **主输入** = net.json 输入列表**首个**（slot0，标 EXT_IN） | `gehtp run --input` 注入 | **能** |
| 其余图输入 | 编译期 `--input-f16` 后续文件固化进 blob；不给则全零 | 不能（改了要重编译） |

两个输入的模型（如 add+conv，x0 变、x1 不变）：

```bash
gehtp compile m.onnx -o m.wtop --input-f16 x0.f16.raw,x1.f16.raw
gehtp run m.wtop --input x0_new.f16.raw --output out.f16.raw   # x1 用编译时那份
```

两个输入都要运行时变 = 当前**不支持**（引擎单注入指针），扩展方案已设计（引擎指针数组化 + emit 多 EXT_IN + CLI 逗号输入），待排期。

## 2.1 host_run 输入铁律（全模型 host 参考链；2026-09-17 C1 线双实锤）

- **多输入顺序 = 图张量 id 升序，不是命名直觉序**。0.8B 三输入张量 id：input_ids=1、
  attention_mask=2、position_ids=3 → 喂法 `ids,mask,pos`；按直觉 `ids,pos,mask`
  喂反 = pad 掩码列 0 -inf → 注意力 row 0 全 NaN → logits 全 NaN（无报错）。
- **布局 = 图声明布局**（ncf 族）。资产文件与图声明不一致时须先转置：
  L3 的 hidden（声明 [1,1024,32]）与 cos/sin（声明 [1,64,32]，文件 [1,32,64]）
  都要转置喂入；不转 = RoPE 全烂（cos 0.97 级假绿，无报错）。
- f16 权重域与 fp32 golden 的差异已实测：bf16 源权重 f16 往返恒等（scripts/golden_f16_weights.py），
  host 链 vs fp32 golden 的 cos 漂移来自 f32 归约序积累（0.9987 级均匀漂移，top1 保）。

## 3. 输出规则

- 输出 = manifest `output_temp` 指定的 temp，f16 raw 字节流（元素数 × 2 字节，由引擎最后写入大小定长）
- 多输出图：只取 manifest 的主输出（扩展待排期）
- 对拍纪律：f16 判 **≤1 ULP**（短链 conv_add 实测逐字节全同）；长链（81+ op）用**值差**判据，不要用位差（f16 舍入积累会在小值区放大位差，M4.2 实锤）

## 4. 设备纪律与互斥（三窗口约定）

1. **仅 52f67807**；跑前先查占用：`adb -s 52f67807 shell "ps | grep run_main_on_hexagon"`，有主就等（quickstart 已内置此检查）
2. `adb devices` 列表突然空 = host adb server 抽风，`adb kill-server` 即愈，**板子没坏**，别重启板
3. adb 掉线会孤儿设备侧 `run_main_on_hexagon` + DSP main（卡 fastrpc 无进展）；排查设备异常先 ps 看有没有上个窗口的孤儿
4. CMA 耗尽 / `failed to get map da` → 停下通知用户，不自行 reboot
5. **0x39 全域 unsigned PD 拒绝**（"FastRPC Capability API failed ... Unsigned PD is not
   supported on domain 3"）= CDSP (gunyah VM) 未启动（remoteproc 未注册），板方恢复，
   勿自重启（2026-09-17 双会话独立实锤；与 SELinux/skel 无关——两版 skel
   380f3cb 正典/d0bfbc third_party 同败）。预检：`ls /sys/class/remoteproc/` 非空。
6. `build_examples.sh` 会推 `third_party/run_main_on_hexagon/ship/` 装载器+skel 覆盖设备
   正典件（md5 与全 V81Dev 谱系不同）；设备套件异常时先核 md5。

## 5. 排错速查

| 症状 | 位置/原因 |
|---|---|
| runner 判定日志 | 设备 `/data/local/tmp/hvxhmx23/42_gehtp_runner.txt` |
| 逐 op 轨迹 | 设备 `/data/local/tmp/hvxhmx23/optrace.txt` |
| `no output on device` | 看 runner 日志 `[FAIL]` 行（如 `transpose_gen rank 5` = blob 含 rank>4 转置，用当前 build 重新 compile，16af35d 已修） |
| 拉到陈旧输出 | 不可能（run 前自动 rm）；若见即 bug |
| 编译器覆盖 | 20 型 opcode 家谱（0.8B 全模型）；新 op 报 `opcode N unhandled` 是编译器缺口不是门户问题 |

## 6. 已验证矩阵（52f67807 实测）

测试资产仓内暂存：[test_assets/](../test_assets/README.md)(/tmp 一日两清，资产一律入此；设备 g40/ 为兜底源）。

| 模型 | 规模 | 结果 |
|---|---|---|
| conv_add × 3 组输入轮换 | 5 op / 3 槽 | **byte-exact 32768/32768 × 3** |
| conv_add spill 变体（`--ddr-budget 4096`） | 13 op 含 6×SPILL/FILL | **byte-exact 32768/32768** |
| L3 注意力层（[test_assets/l3/](../test_assets/)） | 81 op / 45MB blob | **max_valdiff=0.000488 = M4.2 golden 判据** |
| L0 GDN 层 | 903 op / 119MB blob | 前 91 op 通过；rank 三修复（16af35d + c5d4c2a）已落，新编码资产由例41线收口后复跑 |

已知边界：单运行时主输入、单输出、f16 only、编译器 20 型 opcode 覆盖。
