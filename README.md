# GEHTP — 通用 HTP 图编译 + V81 设备执行工具链

目标:**通用的 compiler + runtime**——ONNX/QNN IR 经图编译产出我方 .bin(含 runlist),
V81 板(52f67807)自研引擎按 runlist 派发 kernel 执行,不用官方 qnn-net-run。
支持任意模型、新模型零适配、不硬编码。

## 路线

```
ONNX → qairt-converter → net.json (QNN IR)
  → compiler/ (B线图编译器): qnn_ir_loader → OpDef IR → do_prepare1
      → ST-Cut 调度 → 单算子 tiling → FancyAllocator/cp_solver (spill/fill)
      → 序列化 (tagged-record runlist) → 我方 .bin
  → [host] wtop_emit (读 TAG_MEM_PLAN 照抄, 无自算) → WTOP blob
  → V81 板 oplist_exec 引擎执行 (kernels/: hmx/hvx kernel 库)
```

## 目录

| 目录 | 内容 | 来源 |
|---|---|---|
| `compiler/` | B线图编译器(REQNN,字节级重实现 libHtpPrepare 前端+IR+调度+序列化) | REQNNFRAME/REQNN |
| `kernels/` | V81 设备 kernel 库 + WTOP 执行引擎(oplist_parse/exec)+ 36 例 | V81Dev/hvxhmx_libsV2.3 |
| `scripts/` | 上板脚本模板(阶段 9/10 改写为 GEHTP 原生版) | V81Dev/vtcm_engine_probe |
| `test_models/` | 测试模型(conv_add ONNX/net.json/输入/golden) | 新建 |
| `docs/` | PROVENANCE.md 等 | 新建 |

来源与拷贝细节见 [docs/PROVENANCE.md](docs/PROVENANCE.md)。

## 构建

- 编译器(host,Linux):`cmake -S compiler -B compiler/build_linux -G Ninja && cmake --build compiler/build_linux`
- 内核库(设备,V81):`cd kernels && ./build_libs.sh`(需 Hexagon SDK 6.6.0.0 + SWIV)
- 设备上板:仅 52f67807;见 `kernels/TUTORIAL.md` 与 `scripts/device_run.sh`

## 里程碑

M0-M3c ✓(probe 全链上板 cos=1.0 top1 256/256,75a740f)→ M4 0.8B prefill → M5 decode
→ M6 4B → M7 性能收敛 + 第 7 步编译期内存规划(阶段一静态 DDR 偏移 = M4/M5,
阶段二 VTCM 驻留 + DMA 算子 runlist = M6 尾/M7)。详见实施计划(plan 文档)。
