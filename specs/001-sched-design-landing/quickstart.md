# Quickstart: 调度设计原则落地 — 验证指南

> 2026-09-17。全部为 host 先行命令；设备项在 52f67807 恢复后补跑（挂起不阻塞）。
> 实施细节属于 tasks.md/实现期，本文只定义"怎么证明它对了"。

## 前置

```bash
cd /disk1/GEHTP
cmake -S compiler -B compiler/build_linux -G Ninja && cmake --build compiler/build_linux
```

## S1 访问分类谓词（US1/FR-001/FR-002）

```bash
# 新增单测 + 存量回归
cd compiler/build_linux && ctest --output-on-failure          # 期望 44/44 + 新增分类测试全绿
# 合成图验证（tests 内新增用例产出审计日志）：
#   MatMul 激活 → verdict=vtcm-ok；Gather 索引 scratch → verdict=ddr-scalar
# 行为保持门：
bash scripts/gehtp compile <conv_add.onnx> -o /tmp/cc.wtop   # 默认预算 → blob 与基线逐字节一致
```

## S2 去重门禁（US2/FR-003）

```bash
cd compiler/build_linux && ctest -R dedup_guard --output-on-failure
# 用例：两全同可写 scratch → 规划条目独立；两全同只读权重 → 去重不变
```

## S3 单元指派审计（US3/FR-004）

```bash
scripts/gehtp compile <含大方阵+窄输出算子的.onnx> -o /tmp/x.wtop --audit 2>audit.log
# 期望：每算子一行 unit/rule_match；窄输出→HMX 的算子带 narrow-on-hmx 警告
# 派发行为与基线一致（blob 逐字节不变）
```

## S4 dominant-path 下界（US4/FR-005/FR-006）

```bash
scripts/gehtp compile <conv_add.onnx> -o /tmp/cc.wtop
python3 -c "import json; d=json.load(open('/tmp/cc.wtop.manifest.json'))['dompath_estimate']; \
    assert d['source']=='table' and d['cycles']>0"
# 旧 blob 兼容：取一份本特性前的 .wtop.manifest.json，gehtp 消费链不报错
# 不变式（设备恢复后复核）：dompath_estimate.cycles ≤ 该图实测 wall
```

## S5 报告门禁（US5/FR-007）

```bash
scripts/perf_report_lint.py init > /tmp/rpt.md && scripts/perf_report_lint.py check /tmp/rpt.md
# 期望：通过。构造缺 category 或 cycles_used/N 表述的报告 → 退出 1 且点名缺失/违规项
```

## 全量行为保持门（每个阶段合入前必跑）

```bash
cd compiler/build_linux && ctest --output-on-failure     # 44/44 + 新增
bash scripts/gehtp_quickstart.sh                          # 设备恢复后：=== SAMPLE ALL GREEN ===
# conv_add ×3 byte-exact；spill 变体 byte-exact；L3 81-op |d|≤0.001
```

## 设备挂起项（52f67807 恢复后补）

1. `bash scripts/gehtp_quickstart.sh` → `=== SAMPLE ALL GREEN ===`
2. `cd kernels && ./examples/build_examples.sh 43` → 43_hwinfo 四门 PASS，
   数字回填 `docs/GEHTP_SCHED_HWFACTS.md` §1 表格 ⏳ 项
3. conv_add 的 manifest `dompath_estimate.cycles` ≤ 实测 wall 复核
