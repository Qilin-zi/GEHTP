# Contract: 性能报告三分类口径门禁（perf-report-lint）

> 落点：`scripts/perf_report_lint.py`（新文件）。2026-09-17。

## CLI

```
scripts/perf_report_lint.py check <report.md|txt>   # 缺项 → 退出 1 + 逐条列出
scripts/perf_report_lint.py init  > report_template.md  # 生成合规模板
```

## 规则（每个性能数字块）

1. 必须含 `category:` ∈ {真算, 装料, 卸料}
2. 必须含 `metric:` ∈ {num_dominant_path, cycles_used, graph-wall, retire-interval}
3. 必须含 `shape:`（非空）与 `scenario:` ∈ {single, sequence, graph}
4. 必须含 `sample:` ∈ {reps2-N median, ACAC paired}，或显式 `single-rep (structure-only)`
5. `single-rep` 块禁止出现 utilization 百分比（结构专用口径）
6. `cycles_used/N` 形式的吞吐表述 → 直接拒绝（必须用 retire-interval 或 graph-wall/N）

## 适用范围

- 拦：本特性合入后**新增/修改**的性能报告（git diff 触及的报告文件）
- 不追溯：`kernels/PERF_REPORT.md`、`kernels/results/*.txt` 等存量
- 词汇定义引用 `docs/GEHTP_SCHED_HWFACTS.md` §4，脚本内不复制语义定义

## 验收

- 合规模板 `init` 输出 → `check` 通过
- 缺 category 的报告 → 退出 1 且点名缺项
- 含 `cycles_used/N` 吞吐表述的报告 → 退出 1
