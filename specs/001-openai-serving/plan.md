# Implementation Plan: OpenAI 兼容推理服务接口

**Branch**: `task/prof-wp` | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-openai-serving/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

为 GEHTP 增加 OpenAI 兼容的推理服务接口，支持多模型注册、流式 SSE 响应、host 侧 KV-cache 管理和采样。目标是在 V81 设备上高效地比较不同编译模型的推理效果与性能，适用于内部测试而非生产部署。

## Technical Context

**Language/Version**: Python 3.10（与现有 `scripts/venv_qpm210` 环境一致）

**Primary Dependencies**:
- FastAPI + uvicorn（HTTP 服务）
- tokenizers（HF 兼容）
- 现有 GEHTP toolchain（`hnnx_compile`, `wtop_emit`, `gothp run`）

**Storage**: 无持久化数据库，内存中管理 model/session/request/KVCacheHandle；日志输出到文件

**Testing**: pytest（API 单元测试、对拍测试）

**Target Platform**: Linux server（V81 板通过 adb 连接）

**Performance Goals**:
- 首 token 延迟 ≤ 500ms（warm device runtime）
- 单次推理（prompt ≤ 128 tokens, 生成 ≤ 64 tokens）≤ 2s
- 并发 ≥ 4（请求队列 + 串行调度）

**Constraints**:
- KV-cache 元数据在 host，实际数据在 device
- 单设备，独占资源
- 最多 5-10 个模型，内部测试场景

**Scale/Scope**: 5-10 个模型，并发 ≤ 4，运行时长 ≥ 8 小时无需重启

## Constitution Check

*GATE: 必须通过才能进行 Phase 0 research。Phase 1 design 后重新检查。*

本项目所在目录 `.specify/memory/constitution.md` 目前为空模板占位，未发现实际治理约束。因此跳过宪法检查，无违规项。

若未来引入宪法规则（如「Library-First」、「Test-First」等），则需在此处填写：

| Gate | Result | Justification |
|------|--------|---------------|
| Library-First | N/A | 宪法未定义此原则 |
| Test-First | N/A | 宪法未定义此原则 |

## Project Structure

### Documentation (this feature)

```text
specs/001-openai-serving/
├── spec.md                # Feature specification
├── plan.md                # This file (/speckit-plan command output)
├── research.md            # Phase 0 research
├── data-model.md          # Phase 1 data entities
├── quickstart.md          # Phase 1 validation guide
├── contracts/             # Phase 1 external contracts
│   └── openai-api.md
└── tasks.md               # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
scripts/
├── gehtp_server.py        # FastAPI server and API handlers
├── device_runtime.py      # Device session management
├── tokenizer.py           # Tokenizer and chat template
├── sampler.py             # Sampling logic
├── sessions.py            # Session and KV-cache metadata
└── metrics.py             # Metrics collection and endpoint

tests/
├── test_api.py
├── test_device_runtime.py
a├── test_tokenizer.py
├── test_sampler.py
└── conftest.py
```

**Structure Decision**: 服务端和测试代码放在现有 `scripts/` 与 `tests/` 顶层目录，复用 GEHTP 当前脚本入口和 Python 测试布局；功能设计文档集中在 `specs/001-openai-serving/`。

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
## Project Structure

### Source Code

```text
scripts/
└── gehtp_server.py        # FastAPI server, API handlers
├── device_runtime.py      # Device session management (init/run/release)
├── tokenizer.py           # Tokenizer loader, chat template
├── sampler.py             # Sampling logic (greedy/top-p/temperature)
├── sessions.py            # Session/KVCacheHandle manager
└── metrics.py             # Metrics collection and /metrics endpoint

tests/
├── test_api.py            # API contract tests
├── test_device_runtime.py # Device runtime tests
├── test_tokenizer.py      # Tokenizer tests
├── test_sampler.py        # Sampling tests
└── conftest.py            # Fixtures

specs/001-openai-serving/
├── spec.md                # Feature spec
├── plan.md                # This file
├── research.md            # Research findings
├── data-model.md          # Data entities
├── quickstart.md          # Validation guide
└── contracts/
    └── openai-api.md      # OpenAI API contract
```

## Phase Breakdown

### Phase 0: Research (已完成)

- [x] KV-cache 持久化机制调研（research.md R1）
- [x] 常驻 device runtime 路径选择（research.md R2）
- [x] 多模型并发策略（research.md R3）
- [x] Host 侧 tokenizer 方案（research.md R4）
- [x] 指标与日志方案（research.md R5）
- [x] SSE 流式响应实现（research.md R6）

**Deliverable**: research.md

---

### Phase 1: Design

- [x] 数据实体定义（data-model.md）
- [x] API 合同（contracts/openai-api.md）
- [x] 快速验证指南（quickstart.md）
- [ ] Extension Hooks 处理（需手动确认）

**Deliverable**: data-model.md, contracts/, quickstart.md, plan.md（本文档）

---

### Phase 2: Implementation (下一步)

使用 `/speckit-tasks` 生成可执行的任务列表，包括：

1. 实现 `device_runtime` 模块（原型期 adb/file 包装）
2. 实现 FastAPI server 框架（路由、中间件、错误处理）
3. 实现 tokenizer 和采样逻辑
4. 实现请求队列和并发控制
5. 实现 metrics 收集
6. 编写 API 测试和集成测试
7. 验证端到端流程

**Next step**: 运行 `/speckit-tasks` 生成 tasks.md

## Next Steps

1. ✅ Research 完成（本文档包含研究结论）
2. ✅ Design 完成（data-model.md、contracts/、quickstart.md 已生成）
3. ➡️ 执行 `/speckit-tasks` 生成任务列表并开始实现
