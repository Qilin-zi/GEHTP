---
description: "Task list for OpenAI 兼容推理服务接口"
---

# Tasks: OpenAI 兼容推理服务接口

**Input**: Design documents from `/specs/001-openai-serving/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks included — spec.md user stories define coverage requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `scripts/`, `tests/` at repository root
- Source code: `scripts/` for service modules, `tests/` for test files
- Spec artifacts: `specs/001-openai-serving/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create `scripts/gehtp_server.py` with FastAPI app skeleton, main entry point, middleware
- [X] T002 Initialize Python environment with `fastapi uvicorn tokenizers pytest` dependencies in `scripts/requirements.txt`
- [X] T003 [P] Configure linting and formatting tools (black, isort, ruff) in `pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create `scripts/device_runtime.py` — prototype adb/file runner wrapper exposing `init(model_id, kv_max)`, `run(input_ids, position, kv_position)`, `cancel()`, `release()` methods
- [X] T005 Create `scripts/tokenizer.py` — tokenizer loader from `tokenizer.json`, chat template application, encode/decode methods
- [X] T006 Create `scripts/sampler.py` — greedy / top-p / temperature sampling logic
- [X] T007 [P] Create `scripts/sessions.py` — Session 和 KVCacheHandle manager with UUID generation, state machine, KV position tracking (fields: session_id, model_id, status, messages[], kv_position, kv_handle per data-model.md)
- [X] T008 [P] Create `scripts/metrics.py` — request counter, latency histogram, error tracker, `/metrics` endpoint data structure
- [X] T009 [P] Setup error handling framework — custom exceptions (ModelError, DeviceError, ValidationError), HTTP status code mapping (400/404/409/429/500/503/504), structured error response format per contracts/openai-api.md
- [X] T010 [P] Configure logging infrastructure — structured JSON logs with request_id, model, tokens, latency fields

**Checkpoint**: Foundation ready — no user story work can begin until this phase completes

---

## Phase 3: User Story 1 - 提交单次对话请求并获得回复 (Priority: P1) 🎯 MVP

**Goal**: 研究人员能提交一个对话请求并获得流式回复

**Independent Test**: 调用 `/v1/chat/completions`，验证返回 SSE 流并能正确解码完整回复

### Tests for User Story 1

- [X] T011 [P] [US1] Contract test for `/v1/chat/completions` endpoint in `tests/contract/test_chat_completions.py` — validates request/response format per contracts/openai-api.md
- [X] T012 [P] [US1] Integration test for single request → stream response in `tests/integration/test_us1_single_request.py` — end-to-end flow through device_runtime

### Implementation for User Story 1

- [X] T013 [P] [US1] Create `Model` entity in `scripts/model_registry.py` — model registration, status tracking (loading/available/unavailable/error), manifest parsing (input_length from manifest)
- [X] T014 [US1] Implement `GET /v1/models` endpoint returning registered models and status per contracts/openai-api.md response format
- [X] T015 [US1] Implement `POST /v1/chat/completions` endpoint — validate model, encode messages via tokenizer.py, run prefill/decode via device_runtime.py, produce output
- [X] T016 [US1] Implement non-streaming response path (return JSON with `choices[0].message.content`, `finish_reason`, `usage` per contracts/openai-api.md)
- [X] T017 [US1] Add validation and error handling for invalid model ID, malformed messages, missing parameters — mapping to HTTP status codes per contracts/openai-api.md
- [X] T018 [US1] Add logging for request lifecycle (queued → running → completed/error) using metrics.py and logging infrastructure

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - 注册和管理多个模型 (Priority: P2)

**Goal**: 支持多个模型注册，查询列表，对比性能和效果

**Independent Test**: 注册多个模型，`GET /v1/models` 返回所有模型信息和状态

### Implementation for User Story 2

- [X] T019 [US2] Implement `POST /v1/models/register` endpoint — load model artifacts (wtop_path, manifest_path, tokenizer_path), allocate device slot via device_runtime.py init(), set status to `loading` → `available`
- [X] T020 [US2] Implement model status machine — `loading` → `available` | `unavailable` | `error` transitions per data-model.md state transitions
- [X] T021 [US2] Implement KV-cache allocation strategy (per-model slot, memory budget tracking) — kv_base_addr, slot_id, current_position, max_position per KVCacheHandle entity
- [X] T022 [US2] Implement graceful model unload/release — device_runtime.py release(), status transition to unavailable

---

## Phase 5: User Story 3 - 流式响应以实时观察生成分 (Priority: P3)

**Goal**: 支持 `stream=true`，实时输出增量 token 的 SSE 事件

**Independent Test**: 发起流式请求，验证每秒收到 1-3 条增量 SSE 事件，断开连接时停止生成

### Implementation for User Story 3

- [X] T023 [P] [US3] Implement SSE streaming response in `POST /v1/chat/completions` — FastAPI StreamingResponse, `data: {json}\n\n` format per contracts/openai-api.md, `[DONE]` final event
- [X] T024 [US3] Implement client disconnect detection — stop generation via device_runtime.py cancel(), release or recycle KV-cache resources
- [X] T025 [US3] Add `finish_reason` (stop / length / error) and `[DONE]` final event per contracts/openai-api.md stream response format

---

## Phase 6: User Story 4 - 查看请求指标和日志 (Priority: P4)

**Goal**: 管理员了解系统负载、各模型响应时间和成功率

**Independent Test**: 发送若干请求后，`GET /metrics` 返回请求计数、延迟、错误率

### Implementation for User Story 4

- [X] T026 [US4] Implement `GET /metrics` endpoint — per-model request counts, avg/p50/p95 latency, success/failure rates, queue length, active sessions, device runtime status per contracts/openai-api.md
- [X] T027 [US4] Implement request lifecycle logging — structured JSON logs with request_id, model, tokens, latency per research.md R5

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T028 [P] Implement request queue and concurrency limit (max 4 concurrent) — `scripts/queue.py` or integration in gehtp_server.py, per research.md R3 (host-side queue + limit)
- [X] T029 [P] Add model health check (device runtime ping, memory pressure check) — extend device_runtime.py with `health()` method
- [X] T030 [P] Add graceful shutdown handling (close device sessions, finish in-flight requests, release KV resources) — signal handlers in gehtp_server.py
- [X] T031 Run quickstart.md validation — verify end-to-end flow works (register model → submit request → check metrics)
- [X] T032 Document edge cases and limitations in `docs/` — KV-cache capacity, device timeout, tokenizer mismatch, stream disconnect

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (P1) → US2 (P2) → US3 (P3) → US4 (P4) in priority order
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) — No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) — May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) — Extends US1 with SSE streaming; depends on T023 (same endpoint)
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) — Extends all stories with metrics/logging

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- Phase 1 tasks marked [P] can run in parallel (T003)
- Phase 2 tasks marked [P] can run in parallel (T007, T008, T009, T010)
- Within US1 implementation, T013 can run in parallel with contract/integration test writing (T011, T012)
- T023 (SSE streaming) and T029 (health check) can run in parallel with other Phase 5/7 tasks

### MVP First

1. Complete Phase 1 (Setup)
2. Complete Phase 2 (Foundational) — CRITICAL: blocks all stories
3. Complete Phase 3 (US1)
4. **STOP and VALIDATE**: Test single request end-to-end
5. Deploy/demo MVP

### Parallel Example: US1

```bash
# Launch all tests for User Story 1 together:
Task: "Contract test for /v1/chat/completions in tests/contract/test_chat_completions.py"
Task: "Integration test for single request → stream response in tests/integration/test_us1_single_request.py"

# Launch models in parallel:
Task: "Create Model entity in scripts/model_registry.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
