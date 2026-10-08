# Data Model: Compiler-Driven Scheduling

Phase 1 输出。实体来自 [spec.md](./spec.md) 的 Key Entities，并落成 blob 侧具体字段（编码细节见 [contracts/](./contracts/)）。

## 实体

### 1. Operator（算子）

模型中的一个计算步骤。

| 字段 | 类型 | 说明 |
|------|------|------|
| op_index | u32 | op 表内序号，全 blob 唯一 |
| opcode | u16 | 计算类型（matmul/conv/unary/...） |
| shape | {M,K,N,...} | 成本模型输入 |
| engine | u8 | 编译期指派单元（0=scalar/1=HVX/2=HMX） |
| fallback_allowed | bool | FR-008 开关 |
| fallback_engine | u8 | 允许回落时的替代单元 |

**校验**：`engine`/`fallback_engine` ∈ {0,1,2}；`fallback_allowed=0` 时 `fallback_engine` 无意义（置 0）。

### 2. ComputeUnit（计算单元）

平台三种执行资源：scalar / HVX（向量）/ HMX（矩阵加速）。枚举 `wt_engine {WT_ENG_SCALAR=0, WT_ENG_HVX=1, WT_ENG_HMX=2}`。

### 3. EngineAnnotation（引擎注解，blob footer，ver=3 新增）

逐算子一条，打包 `u32`（R1）。与 Operator 一一对应（按 op_index 平行数组）。

### 4. OnChipMemorySchedule（片上内存时间表）→ VtcmCarveEntry

| 字段 | 类型 | 说明 |
|------|------|------|
| op_index | u32 | 归属算子 |
| vtcm_offset | u32 | 片内偏移 |
| size_bytes | u32 | 占用字节 |
| residency_start | u32 | 生命周期首 op |
| residency_end | u32 | 生命周期末 op（含） |

**校验**：生命周期重叠的任意两 entry 区间不重叠（`[offset, offset+size)` 不相交）——对应 spec Edge Case「生命周期重叠」。

### 5. DataMovementStep（数据搬移步骤）→ 复用 OP_DMA

| 字段 | 类型 | 说明 |
|------|------|------|
| src_ref | u32 | 源引用（DDR `0x8000\|slot` / VTCM `0x4000\|temp`） |
| dst_ref | u32 | 目标引用（同上） |
| bytes | u32 | 搬移字节数 |
| fence | — | CPU→DMA→HMX 同步序（沿用现 fence 纪律） |

### 6. ExecutionReport（执行报告）→ 扩 op_ts.bin

| 字段 | 类型 | 说明 |
|------|------|------|
| op_index | u32 | 算子 |
| planned_engine | u8 | footer 里编译器指派值 |
| actual_engine | u8 | 运行期实际所用 |
| deviation | bool | `planned != actual` |

**不变量（FR-004）**：`deviation=1` 仅当 `fallback_allowed=1` 且 `actual_engine == fallback_engine`；否则运行期必须已中止（`fallback_allowed=0` 硬报错）。

## 关系

- Operator 1—1 EngineAnnotation（footer 平行数组）。
- Operator 0—1 VtcmCarveEntry（需片内的 op 才有）。
- Operator 1—N DataMovementStep（一个 op 可多次搬进/搬出）。
- Operator 1—1 ExecutionReport 记录（执行后）。

## 状态转移

**Operator 执行期状态**：

```
compiled(engine 已指派) ──按计划执行──▶ executed(planned==actual, deviation=0)
       │
       └─ 计划单元不可用 ──┬─ fallback_allowed=1 ──▶ executed(deviation=1, 已上报)
                          └─ fallback_allowed=0 ──▶ aborted(硬报错)
```

无"静默换单元"状态（FR-004/FR-010）。
