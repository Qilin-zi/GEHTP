# Contract: 执行报告（逐算子实际引擎 + 偏差）

**面向接口**：运行时（设备）→ 宿主工具链（prof/判分）。载体为 `op_ts.bin`（现 ver=2, 20B/op 含 engine u32），本契约定义 **ver=3** 增量。

## 记录结构（ver=3）

| 字段 | 类型 | 说明 |
|------|------|------|
| op_index | u32 | op 序号 |
| planned_engine | u8 | 编译器指派值（来自 blob 注解表） |
| actual_engine | u8 | 运行期实际所用 |
| deviation | u8 | 0=无偏离，1=按回落策略替换，2=硬报错中止 |
| 其余 | — | 沿用 v2 现有字段（耗时等） |

## 完整性要求（FR-007 前置）

`g_last_engine`（`kernels/src/runtime/oplist_exec.c:82-83`）当前只在 3 类路径记录：W4A16 DDR-DMA matmul（:494）、f16 HMX crouton（:1554）、HVX add（:794-797）。**5 类路径漏记成 scalar**：
- W4A16 VTCM 预置 act 路径（:472-487，HMX 但未记录）
- conv2d HMX（:627-727,756-760）
- HVX unary/binary/softmax/rmsnorm2（:1008-1015,1102-1114,1142-1153,1686-1697）

本契约要求**每个 op 执行结束时统一记录 `actual_engine`**，不再有缺漏。

**载体锚点**：记录结构 `wt_op_ts`（`oplist_exec.h:81-87`，5×u32=20B）；填充点 `run_range`（`oplist_exec.c:2147-2155`）；落盘 `write_op_ts`（`kernels/examples/42_gehtp_runner/main.c:42-58`，header 硬编码 ver=2）。v3 需同时改这三处。注意 op_ts.bin 的 ver 与 blob 的 ver 是**两套独立版本号**。

## 不变量

- `deviation == 0` ⟺ `planned_engine == actual_engine`。
- `deviation == 1` ⟹ `fallback_allowed == 1` 且 `actual_engine == fallback_engine`（计划单元确实不可用）。
- `deviation == 2` ⟹ `fallback_allowed == 0` 且执行已中止（该 op 之后的记录不再产生）。
- 报告 MUST 覆盖 blob 内**全部** op（SC-001 的 100% 可验证性）。

## 消费方

- `judge_logits.py` / prof 工具：读 `actual_engine` 分布，核对"指派==实际"（SC-001）、"零静默替换"（SC-002）。
- 数值门：`deviation==1` 的算子若有数值漂移需在报告可见（spec Edge Case「回落改变数值」）。
