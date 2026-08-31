# api_v23_harness — U11 对拍框架 (ex_* + SHA-256 case 流)

源: `src/runtime/harness.c` · 头: `include/harness.h` · 例: `25_harness` (13 门 PASS)

## 契约

V2.3 库内自带轻量对拍层, 不再要求每个新例手写金标循环:

```c
/* harn_case: 包一个 "被测函数", 自动计时 + 对发射流 sha256 */
typedef int (*harn_fn)(void* ud);
int  harn_case(const char* name, harn_fn fn, void* ud);
/* fn 内部用 harn_emit(buf, bytes) 声明 "输出语料" (进 sha 流) */
void harn_emit(const void* buf, uint32_t bytes);

void harn_begin(void);
void harn_expect(const char* label, int err, int tol);   /* 门 */
void harn_note(const char* fmt, ...);                    /* 缩进说明行 */
const char* harn_last_sha(void);  uint32_t harn_last_bytes(void);
int  harn_summary(void);
```

`ex_log/ex_check/ex_summary` 以 **extern 跨 .so 运行时解析** (example_util.c 编入
每个 test_XX.so — 与 qurt/HAP 同款约定), 库本身零依赖。

## 用法 (对拍 = 金标重跑)

框架价值在 "同一份代码在两处跑":

1. 例内 `harn_case("solve_tri", ...)` → 设备输出 `[CASE] solve_tri us=383 sha=a411…  bytes=512`
2. 同 seed 复跑 case → **sha 必须逐位一致** (确定性门, 替代存盘金标)
3. 例 25 用它重跑了例 17 金标 (w4a16, 65536/65536 byte-exact) 与例 19 gdn_sm
   (49152B, sha 与手写版逐字节一致): `gdn_sm_rerun` / `w4a16_gold_rerun` 两门。

## 设备门 (25_harness)

solve_tri cos / gdnsm oracle cos / gdnsm 字节恒等 / w4a16 65536 byte-exact /
两 case 复跑 sha 恒等 (共 13 门)。

## 坑

- `harn_emit` 只在 `harn_case` active 时计入 (active 标志), case 外调用是 no-op。
- sha 流是**语料指纹**不是正确性 — 正确性仍需 oracle/金标门, sha 门只锁确定性。
