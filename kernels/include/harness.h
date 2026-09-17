/* harness.h — V2.3 金标对拍框架 (U11; 例 17/18/21 手写对拍流程的统一化)
 *
 * 三件套:
 *   harn_case(name, fn, ud)   运行一个用例: 计时 + 输出流 sha256 + PASS/FAIL
 *   harn_emit(buf, bytes)     用例内提交"被钉字节" → 喂 sha256 (wt_sha256 复用)
 *   harn_expect(err, tol)     用例内判定 (ex_check 语义)
 *
 * 用例 fn 返回 0 = 通过; 非 0 = 失败码。框架产出行:
 *   [CASE] <name> us=<t> sha=<hex16> bytes=<n>
 *   [PASS] <name> err=0
 * 判定真值 = fn 内部自检 (oracle 对拍 / 金标位恒等), sha 供跨轮次/跨进程
 * 逐字节比对 (确定性钉住: 同输入两跑 sha 必须相等)。
 */
#ifndef HVXHMX_V23_HARNESS_H
#define HVXHMX_V23_HARNESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*harn_fn)(void* ud);

/* 框架会话: ex_open_result 之后调 harn_begin; 结束 harn_summary (替代 ex_summary) */
void harn_begin(void);
int  harn_summary(void);

/* 运行用例并记录。返回 fn 的返回值。 */
int  harn_case(const char* name, harn_fn fn, void* ud);

/* 用例内: 提交钉住字节 (累计进本例 sha) / 判定 / 记录自由文本 */
void harn_emit(const void* buf, uint32_t bytes);
void harn_expect(const char* label, int err, int tol);
void harn_note(const char* fmt, ...);

/* 上一例的 sha (16 hex) 与字节数 — 供确定性门取用 */
const char* harn_last_sha(void);
uint32_t    harn_last_bytes(void);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_HARNESS_H */
