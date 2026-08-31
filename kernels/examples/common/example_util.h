/*
 * example_util.h — hvxhmx 示例共享工具 (result 文件 I/O + pass/fail 日志)
 * =====================================================================
 * 每个 example main.c 都用这套工具把结果写到设备上的 result 文件,
 * 供 build_examples.sh 拉回检查. 用法见任一 example.
 */
#ifndef HVXHMX_EXAMPLE_UTIL_H
#define HVXHMX_EXAMPLE_UTIL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>   /* abs() — 供各 example main.c 直接用 */
#include <string.h>   /* memset/memcpy — 供各 example main.c 直接用 */

#ifdef __cplusplus
extern "C" {
#endif

/* 打开结果文件 /data/local/tmp/hvxhmx_libs/<name>.txt. 在 main 开头调一次. */
void ex_open_result(const char *name);

/* 关闭结果文件 (main 末尾). */
void ex_close_result(void);

/* 往结果文件写一行 (printf 语义). */
void ex_log(const char *fmt, ...);

/* 记一条 PASS/FAIL: err <= tol → PASS. 自动累计 npass/nfail. */
void ex_check(const char *label, int err, int tol);

/* 打印汇总并返回: 有 fail 返回 1, 全 pass 返回 0. main 末尾调, 用其返回值. */
int  ex_summary(void);

/* 用固定种子填 buffer (可复现). */
void ex_fill_u8 (uint8_t  *p, int n, int seed, int mod);
void ex_fill_i8 (int8_t   *p, int n, int seed, int mod);
void ex_fill_u16(uint16_t *p, int n, int seed, int mod);
void ex_fill_i16(int16_t  *p, int n, int seed, int mod);
void ex_fill_i32(int32_t  *p, int n, int seed, int mod, int off);
void ex_fill_f16(__fp16   *p, int n, int seed, float scale);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_EXAMPLE_UTIL_H */
