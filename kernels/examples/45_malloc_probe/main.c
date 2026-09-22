/*
 * 45_malloc_probe — PD 堆 malloc 上限探测 (⑤ 前置: blob 需 2.63GB 驻留)
 * 二分试 malloc [128MB, 3GB]: 找到最大可分配大小。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "example_util.h"

int main(int argc, char** argv) {
    ex_open_result("45_malloc_probe");
    (void)argc; (void)argv;
    size_t lo = 128u << 20, hi = 3u << 30;   /* 已知 128MB 可, 3GB 不可 */
    size_t best = 0;
    for (int i = 0; i < 12; i++) {
        size_t mid = lo + (hi - lo) / 2;
        void* p = malloc(mid);
        if (p) {
            /* 写几个点验证真实可用 (虚分配探测) */
            memset(p, 0x5A, 4096);
            memset((uint8_t*)p + mid - 4096, 0x5A, 4096);
            best = mid;
            lo = mid;
            free(p);
            ex_log("malloc(%zu) OK", mid);
        } else {
            hi = mid;
            ex_log("malloc(%zu) FAIL", mid);
        }
    }
    ex_log("max malloc ~= %zu bytes (%.2f GB)", best, (double)best / (1 << 30));
    ex_check("probe_done", best > 0 ? 0 : 1, 0);
    return ex_summary();
}
