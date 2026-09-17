/* wt_inspect.c — host 侧 blob 检查器 (与设备 wt_main parse 模式同一份解析/报告源码)
 *
 *   gcc host/wt_inspect.c dsp/oplist_parse.c dsp/wt_sha256.c dsp/wt_w3.c \
 *       -Idsp -O2 -o build_host/wt_inspect
 *   ./build_host/wt_inspect <blob.wtop>   # stdout = W3 行 (与设备 result jsonl 逐行比对)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oplist_parse.h"

void wt_w3_report(const char* blob_name, const uint8_t* buf, size_t size,
                  const struct wt_blob* w,
                  void (*emit)(const char* line, void* ud), void* ud);

static void emit_stdout(const char* line, void* ud) {
    (void)ud;
    puts(line);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <blob>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);
    struct wt_blob w;
    int rc = wt_parse(buf, (size_t)n, &w);
    if (rc != WT_OK) {
        printf("{\"t\":\"W3\",\"verdict\":\"FAIL\",\"rc\":%d,\"err\":\"%s\"}\n", rc,
               wt_err_str(rc));
        return 1;
    }
    const char* base = strrchr(argv[1], '/');
    base = base ? base + 1 : argv[1];
    wt_w3_report(base, buf, (size_t)n, &w, emit_stdout, NULL);
    return 0;
}
