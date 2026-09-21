/* wt_host_exec.c — WTOP blob 的 host 执行器 (与设备 oplist_exec 同源)
 *
 * 用途: 设备输出异常时在 host 侧复现/排除 ——
 *   host 对 = blob/编码问题 (bisect --upto 找首个发散 op);
 *   host 对不上 golden 但和设备一样错 = 设备侧问题 (内存/cache/版本);
 *   host 对得上 golden 而设备错 = 设备侧问题。
 *
 * 用法:
 *   wt_host_exec blob.wtop input.f16.raw out.f16.raw [out_temp] [--upto N] [--dump-temp T:file]
 *     --upto N:     只跑前 N 个 op (run_range(0,N))
 *     --dump-temp:  跑完后把 temp T 内容写 file (f16 raw)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oplist_parse.h"
#include "oplist_exec.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s blob in.raw out.raw [out_temp] [--upto N] [--dump-temp T:file]\n", argv[0]);
        return 64;
    }
    const char* blob_p = argv[1];
    const char* in_p = argv[2];
    const char* out_p = argv[3];
    uint32_t out_temp = argc > 4 && argv[4][0] != '-' ? (uint32_t)atoi(argv[4]) : 7;
    long upto = -1;
    const char* dump_spec = NULL;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--upto") && i + 1 < argc) upto = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dump-temp") && i + 1 < argc) dump_spec = argv[++i];
    }

    FILE* f = fopen(blob_p, "rb");
    if (!f) { perror("blob"); return 66; }
    fseek(f, 0, SEEK_END); long bsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* bb = malloc(bsz);
    if (fread(bb, 1, bsz, f) != (size_t)bsz) { perror("read blob"); return 66; }
    fclose(f);

    f = fopen(in_p, "rb");
    if (!f) { perror("input"); return 66; }
    fseek(f, 0, SEEK_END); long isz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* ib = malloc(isz);
    if (fread(ib, 1, isz, f) != (size_t)isz) { perror("read input"); return 66; }
    fclose(f);

    static struct wt_blob b;
    int prc = wt_parse(bb, (size_t)bsz, &b);
    if (prc != 0) { fprintf(stderr, "wt_parse rc=%d\n", prc); return 65; }
    fprintf(stderr, "[host_exec] slots=%u ops=%u weight_bytes=%zu\n", b.n_slots, b.n_ops, b.weight_bytes);

    char err[256] = {0};
    uint32_t engine_m = 0;
    int rc;
    if (upto >= 0)
        rc = wt_exec_run_range(&b, 0, (uint32_t)upto, &engine_m, NULL, err, sizeof(err));
    else
        rc = wt_exec_run_io(&b, ib, NULL, out_temp, NULL, NULL, err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "[host_exec] exec rc=%d err=%s\n", rc, err); return 1; }

    if (dump_spec) {
        uint32_t t = 0; char path[256] = {0};
        if (sscanf(dump_spec, "%u:%255s", &t, path) == 2) {
            const uint8_t* tp = wt_exec_temp(t);
            uint32_t tb = wt_exec_temp_last_bytes(t);
            FILE* df = fopen(path, "wb");
            if (tp && df) fwrite(tp, 1, tb, df);
            if (df) fclose(df);
            fprintf(stderr, "[host_exec] dump temp %u = %u bytes -> %s\n", t, tb, path);
        }
    }

    if (upto < 0) {
        const uint8_t* op = wt_exec_temp(out_temp);
        uint32_t ob = wt_exec_temp_last_bytes(out_temp);
        FILE* of = fopen(out_p, "wb");
        if (op && of) fwrite(op, 1, ob, of);
        if (of) fclose(of);
        fprintf(stderr, "[host_exec] out temp %u = %u bytes -> %s\n", out_temp, ob, out_p);
    }
    return 0;
}
