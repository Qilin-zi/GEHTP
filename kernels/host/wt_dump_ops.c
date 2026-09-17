/* wt_dump_ops — 最小 blob op 记录转储 (结构检查/排错用)
 * 用法: wt_dump_ops blob.wtop [-v]   默认输出 opcode 直方图+首末 op; -v 逐 op
 * 构建: gcc -O2 -o wt_dump_ops wt_dump_ops.c ../src/runtime/oplist_parse.c -I../include
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oplist_parse.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: wt_dump_ops blob.wtop [-v]\n"); return 1; }
    int verbose = (argc > 2 && strcmp(argv[2], "-v") == 0);
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(sz);
    if (!buf || fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(f);
    struct wt_blob* w = calloc(1, sizeof(*w));
    int rc = wt_parse(buf, sz, w);
    if (rc != WT_OK) { fprintf(stderr, "parse rc=%d %s\n", rc, wt_err_str(rc)); return 1; }
    printf("slots=%u ops=%u ver=%u weight_off=%u\n", w->n_slots, w->n_ops, w->ver, w->weight_off);
    if (verbose) {
        for (uint32_t i = 0; i < w->n_ops; i++) {
            printf("op%u opcode=%u args=[", i, w->ops[i].opcode);
            for (uint16_t j = 0; j < w->ops[i].n_args; j++)
                printf("%s%u", j ? "," : "", w->ops[i].args[j]);
            printf("]\n");
        }
    } else {
        uint32_t hist[64] = {0};
        for (uint32_t i = 0; i < w->n_ops; i++)
            if (w->ops[i].opcode < 64) hist[w->ops[i].opcode]++;
        for (int c = 0; c < 64; c++)
            if (hist[c]) printf("  opcode %-3u × %u\n", c, hist[c]);
        for (uint32_t i = 0; i < w->n_ops && i < 3; i++) {
            printf("first op%u opcode=%u nargs=%u | ", i, w->ops[i].opcode, w->ops[i].n_args);
        }
        printf("\n");
        for (uint32_t i = w->n_ops > 3 ? w->n_ops - 3 : 0; i < w->n_ops; i++)
            printf("last op%u opcode=%u nargs=%u\n", i, w->ops[i].opcode, w->ops[i].n_args);
    }
    free(w); free(buf);
    return 0;
}
