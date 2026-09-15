/* check_offsets —— 编译期静态偏移表的正确性裁判(第7步阶段一)
 * =====================================================================
 * 用途: 对编译器(DDR 区间分配器)输出的 temp 偏移表做三类检查:
 *   1. 越界:  offset + size > pool_cap
 *   2. 对齐:  offset 必须 128B 对齐(HMX/DMA 契约)
 *   3. 别名:  两个生命期相交的区间不得共享地址空间(偏移区间相交)
 * 生命期 = 执行序区间 [life_begin, life_end]: 张量从生产 op 写入
 * 到最后一个消费 op 读完。两区间可共享偏移 ⇔ 生命期不相交。
 *
 * 输入格式 (TSV, # 为注释):
 *   pool_cap <bytes>              # 第 0 行(必须)
 *   <op_id> <offset> <size> <life_begin> <life_end>
 *   ...                            # 每行一个张量(temp id = op_id)
 *
 * 用法: check_offsets <table.tsv> [--align N] [--pool N]
 * 输出: 每项检查 OK/FAIL; 别名冲突列出相交对。退出码 0=全过。
 * 构建: gcc -O2 -o check_offsets check_offsets.c (纯 C, 无依赖)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_ROWS 65536
#define MAX_LINE 512

struct row {
    uint64_t op_id;
    uint64_t offset;
    uint64_t size;
    uint64_t life_begin;
    uint64_t life_end;
};

static struct row rows[MAX_ROWS];
static size_t n_rows = 0;
static uint64_t pool_cap = 0;
static uint64_t align = 128;
static int g_fail = 0;

static int parse_table(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[MAX_LINE];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        if (strncmp(p, "pool_cap", 8) == 0) {
            if (sscanf(p + 8, "%llu", (unsigned long long*)&pool_cap) != 1) {
                fprintf(stderr, "line %d: bad pool_cap\n", lineno);
                fclose(f); return -1;
            }
            continue;
        }
        struct row r;
        if (sscanf(p, "%llu %llu %llu %llu %llu",
                   (unsigned long long*)&r.op_id,
                   (unsigned long long*)&r.offset,
                   (unsigned long long*)&r.size,
                   (unsigned long long*)&r.life_begin,
                   (unsigned long long*)&r.life_end) != 5) {
            fprintf(stderr, "line %d: parse error: %s", lineno, line);
            fclose(f); return -1;
        }
        if (n_rows >= MAX_ROWS) { fprintf(stderr, "too many rows\n"); fclose(f); return -1; }
        rows[n_rows++] = r;
    }
    fclose(f);
    return 0;
}

/* 生命期相交: [a1,a2) 与 [b1,b2) 有公共执行序时刻 */
static int lives_overlap(const struct row* a, const struct row* b) {
    return a->life_begin < b->life_end && b->life_begin < a->life_end;
}

/* 地址相交 */
static int addr_overlap(const struct row* a, const struct row* b) {
    return a->offset < b->offset + b->size && b->offset < a->offset + a->size;
}

int main(int argc, char** argv) {
    const char* path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--align") == 0 && i + 1 < argc) align = strtoull(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) pool_cap = strtoull(argv[++i], NULL, 0);
        else if (!path) path = argv[i];
        else { fprintf(stderr, "usage: check_offsets <table.tsv> [--align N] [--pool N]\n"); return 2; }
    }
    if (!path) { fprintf(stderr, "usage: check_offsets <table.tsv> [--align N] [--pool N]\n"); return 2; }
    if (parse_table(path) != 0) return 2;
    if (n_rows == 0) { fprintf(stderr, "empty table\n"); return 2; }

    printf("=== check_offsets: %zu rows, pool_cap=%llu, align=%llu ===\n",
           n_rows, (unsigned long long)pool_cap, (unsigned long long)align);

    /* 1. 越界 + 对齐 */
    for (size_t i = 0; i < n_rows; i++) {
        const struct row* r = &rows[i];
        if (pool_cap && r->offset + r->size > pool_cap) {
            fprintf(stderr, "FAIL oob: op %llu off=%llu size=%llu > cap=%llu\n",
                    (unsigned long long)r->op_id, (unsigned long long)r->offset,
                    (unsigned long long)r->size, (unsigned long long)pool_cap);
            g_fail++;
        }
        if (r->offset % align != 0) {
            fprintf(stderr, "FAIL align: op %llu off=%llu %% %llu != 0\n",
                    (unsigned long long)r->op_id, (unsigned long long)r->offset,
                    (unsigned long long)align);
            g_fail++;
        }
        if (r->life_end <= r->life_begin) {
            fprintf(stderr, "FAIL lifetime: op %llu end=%llu <= begin=%llu\n",
                    (unsigned long long)r->op_id, (unsigned long long)r->life_end,
                    (unsigned long long)r->life_begin);
            g_fail++;
        }
    }
    printf("bounds/align: %s\n", g_fail ? "FAIL" : "OK");

    /* 2. 别名: 共活区间地址相交 */
    int aliases = 0;
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = i + 1; j < n_rows; j++) {
            const struct row* a = &rows[i];
            const struct row* b = &rows[j];
            if (lives_overlap(a, b) && addr_overlap(a, b)) {
                fprintf(stderr,
                        "FAIL alias: op %llu [%llu,%llu) life[%llu,%llu) vs "
                        "op %llu [%llu,%llu) life[%llu,%llu)\n",
                        (unsigned long long)a->op_id, (unsigned long long)a->offset,
                        (unsigned long long)(a->offset + a->size),
                        (unsigned long long)a->life_begin, (unsigned long long)a->life_end,
                        (unsigned long long)b->op_id, (unsigned long long)b->offset,
                        (unsigned long long)(b->offset + b->size),
                        (unsigned long long)b->life_begin, (unsigned long long)b->life_end);
                aliases++;
                g_fail++;
                if (aliases > 50) {
                    fprintf(stderr, "(alias 报告截断)\n");
                    i = n_rows; break;
                }
            }
        }
    }
    printf("alias: %s (%d conflicts)\n", aliases ? "FAIL" : "OK", aliases);

    /* 3. 池用量统计 */
    uint64_t used = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint64_t end = rows[i].offset + rows[i].size;
        if (end > used) used = end;
    }
    printf("pool used: %llu / %llu (%.1f%%)\n",
           (unsigned long long)used, (unsigned long long)pool_cap,
           pool_cap ? 100.0 * (double)used / (double)pool_cap : 0.0);

    printf("%s\n", g_fail ? "VERDICT: FAIL" : "VERDICT: OK");
    return g_fail ? 1 : 0;
}
