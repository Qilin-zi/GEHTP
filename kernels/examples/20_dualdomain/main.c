/*
 * 20_dualdomain — U7 dd_worker 双域 step-list 单元 设备验证
 * =====================================================================
 * 本 .so 是"一个分片执行器": ./run_main_on_hexagon <dom> test_20.so dd <tag> <start> <len>
 * host 侧 build_examples.sh 编排三段:
 *   ser: dom3 一次跑 [0,16)                → 16 步基线 (wall_us + 每步 sha)
 *   a/b:  dom3 [0,8) 与 dom4 [8,16) 并发   → 双域 (各自的 wall_us)
 * 判据 (host 汇总): a/b 每步 sha == ser 对应步 (切分等价) + 加速比 = ser/min-max。
 * 模块内不做跨域通信; 输出 = result_20_dualdomain_<tag>.txt。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hvxhmx_v22.h"
#include "example_util.h"
#include "wt_sha256.h"

#define ASSET "/data/local/tmp/hvxhmx23/assets/s256"
#define OUT   "/data/local/tmp/hvxhmx23/dd_out"
#define STEPS 16
#define SHA_BYTES (256u * 256u * 2u)

int main(int argc, char** argv) {
    const char* sel = (argc > 1) ? argv[1] : "dd";
    if (strcmp(sel, "dd") != 0) {
        ex_open_result("20_dualdomain_err");
        ex_log("unknown selector %s", sel);
        return 0xFE;
    }
    const char* tag   = (argc > 2) ? argv[2] : "x";
    int start = (argc > 3) ? atoi(argv[3]) : 0;
    int len   = (argc > 4) ? atoi(argv[4]) : STEPS;
    char name[64];
    snprintf(name, sizeof(name), "20_dualdomain_%s", tag);
    ex_open_result(name);

    if (start < 0 || len <= 0 || start + len > STEPS) {
        ex_log("range %d+%d > %d", start, len, STEPS);
        return ex_summary();
    }
    /* OUT 目录必须由 host 侧预建 (DSP 上 system() 无 shell, fopen 不会建目录) */

    struct dd_cfg cfg = { ASSET, 256, 256, 256, 8, 0, 1 };  /* n_act_files=0 → v%4, dump=1 */
    struct dd_stats st;
    char err[128];
    int rc = dd_run(&cfg, tag, start, len, OUT, &st, err, sizeof(err));
    if (rc) { ex_log("dd_run FAIL: %s", err); return ex_summary(); }
    ex_log("tag=%s start=%d len=%d wall_us=%llu e2e_us=%llu first_us=%lld per_step=%.1f",
           tag, start, len, (unsigned long long)st.wall_us,
           (unsigned long long)st.e2e_us, (long long)st.first_step_us, st.per_step_us);

    /* 每步输出 sha256 → host 与 ser 对拍 (切分等价证据) */
    uint8_t* buf = memalign(128, SHA_BYTES);
    int shas = 0;
    for (int k = start; k < start + len; k++) {
        char p[160], hex[65];
        snprintf(p, sizeof(p), OUT "/dd_%s_step%d.raw", tag, k);
        FILE* f = fopen(p, "rb");
        if (!f || fread(buf, 1, SHA_BYTES, f) != SHA_BYTES) {
            ex_log("step %d read FAIL", k);
            if (f) fclose(f);
            free(buf);
            return ex_summary();
        }
        fclose(f);
        remove(p);                       /* 读完即删, 双域并发不挤设备空间 */
        wt_sha256_hex(buf, SHA_BYTES, hex);
        ex_log("sha step%d %s", k, hex);
        shas++;
    }
    free(buf);
    ex_check("steps_dumped", shas == len ? 0 : 1, 0);
    return ex_summary();
}
