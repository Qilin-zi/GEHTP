/*
 * 21_oplist_exec — U6 oplist 单元 设备验证 (解析负例 + 执行 + 数值金标)
 * =====================================================================
 * 输入 (host build 脚本准备并 push):
 *   /data/local/tmp/hvxhmx23/blob_w4.wtop  (NOP/PIN/PIN/MATMUL)
 *   /data/local/tmp/hvxhmx23/blob_w5.wtop  (w4 + RMSNORM, 共享同一 weight 区)
 *   /data/local/tmp/hvxhmx23/rms_w.f16.raw (slot5 参考镜像, n=2560)
 *   /data/local/tmp/hvxhmx23/Y_gold.raw    (K2560 标量金标 (N,M), tol 40 LSB)
 * 判据:
 *   W3-  负例 5 项 (magic/ver/endian/n_slots/n_ops) 均返回期望错误码
 *   W3   设备 W3 报告行 (host 与 wt_inspect 逐行比对在 build 脚本)
 *   W4   blob_w4 MATMUL 解码 vs Y_gold max ≤ 40 LSB (K2560 规范 37)
 *   W4+  blob_w5 重开引擎后 temp0 与 blob_w4 temp0 byte-exact (重初始化确定性)
 *   W5   blob_w5 RMSNORM vs 设备内同算法标量镜像 bit-exact (0 ULP)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define D "/data/local/tmp/hvxhmx23"
#define MM 256u
#define KK 2560u
#define NN 2560u
#define TOL_LSB 40

/* ---- 与 src/v22/oplist_exec.c 逐位同源的镜像 (判定真值必须同算法) ---- */
static float mf16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(127 - 15 + 1 - e - 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static uint16_t mf32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> (shift + 1);
        uint32_t rest = man & ((1u << shift) - 1u);
        if ((man >> shift) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}
/* crouton16_row4 面 (M,cols) → row-major (值不变; 存 u16 防符号扩展) */
static void minv_crouton(const uint16_t* surf, uint16_t* dst, uint32_t M, uint32_t cols) {
    uint32_t n_m32 = M / 32, n_kt = cols / 32, out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++)
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    const uint16_t* p = surf + out;
                    for (uint32_t c = kt * 32; c < kt * 32 + 32; c++) {
                        dst[(size_t)row0 * cols + c] = p[0];
                        dst[(size_t)(row0 + 1) * cols + c] = p[1];
                        p += 2;
                    }
                    out += 64;
                }
}

static void w3_emit(const char* line, void* ud) { (void)ud; ex_log("%s", line); }

static uint8_t* read_all(const char* p, uint32_t* b) {
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *b = (uint32_t)n;
    return buf;
}

/* 负例: 好 blob 拷贝 → 定点破坏 → wt_parse 必须返回期望错误码 */
static int negative_cases(const uint8_t* good, size_t n) {
    struct wt_blob w;
    uint8_t* c = malloc(n);
    int fails = 0;
#define NEG(expect, mut) do { memcpy(c, good, n); do { mut; } while (0); \
    int rc = wt_parse(c, n, &w); \
    if (rc != (expect)) { \
        ex_log("neg(%s): rc=%d expect=%d (%s)", #mut, rc, (int)(expect), wt_err_str(rc)); \
        fails++; \
    } } while (0)
    NEG(WT_ERR_MAGIC,  memcpy(c, "XTOP", 4));
    NEG(WT_ERR_VER,    c[4] = 0xEE);
    NEG(WT_ERR_ENDIAN, c[6] ^= 0xFF);
    NEG(WT_ERR_NSLOTS, { c[8] = 0xFF; c[9] = 0xFF; c[10] = 0; c[11] = 0; });
    NEG(WT_ERR_NOPS,   { uint32_t no = 100000; memcpy(c + 12, &no, 4); });
#undef NEG
    free(c);
    return fails;
}

/* 单 blob 执行: matmul 解码后比对金标; want_rms=1 时再比对 rmsnorm 镜像。
 * t0_copy 非空时与 temp0 byte 比对 (重初始化确定性)。 */
static int run_blob(const uint8_t* blob, uint32_t bb, const uint16_t* gold,
                    const uint16_t* w16, const uint8_t* t0_copy, int want_rms) {
    struct wt_blob w;
    if (wt_parse(blob, bb, &w) != WT_OK) { ex_log("wt_parse FAIL"); return 1; }
    uint32_t em = 0;
    int64_t op_us[8] = {0};
    char err[128];
    int rc = wt_exec_run(&w, &em, op_us, err, sizeof(err));
    if (rc) {
        ex_log("wt_exec_run FAIL at op %d: %s", rc, err);
        wt_exec_shutdown();                 /* 铁律④: FAIL 路径也关 */
        return 1;
    }
    uint8_t* t0 = wt_exec_temp(0);
    uint8_t* t1 = wt_exec_temp(1);
    if (!t0 || wt_exec_temp_bytes(0) != MM * NN * 2u || em != MM) {
        ex_log("temp0 bad (t0=%p bytes=%lu m=%lu)", t0,
               (unsigned long)wt_exec_temp_bytes(0), (unsigned long)em);
        wt_exec_shutdown();
        return 1;
    }

    if (t0_copy) {
        int bad = memcmp(t0, t0_copy, MM * NN * 2) != 0;
        ex_check("matmul_reinit_byteexact", bad, 0);
    }

    uint16_t* dev = malloc((size_t)MM * NN * 2);
    minv_crouton((const uint16_t*)t0, dev, MM, NN);
    int max_lsb = 0;
    for (uint32_t m = 0; m < MM; m++)
        for (uint32_t n = 0; n < NN; n++) {
            int d = (int)dev[(size_t)m * NN + n] - (int)gold[(size_t)n * MM + m];
            if (d < 0) d = -d;
            if (d > max_lsb) max_lsb = d;
        }
    ex_check("matmul_vs_gold_lsb", max_lsb <= TOL_LSB ? 0 : 1, 0);
    ex_log("matmul max_lsb=%d (tol %d, K2560 规范 37)", max_lsb, TOL_LSB);

    int fret = 0;
    if (want_rms) {
        if (!t1 || wt_exec_temp_bytes(1) != MM * NN * 2u) {
            ex_log("temp1 bad");
            fret = 1;
        } else {
            uint16_t* yref = malloc((size_t)MM * NN * 2);
            float* xf = malloc(NN * 4);
            for (uint32_t r = 0; r < MM; r++) {
                const uint16_t* row = dev + (size_t)r * NN;
                double acc = 0.0;
                for (uint32_t i = 0; i < NN; i++) {
                    xf[i] = (float)((int32_t)row[i] - 32768);
                    acc += (double)xf[i] * (double)xf[i];
                }
                float rms = (float)sqrt(acc / (double)NN + 1e-6);
                for (uint32_t i = 0; i < NN; i++)
                    yref[(size_t)r * NN + i] =
                        mf32_to_f16(xf[i] / rms * mf16_to_f32(w16[i]));
            }
            const uint16_t* y1 = (const uint16_t*)t1;
            size_t bitbad = 0;
            for (size_t z = 0; z < (size_t)MM * NN; z++)
                if (y1[z] != yref[z]) bitbad++;
            ex_check("rmsnorm_bitexact", bitbad == 0 ? 0 : 1, 0);
            ex_log("rmsnorm bit-diff=%lu / %lu", (unsigned long)bitbad,
                   (unsigned long)MM * NN);
            free(yref); free(xf);
        }
        ex_log("op_us matmul=%lld rmsnorm=%lld",
               (long long)op_us[3], (long long)op_us[4]);
    } else {
        ex_log("op_us matmul=%lld", (long long)op_us[3]);
    }
    free(dev);
    wt_exec_shutdown();                     /* 铁律④ */
    return fret;
}

int main(void) {
    ex_open_result("21_oplist_exec");
    uint32_t b4 = 0, b5 = 0, brw = 0, bg = 0;
    uint8_t* blob4 = read_all(D "/blob_w4.wtop", &b4);
    uint8_t* blob5 = read_all(D "/blob_w5.wtop", &b5);
    uint8_t* rmsw  = read_all(D "/rms_w.f16.raw", &brw);
    uint16_t* gold = (uint16_t*)read_all(D "/Y_gold.raw", &bg);
    if (!blob4 || !blob5 || !rmsw || !gold ||
        brw != NN * 2u || bg != NN * MM * 2u) {
        ex_log("inputs missing/mismatch (b4=%lu b5=%lu rms=%lu gold=%lu)",
               (unsigned long)b4, (unsigned long)b5,
               (unsigned long)brw, (unsigned long)bg);
        goto out;
    }
    dc_clean_ddr(blob4, b4);                /* 铁律①: blob 是 DMA bypass 源 */
    dc_clean_ddr(blob5, b5);

    /* ---- 解析 + 负例 + W3 报告 (用 w5 blob, 覆盖 slot 最多形态) ---- */
    struct wt_blob w;
    if (wt_parse(blob5, b5, &w) != WT_OK) { ex_log("wt_parse(w5) FAIL"); goto out; }
    ex_check("negatives_rejected", negative_cases(blob5, b5), 0);
    wt_w3_report("blob_w5.wtop", blob5, b5, &w, w3_emit, NULL);

    /* ---- W4: blob_w4 单 op 路径, 留 temp0 快照 ---- */
    {
        struct wt_blob w4;
        if (wt_parse(blob4, b4, &w4) != WT_OK) { ex_log("wt_parse(w4) FAIL"); goto out; }
        uint32_t em = 0;
        int64_t op_us[8] = {0};
        char err[128];
        if (wt_exec_run(&w4, &em, op_us, err, sizeof(err))) {
            ex_log("w4 exec FAIL: %s", err);
            wt_exec_shutdown();
            goto out;
        }
        uint8_t* t0 = wt_exec_temp(0);
        if (!t0 || wt_exec_temp_bytes(0) != MM * NN * 2u) {
            ex_log("w4 temp0 bad");
            wt_exec_shutdown();
            goto out;
        }
        uint8_t* snap = malloc((size_t)MM * NN * 2);
        memcpy(snap, t0, MM * NN * 2);
        ex_log("w4 op_us matmul=%lld", (long long)op_us[3]);
        wt_exec_shutdown();

        /* ---- W4 数值 + W4+ 重初始化确定性 + W5 rmsnorm ---- */
        if (run_blob(blob5, b5, gold, (const uint16_t*)rmsw, snap, 1))
            { free(snap); goto out; }
        free(snap);
    }
    ex_check("suite_complete", 0, 0);

out:
    free(blob4); free(blob5); free(rmsw); free(gold);
    return ex_summary();
}
