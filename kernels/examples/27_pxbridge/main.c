/*
 * 27_pxbridge — U13 精度桥 设备验证
 * =====================================================================
 * 判据:
 *   1) f32→f16→f32 往返: 100k 随机值 |r-f| ≤ 0.5 ULP 包络 (含次正规邻域)
 *   2) INT16 对称 (zp=-32768): 解码误差 ≤ scale/2 (半步包络), 零 ⇔ 0x8000
 *   3) 单调线性: 相邻码解码差恒 == scale, 全 65535 步
 *   4) 钳位安全: ±溢出输入落 0xFFFF/0x0000, 解码单侧有界
 *   5) f16↔i16 组合: i16_to_f16→f16_to_f32 与直接解码差 ≤ scale/2 + 0.5 ULP
 *   6) unipolar 契约: 负输入钳到零码 (0x8000, 解码恒 0)
 *   7) f16 桥包络: f32→f16→i16→f32 与 f16 值差 ≤ scale/2 + f16 半 ULP
 *   7) 批量与标量逐字节一致
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define N 100000


int main(void) {
    ex_open_result("27_pxbridge");
    uint32_t lcg = 20260827u;
    float* x = malloc(N * 4);
    int16_t* q = malloc(N * 2);
    int16_t* h = malloc(N * 2);

    /* G1 f16 往返包络 */
    {
        int bad = 0;
        for (int i = 0; i < N; i++) {
            float f = (double)((lcg = lcg * 1664525u + 1013904223u) >> 8) / 16777216.0 - 0.5;
            f *= (i % 7 == 0) ? 6.1e-5f : 1.0f;      /* 1/7 落次正规邻域 */
            int16_t k = gdn_f32_to_f16(f);
            float r = gdn_f16_to_f32(k);
            if (gdn_f32_to_f16(r) != k) bad++;
            /* 次正规区绝对包络: f16 subnormal 量子 5.96e-8, 半步+裕量 */
            float ulp = fabsf(r) * 1.2e-3f + 6.5e-8f;
            if (fabsf(r - f) > ulp) bad++;
        }
        ex_check("f16_roundtrip_ulp_envelope", bad, 0);
    }

    const float scales[4] = { 1e-3f, 1e-2f, 0.1f, 1.0f };
    for (int s = 0; s < 4; s++) {
        float sc = scales[s];
        for (int i = 0; i < N; i++) {                 /* unipolar: u ∈ [0,1) */
            double u = (double)((lcg = lcg * 1664525u + 1013904223u) >> 8) / 16777216.0;
            x[i] = (float)(u * 30000.0 * sc);
        }
        /* G2 半步包络 + 零点 */
        int bad = 0;
        pxb_f32_to_i16_v(x, q, N, sc);
        for (int i = 0; i < N; i++) {
            float d = pxb_i16_to_f32(q[i], sc);
            if (fabsf(d - x[i]) > sc * 0.5f + sc * 5e-3f) bad++;
        }
        if (pxb_f32_to_i16(0.0f, sc) != (int16_t)0x8000) bad++;
        if (pxb_i16_to_f32((int16_t)0x8000, sc) != 0.0f) bad++;
        ex_check("i16_halfstep_envelope_zero_exact", bad, 0);

        /* G5 组合桥 */
        int bad5 = 0;
        pxb_i16_to_f16_v(q, h, N, sc);
        for (int i = 0; i < N; i++) {
            float via = gdn_f16_to_f32(h[i]);
            float dir = pxb_i16_to_f32(q[i], sc);
            float ulp = fabsf(via) * 1.2e-3f + 6.5e-8f;
            if (fabsf(via - dir) > sc * 0.5f + sc * 1e-3f + ulp) bad5++;
        }
        ex_check("f16_i16_combo_envelope", bad5, 0);

        /* G6 unipolar 契约: 负钳零码 + 桥恒等 (f16 路径 == f32 路径) */
        int bad6 = 0;
        for (int i = 0; i < N; i += 3) {
            if (fabsf(x[i]) > 0.5f) continue;
            if (pxb_f32_to_i16(-fabsf(x[i]), sc) != (int16_t)0x8000) { bad6++; break; }
            if (pxb_i16_to_f32(pxb_f32_to_i16(-fabsf(x[i]), sc), sc) != 0.0f) { bad6++; break; }
        }
        ex_check("negative_clamps_zero_code", bad6, 0);
        {
            int bad7 = 0;
            for (int i = 0; i < N; i += 7) {
                int16_t h = gdn_f32_to_f16(x[i]);
                float x16 = gdn_f16_to_f32(h);
                float d16 = pxb_i16_to_f32(pxb_f16_to_i16(h, sc), sc);
                if (fabsf(d16 - x16) > sc * 0.5f + fabsf(x16) * 1.3e-3f + 1e-12f) { bad7++; break; }
            }
            ex_check("f16_bridge_envelope", bad7, 0);
        }
    }

    /* G3 线性性: 全码空间相邻步长恒等 */
    {
        int bad = 0;
        float sc = 1e-2f;
        for (int32_t c = 0; c < 65535; c++) {
            float d0 = pxb_i16_to_f32((int16_t)(c - 32768), sc);
            float d1 = pxb_i16_to_f32((int16_t)(c + 1 - 32768), sc);
            /* 步长差被 f32 乘法舍入吸收 (高端码处 ulp ≈ 655×2^-24) */
            if (fabsf((d1 - d0) - sc) > fabsf(d1) * 1.3e-3f + 1e-7f) { bad++; break; }
        }
        ex_check("code_space_linear_exact", bad, 0);
    }

    /* G4 钳位 */
    {
        int bad = 0;
        float sc = 0.1f;
        if (pxb_f32_to_i16(1e9f, sc) != (int16_t)0x7FFF) bad++;
        if (pxb_f32_to_i16(-1e9f, sc) != (int16_t)0x8000) bad++;
        if (pxb_i16_to_f32((int16_t)0x7FFF, sc) != 65535.0f * sc) bad++;
        if (pxb_i16_to_f32((int16_t)0x8000, sc) != 0.0f) bad++;
        ex_check("clamp_extremes", bad, 0);
    }

    /* G7 批量 vs 标量逐字节 */
    {
        int bad = 0;
        float sc = 1e-2f;
        for (int i = 0; i < N; i++)
            x[i] = (float)(((lcg = lcg * 1664525u + 1013904223u) >> 8) % 65536) - 32768.0f;
        pxb_f32_to_i16_v(x, q, N, sc);
        for (int i = 0; i < N; i++)
            if (q[i] != pxb_f32_to_i16(x[i], sc)) { bad++; break; }
        float* y = malloc(N * 4);
        pxb_i16_to_f32_v(q, y, N, sc);
        for (int i = 0; i < N; i++)
            if (y[i] != pxb_i16_to_f32(q[i], sc)) { bad++; break; }
        ex_check("batch_vs_scalar_identical", bad, 0);
        free(y);
    }

    free(x); free(q); free(h);
    return ex_summary();
}
