/*
 * 09_activation — HVX 激活 (HardSwish + PReLU) + golden
 * =====================================================================
 * 教学:
 *   - hvhx_hardswish_flat_u16: 输入 u16 按 int16 二补码解读 (Q12 定点),
 *     f(x)=x*clamp(x+3,0,6)/6. golden 用库内联 hvhx_hardswish_scalar.
 *     向量路径用 1/6≈2731/16384 近似 + 截断, 与标量相差 ≤2 LSB.
 *   - hvhx_prelu_u8: u8 域零点 0x80, slope Q7. golden 用 hvhx_prelu_scalar_u8.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define N 1024

static uint16_t in_u16[N]  __attribute__((aligned(128)));
static uint16_t out_u16[N] __attribute__((aligned(128)));
static uint8_t  in_u8[N]   __attribute__((aligned(128)));
static uint8_t  out_u8[N]  __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("09_activation");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("WARN: setup FAIL"); }

    /* HardSwish flat u16: 用 Q12 有趣区间 [-3,3] → x=(i%256)*96-12288, int16 二补码 */
    for (int i=0;i<N;i++) in_u16[i] = (uint16_t)((int16_t)((i % 256) * 96 - 12288));
    hvhx_hardswish_flat_u16(in_u16, out_u16, N);
    int e=0;
    for (int i=0;i<N;i++){
        int16_t r = hvhx_hardswish_scalar((int16_t)in_u16[i]);
        int d = abs((int)(int16_t)out_u16[i] - (int)r);
        if (d>e) e=d;
    }
    ex_check("hvhx_hardswish_flat_u16 (Q12)", e, 2);

    /* PReLU u8: 输入 u8 偏移二进制 (0x80=零点), slope_q7=128 (≈1.0 → 近线性) */
    ex_fill_u8(in_u8, N, 13, 256);
    uint8_t slope = 128;
    hvhx_prelu_u8(in_u8, slope, out_u8, N);
    e=0;
    for (int i=0;i<N;i++){
        int8_t x = (int8_t)(in_u8[i] - 0x80);   /* u8 偏移二进制 → s8 */
        uint8_t r = hvhx_prelu_scalar_u8(x, slope);
        int d = abs((int)out_u8[i]-(int)r);
        if (d>e) e=d;
    }
    ex_check("hvhx_prelu_u8 (slope=128)", e, 1);

    hmx_runtime_teardown();
    return ex_summary();
}
