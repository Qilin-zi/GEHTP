/*
 * 11_lookup_unpack — HVX 查表 + 权重解包 + golden
 * =====================================================================
 * 教学:
 *   - hvhx_table_lookup_flat_u8: out[i]=table[in[i]], 256 元素 LUT. exact.
 *   - hvhx_unpack_weights: 4-bit → 8-bit, 每输入字节出 2 个 nibble.
 *       out[2i]   = (in[i] >> 4) & 0x0F
 *       out[2i+1] =  in[i]       & 0x0F
 *     输入 n 字节 → 输出 2n 字节. exact.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define N 1024

static uint8_t idx[N]     __attribute__((aligned(128)));
static uint8_t table[256] __attribute__((aligned(128)));
static uint8_t out[N]     __attribute__((aligned(128)));
static uint8_t gold[N]    __attribute__((aligned(128)));
static uint8_t packed[N]  __attribute__((aligned(128)));
static uint8_t unpacked[2*N] __attribute__((aligned(128)));
static uint8_t gold_un[2*N]  __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("11_lookup_unpack");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("WARN: setup FAIL"); }

    /* table_lookup_flat_u8: out[i] = table[idx[i]], idx ∈ [0,255] */
    for (int i=0;i<256;i++) table[i]=(uint8_t)(i*2);
    ex_fill_u8(idx, N, 42, 256);
    hvhx_table_lookup_flat_u8(idx, table, out, N);
    int e=0;
    for (int i=0;i<N;i++){gold[i]=table[idx[i]]; int d=abs((int)out[i]-(int)gold[i]); if(d>e)e=d;}
    ex_check("hvhx_table_lookup_flat_u8", e, 0);

    /* unpack_weights: n 输入字节 → 2n 输出字节 */
    ex_fill_u8(packed, N, 55, 256);
    hvhx_unpack_weights(packed, unpacked, N);
    e=0;
    for (uint32_t i=0;i<N;i++){
        gold_un[2*i+0] = (packed[i] >> 4) & 0x0F;
        gold_un[2*i+1] =  packed[i]       & 0x0F;
        int d0 = abs((int)unpacked[2*i+0]-(int)gold_un[2*i+0]);
        int d1 = abs((int)unpacked[2*i+1]-(int)gold_un[2*i+1]);
        if(d0>e)e=d0; if(d1>e)e=d1;
    }
    ex_check("hvhx_unpack_weights (4bit→8bit)", e, 0);

    hmx_runtime_teardown();
    return ex_summary();
}
