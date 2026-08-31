/* w4a16_driver_dc.c — W4A16 HMX kernel driver, 参数化改造版
 *
 * 拷贝自 htpw4a16_v81/closure_dsp/hmx_v81_main.c (已闭合), 按 T10 plan §4.2 改造:
 *   1. 面/表/输出指针全部外部传入 (wtcache 布局), 不再自己 carve VTCM
 *   2. 退役 vtcm_setup/acquire/power (wtcache_open 独占)
 *   3. 表重写按传入地址 (act_table_rw/out_table_rw 是可写 VTCM 区,
 *      内容先由调用方从 host 表 memcpy 进来, 这里把 u32 offset → 绝对指针)
 *   4. descriptor 段 (ABI 常量) 逐字保留 — 改了等于重验 2-B
 *   5. mask/extra 由调用方提供可写 VTCM 区, 这里填静态字 + 动态 word[14]
 *
 * kernel 本体 (w4a16_v81deep_conv1x1_kernel.inc) 原样, 零改动。
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include <qurt.h>

#include "dc_parts.h"

/* ===== v73 ABI descriptors (逐字保留, _Static_assert 同原) ===== */
typedef struct {
  int32_t  *out_tile_ptr_table;
  uint32_t  out_table_stride_dwords;
  uint32_t  out_y_stride_words;
  uint32_t  n_tiles_pow2;
  int32_t   m_total_minus_step;
  uint32_t  k_total_bytes;
} HmW4A16OutDescT10;

typedef struct {
  int32_t  *act_ptr_pairs;
  uint32_t  n_act_pairs;
  uint32_t  act_table_y_stride_words;
} HmW4A16ActDescT10;

typedef struct {
  int32_t   out_check;
  uint32_t  out_rt_mask;
  int32_t   act_check;
  uint32_t  act_rt_base;
  uint32_t  filter_x_stride;
  uint32_t  pad14;
  uint32_t  alt_rt;
} HmW4A16MaskDescT10;

_Static_assert(sizeof(HmW4A16OutDescT10)  == 24, "out desc size");
_Static_assert(sizeof(HmW4A16ActDescT10)  == 12, "act desc size");
_Static_assert(sizeof(HmW4A16MaskDescT10) == 28, "mask desc size");
_Static_assert(offsetof(HmW4A16OutDescT10, out_tile_ptr_table)       == 0,  "ot0");
_Static_assert(offsetof(HmW4A16OutDescT10, out_table_stride_dwords)  == 4,  "ot4");
_Static_assert(offsetof(HmW4A16OutDescT10, out_y_stride_words)       == 8,  "ot8");
_Static_assert(offsetof(HmW4A16OutDescT10, n_tiles_pow2)             == 12, "ot12");
_Static_assert(offsetof(HmW4A16OutDescT10, m_total_minus_step)       == 16, "ot16");
_Static_assert(offsetof(HmW4A16OutDescT10, k_total_bytes)            == 20, "ot20");
_Static_assert(offsetof(HmW4A16ActDescT10, act_ptr_pairs)            == 0,  "at0");
_Static_assert(offsetof(HmW4A16ActDescT10, n_act_pairs)              == 4,  "at4");
_Static_assert(offsetof(HmW4A16ActDescT10, act_table_y_stride_words) == 8,  "at8");

#define HM_ROW4_PHASES 8u
#define HM_MASK_BYTES  32u
#define HM_EXTRA_BYTES 16u

extern void hmx_v81_w4a16_kernel(const HmW4A16OutDescT10 *out_desc,
                                 const HmW4A16ActDescT10 *act_desc,
                                 const uint8_t *packed_weight,
                                 const uint8_t *folded_bias,
                                 const HmW4A16MaskDescT10 *mask_desc,
                                 const uint32_t *extra_param);

/* kernel 本体: 原 closure_dsp/hmx_v81_w4a16_kernel.c 的 naked 函数, 原样 */
__attribute__((naked, aligned(64), noinline))
void hmx_v81_w4a16_kernel(const HmW4A16OutDescT10 *out_desc,
                          const HmW4A16ActDescT10 *act_desc,
                          const uint8_t *packed_weight,
                          const uint8_t *folded_bias,
                          const HmW4A16MaskDescT10 *mask_desc,
                          const uint32_t *extra_param) {
  __asm__ volatile(
      "{ r8 = memw(r4+#0x30) }\n"
      "{ p1 = tstbit(r8,#0x5)\n"
      "  if (p1.new) jump:t 1f\n"
      "  r7:6 = memd(r4+#0x8)\n"
      "  r11:10 = memd(r4+#0x0) }\n"
      "{ jump 1f }\n"
      "1:\n"
#include "w4a16_v81deep_conv1x1_kernel.inc"
  );
}

/* mask 静态字: contract.json native_static_words, word[14] 动态控制指针 */
static const uint32_t k_w4a16_mask[16] = {
  0u, 0x700u, 0u, 0x77Cu, 0u, 0u, 0x3FFu, 0u,
  0u, 0u, 0u, 0u, 0xA0u, 0u, 0u, 0u
};
static const uint32_t k_w4a16_extra[4] = { 1u, 1025u, 524u, 0u };

/* ===== invoke: 指针全外部, 表重写按本次槽地址, mask word[14] patch ===== */
int w4a16_invoke(const uint8_t* vtcm_act, const uint8_t* vtcm_weight,
                 const uint8_t* vtcm_bias, uint8_t* vtcm_out,
                 uint8_t* act_table_rw, uint8_t* out_table_rw,
                 uint8_t* mask_rw, uint8_t* extra_rw,
                 uint32_t m, uint32_t k, uint32_t n) {
  if (!vtcm_act || !vtcm_weight || !vtcm_bias || !vtcm_out ||
      !act_table_rw || !out_table_rw || !mask_rw || !extra_rw)
    return 0xA001;

  uint32_t k_t = k / 32, n_t = n / 32;
  if (m % 32 || k % 32 || n % 32) return 0xA002;
  uint32_t act_tbl_entries = HM_ROW4_PHASES * k_t;
  uint32_t out_tbl_entries = HM_ROW4_PHASES * n_t;

  /* 表重写: host 打包的 u32 byte-offset → 绝对 VTCM 指针 (kernel 逐 entry
   * mxmem(addr,route) 读)。act 表配本次激活槽, out 表配输出区。 */
  {
    int32_t* at = (int32_t*)act_table_rw;
    int32_t* ot = (int32_t*)out_table_rw;
    uintptr_t act_base = (uintptr_t)vtcm_act;
    uintptr_t out_base = (uintptr_t)vtcm_out;
    for (uint32_t i = 0; i < act_tbl_entries; ++i)
      at[i] = (int32_t)(act_base + (uint32_t)at[i]);
    for (uint32_t i = 0; i < out_tbl_entries; ++i)
      ot[i] = (int32_t)(out_base + (uint32_t)ot[i]);
  }

  /* mask 28B + word[14] → extra 区; extra 4 dwords (ABI 静态) */
  memset(mask_rw, 0, HM_MASK_BYTES);
  memcpy(mask_rw, k_w4a16_mask, sizeof(k_w4a16_mask));
  ((uint32_t*)mask_rw)[14] = (uint32_t)(uintptr_t)extra_rw;
  memset(extra_rw, 0, HM_EXTRA_BYTES);
  memcpy(extra_rw, k_w4a16_extra, sizeof(k_w4a16_extra));

  /* C4: CPU 写完表/mask/extra, FLUSH 到内存再让 HMX 读 */
  qurt_mem_cache_clean((qurt_addr_t)act_table_rw, act_tbl_entries * 4u,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t)out_table_rw, out_tbl_entries * 4u,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t)mask_rw, HM_MASK_BYTES,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t)extra_rw, HM_EXTRA_BYTES,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

  /* descriptor (ABI 常量段, 逐字保留: n_tiles_pow2=32, m_total_minus_step=8) */
  HmW4A16OutDescT10 out_desc = {
    (int32_t *)out_table_rw,
    n_t,
    n_t,
    HM_ROW4_PHASES * 4u,
    8,
    n_t * 32u
  };
  HmW4A16ActDescT10 act_desc = {
    (int32_t *)act_table_rw,
    k_t,
    k_t
  };

  hmx_v81_w4a16_kernel(&out_desc, &act_desc, vtcm_weight, vtcm_bias,
                       (HmW4A16MaskDescT10 *)mask_rw, (uint32_t *)extra_rw);
  return 0;
}
