/*
 * hvxhmx_v2.h — V2 LLM 算子公共库 umbrella 头
 * =====================================================================
 * 一行 include 拉入全部 V2 公共 API:
 *   #include "hvxhmx_v2.h"
 *
 * 模块组织 (P0/P1/P2):
 *   P0 (基础): norm + matmul
 *   P1 (高价值): flash_attn + rope
 *   P2 (函数族): unary + binary + softmax + ssm
 *
 * 设计原则:
 *   - 每个 .h 声明 hvhx_v2_* 公共符号, extern "C" 保护.
 *   - 每个 .c 用 #if defined(__HVX__) || defined(__hexagon__) 选 DSP HVX
 *     内核路径, #else 走标量 host fallback (数值一致, 供回归对比).
 *   - 内部 kernel 头 (internal 模块) 仅在 .c 的 DSP 分支里 include,
 *     不泄漏 HVX 原语到公共接口.
 *   - tile-major interleaved 数据布局 (HMX) 由 hvhx_v2_matmul.h 文档.
 *
 * 依赖: 仅 hvxhmx_types.h (uint32_t 等). 无 ggml / HAP / qurt 依赖.
 *
 * 编译: hexagon-clang -mv81 -O2 -mhvx -mhvx-length=128B -mhmx
 *        + V1 的 hvxhmx_runtime (hmx_enable_execution / hmx_runtime_setup).
 */
#ifndef HVXHMX_V2_H
#define HVXHMX_V2_H

#include "hvxhmx_v2_norm.h"        /* P0: RMSNorm/Norm/L2Norm */
#include "hvxhmx_v2_matmul.h"      /* P0: HMX tiled GEMM + dequant */
#include "hvxhmx_v2_flash_attn.h"  /* P1: FA tile 内核 (HVX+HMX) + ALiBi */
#include "hvxhmx_v2_rope.h"        /* P1: RoPE (NEOX/Normal/YaRN) */
#include "hvxhmx_v2_unary.h"       /* P2: exp/log/sqrt/sigmoid/sin-cos/... */
#include "hvxhmx_v2_binary.h"      /* P2: add/sub/mul/div */
#include "hvxhmx_v2_softmax.h"     /* P2: online softmax + mask prep */
#include "hvxhmx_v2_ssm.h"         /* P2: solve_tri + gated delta net */

#endif /* HVXHMX_V2_H */
