/*
 * hvxhmx.h — hvxhmx 库公共总入口
 * =====================================================================
 * 调用方只需:
 *     #include "hvxhmx.h"
 * 即可获得全部公共 API: 运行时 (hvxhmx_runtime.h) + HMX 算子 (hmx_kernels.h)
 * + HVX 算子 (hvx_kernels.h) + crouton 打包/解包 (hmx_crouton.h) + 公共类型.
 *
 * 使用流程:
 *   1. hmx_runtime_setup(2*1024*1024);          // HMX kernel 前必须调一次 (幂等)
 *   2. 调具体算子 (hmx_convf16 / hvhx_divide_u8 / ...);
 *   3. hmx_runtime_teardown();                   // 退出前释放
 *
 * 内部实现头 (hmx_common.h / hmx_fields.h / q6_intrinsics.h) 不在本头中,
 * 仅供库本身使用; 高级用户可单独 include hmx_fields.h 手工构造 crouton 字段.
 *
 * 完整使用说明见 USERGUIDE.md; 分模块 API 参考见 docs/api_*.md.
 */
#ifndef HVXHMX_H
#define HVXHMX_H

#include "hvxhmx_types.h"     /* 公共类型 / 常量 / 饱和宏 / 容差 */
#include "hvxhmx_runtime.h"   /* runtime 生命周期 + 电源 + 计时 */
#include "hmx_crouton.h"      /* crouton 打包/解包 helper (高级) */
#include "hmx_kernels.h"      /* HMX GEMM / 卷积 / depthwise / elementwise 族 */
#include "hvx_kernels.h"      /* HVX divide / activation / reduction / lookup_unpack / int8gemm */

#endif /* HVXHMX_H */
