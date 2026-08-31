# hvxhmx_libsV2.3 — V2.2 全量 (融合布局) + 9 个新单元

Hexagon V81 (52f67807) 单元功能库。**V2.3 = V2.2 全部功能 + src/v22 融合进原目录 +
9 个新工程单元 (U9-U17)**, 单库单签名 `lib/libhvxhmx_v23.so`:

- **V2.1 算子层**: `hvhx_v2_*` 60 符号 (norm/GEMM/FA/RoPE/unary/binary/softmax/SSM),
  见 `docs/api_v2_llm.md` 等
- **V2.2 工程单元 (U1-U8)**: wtcache / dcmem / dcthread / w4a16 / gdnsm / oplist /
  dualdom / smallm — 手册 `docs/api_v22_*.md` (源已并入 runtime/hmx/hvx 原目录)
- **V2.3 新单元 (U9-U17)**: 见下表, 总览 [docs/api_v23_overview.md](docs/api_v23_overview.md)

| 单元 | 一句话 | 手册 |
|------|--------|------|
| U9 fence | 方向对偶 cache fence 决策表 (四铁律收敛) | [api_v23_fence.md](docs/api_v23_fence.md) |
| U10 arena | DDR/VTCM 双池对齐 arena (free+合并) | [api_v23_arena.md](docs/api_v23_arena.md) |
| U11 harness | 对拍框架 (ex_* + sha256 case 流) | [api_v23_harness.md](docs/api_v23_harness.md) |
| U12 wpool | 常驻 worker 池 + hmx_lock 交接 (4.4× vs spawn) | [api_v23_wpool.md](docs/api_v23_wpool.md) |
| U13 pxbridge | f32↔f16↔INT16 桥 (unipolar 契约) | [api_v23_pxbridge.md](docs/api_v23_pxbridge.md) |
| U14 gdntree | 树形 GDN 闭式 oracle + f16 kernel (cos=1.0) | [api_v23_gdntree.md](docs/api_v23_gdntree.md) |
| U15 kvcache | KV 槽缓存 (append/scatter/posmap/DMA 读回) | [api_v23_kvcache.md](docs/api_v23_kvcache.md) |
| U16 graphstep | oplist 整步/分段执行恒等 + 统计 | [api_v23_graphstep.md](docs/api_v23_graphstep.md) |
| U17 gemmdispatch | MatMul 三路由决策 (M=512 拆块语义) | [api_v23_gemmdispatch.md](docs/api_v23_gemmdispatch.md) |

## 30 秒上手

```c
#include "hvxhmx_v23.h"             /* V2.1 算子 + U1-U17, 一个头 */

hmx_runtime_setup(2 * 1024 * 1024);           /* V2.1 路径照旧 */
hvx_v2_rmsnorm_f16(...);

/* V2.2 路径照旧 (wtcache/dc_w4_invoke 见 api_v22_*) */
/* V2.3 新路径 */
fence_handoff(buf, bytes, FC_CPU, FC_DMA, FM_DDR);   /* 铁律① 一行 */
int r = gemm_route_for(m, k, n);                     /* W4A16/SMALLM/DENSE */
```

## 构建 / 测试

```bash
./build_libs.sh                     # lib/libhvxhmx_v23.so + SWIV 签名 (vgather=0)
cd examples && ./build_examples.sh  # 01-33 编译/签名/推送/实机跑 + results/ 汇总
gcc host/gtest_v23.c -o /tmp/g && /tmp/g    # host 模拟 18 门
```

- 设备仅 **52f67807**; 链接 `-lc -ldl -lgcc` (**无 -lqurt**, 运行时解析)。
- 验收 (2026-08-16): **33/33 例全绿, 181 门 PASS 0 FAIL** (+例 15 内嵌 49 项, 共 230)。
- 性能报告: [PERF_REPORT.md](PERF_REPORT.md)。新手上机: [TUTORIAL.md](TUTORIAL.md)。

## 目录

```
include/   公共头 (V2.1 全部 + U1-U17 单元头 + hvxhmx_v22.h + hvxhmx_v23.h)
src/runtime  wtcache/dc_*/oplist_*/wt_*/dma_utils + fence/arena/wpool/kvcache/
             harness/gemm_dispatch        ← 融合布局, 无 v22 子目录
src/hmx/   conv 族 + w4a16_driver_dc (+.inc)     src/compat/  V2.1 兼容层
src/hvx/   V2.1 算子 + gdn_kern/gdn_ref + gdn_tree/pxbridge
host/      pack_oplist / wt_inspect / analyze_dd / gtest_v23 / 资产生成
assets/    s256 · s2560 · smallm                build/  .o 与 test_XX.so
examples/  01-15 V2.1 回归 + 16-22 V2.2 单元 + 23-31 V2.3 单元; README/EXAMPLES/TEST_CASES
docs/      V2.1 文档 + api_v22_*.md ×9 + api_v23_*.md ×10
```
