# V2.2 单元功能库总览

`libhvxhmx_v22.so` = **V2.1 算子层** (`hvhx_v2_*` 60 符号, 原样) + **8 个已设备闭合的
工程单元** (src/v22/, 从 4 个闭合模块移植, 源模块不动)。一个头拉全:

```c
#include "hvxhmx_v22.h"
```

## 单元 ↔ 头 ↔ 源模块 ↔ 测试

| 单元 | 头 | 源模块 (设备 PASS 存档) | API 手册 | 设备用例 |
|------|----|------------------------|----------|----------|
| U1 wtcache (VTCM 权重 pin/ring) | `wtcache.h` | wtcache_pin_v81 (T1-T9 9/9) | [api_v22_wtcache.md](api_v22_wtcache.md) | examples/16 |
| U2 dcmem (arena/文件/DMA 流/W4 引擎封装) | `dc_parts.h` | dualcore_v81 · 4C | [api_v22_dcmem.md](api_v22_dcmem.md) | examples/17,20,22 |
| U3 dcthread (QURT 线程/同步) | `dc_threads.h` | dualcore_v81 · 4C | [api_v22_dcthread.md](api_v22_dcthread.md) | examples/22 |
| U4 w4a16 (W4A16 HMX GEMM 引擎) | `dc_parts.h` | t10 + htpw4a16_v81 (.inc 同源) | [api_v22_w4a16.md](api_v22_w4a16.md) | examples/17,21 |
| U5 gdnsm (GDN 递归状态机) | `gdn_sm.h` | gdn_sm_v81 (G1-G9) | [api_v22_gdnsm.md](api_v22_gdnsm.md) | examples/19 |
| U6 oplist (blob v1 解析/执行) | `oplist_parse.h` `oplist_exec.h` `wt_sha256.h` | wt_repack_v81 (W1-W5) | [api_v22_oplist.md](api_v22_oplist.md) | examples/21 |
| U7 dualdom (双域分片执行) | `dd_worker.h` | dualdomain_v81 (D1-D7, 2.001×) | [api_v22_dualdom.md](api_v22_dualdom.md) | examples/20 |
| U8 smallm (小-M pad-256 GEMV) | (U4 用法) | htpw4a16_v81 MODULE A | [api_v22_smallm.md](api_v22_smallm.md) | examples/18 |

## 构建

```bash
./build_libs.sh                    # → lib/libhvxhmx_v22.so + 签名版 (vgather 检查)
cd examples && ./build_examples.sh # 编译/签名/推送/实机跑 01-22, 结果落 results/
./build_examples.sh 17             # 单跑一个
```

- 链接: `-lc -ldl -lgcc`。**不要** `-lqurt` — libqurt.a 非 PIC (R_HEX_32_6_X),
  qurt/HAP 符号运行时由 CDSP 进程解析 (全部闭合模块同款链法)。
- V2.1 例程 01-15 对 V2.2 库全量回归 (编译已验证, 设备结果见 PERF_REPORT.md)。

## cache 协议四铁律 (混访 VTCM/DDR/DMA/HMX 前必过)

| # | 铁律 | 违反后果 (实测) |
|---|------|----------------|
| ① | CPU 写完 DDR、DMA bypass 读之前 → `dc_clean_ddr` | DMA 读到 dcache 驻留旧行 |
| ② | `wtcache_open` 末尾 memset(VTCM) 留脏零行 → 已内置全 VTCM `FLUSH` (V2.2 修复) | 脏行驱逐覆盖 HMX 直写面 (dualdomain run3 根因) |
| ③ | DMA 写完的 DDR、CPU 读之前 → `QURT_MEM_CACHE_INVALIDATE` | CPU 读回 dcache 驻留旧值 (400× 假读) |
| ④ | 退出必 `wtcache_close` / `wt_exec_shutdown` — **PASS/FAIL 两路都要** | VTCM/HMX 占死域, 下进程连文件都建不出 |

## VTCM/引擎硬约束 (实测)

- HMX `mxmem` 操作数必须 **2KB 对齐** (T10 集成门发现; `dc_w4_carve` 已内置)。
- W4A16 引擎 **M 必须 256 倍数** (`m_t=8` 硬约束; 小 M 用 U8 pad-256)。
- VTCM 16MB @0xFF000000; DMA 引擎单实例 (跨线程 submit 必须 mutex);
  HMX 全局锁 (线程并发不缩放, 缩放靠双域两进程)。
- `alloc_data`/DMA 描述符 **128B 对齐** (HVX aligned load / desc 硬件要求)。

## 性能基线 (源模块设备闭合值; V2.2 复测见 PERF_REPORT.md)

| 项 | 值 |
|----|----|
| UserDMA DDR→VTCM / VTCM→DDR | 65.5 / 69.9 GB/s |
| ring overlap (DMA×compute) | 0.968 |
| W4A16 256³ (K=256) | invoke ~µs 级, bit-exact vs QNN gold |
| W4A16 K=2560 N=2560 | invoke 1047 µs ≈ 3.2 TFLOPS (op 全路径 32 ms 含权重 restage) |
| 双域并发 | 2.001× (对半切) |
