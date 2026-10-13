// requant_w4a16.h — W4A16 供给侧生成器 (B 段 T004, 013-executor-hoist)
//
// OP_MATMUL_W4A16 (opcode 1) 的供给契约: 编译器须产四张表, 运行期 exec_matmul
// 按 9-arity 直读 (args[6..8] = bias/atbl/otbl 裸 slot id, 见 op_matmul.cpp
// T006 与 oplist_exec.c T009)。本文件是这四张表字节布局的唯一事实源 (与
// kernels/docs/api_v22_w4a16.md + kernels/host/vendor/PROVENANCE.md Route B 互证)。
//
// ── 四表契约 ────────────────────────────────────────────────────────────────
//   wt    (packed_w)   int4[-7,7] 对称权重, K*N/2 字节
//                      布局 = pack_w4_kblock32_nmajor_k4_lohi XOR 0x88
//   bias  (folded_bias) (N/32)*512 字节, 每 32 列 tile 512B (= 8 u16/列)
//                      kernel 语义: out[m,n] += bias_fold[n]
//   atbl  act 表       8*(K/32)*4 字节, 每项 u32 LE = i*(M*8)  (byte offset 进 act 面)
//   otbl  out 表       8*(N/32)*4 字节, 每项 u32 LE = i*(M*8)  (byte offset 进 out 面)
//   act   (运行时输入) M*K*2 字节 crouton16_row4 面, 非供给侧槽
//
// kernel 数值契约: out[m,n] = round(Σ_k (act[m,k]-32768)·W[k,n]/7) + bias_fold[n] + 32768
//   (act u16 对称 offset 32768; 权重 int4 对称 /7 归一)
//
// ── 降级范围 (2026-10-13 重定 B 段 = "机制 only") ───────────────────────────
// 原生 W4A16 因 M%256 硬约束 (M=32/128 实测 0xA002 失败) 判 NO-GO, 全图 0.8B
// E2E 移出 013 范围。故本生成器只交付**可位级验证**的部分:
//
//   ✅ atbl_for / otbl_for  — 纯闭式 (atbl[i]=i*(M*8)), 已对 s256/s2560 oracle
//                              bit-exact 命中 (见 test_requant_w4a16.cpp)
//   ⛔ packed_w (wt)        — GAP: 重量化 (Q4_0 per-32 F16 scale → 对称
//                             int4[-7,7]) 不可逆, 且 pack_w4_kblock32_nmajor_k4_lohi
//                             的字节序 + XOR 0x88 的 nibble 映射需 oracle 反推
//                             (原生成器 DFlashSpecCppRunner/htpw4a16_v81 本机不存在)
//   ⛔ folded_bias (bias)   — GAP: 折叠语义 = 吸收后续 Add 节点 (对齐 fc_bias_fold
//                             口径) + 每列 8 u16 的 tile 内部布局, oracle 已失
//
// packed_w / folded_bias 的精确格式见本文件尾注; 合成 M=256 blob 门 (T010) 的
// 这两槽直接复用板上 oracle .raw, 不经本生成器。

#pragma once

#include <cstdint>
#include <vector>

namespace requant_w4a16 {

// act 表: 8*(K/32) 项, 每项 u32 = i*(M*8) (byte offset 进 act 面)。
// 前置: K % 32 == 0 (W4A16 形状契约, 非对齐由 emit 侧 M/K/N 门槛拦截)。
std::vector<uint32_t> atbl_for(uint32_t M, uint32_t K);

// out 表: 8*(N/32) 项, 每项 u32 = i*(M*8) (byte offset 进 out 面)。
// 前置: N % 32 == 0。注意 K≠N 时表长随 N 走 (历史 otbl 误按 K 计长的 bug 修正)。
std::vector<uint32_t> otbl_for(uint32_t M, uint32_t N);

} // namespace requant_w4a16

// ── GAP 文档 (packed_w / folded_bias, oracle 已失, 供 T010 及后续 oracle 反推) ──
//
// packed_w (wt) 目标字节布局, K*N/2 字节:
//   · 对称量化: 每 (k,n) 权值 w ∈ Q4_0 反量化后按 round 归到 int4[-7,7];
//     反量化用 Q4_0 per-32 的 F16 scale (沿 K 维每 32 元素一个 scale), 与
//     GGUF Q4_0 block (18B/32 元素) 的读取层 gguf_wtlink 对接。
//   · 打包序: kblock32 (K 按 32 分块) → nmajor (N 维主序) → k4_lohi (块内 4-bit
//     nibble 低-高)。每个 byte 装两个相邻 nibble (lo, hi), 全表 XOR 0x88
//     (= 每 nibble XOR 0x8, 把 int4 有符号补码翻成 +8 偏置编码)。
//   · 不可逆: 是"重量化" (Route B) 而非"重排" (Route A tile-major), 无 bit-exact
//     round-trip 验收, 只能 dequant 误差有界 + 喂闭合 kernel 对 scalar gold。
//
// folded_bias (bias) 目标字节布局, (N/32)*512 字节:
//   · 每 32 列 tile 一段 512B = 8 u16 × 32 列。内容 = 该 tile 的 per-column
//     bias_fold[n] (含 Q4_0 scale 归一 + 后续 Add 节点折叠进来的偏置)。
//   · 折叠口径: 对齐 fc_bias_fold 的"吸收下游 Add"语义, 0.8B 正典须
//     --no-pass fc_bias_fold 防双折 (contract-w4a16-op.md 供给侧节)。
//   · 内部 u16 排布 (8 u16/列的哪些位是 scale、哪些是 bias、含 32768 offset 与否)
//     需 oracle 反推。
