# HVX Q4_0 反量化核设计 — 产出正确 pair-interleave（crouton）面

> 状态：**已落地全绿**（见 §6；2026-10-14 板 4/4 双门 + exec 3724ms）。原目标：把 w4 线 12.5s 反量化税（MATMUL 8.36s + lm_head 4.16s）从标量 LUT 换成 HVX，
> 且**直接产出 crouton 序**，规避 ggml HVX dequant 的 lane 序错误。

## 1. 问题定位：ggml HVX dequant 为什么 lane 序不对

现有标量 T017 反量化（`oplist_exec.c` w_q4 分支）已经在**写的时候就**产出 crouton 序（见 §3 的 little-endian 分析），
所以当前板跑 cos 0.9998 是对的。ggml 那条 HVX dequant 是为 `core_dot` HMX 核服务的，它产出的是
**row-major pair 面**（一对 K 行不交织、分成两段 32 元素块，即「hi/lo 平面」），而我们 crouton GEMM
（`hmx_matmul`）要吃的是 **element-interleaved**（一对 K 行逐列交织）。

- 平面形（ggml core_dot 要的，错）：`[e0 e1 ... e31 | o0 o1 ... o31]`
- crouton 形（我们要的，对）：`[e0 o0 e1 o1 ... e31 o31]`

一句话：ggml 用的是 `vdeal`（把交织向量拆成奇偶两段 = 反交织），crouton 要的是 `vshuff(v1,v0,rt8)`
（两段拼回逐元素交织）。本核在最后一拍用后者即可，lane 序**完全由我们掌控**。

## 2. 输入/输出契约（三条独立来源互相印证）

**输入：每个 32×32 权重 tile = 640B**（`repack_q4_0_tiles` 发射格式）：

- 512B 量化：`qt[cp*32 + col]`，`cp=0..15`（K 对）、`col=0..31`（oc/N 列）
  - 字节 = `[hi nibble = q(2cp+1, col)][lo nibble = q(2cp, col)]`
  - 即：低 nibble = K 行 `2cp` 的量化值，高 nibble = K 行 `2cp+1`。
  - 定论来源：`weight_pack.cc` `unpack_tiled_q4_0` L218-219（`tile_quants[row][2cp]=byte&0xf`）。
- 128B 缩放：`dv = qt + 512`，32 个 f16，`dv[col]` = N 列 col 的缩放因子。

**反量化（逐元素）**：`deq(k, col) = f32_to_f16_RNE( f16_to_f32(d_col) × (q(k,col) − 8) )`，
`q ∈ {0..15}`。f16→f32 精确、`(q−8)` 小整数精确、f32 乘积精确，唯一舍入点 = f32→f16 RNE
（与 `hb_f32_to_f16` / `hvx_vec_f32_to_f16` 同算法，见 `hvx_binary.c`）。

**输出：2048B crouton = 16 个 HVX 向量（128B/64f16）**，向量 `cp`（K 对 cp）：

```
V[cp] = [deq(2cp,0), deq(2cp+1,0), deq(2cp,1), deq(2cp+1,1), …, deq(2cp,31), deq(2cp+1,31)]
```

crouton 位公式（`hmx_crouton.h`）：`pos(k,col) = (k/2)*64 + 2*col + (k&1)`，
权重面 row=ic(K)、col=oc(N)。校验：`pos(2cp,col)=cp*64+2col`（偶 lane）、`pos(2cp+1,col)=cp*64+2col+1`（奇 lane）。
little-endian 下标量 `u32[cp*32+col] = (deq_hi<<16)|deq_lo` 作为两 f16 即 `[deq_lo, deq_hi] = [deq(2cp), deq(2cp+1)]`，
与上面交织完全一致 —— 这就是标量版正确、ggml HVX 版错误的根。

## 3. HVX 指令编排（每个 K 对 cp → 1 个 128B 输出向量）

对每 cp（32 列 → 64 f16 输出），一次 32 字节量化读 + 32 f16 缩放读：

1. **nibble 拆**：`Vlo = vand(Vq, 0x0F)`（u8，32 lane）、`Vhi = vlsr(Vq, 4)`（u8）。
2. **u8→u16→f32，减 8**：`Vlo_f32 = f32(q_lo) − 8.0`，同 `Vhi`（u16 零扩展 `vzxt` + int→f32，或直接 vlut4 把 nibble 映射成 `f16(q−8)` 再 f16→f32）。
3. **缩放（f32，逐列）**：`D_f32 = hvx_vec_f16_to_f32(d)`（32 f32，lane=col）；`Vlo_prod = hvx_vec_mul_f32_f32(Vlo_f32, D_f32)`，同 `Vhi`。
4. **f32→f16 RNE**：`Vlo_f16 = hvx_vec_f32_to_f16(...)`、`Vhi_f16`（32 f16 各）。
5. **交织（关键一拍）**：`W = Q6_W_vshuff_VVR(Vhi_f16, Vlo_f16, rt8)` → 64 f16 `[lo0,hi0,lo1,hi1,…]`，即 crouton 序。
   - 惯用 rt8 = `-2`（`hmx-utils.h:187` `Q6_W_vshuff_VVR(v1,v0,-2)` 现成），落定以 disasm 对照为准。
6. **存**：128B 对齐写。

**向量化放大**：一次读 4 个 cp 的量化（128B = 1 HVX 向量），拆出 4×32 nibble，缩放共享同一份 32 f16（splat 到 4 段），
一次产 4 个输出向量 —— 每 tile 4 次迭代覆盖 16 cp，摊销缩放装载。第一版可先按 cp 循环（正确性优先），再向量化。

每 tile 总 op 预算（向量化后粗估）：~4×(拆 nibble + 2 转 f32 + 2 乘 + 2 转 f16 + 1 交织 + 1 存) ≈ 40 拍级，
对 12.5s 的 199+32 个 tile 是微秒级，预期 q4 线 15.09s → 逼近 f16 线 2.76s 地板（MATMUL 反量化税 8.36s 应清零）。

## 4. 位级保真是硬门（不是可选项）

反量化值直接喂 HMX GEMM。两个必须满足：

1. **f32→f16 舍入必须与标量逐 bit 一致**：复用 `hvx_vec_f32_to_f16` / `hb_f32_to_f16`（RNE，与
   `oplist_exec` 标量 `f32_to_f16` 同源同算法，subnormal 已一起修，见 `hvx_binary.c` L162-163）。
   不要用 `Q6_Vhf_vmpy_VhfVhf`（f16×f16 直接乘，内部舍入非 f32 中间，会引入 ULP 差）。
2. **nibble 面映射不能 swap**：lo nibble = K 行偶、hi nibble = K 行奇，落到偶/奇 lane（§2 已锁死）。
   一旦 swap，cos 会崩（等价于权重 K 维相邻行互换）。

## 5. 验证方案

1. **host 位级对拍**（零硬件）：新核 vs 标量 T017 参考，随机 tile（含 subnormal scale）逐 f16 位级全等。
   标量参考 = 现板跑路径，是唯一真值。
2. **探针微 blob**（板 d0f1784）：单 matmul op 喂合成 Q4_0 权重，host 侧比对输出 crouton 字节。
3. **全图双门**：重跑 q4 线 4 prompt，冻结门 pooled cos≥0.98 + top1≥20/32 不回退；op_ts 确认
   code24/35 反量化税清零。

## 6. 风险

- **rt8 常数**：`vshuff` 交织控制字需 disasm 对照（`-2` 是现成惯用，但 f16 半字粒度要实测确认）。
- **u8→f32 的转换路径**：`vzxt`+int→f32 与 vlut4 两条路二选一，vlut4 更省但需确认 4-bit 索引语义。
- **负 scale 的 subnormal**：`d×(q−8)` 可能落 subnormal，务必走 RNE 路径（已修）不能回退旧 `man>>(shift+1)`。

---

## 6. 落地结果（2026-10-14，commit 待记）

> 状态：**已落地，全绿**。核 = `kernels/src/hvx/hvx_binary.c` `hvhx_v2_dequant_q4_0_crouton`，
> 调用点 = `oplist_exec.c` w_q4 分支（MATMUL code24 与 lm_head code35 共用，唯一反量化位点）。

### 6.1 板探针实证的 HVX 语义表（调试中裁决，防再踩）

| 原语 | 实测语义（板上 dump 证实） |
|---|---|
| `Q6_W_vshuff_VVR(A,B,-1)` | 字节交织：out.lo[2j]=B[j], [2j+1]=A[j]（前后半各 64B） |
| `Q6_W_vshuff_VVR(A,B,-2)` | u16 粒度交织 |
| `Q6_W_vshuff_VVR(A,B,-3)` | **仍是字节交织**（不是 u32！） |
| `Q6_W_vshuff_VVR(A,B,-4)` | **u32 粒度交织**（lo=[B0,A0,B1,A1..]，hi 取后半） |
| `Q6_Wuh_vzxt_Vub` / `Q6_Wuw_vzxt_Vuh` | **奇偶拆分**（vunpack），非位置序：lo=偶字节/lane，hi=奇 |
| `hvx_vec_f16_to_f32(v)` | 位置序 C 序：lo = f32(v[0..31]) |
| `hvx_vec_f32_to_f16(v0,v1)`（vdeal 版） | **拼接** [f16(v0)‖f16(v1)] |
| `hvx_vec_f32_to_f16_shuff(v0,v1)` | **交织** [f16(v0)[j], f16(v1)[j], …] = crouton 序 |

三个 bug 依次：①vzxt 当位置序用（实为奇偶拆分）→ 重排数据流；②`-3` 当 u32 粒度（实为字节，
u32 是 `-4`）→ 换立即数；③收尾误用 vdeal 版（拼接，实需交织）→ 换 `_shuff` 版。
设计 §1 的预判（ggml 用 vdeal 产出平面形）方向正确，但本库 helper 命名易反着记：**_shuff=交织、
vdeal=拼接**。

### 6.2 最终数据流（全部原语板实证）

每 tile（640B→2KB）：`v_sc=vmemu(qt+512)`；`scale=lo(f16_to_f32(v_sc))`（lane n=d_n）。
每 g（128B）：`v_e=vand(vq,0x0F)`（K 偶行）、`v_o=vlsr(vq,4)`；`mix=vshuff(v_o,v_e,-1)`。
每 h：`w16=vzxt_u8(half)`（lo=K 偶行 u16 [cpA‖cpB]，hi=K 奇行）；
`w32=vzxt_u16(·)` 按 n 奇偶再拆；`rec=vshuff(hi,lo,-4)` 复原 n 序（lo=cpA，hi=cpB）。
每 c：magic vor(0x4B000000)−(2^23+8) 得精确 f32(q−8)，×scale（f32 精确），
`f32_to_f16_shuff(pe,po)` 交织收尾 = `V[cp]`，`vmem` 写 `dst[(t*16+g*4+h*2+c)*64]`。

### 6.3 门与锚

- host 位级门（`.scratch_013/test_dequant_q4_crouton.c`，vs T017 LUT 逐字参考）：**PASS**。
- 板整核探针（`.scratch_013/probe_kernel.c`，内联全核 vs 核内 ref，合成布局 tile+LCG 随机 tile
  含 ±0/subnormal/min normal/±max scale 与尾区垃圾）：**位级一致 + 布局锁死 OK**。
- q4 线 4/4 双门（judge vs cpu_golden/q4_0）：cos 0.999802/0.999763/0.999963/0.999954，
  top1 31/32、31/32、32/32、31/32（冻结门 0.98/20 大幅超线）。
- 性能（op_ts, `gehtp_prof.py`）：**exec wall 3724ms**；对 a08ab8b q4 锚 15.09s = **−75.3%**，
  对 007 q4 锚 25.7s = −85.5%；离 f16 线地板 2.76s 仅 1.35×。12.5s 标量反量化税并入
  HVX 核后消失（MATMUL_F16 合计 1.73s / SPLIT 0.66s，无独立 dequant 项）。
- 板资产：`hvxhmx23/cl020/{out_q4v13q_0..3}.f16.raw, op_ts_q4v13q_0..3.bin}`。
