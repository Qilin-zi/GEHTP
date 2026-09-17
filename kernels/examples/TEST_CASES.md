# 示例测试用例详解 (TEST_CASES)

每个示例**在设备上实际验证了什么**. 配合 [README.md](README.md) 的索引和运行说明阅读 ——
README 回答"怎么跑", 本文回答"每个用例到底测了什么、PASS 代表什么".

> 共 14 个示例, 设备实测 **66 项 PASS / 0 FAIL** (52f67807, 2026-08-10).
> 容差口径与权威回归 `hvxhmx_kernel/test/test_all_hmx.c` 一致: 整数族 exact (err=0), fp16 族 ≤1 ULP.

---

## 通用模式

所有示例的 `main.c` 共用同一骨架 (见 [common/example_util.h](common/example_util.h)):

1. `ex_open_result("示例名")` — 在设备上打开 `/data/local/tmp/hvxhmx_libs/<名>.txt`
2. `hmx_runtime_setup(2*1024*1024)` — HMX/HVX 上电 + 申请 VTCM (HVX-only 示例也调, 顺带上电)
3. `ex_fill_*` 用**固定种子 LCG** 填 buffer (可复现, 不依赖 rand)
4. 调被测 kernel
5. **标量 golden**: 普通 C 嵌套循环算期望值 (不调任何库函数, 纯标量)
6. 逐元素比 `|out - gold|`, 取 max → `ex_check(label, maxerr, tol)` (≤ tol 即 PASS)
7. `return ex_summary()` — 有 fail 返回 1, 全 pass 返回 0

**关键**: golden 是独立标量实现, 不是"用库算两遍". 所以 PASS 真正证明"库的 HVX/HMX 输出 == 数学定义".

---

## 01_runtime_init — runtime 生命周期冒烟

**测什么**: 不调任何 kernel, 只验证 runtime 三件套工作.

| 检查项 | PASS 判据 | 证明 |
|--------|-----------|------|
| `hmx_runtime_setup(2MB)` | 返回 0 | CDSP/fastrpc 就绪, 上电+VTCM 申请成功 |
| `get_vtcm_base()` | 非 NULL | VTCM 真的拿到了 (本机固定给整块 16MB) |
| `get_vtcm_size()` | ≥ 2MB | 申请量被满足 |
| `hmx_perf_now_us()` 两次 | 单调递增 | HAP qtimer 计时器可用 (后面所有 benchmark 依赖它) |

**为什么单独一个示例**: 任何 HMX kernel 前必须 setup (否则 CX_FAULT). 这个示例是"环境健康度"探针 ——
如果它 FAIL, 后面 13 个全跑不了, 先修环境 (host 侧重连 CDSP).

---

## 02_convf16_gemm — fp16 GEMM 单 tile

**测什么**: `hmx_convf16(act, wgt, bias, out, 32, 32, 32)` 一个 32×32×32 fp16 矩阵乘.

- **数学**: `out[m,n] = bias[n] + Σ_k act[m,k]·wgt[k,n]` (fp32 acc → cvt.hf 截断)
- **golden**: 标量 fp32 累加后 `(__fp16)` 强转
- **容差**: raw `fabsf(out-gold) ≤ 1` (fp16 systolic 截断到 1 ULP)
- **数据**: act/wgt ∈ ~[-0.5, 0.5] (种子 7/9, scale 0.01), 避开 ±1.0 denormal 边界
- **PASS 证明**: 真 HMX fp16 硬件路径数值正确 (不是 HVX 仿真)

**这是 fp16 族的基础正确性**. 其余 fp16 几何变体 (1x1/stride2/5x5/dilate/...) 在单 tile GEMM 层数学等价,
test_all_hmx 覆盖了它们的几何差异.

---

## 03_convbbb_int8 — u8×u8→u8 GEMM

**测什么**: int8 族基础型 `hmx_convbbb`.

- **数学**: `out[m,n] = sat_u8( bias[n] + Σ_k act[m,k]·wgt[k,n] )`, int32 累加 + u8 饱和
- **golden**: 标量 int32 累加, 手写 `<0→0, >255→255` 饱和
- **容差**: **0 (exact)**
- **PASS 证明**: int8 走 HVX `vmpyacc` + `vasr_sat` 路径逐元素 exact (本设备 HMX int8 silent NOP, 库已切 HVX)

注意 bias 是 `int32_t` (int8 族的 bias 类型, 与 fp16 族的 `__fp16` bias 不同).

---

## 04_convhbh_u16 — u8×i8→u16 GEMM (宽动态)

**测什么**: 两个 u16 输出族 —— `hmx_convhbh` 和 `hmx_convhhh` (同数学, 写回格式 :2x1 vs :2x2 不同).

- **数学**: `out[m,n] = sat_u16( bias[n] + Σ_k act_u8[m,k]·wgt_i8[k,n] )`
- **golden**: int32 累加 + u16 饱和
- **容差**: **0 (exact)**
- **PASS 证明**: u16 输出族 (宽动态范围, 不 sat 到 u8) 数值正确; 两种写回格式数学一致

wgt 是 `int8_t` (带符号), act 是 `uint8_t`. 这是 hbh/hhh 族的精度组合.

---

## 05_i16_weight_convs — i16 权重族全家桶

**测什么**: 一次验证 4 个 i16 权重族, 都是 `u8 act × i16 wgt`:

| 函数 | 输出 | 格式 |
|------|------|------|
| `hmx_convbcb` | u8 (sat) | — |
| `hmx_convbnb` | u8 (sat) | 基本型, 无 sparsity |
| `hmx_convhch` | u16 | :2x2 写回 |
| `hmx_convhnh` | u16 | :2x1 写回 (数学同 hch) |

- **golden**: int32 累加 + u8/u16 饱和 (两套 golden 函数)
- **容差**: **0 (exact)**
- **PASS 证明**: i16 权重族 (HVX 内部把 i16 拆半处理) 全部 exact; bcb/bnb 数学等价, hch/hnh 数学等价

i16 权重比 i8 权重慢 (~11-16us vs 6us, 见 performance.md), 因为双倍带宽. 能用 i8 就别用 i16.

---

## 06_dwconv — 深度卷积 (fp16 + u8)

**测什么**: 3×3 深度可分离卷积, 每通道独立空间卷积.

- **布局**: `act[H][W][C]` 行主序 (无 padding), `wgt[C][9]` 通道主序, `bias[C]`
- **数学**: `out[h,w,c] = bias[c] + Σ_{kh,kw} act[h+kh, w+kw, c]·wgt[c,kh,kw]` (越界样本跳过, clamped 不补零)
- **golden**: 标量嵌套 h/w/c/kh/kw 循环, 边界 if 检查
- **容差**: fp16 ≤1, u8 exact
- **PASS 证明**: depthwise 路径 (空间卷积, 不是 GEMM) 两精度都对

H=5, W=5, C=9 (小空间, 9 通道). depthwise 在网络中占比小, 标量路径性能可接受 (见 performance.md).

---

## 07_add — 元素级 fp16 加法 (残差)

**测什么**: `hmx_add(a, b, bias, out, 32, 32)`.

- **数学**: `out = max(0, a + b + bias)` (含 ReLU, 残差连接用)
- **golden**: 标量加法 + ReLU
- **容差**: Q10 定点误差 ≤ 1 (`err = (int)(fabsf(d)·1024)`, 即 |d| < 1/1024)
- **PASS 证明**: elementwise fp16 加法 + ReLU 正确

---

## 08_divide — HVX 整除 5 变体

**测什么**: 全部 5 个 HVX 除法函数, 含除零饱和.

| 函数 | 精度 | 除零行为 | 舍入 |
|------|------|----------|------|
| `hvhx_divide_u8` | u8 | →0xFF | 截断 (=floor, 无符号) |
| `hvhx_floor_divide_u8` | u8 | →0xFF | floor |
| `hvhx_divide_u16` | u16 | →0xFFFF | 截断 |
| `hvhx_floor_divide_u16` | u16 | →0xFFFF | floor |
| `hvhx_divide_flat_i32` | i32 | →±INT32_MAX | **round-to-nearest** (不是截断!) |

- **golden**: 标量 C `/` 运算 (u8/u16 截断 = floor; i32 用 `·2 + b ±|b|) / (2b)` 四舍五入)
- **容差**: u8/u16 exact; i32 ≤1 (round 实现差异)
- **数据**: b ∈ [1,15]/[1,1000] 避免大量除零, 但**故意塞 8 个 b=0** 测饱和
- **PASS 证明**: 倒数法除法 (u8/u16) 和 shift-subtract 除法 (i32) 全部正确, 除零饱和到位

⚠️ **i32 是 round-to-nearest 不是截断** —— 这是历史上踩过的坑 (见上层 SUMMARY). golden 必须按四舍五入算.

---

## 09_activation — HardSwish + PReLU

**测什么**: 两个 HVX 激活函数.

- **`hvhx_hardswish_flat_u16`**: MobileNetV3 HardSwish. 输入 u16 按 **int16 二补码**解读 (Q12 定点),
  `f(x) = x·clamp(x+3, 0, 6) / 6`. golden 用库内联 `hvhx_hardswish_scalar`. 向量路径用 `1/6 ≈ 2731/16384`
  近似 + 截断, 与标量差 ≤2 LSB (容差 2).
- **`hvhx_prelu_u8`**: PReLU. u8 域零点 0x80 (偏移二进制), slope Q7. golden 用 `hvhx_prelu_scalar_u8`.
  slope=128 (≈1.0, 近线性).
- **容差**: hardswish ≤2 LSB (近似误差), prelu ≤1
- **PASS 证明**: 激活函数定点近似在容差内

⚠️ 输入 u16 **不是无符号数**, 是 int16 二补码的位模式. golden 必须按 `(int16_t)in` 解读.

---

## 10_reduction — 沿 depth 归约 5 变体

**测什么**: 沿 depth 维归约, 输入 flat `[hw][d]`, 输出每行一个结果.

| 函数 | 精度 | 输出 |
|------|------|------|
| `hvhx_argminmax_depth_crouton_b` | u8 | per-row min/max + 各自 idx |
| `hvhx_argminmax_depth_flat_h` | u16 | per-row min/max + idx |
| `hvhx_find_max_and_index_in_depth_b` | u8 | per-row max + idx |
| `hvhx_top1_qu8_dLE32_cr2flt` | u8 | per-row top-1 (val + idx) |
| `hvhx_reducesum_depth_u8` | u8 | per-row Σ |

- **golden**: 标量扫描每行找 min/max/sum
- **容差**: **0 (exact)**
- **数据**: HW=8 行, D=32 depth
- **PASS 证明**: HVX 树形归约 (vmax/vmin 旋转降维) 数值与标量一致

⚠️ 函数名含 "crouton" 指内部处理方式, **输入仍是 flat 行主序**, 不是 crouton 布局.

---

## 11_lookup_unpack — 查表 + 权重解包

**测什么**: 两个 HVX 数据搬运算子.

- **`hvhx_table_lookup_flat_u8`**: `out[i] = table[in[i]]`, 256 元素 LUT. golden 直接标量查表. exact.
- **`hvhx_unpack_weights`**: 4-bit → 8-bit 解包, 每输入字节出 2 个 nibble:
  `out[2i] = (in[i]>>4)&0x0F`, `out[2i+1] = in[i]&0x0F`. n 输入字节 → 2n 输出字节. exact.
- **PASS 证明**: HVX gather 查表 + nibble 解包逐字节正确

---

## 12_multitile_gemm — 大尺寸 fp16 GEMM

**测什么**: `hmx_convf16` 在 M/N/K > 32 时的 4 种维度组合 (库内部多 tile 循环).

| run | 维度 | 测什么 |
|-----|------|--------|
| 1 | 64×32×32 | M > 32 (多 M tile) |
| 2 | 32×32×64 | N > 32 (多 N tile) |
| 3 | 32×64×32 | K > 32 (多 K tile, acc 跨 tile 累加) |
| 4 | 64×64×64 | 全维度 > 32 |

- **golden**: 标量 fp32 全累加
- **容差**: ≤1 ULP
- **PASS 证明**: 多 tile K-loop 累加 (acc 跨 activation+weight pair 持续, 只 clracc 一次) 数值正确

⚠️ 这是**公开 wrapper 的多 tile 正确性**, 不是峰值性能. 性能见 14.

---

## 13_compat_dlsym — v73/v75/v79 兼容层 dlsym

**测什么**: `dlopen("libhvxhmx.so")` 后 `dlsym` 6 个老版本/新几何符号, 调用 + golden 对比.

```
hmx_v73_convbbb1x1_stride1, hmx_v73_convbbb_stride2,
hmx_v73_convbbb1x1deep_stride1, hmx_v73_convbbb_dilate_stride1,
hmx_convbbb1x1_stride1 (v81 新增), hmx_convbbbNx1_stride2 (v81 新增)
```

- **golden**: u8×u8→u8 标量 (与 03 同)
- **容差**: **0 (exact)**
- **PASS 证明**: 
  1. 库能被 `dlopen` 加载 (符号全部 resolved, 无 UNDEF)
  2. v73/v75/v79 wrapper 尾调 v81 族函数, 行为由构造正确
  3. v81 新几何变体符号存在且功能对

这是 `test_compat_sym.c` (39 符号 + 25 功能) 的简化教学版.

---

## 14_hmx_peak_gemm — 裸 HMX K-loop 达峰值 (高级)

**测什么**: 同一个 M=32 N=32 K=256 fp16 GEMM, 对比两条路径的**正确性 + 吞吐**:

1. **裸 K-loop** (本示例核心): 一次性把 NK=8 个 act/wgt 32×32 slice 打包进 VTCM crouton,
   然后 `clracc → bias → NK 条背靠背 activation.hf/weight.hf (共享 acc) → cvt`.
2. **公开 `hmx_convf16` wrapper**: 同尺寸对照.

- **正确性**: 裸 K-loop vs 标量 golden, ≤1 ULP → **PASS** (证明低级路径数值对)
- **吞吐** (实测, 52f67807):
  - 裸 K-loop: **0.04 us/call, 12.34 TFLOPS**
  - wrapper: 269.5 us/call, 0.0019 TFLOPS (1.9 GFLOPS)
  - **speedup ≈ 6341×**
- **PASS 证明**: 
  1. 裸 K-loop 数值正确 (不是只有速度, 结果也对)
  2. HMX 硬件峰值真实可达 (~12 TFLOPS @ NK=8, README 声称 20.4 是 NK→大 + 理论峰)
  3. 公开 wrapper 慢 4 个数量级是因为每 tile 重新 gather+pack (HMX 被 CPU 搬数据饿死)

⚠️ **这是低级/高级路径**. 一般应用用 `hmx_convf16` 即可 (数值相同); 只有大 GEMM 吞吐敏感时才走本例模式
(预打包 + 裸 K-loop). 详见 [docs/api_lowlevel.md](../docs/api_lowlevel.md) 和
[docs/performance.md](../docs/performance.md) §性能现实.

调 `NK` 宏 (默认 8) 可加大 K 维 —— NK 越大, 每轮 K-loop 摊薄的 clracc/cvt 开销越少, 越逼近 20.4 TFLOPS 峰值.

---

## 容差矩阵汇总

| 精度族 | 容差 | 涉及示例 |
|--------|------|----------|
| int8 (bbb/bbh/bcb/bnb) | **0 exact** | 03, 05, 13 |
| u16 输出 (hbh/hhh/hch/hnh) | **0 exact** | 04, 05 |
| u8 depthwise | **0 exact** | 06 |
| HVX divide u8/u16, reduction, lookup | **0 exact** | 08, 10, 11 |
| HVX divide i32 (round) | ≤1 | 08 |
| HVX activation hardswish | ≤2 LSB (近似) | 09 |
| HVX activation prelu | ≤1 | 09 |
| fp16 GEMM/depthwise/add/K-loop | ≤1 ULP | 02, 06, 07, 12, 14 |

## 一键复跑

```bash
cd /disk1/V81Dev/hvxhmx_libs/examples
./build_examples.sh all          # 编译/签名/部署/运行全部 14 个
# 结果在 52f67807:/data/local/tmp/hvxhmx_libs/<名>.txt
adb -s 52f67807 shell "cat /data/local/tmp/hvxhmx_libs/14_hmx_peak_gemm.txt"
```

预期: 全部 `--- summary: N pass, 0 fail ---`, 无 FAIL.

---

# V2.2 单元用例 (16-22)

库: `lib/libhvxhmx_v22.so` · 设备目录 `/data/local/tmp/hvxhmx23/` ·
结果行格式与 01-15 相同 (`[PASS] label err=.. tol=..`) · 脚本自动拉回 `results/`。

| # | 用例 | 单元 | 判据 (门) |
|---|------|------|-----------|
| 16 | wtcache_pin | U1 | pin_bitexact ×3 / pin 槽独立 / pin 区在 ring 活动后完好 / ring prefetch+move_back byte-exact / 1MB pin 带宽 |
| 17 | w4a16_gemm | U2+U4 | 解码后 vs oracle Y_gold_2563 **byte-exact** (65536/65536) / 复跑 byte-exact / 100 次中位 invoke → GFLOPS (Y_ref_v0 是 int8 全精度金标, 非 w4 引擎参考, 不作门) |
| 18 | smallm_gemv | U8 | 输出面解码后: p1 行0==full 行0 / p16 行0..15==full / pad 行不变 / 行0 随输入变 / M=1 与 M=256 同价 (中位) |
| 19 | gdn_sm | U5 | f16 往返幂等 / conv·delta·solve cos≥0.9999..0.99995 / 状态 byte-exact+64KB 守卫 / chunk 16=8+8 |
| 20 | dualdomain | U7 | ser 16 步 → dom3/dom4 并发 a[0,8)+b[8,16): 每步 sha256 与 ser 同步 (host analyze_dd.py); 加速比只报告 |
| 21 | oplist_exec | U6 | 5 负例返回期望错误码 / W3 行 host==设备 (逐行 diff) / MATMUL vs K2560 金标 ≤40 LSB / 重开引擎 byte-exact / RMSNORM 0 ULP |
| 22 | dualcore_threads | U3 | 并发双引擎 out==串行 (byte-exact) / hvx·norm·dot 跨线程确定 / 旗标握手完成; 加速比只报告 (HMX 锁) |

容差矩阵追加:

| 精度族 | 容差 | 涉及示例 |
|--------|------|----------|
| W4A16 vs 4C 基线 / pad-256 | **0 (byte-exact)** | 17, 18, 20, 22 |
| W4A16 vs K2560 标量金标 | ≤40 LSB (规范 37) | 21 |
| RMSNORM vs 同算法标量镜像 | **0 ULP** | 21 |
| GDN conv/delta | cos ≥ 0.9999 | 19 |
| GDN solve-tri | cos ≥ 0.99995 | 19 |

复跑:

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.2/examples
./build_examples.sh            # 01-22 全部 (V2.1 回归 + 新单元)
./build_examples.sh 20         # 单跑 (含双域编排 + host 对拍)
# 结果: ../results/<名>.txt; 例 20 另有 host 对拍文件 20_dualdomain_host.txt
```

预期: 全部 `--- summary: N pass, 0 fail ---`, 末尾 `ALL GREEN`。

---

# V2.3 单元用例 (23-31)

库: `lib/libhvxhmx_v23.so` · 设备目录 `/data/local/tmp/hvxhmx23/` ·
手册: `../docs/api_v23_*.md` · 依赖: U9/U10/U11/U13 无依赖, 其余见各行。

| # | 用例 | 单元 | 判据 (门) |
|---|------|------|-----------|
| 23 | fence | U9 | 4×4×2 决策表 32 组合 host 对拍 / 非法组合拒绝 / CPU→DMA DDR 铁律① 200 轮 bypass 读回 / CPU→HMX VTCM 20 轮 / HMX 输入敏感 / CPU↔HVX VTCM 可见 / DMA moveback 100 轮 |
| 24 | arena | U10 | 1000 轮随机 size/align 指针恒对齐无泄漏 / 全释放完全合并 / gdn_sm+W4A16 双池 100 轮共存 (frag≈98.9%) / 碎片上界 |
| 25 | harness | U11 | solve_tri cos / gdnsm oracle cos + 字节恒等 / w4a16 65536 byte-exact / gdn_sm 与 w4a16_gold **复跑 sha 逐位一致** (共 13 门) |
| 26 | wpool | U12 | 24 job 随机到达值精确 (norm/dot/hvxload 混合) / hmx 门交接池==串行 byte-exact / spawn16 值精确 / **池 110us vs spawn 488us (4.4×)** / 5 轮压力 done==executed==total |
| 27 | pxbridge | U13 | f16 往返 ULP 包络 / i16 半步包络+零码↔零值 / 负钳零码 / 组合桥包络 / f16 桥恒等 / code 空间线性 / 端点钳位 / 批量==标量 (sc×4 组 ×20 门) |
| 28 | gdn_tree | U14 | 闭式 vs 串行 y/state cos=1.0 (host 数学闭合) / 设备 f16 kernel vs 闭式 cos=1.0 / 复跑 bit-exact / 违序拓扑拒绝 / INT16 衰减曲线 (深度 0..7, ≤2.5e-5) |
| 29 | kvcache | U15 | 非 128 倍数 slot 拒绝 / 64 槽 append+查读+posmap 逐字节 / 回绕旧 miss 新 hit + evict 精确 / scatter 邻槽金丝雀 / 槽 128B 对齐 / **DMA bypass 真读回** / verify 同槽校正语义 |
| 30 | graph_step | U16 | blob 解析 / 整步跑 / stats 精确 (ops=6 mm=1 rms=1 silu=1 pin=2 skipped=1) / 引擎未建 pin-skip / 分段跑 / **分段 vs 整步三 temp 面逐字节恒等** / pin-skip 归零 / silu 半 ULP |
| 31 | gemm_dispatch | U17 | 决策表 600×3+5 边界 0 mismatch / dense f16 vs 标量 oracle / 256³ vs 金标 65536 byte-exact / smallm 行0 恒等×2 + 同价 (1510 vs 1029us) / crouton512 编解码往返恒等 / **M=512 拆块上半==A 金标 下半==B 金标** |
| 32 | rbr | U18 | L1 契约: 快照污染恢复逐字节 / SKIP 一次性+CLEAR 作废 (m4) / NOSNAP·STALE·FROZEN·rewind 参数域拒绝 / L2 指纹: **等式1 缺陷引擎可检出** (replay_in==phantom_out 且偏离基线) / 等式2 replay_in==snapshot_src / 重放 KV 行覆写幂等 / **L3: 16 轮混合接受 (j=0 强拒绝~全接受) 终态+91 行+双账本 bitwise==基线** / 计数器精确 (snap16 restore=rewind=6 skip=5) |
| 33 | bledger | U19 | L1 契约 (参数域/状态机/统计) / T1 冷启动断链 verify=NEVER 即时报错+写入收敛修复 / T2 行错位 verify=NEVER + **E2 rank 反查定位错位行** / T3 canary: CANARY+0xAA 位样+全行深 rank (H2 与 H1a 判别) / T4 归还后禁读+归还前 canary / T7 未读异写覆写计数 (读后覆写须再读) / T5+T6 新路径 NEVER + qtag 对账 QTAG / **T8 P1§5 复刻: d1 垃圾 rank127+d2 正确 → 每轮 break+接受率崩塌 hist 全1; 修复 breaks=0+回归锚成立+hist 全3** / L1 timeline 文本 |

容差矩阵追加:

| 精度族 | 容差 | 涉及示例 |
|--------|------|----------|
| gdn_tree closed vs serial / kernel vs closed | cos = 1.0000000 (浮点精确) | 28 |
| kvcache 全部 / wpool HMX / graph_step temp | **0 (byte-exact)** | 26, 29, 30 |
| pxbridge i16 半步 | sc·0.5 + sc·5e-3 (量子+商舍入) | 27 |
| pxbridge f16 | \|r\|·1.2e-3 + 6.5e-8 (subnormal 底) | 27 |
| gemm 512 拆块 vs 单独金标 | **0 (byte-exact)** | 31 |
| rbr 全部 (L1/L2/L3) | **0 (byte-exact)** | 32 |
| bledger 全部 (状态机/计数) | **0 (精确匹配)** | 33 |

复跑:

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh all          # 01-33 全部 (V2.1 回归 + V2.2 + V2.3)
./build_examples.sh 28           # 单跑
```

预期: 末尾 `TOTAL 164 pass 0 FAIL / ALL GREEN` (另例 15 内嵌 49 项 PASS)。
