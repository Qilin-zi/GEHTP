# hvxhmx_libsV2.2 设备实测性能报告

- 设备: 52f67807 (unsigned PD, CDSP dom3/dom4), 2026-08-15
- 库: `lib/libhvxhmx_v22.signed.so` (184,624 B, vgather=0)
- 用例: examples 01-22 全量实机, **83 门全 PASS / 0 FAIL** (`build/run_all_final.log`)
- 运行: `./examples/build_examples.sh` (一键 编译→SWIV 签名→部署→实跑→results/)

## 1. 总览

| 范围 | 用例 | 门数 | 结果 |
|------|------|------|------|
| V2.1 算子回归 (01-14) | 14 | 41 | ALL PASS (含 13 dlopen 兼容层 7/7) |
| V2.2 新单元 (16-22) | 7 | 42 | ALL PASS (例 20 含 ser/a/b 各 1 门) |
| 例 15 (列式输出) | 1 | 48 | ALL PASS (不计入脚本 83 门统计) |

## 2. 新单元性能 (V2.2 核心)

| 单元 | 用例 | 关键性能 | 判据结果 |
|------|------|----------|----------|
| U1 wtcache (pin/ring) | 16 | 1MB pin = **16.38 GB/s** (64 µs) | 9/9, ring 4+4 prefetch/move_back byte-exact, pin 区在 ring 流量后完好 |
| U4 W4A16 引擎 256³ | 17 | invoke 中位 **26 µs = 1290.6 GFLOPS** (100 次采样) | **vs oracle Y_gold_2563 位恒等 65536/65536**, 复跑 byte-exact |
| U8 small-M GEMV | 18 | M=1(pad-256) 与 M=256 **同价** (26 µs, Δ=0%) | 5/5, pad 行不变性 + 行0==full 行0 (解码后) |
| U5 GDN 状态机 | 19 | (正确性单测, 8 门) | 8/8, conv/delta/solve cos 达标 + 状态 byte-exact |
| U7 双域 dom3/dom4 | 20 | **2.004×** (ser 543 µs → a/b 各 271 µs), per_step 33.9 µs | 切分等价: a[0,8)+b[8,16) 每步 sha256 == ser, 32/32 步 |
| U6 oplist 引擎 | 21 | w4 MATMUL op **27.1 ms** (含 3.2MB 权重 restage); w5 45.4 ms | MATMUL vs K2560 金标 ≤40 LSB, RMSNORM 0 ULP, W3 报告 host==设备逐行 |
| U3 双线程 dcthread | 22 | 并发/串行 ratio **0.963** (~1, 符合 4C 预期: 单 DMA+HMX 锁) | 6/6, 双引擎并发 byte-exact + hmx/norm/dot 跨线程确定 |

## 3. V2.1 回归要点 (01-15)

| 用例 | 要点 |
|------|------|
| 14 hmx_peak_gemm | raw K-loop 12.34 TFLOPS (wrapper 路径受 per-tile gather 限制, 例内说明) |
| 15 v2_llm_ops | rms_norm_mul n=1024: 0.30 µs (43.9 GB/s); mul 0.12 µs (102.4 GB/s); softmax 1.52 µs; sigmoid 0.95 µs |
| 01-12 | 全部数值门 PASS (V2.1 判据不变, 链接 libhvxhmx_v22) |
| 13 compat | dlopen V2.1 `libhvxhmx_v2.so` + dlsym 6/6 老符号解析成功 |

## 4. 与闭合模块基线对照

| 项 | 模块闭合基线 | V2.2 实测 | 一致性 |
|----|--------------|-----------|--------|
| W4A16 256³ 数值 | t10-a bit-exact 65536/65536 | 65536/65536 | **位恒等复现** |
| W4A16 K2560 数值 | 37 LSB (tol 40) | ≤40 LSB (例 21) | 一致 |
| 双域切分等价 | 模块 D 2.001× | 2.004× | 一致 |
| 双线程 HMX | 模块 C P3: 需 hmx_lock 交接 | 交接后 6/6, ratio 0.963 | 结论复现 |
| wtcache ring | T4 move_back 契约 | byte-exact 8/8 tiles | 一致 |

## 5. 测试环境要点 (踩坑记录)

1. **库 UNDEF 符号拒载**: FUSA/unsigned PD 下 .so 存在不可解析 UNDEF (如缺
   `dma_utils.c` 的 `dma_desc_*`) 时 dlopen 静默失败, skel 只报 "returned 1"。
   修复: `src/v22/dma_utils.c` 必须入库 (SDK 拷贝件, 只编译不改)。
2. **skel 版本**: 必须用 `/data/local/tmp/hvxhmx_libs/librun_main_on_hexagon_skel.so`
   (2026-08-10 版, 支持 dom3/4); `/data/local/tmp` 顶层 2022 旧版报 0x80000406。
3. **hmx_lock 持有线程属性** (模块 C P3): invoke HMX 的工作线程必须先
   `wtcache_hmx_lock`; 模式 = 主线程 `wtcache_hmx_unlock` 交接 → worker 引擎段
   batch lock/unlock → join 后主线程 re-lock。不交接直接 PD crash (-2147482611)。
4. **W4A16 金标配对**: `Y_gold_2563.raw` 对应输入是 **`act_surface.raw`**
   (t10 step0 基准面), 不是 `act_variants/v0.raw` (随机变体);
   `Y_ref_v0.raw` 是 v0 的 int8 全精度标量金标, 与 w4 引擎无对应关系, t10 闭合
   从未拿它做门 — V2.2 同样不作门。
5. **输出面布局**: dc_w4 输出是 crouton16_row4 surface, 行偏移在 surface 布局下
   无意义; 消费必须先逆布局 (例 17/18/21 的 `minv_crouton`, 与 host
   `inv_crouton16.py` 逐位一致, 256×256/256×512 双向验证)。
6. **ring move_back 契约**: `wtcache_ring_next` 的第 3 参是 **本轮** out 槽的 DDR
   目标 (ring 内部 pending, 下轮或 drain 才 submit — 保证调用方写完再搬),
   不是上一轮的。
7. **DSP 无 system()**: 设备侧 `system("mkdir -p")` 静默失败 (无 shell);
   dump 目录必须 host 侧 `adb shell mkdir` 预建。

## 6. 复现

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.2
./build_libs.sh                 # 库 (增量)
./examples/build_examples.sh    # 全量 01-22 (约 4 分钟)
./examples/build_examples.sh 17 # 单例
# 结果: results/*.txt; 汇总于尾部
```

---

# V2.3 验收报告 (2026-08-16)

- 设备: 52f67807 (unsigned PD, CDSP dom3), 库 `lib/libhvxhmx_v23.signed.so` (204,552 B, vgather=0)
- 范围: examples 01-33 全量实机 → **33/33 例全绿, 181 门 [PASS] 0 FAIL** (+例 15 内嵌 49 项, 共 230 项)
- host 模拟: `host/gtest_v23.c` 18/18 (决策表/arena 1000 轮/kvcache 影子/pxbridge unipolar/tree 数学/SILU+arity/rbr+bledger 契约)

## V2.3 新单元结果 (23-33, 108 门)

| 单元 | 例 | 门 | 关键实测 |
|------|----|----|----------|
| U9 fence | 23 | 7/7 | 决策表 32 组合; 铁律① 200 轮 bypass 逐字节 |
| U10 arena | 24 | 4/4 | 1000 轮对齐; gdn_sm+W4A16 共存 frag 98.9% |
| U11 harness | 25 | 13/13 | 例 17/19 金标重跑 sha 逐位一致 |
| U12 wpool | 26 | 5/5 | **池 110 µs vs spawn-per-op 488 µs = 4.4×**; hmx 门交接 byte-exact |
| U13 pxbridge | 27 | 20/20 | unipolar 契约 4 scale 全绿 |
| U14 gdntree | 28 | 8/8 | closed/serial/kernel 三方 cos=1.0000000; i16 衰减 ≤2.5e-5 |
| U15 kvcache | 29 | 8/8 | 回绕 evict 精确; DMA bypass 读回逐字节 |
| U16 graphstep | 30 | 8/8 | 整步 vs 逐算子 temp 面逐字节恒等 |
| U17 gemmdispatch | 31 | 8/8 | 256³ 65536/65536; M=512 拆块双金标 byte-exact; smallm 同价 (1510 vs 1029 µs) |
| U18 rbr | 32 | 8/8 | 等式1 缺陷指纹可检出; 16 轮混合接受终态/KV/账本 **bitwise==基线** |
| U19 bledger | 33 | 9/9 | 断链消费即报错 (NEVER/CANARY/RELEASED/WRITER/QTAG); E2 rank 反查定位行错位; P1§5 复刻 d1 垃圾+d2 正确→接受率崩塌, 修复+回归锚 hist 全 3 |

## V2.3 硬教训 (本轮设备调试)

1. **W4 kernel 单 invoke ABI 固定 M=256**: descriptor `n_tiles_pow2=32`/`m_total_minus_step=8`
   是常量段 (逐字保留); 512 面单发高位行全错 (512/512)。M>256 必须在 dispatch 层拆 256 块
   (`gemm_w4a16_m256`)。
2. **crouton decode 禁止原地**: 读流 (phase 外层) 与写行 (g 外层) 顺序交叉, 256 面实测
   32767/65536 败 (曾潜伏于 smallm m=1 — 只验行 0 时行 0 恰在安全带)。
3. **热路径禁每次 memalign/free 128KB**: 设备实测 +38 ms 分配抖动 (smallm 1.5→39 ms);
   scratch 必须进程级单例。
4. **kvcache pos 必须与槽同余**: scatter(slot=5, pos=777) 而 777%64=9 → lookup 永 miss;
   测试 pos 用 64k+slot 形。
5. 例程读回缓冲 G1 前未分配 → memcmp(NULL) fault DSP ("Failed to call main" 0 字节输出)。
6. **一次性 skip 标志必须被所有 setup 分支消费/作废**: rbr CLEAR 分支漏作废挂起 skip,
   标志穿透到下一帧把常规 COPY 错吃成 SKIP (树评估从零态起步, 终态+KV 双偏离,
   首坏行=1)。一次性状态标志的规则: 任何重写 in-state 的分支都要清标志。

## 复现

```bash
./build_libs.sh && cd examples && ./build_examples.sh all   # → TOTAL 181 pass 0 FAIL
```
