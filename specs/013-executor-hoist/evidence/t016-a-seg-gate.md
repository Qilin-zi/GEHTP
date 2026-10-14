# T015/T016/T017 — A 段设备门证据（板 d0f1784，2026-10-14）

## 1. 合成驻留 blob（host 侧逐字节核验）

`/tmp/res016/blob_res_016.wtop`（147840 B，ver=3，n_slots=4，n_ops=2）：链式双 f16 matmul
`input[64,128]×w1[128,256] → t0[64,256]×w2[256,128] → out`，auto TEMPOFF + VTCL 布局段 + n_vtcm=2。

python 按 `oplist_parse.c` 同语义走查核验：

- VTCL `magic 'VTCL'`(0x5654434C) / ver 1
- pin `[0, 262144)` / arena `[262144, 378880)`（arena_sz **116736** == 两 matmul HMX 需求闭式）/ resid `[903168, 16777216)`（== 驻留池 15874048 尾倒划）
- 三区互斥 ✓；carve footer 2 条与 TEMPOFF 双向对拍 ✓
- TEMPOFF **cap=0**（全 VTCM 驻留图，无 DDR temp）——emit 合法产物，旧执行器误拒（`tempoff cap bad`），本次修复为契约允许

## 2. 双跑位级（同 blob 同输入，新旧执行器路径对拍）

| lib | 路径 | 输出 md5 | exec |
|-----|------|----------|------|
| 016o（HEAD 懒初始化+探戈路径） | legacy | `ce338fe6899778aacfc28414f2d9498b` | 27.6 ms |
| 016b/016c（T016 装载期一次分区） | has_layout | `ce338fe6899778aacfc28414f2d9498b` | 28.4 / 29.5 ms |

两 op 均 `[PASS]`（`evidence/t016-synth-device.txt`）。optrace 路径取证：新 lib `hmx attach arena 116736`（表值）vs 旧 lib `hmx attach cap 641024`（运行时反推）——不变量 #2（表驱动有界 attach）实跑证据。

## 3. 计数器门（optrace 文件通道；FARF 不进 logcat，关机双通道打印）

```
[run_io] 013 A-seg summary layout 1
[run_io] 013 A-seg summary full_vtcm_flush 1
[run_io] 013 A-seg summary arena_flush 0
[run_io] 013 A-seg summary tango 0
```

- `layout=1`：VTCL 表被解析并采用
- `full_vtcm_flush=1`：装载期 wtcache_open 恰一次（=唯一一次全 VTCM flush）
- `tango=0`：hmx↔wtcache teardown/re-acquire 探戈零发生（探戈计数全部挂 `!has_layout` 分支）
- `arena_flush=0`：本图为纯 f16 matmul（走 attach，不走 dc_w4_carve）；**W4A16 re-carve 段 flush 路径编译在但本图未触发**——诚实注记，待 C 段 carve 图覆盖

## 4. 0.8B 零回归（最终 lib 016c = T016+capfix+rtrace，md5 3827fdc8）

| 线 | blob | out_temp | 输出 md5 | 锚 | exec |
|----|------|----------|----------|-----|------|
| f16 非融合 | qwen35_08b_010.wtop | 2796 | `f87461e2a5089c48413e70463f4aa7dc` | f87461e2 ✓ 位级同 | 5.99 s |
| gdn 融合 | qwen35_08b_fused_pp.wtop | 96 | `9d1c08777cb9011303d9729a9b2e04e7` | 9d1c0877 ✓ 位级同（=cos 0.999930 / top1 32/32 锚输出） | 2.76 s |

零回归机理：两 blob 编译于 A 段 emit 之前（无 VTCL 段）→ has_layout=0 → legacy 路径逐字节不变；位级双确认（对锚 md5 + 对昨日 HEAD lib 同板输出）。

## 5. 过程坑（后续会话纪律）

1. **fastrpc 按 lib 文件名缓存**：同名 .so 重推不生效（设备 md5 已新、跑的仍是旧码）——每次 lib 重推必须换新私有名（016→016b→016c 实证）。
2. **板上 lib 必须 SWIV 签名版**：推未签名 `libhvxhmx_v23.so` → main 返回 1 且结果文件/optrace 全无（loader 拒载）；推 `lib/libhvxhmx_v23.signed.so`。
3. **FARF 不保证进 logcat**：设备侧计数取证一律走 optrace 文件（关机 rtrace 四行本次补齐）。
4. `build_libs.sh one <base>` 只重编单 .c 再链全 .o——交叉对照 build 时注意 frankenbuild（本次 016o 参照属 HEAD oplist_exec + T016 其余 .o，执行器行为=HEAD，符号面略新；主证据以 016b/016c 与锚位级对照为准）。

## 6. A 段定位（诚实口径）

A 段交付的是**机制性前置设施**：VTCL 表解析/装载期一次分区/探戈消解/flush 收敛/查表 attach 全部实跑验证，但 0.8B 基准图（f16/gdn 线）auto 撤退出驻留、无 VTCL 段，走 legacy 路径——**对本轮基准性能无收益也无回归**（位级双确认）。收益兑现依赖 C 段（carve 序列化）与后续 emit 侧驻留规划打开。
