# T020 — C 段设备门（carve 静态布局表）

日期 2026-10-14；板 d0f1784（多会话共享）；worktree `013-executor-hoist`。

## 板上资产（本次全新私有名，fastrpc 缓存纪律）

| 件 | 名 | md5 (签名版) | 落地方式 |
|---|---|---|---|
| 内核库 | `libhvxhmx_018.so` | ed4e46d6fcbe350560dc36c96352a7a9 | **SWIV 签名版以普通 .so 名**落 `/data/local/tmp/hvxhmx23/` |
| runner | `gehtp_runner_018b.so` | 347a3c56b2fb556a848e012136ccaa02 | 同上；烘焙 `GEHTP_JOB_PATH=/data/local/tmp/hvxhmx23/cl020/job.txt`，`RESULT_NAME=018b_gehtp_runner` |
| 合成 blob | `cl020/blob_w4_{twin,cl,bad}.wtop` | a5e260bd / f0f4aacc / 0435b221 | host mk_w4_cl.py 生成+逐字节核验 |
| 激活输入 | `gehtp/input_013.f16.raw` | 2867f9a19315f4ed19af1f4a79f4fa51 | 复用板上既有 |

blob 家族（v3）：7 slots + 4 ops [NOP, (3,1), (3,2), W4A16 9-arity [out=0, w=1, bias=0, M=256, K=2560, N=2560, bias_s=2, atbl_s=3, otbl_s=4]]，annot 全 0。
- **twin**：到此为止（无 footer）→ 走 legacy carve 路径。
- **cl**：+ TEMPOFF(cap 1310720, pin 4096, n=1) + VTCL 三区(arena 4096..5945344, resid 池 0) + CRVL[n=1 {0,256,2560,2560,0,2621440,5898240,0,1310720,5944464}, n_ref=1 {op 3→lid 0}] → 走装载期对拍+热路径指针直赋。
- **bad**：cl 逐字节同，唯 CRVL 记录 off_wt 字节 ^0xFF（2621440→2621695）→ 哨兵负例。

## 门 1-3：合成三跑（job out_temp 0）

```
twin: [PASS] 4 ops → out 1310720 B   md5 7654a8f64a0d66f2e93985ef34f10016
cl  : [PASS] 4 ops → out 1310720 B   md5 7654a8f64a0d66f2e93985ef34f10016   ← 与 twin/已知答案位级同
bad : [FAIL] wt_exec_run_io rc=-1 cl drift #0: act 0/0 out 1310720/1310720
           wt 2621695/2621440 bias 5898240/5898240 hi 5944464/5944464      ← 装载期拒载,零输出
```

已知答案 `dev_w4.f16.raw` = 7654a8f6…（003/008/B 段历代同锚）。bad 拒载信息五字段对拍中恰好 wt 线值=2621440 算值, blob 值=2621695，即被破坏字段被闭式重算逮住。

## 门 4：计数器（optrace `/data/local/tmp/hrt/gehtp/optrace.txt` 追加段）

```
twin 跑: layout 0 | full_vtcm_flush 1 | arena_flush 0 | tango 0
cl   跑: layout 1 | full_vtcm_flush 1 | arena_flush 1 | tango 0   ← A 段遗留的 re-carve flush 缺口在此补上
bad  跑: layout 1 | ...              | arena_flush 0 | tango 0   ← 拒载于热路径前
```

cl 跑 arena_flush=1 = 布局切换恰一次 arena 段 FLUSH；tango=0 = 布局图探戈零发生。

## 门 5：单 op 时长对照（同板同库紧邻双跑，op_timing code=1）

```
twin (legacy carve): code=1 n=1 total_us=13469   exec total 30832us
cl   (CL 快路径)   : code=1 n=1 total_us= 6916   exec total 30827us   → W4A16 op 1.95×
```
（读权重 25.9ms 摊薄整 exec 差异；op 级机制收益=删 carve 扫描+arena 初始化。）

## 门 6：0.8B 零回归（板上锚 blob 直读，输出落 cl020/）

| 线 | blob (md5) | input | out_temp | 输出 md5 | 锚 |
|---|---|---|---|---|---|
| f16 | qwen35_08b_010.wtop (3f857926) | input_013_08b.i32.raw | 2796 | **f87461e2**a5089c48413e70463f4aa7dc | f87461e2 位级同 ✓ |
| gdn | qwen35_08b_fused_pp.wtop (a7935fdf) | input_012.i32.raw | 96 | **9d1c0877**7cb9011303d9729a9b2e04e7 | 9d1c0877 位级同 ✓ |

f16 线 11504 ops exec 5.97s；gdn 线 884 ops exec 2.75s。

## 过程坑（环境）

1. `/data/local/tmp/hvxhmx23/gehtp` 已变 root:755（另一会话以 root adb 活跃使用）。**绕行**：blob/输入/输出/job 全走私有 `cl020/`（父目录 777）；runner 以新名 018b 重编烘焙 job 路径。
2. **板上库必须推 SWIV 签名版、但以普通 .so 名落地**（签名 25232B vs 未签名 25176B）。推未签名版：DSP dlopen 静默死（"Main returned 1"无任何 DSP 侧痕迹），控制组 SDK 自带 test_39 同样死 → 可快速识别为环境/装载层而非自家二进制。
3. DSP 侧 FARF 默认不可见；`/data/local/tmp/hvxhmx23/run_main_on_hexagon.farf`（内容 0x3f）可开 AP 侧 apps_std 详细开文件日志（用后删除）。
4. runner 结果日志每次运行 "w" 覆盖 + op_ts.bin 共享路径——需留存的跑当面 cat/拉取。

## 结论

C 段（T018/T019/T020）机制、位级、哨兵、零回归、计数器五面全绿；W4A16 单 op 1.95× 机制收益实测。真图收益兑现仍需 W4A16 供给侧在 0.8B emit 打开（与 B 段同口径的"收益未兑"缺口，交接 D 段/后续回合）。
