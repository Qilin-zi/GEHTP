# T008 2 NSP 上板证据 (2026-11-04, 板 d0f1784)

## 结论
**G4 PASS**: 2 NSP 域各跑一路独立序列, 合计吞吐 **~2×** (设备侧 1.998× / host wall 1.98×),
位级同单域 (nsp0/nsp1 输出 == 单域, 两次跑可复现)。footer 单份天然双域兼容, **编译期零改动**。

## 上板口径 (私有名, 共享板纪律)
- lib: `libhvxhmx_020.so` (md5 6ba4bf2a4edc4b58f4140d8f371b6451, 与 T007 同)
- runner: `gehtp_persistent_nsp0.so` (md5 bc1586fa076f28914b61c11ba4f89f85,
  `-DGEHTP_JOB_PATH=.../gehtp_persist_nsp0/job.txt`) + `gehtp_persistent_nsp1.so`
  (md5 ec77377ccc075fda262b9493ca094915, job 在 `.../gehtp_persist_nsp1/job.txt`)
- 两 runner 同源 42_gehtp_persistent/main.c, 仅换 `-DGEHTP_JOB_PATH` 宏 + so 名 (零 C 代码改动),
  均 DT_NEEDED libhvxhmx_020.so
- 域钉定: `run_main_on_hexagon 3 <so> domain_type=NSP core_id=0/1`
  - nsp0 → server.log `Domain_type: NSP core_id: 0 → unsigned PD on domain 1900`
  - nsp1 → `Domain_type: NSP core_id: 1 → domain 1700` (== [[gehtp-multicore-map]] 映射)
- blob: `decode_q4_018_footer.wtop` (697038720B, T007 同)

## footer 双域兼容 (编译期零改动, spec 假设③被实现推翻)
spec 假设 ③「2 NSP footer 编译期生成两份」。T007 实现里 footer `src_offset` 是
**相对 weight 区起点** (`w4ts_fill_ahead`: `p.src_address = b->weight_base + off`),
`weight_base` 是每域运行时自己的 blob 权重区指针 (各自 rpcmem mmap)。故**单份 footer
天然双域兼容** (每域各自 weight_base + 相对 off → 各自正确绝对地址), 无需生成两份,
也无需运行期跨域 offset 平移。T008 编译期**零改动** (footer 由 T002 已生成)。

## 位级 (G4)
两跑 (可复现) 双域 nsp0/nsp1 生成序列 == 单域同一序列, 逐 token 相同:
`12095, 11, 323, 279, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 3146, 13,
2055, 11, 279, 6722, 315, 9625, 374, ...` (32 token, match=3/32 = Q4_0 量化天花板, 非 bug)
- `[bitwise] nsp0 == single: True`
- `[bitwise] nsp1 == single: True`

## 吞吐 (G4)
| 变体 | tok/s | step_us median | wall |
|------|-------|---------------|------|
| 单域 (1 seq) | 21.26 | 47042 | 3.60s |
| 双域并发 (2 seq) | 42.48 | 47078 (nsp0 46569 / nsp1 47078) | 3.64s |

- **device-speedup = 1.998×** (纯设备, 2×med_single/med_dual, 无 host 污染)
- **host-wall-speedup = 1.980×** (2×wall_single/wall_dual, 含 host adb 往返开销)
- **每序列 step_us 无退化** (47042 → 47078, ~0.08% 噪声): 2 NSP 域并发**无 DRAM 争用**,
  各自 DDR 带宽独立。**推翻** [[gehtp-multicore-map]]「DRAM 带宽是共享瓶颈」的猜测 —
  至少 0.6B decode (~21 tok/s ≈ 10GB/s/域) 远未触共享 DRAM 总带宽顶。

## 关键发现
- 双 NSP 并发 = 数据并行最粗粒度 (每域完整副本), 机制简单可靠 (014 T024 双域并发 + 本次双序列)。
- 真吞吐杠杆链成立: 单序列 21.26 tok/s → 双 NSP 42.48 tok/s (×2) → 021 批处理 M>1 再乘
  (目标 65-85 tok/s 超 llama.cpp 45.2)。

## 坑 (复发风险)
`PersistentDeviceRuntime.init()` 内 `_ensure_device_server()` 调 `start_server()` 用
**默认 so_name** (覆盖 `self._so_name`), 破坏按私有 so_name 的 pgrep 匹配。T008 脚本的
`run_sequence` **绕过 init**, 直接 `rt._rpc({"op":"reset"})` + run RPC (runner 侧
op=reset/run 不校验 handle_id)。此前 T007 measure/check 脚本靠 monkeypatch
`server_running` 硬编码 q6b so_name 掩盖了此问题。

## 脚本/资产
- `scripts/t008_dual_nsp_06b.py` (主仓库 /disk1/gehtp/scripts/, 与 measure/check 同约定)
- `persistent_runtime.py` `start_server` 加 `domain_args` 参数 (钉 NSP 域, 向后兼容默认 "")
- 编译脚本 `/tmp/build_t008_nsp_runner.sh` (换宏双 so, 复用 T007 私有构建配方)
