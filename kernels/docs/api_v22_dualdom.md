# U7 dualdom — 双域 step-list 分片执行 功能手册

声明: [`include/dd_worker.h`](../include/dd_worker.h) ·
源: `src/v22/dd_worker.c` (从 dualdomain_v81 dd_main.c 提取的纯函数层) ·
测试: [examples/20_dualdomain](../examples/20_dualdomain/main.c) (host 编排 =
build_examples.sh 20 分支 + `host/analyze_dd.py`)

## 功能

把 "N 步 W4A16 GEMM (权重共享, 每步独立 act)" 的**分片执行**封装成一个单元:
一进程跑一段 `[start, start+len)`。

**双域 = 同一二进制在 dom3/dom4 各起一个进程**, host 侧并发拉起; 模块内不做跨域
通信。对半切拿全部收益 (**2.001×**, MODULE C D2 设备闭合); 不均衡 = 大半边墙钟。

## 切分等价性契约

第 k 步输出只由 (k, 权重) 决定, 与域/并发/次数无关:
`serial[0..16) == halfA[0..8) + halfB[8..16)` 逐步 byte-exact。
example 20 用 `wt_sha256` 在设备上对每步输出取 sha, host 汇总对拍。

## API

```c
struct dd_cfg {
    const char* asset_dir;   /* packed_weight/folded_bias/act_table/out_table 所在目录 */
    uint32_t m, k, n;        /* m 必须 256 倍数 (U4 约束) */
    uint32_t chunk;          /* 每预载块步数 (0 → 默认 8) */
    uint32_t n_act_files;    /* >0: acts/dd_act_%d.raw 按 k%n 取;
                                =0: act_variants/v%d.raw 按 k%4 循环 */
    int dump;                /* 1 = 每步输出 dump 到 out_dir/dd_<tag>_step<k>.raw */
};
struct dd_stats { uint64_t wall_us, e2e_us; int64_t first_step_us;
                  uint32_t steps; double per_step_us; };

int dd_run(const struct dd_cfg* c, const char* tag, int start, int len,
           const char* out_dir, struct dd_stats* st, char* err, size_t errn);
```

`wall_us` = 纯 compute 环 (DMA→invoke→读回), 文件 I/O 不入环。
cache 四铁律全内置 (预载 dc_clean_ddr / open 已 FLUSH / 读回 INVALIDATE / close)。

## 编排 (build_examples.sh 例 20 摘)

```bash
./run_main_on_hexagon 3 test_20_dualdomain.so dd ser 0 16      # 串行基线
./run_main_on_hexagon 3 test_20_dualdomain.so dd a   0  8 &    # 并发
./run_main_on_hexagon 4 test_20_dualdomain.so dd b   8 16 &    # 双域
wait;  python3 host/analyze_dd.py results/
# 判定: dd_split_equiv (a/b 每步 sha == ser 同步) + dd_steps_all; 加速比只报告
```

## 使用说明 (给自己的业务分片)

1. 权重文件放一个目录 (s256/s2560 格式), act 多样性用 `act_variants/v%u.raw` 或
   `acts/dd_act_%d.raw`。
2. 每个域一个进程, 传**不重叠**的 [start,len); dump=1 时步产物即对拍证据。
3. 结果文件名带 tag (`20_dualdomain_<tag>.txt`), 双域互不覆盖;
   dump 读完即删 (example 20 内置), 不挤设备空间。

## 结论出处

- 4C 假设B系列: 双 CDSP 零退化; 单域内线程并发不缩放 (→U3)。
- D2: 切分等价 byte-exact; 对半切 2.001×。
