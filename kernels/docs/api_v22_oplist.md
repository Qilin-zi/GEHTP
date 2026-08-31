# U6 oplist — blob v1 权重格式 / 解析 / 执行 功能手册

声明: [`include/oplist_parse.h`](../include/oplist_parse.h) (格式+解析, host/设备同源) ·
[`include/oplist_exec.h`](../include/oplist_exec.h) (执行引擎) ·
[`include/wt_sha256.h`](../include/wt_sha256.h) (FIPS 180-4, host/设备同源) ·
源: `src/v22/oplist_parse.c` `oplist_exec.c` `wt_sha256.c` `wt_w3.c`
(源模块 wt_repack_v81 W1-W5 闭合) ·
host: `host/pack_oplist.cc` (组包) `host/wt_inspect.c` (检查器) ·
测试: [examples/21_oplist_exec](../examples/21_oplist_exec/main.c)

## blob v1 格式 (小端)

```
off 0  : "WTOP" | u16 ver=1 | u16 endian_chk=0x1234 | u32 n_slots | u32 n_ops
接     : slot[n] × 16B { len, count, offset(128B对齐,相对weight区), addr(恒0) }
接     : op[n] { u16 opcode; u16 n_args; u32 args[n] }   (args 只放 slot/temp id+维度)
接     : weight 区 (128B 对齐, slot 间 pad 128B)
opcode : 0 NOP [] · 1 MATMUL_W4A16 [act_slot,w_slot,out_temp,M,K,N]
         2 RMSNORM_F16 [x_temp,w_slot,y_temp,n] · 3 PIN [slot]
```

解析即全部边界检查 (13 个错误码 WT_ERR_*), host/设备同一份 `wt_parse`。

## 执行引擎

```c
int  wt_exec_run(const struct wt_blob* b, uint32_t* engine_m,
                 int64_t* op_us, char* err, size_t errn);
/* 顺序执行 op 表; 返回 0=全过, >0=失败的 op 序号(1基); temp id 0..7 跨 op 传中间结果 */
uint8_t* wt_exec_temp(uint32_t id);
uint32_t wt_exec_temp_bytes(uint32_t id);
void     wt_exec_shutdown(void);        /* wtcache_close + temps 释放 (铁律④) */
void     wt_w3_report(blob_name, buf, size, &w, emit_cb, ud);   /* W3 报告行 */
```

- MATMUL: 按 slot len 启发定位 bias/atbl/otbl; 引擎惰性建立 (首次 MATMUL 时
  wtcache_open+carve, **V2.2 修复后 open 末尾全 VTCM FLUSH**); 每次 op 重新
  cpu_to_vtcm 权重 (restage); act 由 UserDMA bypass 读 blob 内 DDR (**铁律①:
  blob 读入后必须 `dc_clean_ddr`**)。
- RMSNORM: 标量路径 (inv_crouton 解码 → double 累加 rms → f16 出), 与 example 21
  内嵌镜像 bit-exact (0 ULP)。
- shutdown 后可再次 `wt_exec_run` (引擎重建, temp0 跨重开 byte-exact — example 21 门)。

## host 工具

```bash
g++ host/pack_oplist.cc host/vendor/weight_pack.cc src/v22/oplist_parse.c \
    src/v22/wt_sha256.c -Iinclude -o build/host/pack_oplist     # build_examples.sh 自动
./build/host/pack_oplist --t10 assets/s2560 --out build/blobs --tag w4   # 或 w5
./build/host/wt_inspect build/blobs/blob_w5.wtop                  # W3 行 → stdout
```

- `pack_oplist`: slot 0..6 = act/packed_weight/folded_bias/act_table/out_table/
  rms_w(确定性生成 1.0+0.3·sin(0.37·i))/q8_0 crouton (vendored); w4=4 ops,
  w5=6 ops (MATMUL+RMSNORM+PIN)。产物含 `manifest_<tag>.json` 逐字段对拍清单。
- `wt_inspect` 与设备 `wt_w3_report` 同一份报告源码, 行逐字节一致
  (build_examples.sh 在例 21 后自动 diff)。
- `wt_sha256_hex` 与 python hashlib 对拍 (host 已验证)。

## 判据 (example 21)

| 门 | 判据 |
|----|------|
| negatives_rejected | 5 个定点破坏 (magic/ver/endian/n_slots/n_ops) 各返回期望错误码 |
| W3 行 | host wt_inspect == 设备报告行 (逐行 diff, 脚本内置) |
| matmul_vs_gold_lsb | ≤ 40 LSB (K2560 规范 37) |
| matmul_reinit_byteexact | w4/w5 两 blob 跨引擎重开 temp0 byte-exact |
| rmsnorm_bitexact | vs 同算法标量镜像 0 ULP |

## 性能 (wt_repack_v81 设备实测)

| op | 耗时 |
|----|------|
| MATMUL 全路径 (含 3.2MB restage) | 32.0 ms |
| RMSNORM (256×2560 标量) | 68.5 ms |
| PIN | 5.0 ms |
| blob 传输 (4.6MB, fastrpc) | ~21-24 s → 一次性成本, 建议上机前预推 |
