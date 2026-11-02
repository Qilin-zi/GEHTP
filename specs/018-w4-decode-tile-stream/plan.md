# 018 plan: W4 decode tile 流式搬运

## 改动面

| 文件 | 改什么 |
|---|---|
| `kernels/include/oplist_parse.h` | flags bit5 (0x20) W4TS 契约注释 |
| `compiler/tools/wtop_ops/op_matmul.cpp` | emit: 满足条件 → flags 置 bit5 |
| `kernels/src/runtime/oplist_exec.c` | `exec_matmul_w4_stream` (static) + M<32 分发 + host 标量参考 |
| `kernels/host/pack_w4stream_ab.cc` | 合成 op 两臂组包器 (基线 vs bit5) |
| `kernels/examples/53_w4_stream_ab/` | 设备对拍例 (cos + GB/s) |

## 执行器内核设计 (exec_matmul_w4_stream)

```
op 内一次性:
  arena = hmx_runtime arena 自划: [ring 8×slot | act_q8 | out_f32]
  slot = n_k*640B (K=1024→20KB, K=4096→80KB, K≤8192→160KB)
  激活: f16→f32 (hvx_vec_f16_to_f32) → quantize_row_f32_q8_0_tiled → act_q8 (n_k*1152B)
  ring 预热: submit ct 0..6 (dma_desc_submit 菊花链, 非阻塞)

主循环 ct = 0..n_ct-1:
  等 dma_desc_is_done(desc[ct%8])            // 晚 poll
  tiled_vec_dot_q4_0_32x1(K, out_f32+ct*32, vtcm_slot[ct%8], act_q8, 32, NULL)
  submit ct+7 → desc[(ct+7)%8]               // 槽循环 (is_done 已确认才重写)

收尾:
  dmwait 排空 (desc 静态复用不 free, 链纪律 = 重写前必 is_done)
  out_f32 → f16 → y (HVX store)
```

DMA 纪律 (wtcache_impl.c:32-36 教训): desc 用 static 数组不 free;
重写 desc 前必须 dma_desc_is_done 确认; op 末 dmwait 排空,
不留 g_last_desc 悬垂给下个 op 的 DMA 消费者。

## 数值路径

```
y[n] = Σ_kt scale_w[kt][n%32] × scale_a[kt] × Σ_{k∈kt} (q4[n,k]-8) × q8[k]
```
- q8[k] = round(af[k]/d_kt), d_kt = max|af[kt]|/127 (f16) — llama.cpp 同构
- int8 rmpy 累加 (精确整数) → f32 × (scale_w×scale_a) → f32 跨 kt 累加
- vs 基线 (f16 激活 f32 逐元素): 差异 = q8 激活舍入 (~2^-7 相对) + 累加序

## emit 判别 (op_matmul.cpp)

现 W4A16 int4 判别: `w_is_const && m%256==0 && len==k*n/2` — 不动。
新增 W4TS 判别 (槽长 640B/tile ≠ k*n/2, 与 W4A16 互斥):

```
w4ts = (m < 32) && !batched && t1 && !bias
       && (k%32==0) && (n%32==0)
       && (slots[w].len == (k/32)*(n/32)*640)   // Q4 tile 槽
       && (k <= 8192)
flags |= 0x20
```

## 对拍资产

pack_w4stream_ab.cc (仿 pack_gemv_ab.cc): 同确定性 GGUF Q4_0 权重字节,
两臂 blob 仅 flags 差 bit5:
- A: flags=2 → exec_matmul_q4_0_gemv (基线)
- B: flags=2|0x20 → exec_matmul_w4_stream
形状: K=1024 N=2048 (q_proj 型) + K=1024 N=248320 太大改 N=8192 (流式长链)
例 53: 两臂跑板, 输出 cos/max_rel_err + 各臂 µs → GB/s。

## 风险

1. rmpy 内核对 repack nibble 序的解读兼容性 (A 序假设) → G1 合成门实证,
   不兼容则修 unpack 舞步 (不动 repack — embed/prefill 同吃该布局)。
2. DMA 小 ct 块 launch 开销 → ring 深度 8 可调; K=1024 时 20KB/ct @66GB/s=300ns
   vs 算 ~160ns — DMA  bound 正好, launch 若 ~µs 级则槽数/块大小需上修
   (备选: 1 desc 搬多 ct, slot=2ct)。
3. q8 激活精度 → G3 E2E cos/top1 呈报, 用户裁决。
4. 与 OP_DMA (spill/fill) 共享单 DMA 引擎 — 链 FIFO 保序, 正确性无碍,
   decode 图 OP_DMA 稀少, 性能影响可忽略。
