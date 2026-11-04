# T007 上板证据 (2026-11-04, 板 d0f1784)

## 结论
footer 流 **激活 + 全消费 + 位级零回归**; 单序列性能 **0% 收益** (46.11ms == 基线),
非预估 ~4% — 证实「dma_wait_for_idle 实为 no-op, 单序列不划算」, 真价值 = T008/021 架构前件。

## 上板口径 (私有名, 共享板纪律)
- lib: `libhvxhmx_020.so` (md5 6ba4bf2a4edc4b58f4140d8f371b6451)
- runner: `gehtp_persistent_q6b.so` (DT_NEEDED libhvxhmx_020.so, `-DGEHTP_JOB_PATH=.../gehtp_persist_q6b/job.txt`)
- blob: `decode_q4_018_footer.wtop` (697038720B, 6abbc96cd3f6ec4d7d0c98a528eababe) / 基线 `decode_q4_018.wtop`
- 脚本: `scripts/check_decode_q4_logits_06b.py` / `measure_decode_q4_06b.py` `--blob <板侧路径> --out-temp 1`

## 精度 (G2)
- step0 logits cos = **0.976850** (device-Q4 vs HF-f32, ≥0.9 门, 引擎无 bug)
- **argmax device=12095 == golden=12095** (首生成 token 精确)

## footer 激活取证 (临时 marker, 已删)
`active=1 n_dma=3851 n_mat=194 consume=3851 grp_max=245760`
- **194 matmul 走全局流** == 已知 194 W4TS 权重基址 (fence-flush 记忆, 精确吻合)
- **3851 记录全消费**, 无短填 (fill short) 无漂移 (drift) — footer 与 op 序一致契约成立
- grp_max=245760B = K=3072 gate/up 4-ct 组 (96×640×4)

## 零回归 (位级)
footer vs 基线 (legacy path, 同一 lib) 生成序列 **逐 token 相同**:
`12095, 11, 323, 279, 6722, 315, ...` (两跑完全一致) → 全局流与 legacy 位级同。

## 性能 (G3)
| 变体 | median step_us | tok/s | E2E match |
|------|---------------|-------|-----------|
| footer (全局流) | 46113 | 21.69 | 3/32 |
| baseline (legacy) | 46108 | 21.69 | 3/32 |

- footer == baseline (Δ5µs 噪声) → **0% 单序列收益**, 非预估 ~4%。
- 归因: dma_wait_for_idle 在 legacy 已是 no-op (ring is_done 全消费), 跨 op 预取重叠的
  激活量化 (1-2µs/op) + 首组 DMA (~0.4µs/op) 总计 ~0.2%, 噪声内。
- 46.11ms vs 37.9ms (fence-flush 锚, Oct 30 合入前) = **合并 57ecb84 后既有基线 ~46.7ms**
  (merge G3 gate 同值), 非 footer 回归 — footer 与 baseline 同 lib 同 46.11ms。

## 关键坑 (根因, 已在 [[gehtp-device-lib-stale-arity]] 前科复发)
首轮 footer blob 上板 `state slot table bad (n=0 len=116)` = **oplist_parse.o 陈旧**:
`build_libs.sh one oplist_exec` 只重编单 .o, 而 T001 同时改 oplist_parse.c (footer 解析)。
设备 wt_parse 不认 dma footer → weight_off 落进 footer 内 → 状态槽读到 0。
**修法**: 显式 `build_libs.sh one oplist_parse` + `one oplist_exec` 两刀。复发教训:
T001 跨两个 .o, 上板前必查 build/*.o mtime 是否晚于对应 .c。
