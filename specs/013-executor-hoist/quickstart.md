# Quickstart: 013-executor-hoist 验证指南

**Created**: 2026-10-13 | 前置：[spec.md](spec.md) / [plan.md](plan.md) / contracts/

## 前置条件

- worktree：`/disk1/gehtp-wt/013-executor-hoist`（off `origin/task/prof-wp` 734986b）
- kernels 构建先 `source env.sh`（[[gehtp-build-env]]）；设备 = d0f1784，**私有名纪律**（lib/runner/blob/结果名全私有；共享 op_ts.bin 拉取后按 size 自证）
- 0.8B 正典旗标全程使用（含 `--dma-runlist`、`--no-pass fc_bias_fold`）

## 场景 1：host 回归（每段收口必跑）

```bash
# ctest 基线 49/49
ctest --test-dir build_host   # 全绿 = 序列化/契约无回归
```

预期：B 段后含新增 requant/9-arity 往返用例；既有用例零变化。

## 场景 2：合成单 op 微 blob（B 段第一刀）

```bash
# 003 T014 合成 blob 法：单 W4A16 op(9-arity) 手构 wtop
# 设备跑 17_w4a16_gemm 同源已知答案对拍
```

预期：输出 == P5 dc_w4 已知答案（位级）；同时构造 6-arity 版验证 legacy 路径逐字节一致。

## 场景 3：0.8B 三线设备双门（每段收口 + 终局）

```bash
# 默认/gdn 线
#   门: cos ≥0.9999 (锚 0.999930, cos-min 假 FAIL 坑: 门设锚下)
#       top1 32/32, exec ≤2.77s, 未触路径输出 md5 == f87461e2
# f16 非融合线: md5 f87461e2 / exec 5.99s 锚
# w4 线 (B 段起有): cos ≥0.98, top1 ≥30/32, exec 对 25.7s 锚
```

## 场景 4：op_ts 取证（性能证据）

```bash
python3 scripts/gehtp_prof.py <op_ts.bin>   # 判读纪律: 禁手搓解析
```

取证三件：① re-carve 全 VTCM flush 次数（A 段后 =1）；② W4A16 matmul 族逐 op 前后对照（B/C 段）；③ 各段收益汇总进 results。

## 场景 5：装载期哨兵有效性（C 段）

手工破坏 carve 布局偏移一字段的 blob → 装载期必须硬报错拒绝（非运行期静默错地址）。

## 终局验收（用户口径）

| 门 | 值 |
|---|---|
| 精度不回退 | 默认/gdn 线 cos 0.999930 / 32/32；w4 线 ≥0.98 / ≥30/32 |
| 性能有优化 | w4 线 exec ≤17.1s（≥1.5× vs 25.7s 锚；stretch 5.99s）；默认线 ≤2.77s；各段 op_ts 证据齐 |

注意：feature.json 链末还原 012（备份 /tmp/feature_json_012_backup.json）。
