# Quickstart: 005-graph-identity-passes 验证手册

端到端证明三个 pass 工作的最短路径。前置：编译器已构建（`compiler/build_linux/`），0.8B v2 工作目录在 `/disk1/gehtp/.scratch_a3/qwen35_08b_v2.wtop.work/`（下称 `$W`）。

## 1. 单测门（最快，~20s）

```bash
cd /disk1/gehtp/compiler/build_linux && ctest --output-on-failure
# 期望: 47 + 新增全绿; test_opt_passes 含 [13] view_fold / [14] eltwise / [15] matmul-bias 段
```

## 2. stats 观测（pass 是否命中真图）

```bash
cd $W
/disk1/gehtp/compiler/build_linux/hnnx_compile --net-json qwen35_08b_v2_net.json \
  --weights-bin model.params.bin --format tagged --opt-stats \
  --output /tmp/005_probe.bin 2>&1 | grep opt-stats
# 期望: view_fold=m/r eltwise_const_fold=m/r fc_bias_fold=18/18(FC)+k/r(MatMul)
# 幂等复跑同命令(改 --output): 第二次 rewritten 应为 0 可从 after 数不再下降佐证
```

## 3. 恒等性对拍（数值门）

```bash
# 折叠前基线(全关):
/disk1/gehtp/compiler/build_linux/hnnx_compile ... --no-pass view_fold --no-pass eltwise_const_fold \
  --no-pass fc_bias_fold --output base.bin
# host 双跑对拍:
/disk1/gehtp/compiler/build_linux/host_run --bin base.bin \
  --input-f32 ids_from_golden.i32.raw,mask_ones.f16.raw,pos_arange.i32.raw --out /tmp/o_base.f32.raw
/disk1/gehtp/compiler/build_linux/host_run --bin 005.bin \
  --input-f32 ids_from_golden.i32.raw,mask_ones.f16.raw,pos_arange.i32.raw --out /tmp/o_005.f32.raw
python3 - <<'EOF'
import numpy as np
a=np.fromfile('/tmp/o_005.f32.raw',dtype=np.float32); b=np.fromfile('/tmp/o_base.f32.raw',dtype=np.float32)
assert np.isnan(a).sum()==np.isnan(b).sum()==0
d=np.abs(a.astype(np.float64)-b.astype(np.float64))
print('max|d| =', d.max())   # P1 期望 0(位级); P2 ≤1 ULP f16 ≈ 对 logits 域 <0.01
EOF
```

注意：host_run 输出对输入文件不敏感是已知宿主路径缺陷（见 memory/gehtp-fcbias-device-verify）——**恒等性对拍仍然有效**（同一 host 路径自对自，输入无敏感性不影响 A/B 差分），但绝对精度判据必须走设备。

## 4. 设备门（性能 + 精度）

```bash
# emit + 上板(磁盘检查! 设备常满):
adb -s d0f1784 shell 'df -h /data/local/tmp'   # 需 ≥2.2G 余量, 不足先清本会话旧 blob
/disk1/gehtp/compiler/build_linux/wtop_emit --bin 005.bin --out 005.wtop --manifest 005.wtop.manifest.json
ANDROID_SERIAL=d0f1784 /disk1/gehtp/scripts/gehtp run 005.wtop \
  --input /disk1/gehtp/test_assets/qwen35_08b/input_p0.f16.raw \
  --output $W/out_005_p0.f16.raw --device d0f1784
# 判定:
python3 /disk1/gehtp/test_assets/qwen35_08b/judge_logits.py $W/out_005_p0.f16.raw 0
grep '\[run\]' $W/prof/005.device.txt          # wall 与 ops 对比基线 39.5s/16438
```

**SC-001**: op 数降 ≥6%、wall 降 ≥4%。**SC-006 口径**: 若全链路"前 16 行全零"设备侧问题仍在（cos ~0.2 且行 0-15 全零），以 op 数/wall/host 恒等对拍为准，精度门挂起并记录——这是共享设备状态问题，不是本特性回归。

## 5. 逃生门验证

```bash
hnnx_compile ... --no-pass view_fold --no-pass eltwise_const_fold --no-pass fc_bias_fold ...
# 期望: 编译成功; opt-stats 中三 pass = 0/0; 产物与 e4d1dbb 前行为一致(对照 T012)
```
