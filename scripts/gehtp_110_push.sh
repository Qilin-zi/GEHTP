#!/bin/bash
# gehtp_110_push.sh — 105 → 110 服务器高通板 (d0f1784) 一键推送+冒烟
# 用法: ./gehtp_110_push.sh [conv_add|l0|full]
set -euo pipefail
H110="speech@192.168.0.110"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no $H110"
SCP="scp -o BatchMode=yes -o StrictHostKeyChecking=no"
DEV=d0f1784
BD=/data/local/tmp/hvxhmx23
STAGE=/tmp/gehtp_110
SEL="${1:-conv_add}"

echo "== 1. staging → 110 host =="
$SSH "mkdir -p $STAGE"
case "$SEL" in
  conv_add)
    cp /disk1/GEHTP/test_models/conv_add/conv_add_qnn/blob.wtop $STAGE/
    cp /disk1/GEHTP/test_models/conv_add/in0.f16.raw $STAGE/input.f16.raw
    printf 'blob %s/gehtp/blob.wtop\ninput %s/gehtp/input.f16.raw\noutput %s/gehtp/output.f16.raw\nout_temp 4\n' "$BD" "$BD" "$BD" > $STAGE/job.txt
    ;;
  l0)
    cp /disk1/GEHTP/test_models/qwen35_08b/layers/layer_0/layer_0.wtop $STAGE/
    cp /disk1/GEHTP/test_models/qwen35_08b/layers/layer_0/in_ncf.f16.raw $STAGE/input.f16.raw
    printf 'blob %s/gehtp/layer_0.wtop\ninput %s/gehtp/input.f16.raw\noutput %s/gehtp/output.f16.raw\nout_temp 7\n' "$BD" "$BD" "$BD" > $STAGE/job.txt
    ;;
  full)
    cp /disk1/GEHTP/test_assets/qwen35_08b/qwen35_08b_scatter.wtop $STAGE/ 2>/dev/null || true
    # 主输入 = input_ids int32 (slot0 EXT_IN)
    cp /tmp/gehtp_110_smoke/ids_input.i32.raw $STAGE/input.f16.raw 2>/dev/null || true
    printf 'blob %s/gehtp/qwen35_08b_scatter.wtop\ninput %s/gehtp/input.f16.raw\noutput %s/gehtp/output.f16.raw\nout_temp 1401\n' "$BD" "$BD" "$BD" > $STAGE/job.txt
    ;;
esac

echo "== 2. scp 到 110 host =="
$SCP $STAGE/*.{wtop,raw,txt} $H110:$STAGE/ 2>/dev/null || $SCP $STAGE/* $H110:$STAGE/

echo "== 3. 110 host → 板 ($BD) =="
$SSH "cd $STAGE && adb -s $DEV shell 'mkdir -p $BD/gehtp /data/local/tmp/hrt/gehtp' && \
  for f in *.wtop input.f16.raw job.txt; do adb -s $DEV push \$f $BD/gehtp/\$f; done"

echo "== 4. 板上跑 runner =="
$SSH "adb -s $DEV shell 'cd $BD && ADSP_LIBRARY_PATH=$BD CDSP_LIBRARY_PATH=$BD ./run_main_on_hexagon 3 gehtp_runner.so'"

echo "== 5. 读结果 + 拉输出 =="
$SSH "adb -s $DEV shell 'cat /data/local/tmp/hrt/gehtp/42_gehtp_runner.txt 2>/dev/null | tail -6'"
$SSH "adb -s $DEV pull $BD/gehtp/output.f16.raw $STAGE/output.f16.raw" 2>/dev/null || true
mkdir -p /tmp/gehtp_110_out
$SCP $H110:$STAGE/output.f16.raw /tmp/gehtp_110_out/${SEL}_out.f16.raw 2>/dev/null || echo "(无输出文件)"
echo "== done: /tmp/gehtp_110_out/${SEL}_out.f16.raw =="
