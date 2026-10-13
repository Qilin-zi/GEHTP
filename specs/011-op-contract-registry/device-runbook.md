# 011 设备侧收口 runbook（T008 + T013）

> 目标：把「源级多表漂移已根除」的 011 重构在**设备二进制**上闭环——重编 `.signed.so`（T013）后证明 35 例 + 261 门 + W4A16 65536/65536 仍逐位一致（T008）。
>
> 背景约束（research §4 / [[gehtp-device-lib-stale-arity]]）：本迁移 **byte-exact**，设备回归是「无回归」判据而非「改行为」。host 侧 parity / 行为 / 覆盖率已全绿，本 runbook 只覆盖仍需设备的两步。

## 0. 前置

```bash
cd /disk1/gehtp
source env.sh                    # HEXAGON_SDK_ROOT 默认是失效老路径，必须先 source
export DEVICE=d0f1784            # 0.8B 线 = SA8797P d0f1784；kernels 例集历史 = 52f67807
adb -s "$DEVICE" shell "ps -A | grep -E 'run_main_on_hexagon|gehtp'"   # 有主就等
```

- 设备二选一：`d0f1784`（SA8797P，Hexagon V81，近月 0.8B 线在跑）或 `52f67807`（kernels V2.2 例集历史设备）。`build_examples.sh`/`scripts/gehtp` 默认 `52f67807`，用 `DEVICE`/`--device` 覆盖。
- 设备多会话共享（见 [[gehtp-p5-device-gate]]）：job.txt / op_ts.bin / input 通道会被并发会话互相覆盖。若并发在途，改用私有名（`-DGEHTP_JOB_PATH=... -DRESULT_NAME=...` + 私有 lib 名）。本 runbook 假定独占设备。

---

## 1. T013 —— 重建设备 `.signed.so`

**核心教训**（stale-arity 坑的成因）：`build_libs.sh one <base>` 只重编单 `.o`；`gehtp setup` 里 `[ -f .signed.so ] || build_libs.sh` **只要签名件存在就跳过重建**。011 动了 `oplist_parse.c` + `oplist_parse.h`（新增 `op_spec.inc` include、删 `WT_ARITY_*` 宏），被 `oplist_exec.c`/`wt_w3.c`/`42_gehtp_runner/main.c` 等 include，**必须全量重编**，不能走 `one`。

```bash
cd /disk1/gehtp/kernels && source ../env.sh

# ① 显式全量重编（不要依赖 setup 的 [ -f ] || 短路；全量而非 one，因头文件变了）
./build_libs.sh
```

build_libs.sh 尾部自动：relink `lib/libhvxhmx_v23.so` → vgather grep 必须 0 → UNDEF 白名单扫描 → SWIV 签出 `libhvxhmx_v23.signed.so`。**任一门非 0 即停**（vgather 存在会 crash CDSP）。

```bash
# ② 确认签名件已刷新（时间戳/大小应变化；没变 = 上一步被短路跳过，回去查）
ls -l --time-style=+%H:%M:%S lib/libhvxhmx_v23.so lib/libhvxhmx_v23.signed.so

# ③ 推设备（setup 会再推 lib + runner + skel 治理；[1/5] 见 .signed.so 已在，直接推）
cd /disk1/gehtp && scripts/gehtp setup --device "$DEVICE"
```

**判定**：
- ③ 输出 `device side ready: .../libhvxhmx_v23.so, gehtp_runner.so, ...` 且无 die。
- 设备侧 lib 已更新：`adb -s "$DEVICE" shell "ls -l /data/local/tmp/hvxhmx23/libhvxhmx_v23.so"` 时间戳 = 本次。

**快速冒烟**（可选，确认 wt_parse 在新 lib 上 rc=0）：
```bash
scripts/gehtp run <已知好 blob> --input <in.f16.raw> --output /tmp/o.f16.raw --device "$DEVICE"
# 判据：无 rc=-11 / rc=-10；runner 日志 [PASS] N ops
```

---

## 2. T008 —— 设备 byte-exact 回归

### 2.1 host 侧 wt_parse 负例（已做，可复跑确认）

`oplist_parse.c` 平台无关，负例在 host 单测覆盖：未知 opcode → `WT_ERR_OPCODE`、arity 不匹配 → `WT_ERR_ARITY`。永久覆盖在 `compiler/tests/test_wtop_emit.cpp`（`test_op_spec_registry` + round-trip 解析）。

### 2.2 设备 kernels 例集（主口径：35 例 + 261 门 + W4A16 65536/65536）

```bash
cd /disk1/gehtp/kernels/examples
# 强制 host 工具对拍新源码（pack_oplist/wt_inspect 链 oplist_parse.c，[ -x ] 短路会复用旧件）
rm -f ../build/host/pack_oplist ../build/host/wt_inspect
DEVICE="$DEVICE" ./build_examples.sh all
```

**判定**（`results/` 汇总）：
- `35/35 例全绿`、`261 门 [PASS] 0 FAIL`（api_v23_overview 口径：例 15 内嵌 49 项、例 34=26 门、例 35=54 门，合计 261）。
- 例 31 `gemm_dispatch`：`G2a GR_W4A16 M=256: 65536/65536 exact`（256³ 全引擎 vs Y_gold byte-exact）。
- 任一例 FAIL = 回退定位。本迁移只动解析层元数据，不应触碰任何数值路径——若出现差异，优先怀疑设备 lib 没真正重编（回到 T013 ① 查 `.signed.so` 时间戳）。

### 2.3 0.8B E2E（可选加固，属 0.8B 线口径）

011 只动 kernels 解析层，kernels 例集（2.2）是主裁判。若要顺带确认 0.8B 线不受影响：

```bash
scripts/gehtp run <08b blob> --input <in.f16.raw> --output /tmp/o.f16.raw --device "$DEVICE"
python3 scripts/check_device_golden.py /tmp/o.f16.raw <gold.f16.raw> \
    --rows 32 --top1-min 32/32 --cos-min 0.998646
```

> cos 门 = **0.998646**（共享基线「不回归」口径），不是 `judge_logits.py` 的 0.9999（那是 M4.4 期望值）——见 [[gehtp-validation-line]]。
>
> **已跑**（2026-10-13，FF 到主线 c77665a 后）：`qwen35_08b_gdn_default.wtop`（884 ops，input=ids_from_golden.i32.raw）→ judge_logits.py = cos 0.999930 / top1 32/32 / max|d| 0.2047；exec 2.771s（gdn 默认开锚点 2.76s 持平），wall 7.14s。精度性能双无回归。

---

## 3. 完成后

- [X] T013：`.signed.so` 全量重编 + 推送 + 冒烟 rc=0（lib 278376→300808B，含 op_spec.inc 表 + wt_w3.c 槽行 varargs 修复）。
- [X] T008：`build_examples.sh all` = **312 pass / 10 fail**；10 fail 全为预存范围外例（13_compat_dlsym / 37_conv2d_add / 38_transformer_ops / 39_probe / 40_qwen_layer3 / 41_qwen_layer0），011 相关门（例 21 5/5 + `w3_host_device_lines_identical` PASS、例 19/25/43 全绿）已闭合。
- [X] 勾掉 `tasks.md` 的 T008 / T013。
- [ ] 设备多会话残留：`adb -s "$DEVICE" shell "ps -A | grep gehtp"` 确认无遗留实例（见 [[gehtp-p5-device-gate]] 幽灵会话）。

## 4. 排错速查

| 症状 | 根因 | 解法 |
|---|---|---|
| 回归 FAIL 且 `.signed.so` 时间戳没变 | setup `[ -f ] \|\|` 短路跳过重建 | 回 T013 ① 显式 `./build_libs.sh` 再 setup |
| `Unsigned PD is not supported on domain 3` / `-2147482611` | lib/runner 未签名 或 .so 权限非 755 | 重跑 `gehtp setup`；`adb shell chmod 755 /data/local/tmp/hvxhmx23/*` |
| `domain 3 failed to open (0x80000406)` | skel 版本旧 | `gehtp setup`（[0/5] 会治理 skel SWIV 段） |
| 结果与并发会话串了 | 共享 job.txt/op_ts.bin/input | 独占设备，或全套私有名（见 0.） |
| `adb devices` 空 | host adb server 抽风 | `adb kill-server` 即愈，板子没坏 |
| W3 槽行 host/设备 `offset`/`addr`/`sha256` 漂移（例 21 `w3_host_device_lines_identical` FAIL） | `wt_w3.c` 用 `%u` 吃 `uint64_t offset`：x86-64 寄存器碰巧对齐，Hexagon 栈 varargs 把 u64 拆两字并插 4B 对齐洞 → offset 读洞、addr 读 offset 低字、`%s` 读高字(=NULL→"") | cast `(uint32_t)w->slots[i].offset`，重编 lib + host `wt_inspect`（见 [[gehtp-wt-w3-varargs-align]]；上游 c77665a 已用 `%llu` + `(unsigned long long)` 同根修复，新基线无需再改） |
