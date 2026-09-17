# GEHTP 上板 (52f67807) 操作手册 — 会话交接

> 写给后续会话：从源码到设备跑通的完整链路 + 签名/skel/权限的全部坑。
> 铁律：**设备只用 52f67807**（adb serial），runner 源码在 `kernels/examples/42_gehtp_runner/`。

---

## 0. 一句话架构

```
ONNX → [host] gehtp compile → model.wtop (权重+runlist+槽位+TEMPOFF 静态偏移, 单文件)
     → [device] gehtp run → 通用 gehtp_runner.so (wt_parse→wt_exec_run_io) 零修改执行
```

设备侧执行器是**通用**的：同一份 `gehtp_runner.so` 跑任意编译产出的 blob，不随模型改。

---

## 1. 正门三条命令

```bash
scripts/gehtp setup --device 52f67807      # 一次性: 编译+签名+推送 内核库与 runner
scripts/gehtp compile model.onnx -o model.wtop [--input-f16 in.f16.raw] [--net-json X --weights-bin Y]
scripts/gehtp run model.wtop --input in.f16.raw --output out.f16.raw --device 52f67807
```

依赖绝对路径（`/disk2` 已失效，全部在 `/4090disk2`）：
- QAIRT SDK: `/4090disk2/QCtools/qairt_2.48.40.260702`
- venv310 python: `/4090disk2/Qwen35dev/revlibHtpPrepare/venv310/bin/python3`
- SWIV 签名工具: `/4090disk2/QCtools/swiv_build_utility.py`
- Hexagon SDK: `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0`（本机，仍在）

`scripts/gehtp` 顶部 `SDK=/ PY=/ SWIV=` 三个变量已指向 `/4090disk2`（本树已改；若从 4090 只读树拷贝需重改）。

---

## 2. 设备侧四件套（gehtp setup 全做）

推到 `/data/local/tmp/hvxhmx23/`：

| 文件 | 来源 | 作用 |
|---|---|---|
| `run_main_on_hexagon` | 设备既有（不从源码构建） | FastRPC 装载器，host 侧 adb 执行它 |
| `librun_main_on_hexagon_skel.so` | **`/data/local/tmp/hvxhmx_libs/` 拷贝** | skel（DSP 侧桩），**必须用 hvxhmx_libs 的 2026-08-10 版**，/data/local/tmp 顶层旧版报 `0x80000406` |
| `libhvxhmx_v23.so` | `kernels/lib/libhvxhmx_v23.signed.so`（SWIV 签） | HVX/HMX 算子库 |
| `gehtp_runner.so` | host 交叉编译 `kernels/examples/42_gehtp_runner/main.c` + SWIV 签 | 通用 WTOP runner |

### 2.1 签名（SWIV）— 为什么必须签

- DSP 进程默认跑 **unsigned PD**；未签名的 .so 加载报 `Unsigned PD is not supported on domain 3` / `Error -2147482611 Failed to call main() on DSP`。
- `gehtp setup` 对 runner 和内核库都跑 `python3 $SWIV -i x.so -o x.signed.so`，然后**推签名后的件**（注意推上去时改回不带 `.signed` 的名字：`gehtp_runner.so`、`libhvxhmx_v23.so`）。
- **vgather=0 铁律**：runner 编译后必须 `hexagon-llvm-objdump -d x.so | grep -c vgather` 为 0，否则设备挂。`gehtp setup` 内置此检查。

### 2.2 权限（本次实踩）

- `gehtp setup` 用 `adb push` 推件，落到设备是 `root` 所有、`rw-rw-rw-`。**runner .so 必须 `chmod 755`**，否则 `Failed to call main() on DSP`。
- 若 `adb shell` 是 `shell(2000)` 而非 root，`/data/local/tmp/hvxhmx23/gehtp/` 可能 `Permission denied`。解法：`adb -s 52f67807 root`（adbd 已 root 时直接生效），再操作。
- `gehtp` 工作目录 `/data/local/tmp/hvxhmx23/gehtp/` 与 runner 结果目录 `/data/local/tmp/hrt/gehtp/` 都要可写。

---

## 3. 手动上板（不依赖 gehtp run，调试用）

`gehtp run` 内部就是这些步骤，失败时可逐条手动做：

```bash
D=52f67807
DEVROOT=/data/local/tmp/hvxhmx23
GDIR=$DEVROOT/gehtp

# 1) 推 blob + 输入 + job.txt
adb -s $D push model.wtop        $GDIR/model.wtop
adb -s $D push in.f16.raw        $GDIR/input.f16.raw
# job.txt 4 行: blob/input/output 绝对路径 + out_temp (来自 blob.manifest.json 的 output_temp)
printf 'blob %s/model.wtop\ninput %s/input.f16.raw\noutput %s/output.f16.raw\nout_temp %s\n' \
    "$GDIR" "$GDIR" "$GDIR" "7" > /tmp/job.txt
adb -s $D push /tmp/job.txt $GDIR/job.txt

# 2) 跑 runner (domain 3, 环境变量指库路径)
adb -s $D shell "cd $DEVROOT && ADSP_LIBRARY_PATH=$DEVROOT CDSP_LIBRARY_PATH=$DEVROOT \
    ./run_main_on_hexagon 3 gehtp_runner.so"

# 3) 看结果日志 + 拉输出
adb -s $D shell "cat /data/local/tmp/hrt/gehtp/42_gehtp_runner.txt"
adb -s $D pull $GDIR/output.f16.raw out.f16.raw
```

- runner 结果日志在 **`/data/local/tmp/hrt/gehtp/42_gehtp_runner.txt`**（`ex_open_result` 写的固定路径，不在 `hvxhmx23/gehtp/`）。判据行 `[PASS] N ops, out temp T = B bytes -> ...`。
- 若 `job.txt` 内容不对（比如残留旧的 `prof_wp` 路径），runner 会跑错 blob——`gehtp run` 每次都重写 job.txt，手动跑前 `cat` 确认。

---

## 4. 排错速查（本次全踩过）

| 症状 | 根因 | 解法 |
|---|---|---|
| `Unsigned PD is not supported on domain 3` / `Error -2147482611` | runner/库未签名 **或 .so 权限不是 755** | 重跑 `gehtp setup`（含 SWIV 签）；`chmod 755 gehtp_runner.so libhvxhmx_v23.so` |
| `domain 3 failed to open (0x80000406)` | skel 版本旧 | `cp /data/local/tmp/hvxhmx_libs/librun_main_on_hexagon_skel.so → hvxhmx23/` |
| `Permission denied` on `hvxhmx23/gehtp/` | adb shell 非 root | `adb -s $D root` |
| `no output on device` / runner 日志空 | job.txt 路径错或 out_temp 错 | `cat $GDIR/job.txt` 核对；out_temp 看 `*.wtop.manifest.json` |
| `Failed to call main() on DSP` 但已签名 | .so 权限 `rw-rw-rw-` 非 `rwxr-xr-x` | `chmod 755` |
| push 卡住/EXIT=1 无输出 | blob 大 + 权限或 adb 抽风 | 先 `adb root`；大 blob 用 `gehtp run`（内置尺寸一致免重推） |

---

## 5. 设备纪律（PORTAL §4，铁律）

1. 跑前查占用：`adb -s 52f67807 shell "ps | grep run_main_on_hexagon"`，有主就等。
2. `adb devices` 突然空 = host adb server 抽风，`adb kill-server` 即愈，**板子没坏，别重启**。
3. adb 掉线会孤儿设备侧 `run_main_on_hexagon` + DSP main；排查先 ps 看孤儿。
4. CMA 耗尽 / `failed to get map da` → 停下通知用户，**不自行 reboot**。

---

## 6. 当前 C1 状态（2026-09-17）

- host 参考链 L0 已闭合（cos=1.0）：NCF 输入布局 + Gather compact axis 前导1对齐（commit `2b60f41`）。
- L0 blob（894 op）已重生成，设备 runner 已跑通 894 op 全执行、temp7 写 65536B。
- **设备输出 vs gold cos=0.10** —— 根因 = **ScatterNd 恒等拷贝**（A3 暗雷）：`compiler/tools/wtop_ops/op_copy_sem.cpp` 把 ScatterNd 发射成恒等，attn_iter 在设备上未真正迭代。这是 L0 设备闭合的最后缺口，下一步 = 真 ScatterNd（emit + 设备 opcode + host 对齐）。
