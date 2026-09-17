# third_party — 外部组件收编(避免外部修改污染)

## run_main_on_hexagon/
**来源**: `/disk2/QCtools/Hexagon_SDK_6.6.0.0/libs/run_main_on_hexagon/`(Qualcomm Hexagon SDK 6.6.0.0)
**收编日期**: 2026-09-16
**用途**: host 端 launcher + DSP skel,GEHTP 例子在设备上以 unsigned-PD 跑 DSP .so 的唯一入口

### 文件
- `src/*.c` — host(run_main_on_hexagon_ap.c) + DSP skel(run_main_on_hexagon_dsp.c) + 仿真(_sim.c) + 测试(test_main.c) 源码
- `inc/run_main_on_hexagon.idl` — FastRPC IDL 接口定义
- `Makefile` + `*.min` — SDK 构建系统文件(依赖 HEXAGON_SDK_ROOT 环境变量)
- `ship/run_main_on_hexagon` — 预编译 host 二进制(ARM aarch64 Android,50KB)
- `ship/librun_main_on_hexagon_skel.so` — 预编译 DSP skel(Hexagon DSP6,31KB,支持 unsigned PD dom3/4)

### 重建(如需)
```bash
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0
cd kernels/third_party/run_main_on_hexagon
make V=android_aarch64   # host 端
make V=hexagon_toolv19_v81  # DSP skel
# 产物在 ship/ 对应子目录
```
**注意**: 构建依赖 SDK 头文件(remote.h/dsp_capabilities_utils.h)和 fastrpc 库,SDK 保持外部只读引用。

### 部署
build_examples.sh 会把 `ship/run_main_on_hexagon` 和 `ship/librun_main_on_hexagon_skel.so` 推到设备 `/data/local/tmp/hvxhmx23/`(或 GEHTP_TMP_DIR 指定的临时目录)。

## swiv/
**来源**: `/disk2/QCtools/swiv_build_utility.py`
**收编日期**: 2026-09-16
**用途**: SWIV(Secure World Isolation Vault)签名工具,给 DSP .so 加签使其能在 unsigned PD 加载

### 用法
```bash
python3 kernels/third_party/swiv/swiv_build_utility.py -i input.so -o output.signed.so
```
build_libs.sh 和 build_examples.sh 已改为引用本地副本(原 `SWIV_TOOL` 环境变量默认值)。
