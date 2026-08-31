# PROVENANCE — GEHTP 各目录来源登记

原则:本仓库代码全部来自 /disk2 既有工程(只读)的拷贝迁移,按工作树状态登记,不改写来源。

## compiler/ ← /disk2/REQNNFRAME/REQNN (B线)

- 拷贝日期: 2026-08-31
- 来源无 VCS(REQNNFRAME 不在 git 管理下)
- 排除项: `build/`、`build2/`(Windows MinGW 陈旧 .exe)、`*.exe`、
  `reference/skel_crc/`(25MB Qualcomm 专有 .so,不进 git)、
  `our_simple_linear_context.bin`(构建产物,可由测试重生成)、
  根目录误生成的 `C:\Users\RQILIN\...` 文件(Windows 路径误作文件名)
- 本仓库修改(唯一): 无。后续开发全部在 compiler/ 内进行。

## kernels/ ← /disk2/V81Dev/hvxhmx_libsV2.3

- 拷贝日期: 2026-08-31
- 来源 git: HEAD `877ed37` "V2.3: 新增 U19 bledger — Buffer Ledger 数据流溯源审计, 例 33 设备 9/9, 33/33 例 181 门全绿"
- 来源 remote: https://gitee.com/dumppool/hvxhmx_libs-v2.3.git (master)
- **拷贝按工作树状态**(非 HEAD):来源有未提交修改(M: build_libs.sh、
  docs/api_v23_overview.md、examples/*、include/hvxhmx_v23.h、lib/*、results/*)
  与未跟踪文件(??: docs/api_v23_btrack.md、api_v23_dmaring.md、examples/34_dmaring、
  35_btrack、36_absoak、EXAMPLES_cn.md、include/bflush.h、btrack.h、dcache.h)
- 排除项: `.git/`、`build/`、`lib/`(编译产物 .so)、`results/`(设备运行日志;
  与上游 .gitignore 一致)
- 本仓库修改:
  1. `build_libs.sh`: SWIV 默认路径 `/disk1/swiv_build_utility.py`(已失效)
     → `/disk2/QCtools/swiv_build_utility.py`
- 注意: 例号 22 已被 `examples/22_dualcore_threads` 占用;
  本项目的 conv2d+add 设备例使用 **37_conv2d_add**(阶段 9 新建)。

## scripts/device_run.sh ← /disk2/V81Dev/vtcm_engine_probe/run.sh

- 拷贝日期: 2026-08-31
- 未经修改的模板副本(文件头已加来源注释);阶段 9/10 改写为 GEHTP 原生上板脚本

## 外部只读依赖(不拷贝,按绝对路径调用)

| 依赖 | 路径 | 用途 |
|---|---|---|
| QAIRT SDK 2.48.40.260702 | /disk2/QCtools/qairt_2.48.40.260702 | qairt-converter / qairt-net-run / golden 生成 |
| SWIV 签名工具 | /disk2/QCtools/swiv_build_utility.py | skel 签名 |
| Python 3.10 环境 | /disk2/Qwen35dev/revlibHtpPrepare/venv310 | onnx 生成 + converter |
| Hexagon SDK 6.6.0.0 | /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0 | hexagon-clang (设备编译) |
| 目标设备 | 52f67807 (adb) | V81 上板,项目铁律只此一台 |

## 工作区铁律

- 本机(104)只可写 /disk2/REQNNFRAME 与 /disk2/GEHTP;/disk2/V81Dev、/disk2/QCtools、
  /disk2/Qwen35dev 一律只读,开发只发生在 GEHTP。
- 上游修复不回写来源目录;如需回馈,由用户另行决策。
