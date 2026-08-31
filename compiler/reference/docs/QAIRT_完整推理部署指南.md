# SA8255P 板上 ONNX/Torch 模型推理完整指南

> **平台**: SA8255P (Snapdragon Ride, LeMans), HTP v73, OE Linux aarch64  
> **SDK**: Qualcomm AI Runtime (QAIRT) v2.48  
> **适用模型**: ONNX / PyTorch → QNN → HTP backend 推理  
> **编写日期**: 2026-08-03  
> **状态**: 全流程已验证跑通 ✅

---

## 目录

1. [架构与环境总览](#1-架构与环境总览)
2. [三台机器的分工](#2-三台机器的分工)
3. [第一步：准备 PC 环境](#3-第一步准备-pc-环境)
4. [第二步：部署 QAIRT runtime 到板子](#4-第二步部署-qairt-runtime-到板子)
5. [第三步：生成 ONNX 模型](#5-第三步生成-onnx-模型)
6. [第四步：转换 ONNX → QNN（PC 上）](#6-第四步转换-onnx--qnnpc-上)
7. [第五步：交叉编译 .so（dev133 服务器）](#7-第五步交叉编译-so-dev133-服务器)
8. [第六步：推到板子并运行 HTP 推理](#8-第六步推到板子并运行-htp-推理)
9. [第七步：验证推理结果](#9-第七步验证推理结果)
10. [完整一键脚本](#10-完整一键脚本)
11. [常见问题与排查](#11-常见问题与排查)
12. [关键路径速查表](#12-关键路径速查表)

---

## 1. 架构与环境总览

```
┌─────────────────────────────────────────────────────┐
│                    开发流程                          │
│                                                     │
│  [PC/Windows]          [dev133/Linux]   [SA8255P板] │
│                                                     │
│  Python + ONNX        交叉编译器        QAIRT runtime│
│  qnn-onnx-converter   (aarch64-g++)     libQnnHtp.so │
│       │                    │              libQnnHtp  │
│       ▼                    ▼              V73Skel   │
│  .cpp + .bin  ──scp──→ libxxx.so ──scp──→ /data/qairt│
│                                                    │
│                              qnn-net-run (HTP推理)  │
└─────────────────────────────────────────────────────┘
```

### 三台机器各自的角色

| 机器 | 系统 | 作用 | 关键工具 |
|------|------|------|---------|
| **PC** (你的电脑) | Windows | 模型生成 + ONNX→QNN 转换 | Python, qnn-onnx-converter |
| **dev133** (构建服务器) | Linux x86 | 交叉编译 .cpp → ARM .so | qnn-model-lib-generator, aarch64-linux-gnu-g++ |
| **SA8255P 板** | OE Linux aarch64 | HTP/NPU 推理 | qnn-net-run, libQnnHtp.so |

### 为什么需要三台？

- **PC**：QAIRT 的 converter 是 Python 脚本，PC 上装 Python 即可跑
- **dev133**：把 QNN `.cpp` 编译成 ARM `.so` 需要 aarch64 交叉编译器，PC 上没有，dev133 上有
- **板子**：HTP 推理只能在 NPU 硬件上跑，必须在板子上执行

---

## 2. 三台机器的分工

### dev133 服务器信息

```
IP:       10.137.185.133
SSH 账号: rqilin
密码:     Wangba521.    (注意: keyboard-interactive 认证)

QAIRT SDK 路径:
  /opt/qcom/aistack/qairt/2.45.41.260507    (推荐，与板子 v2.48 兼容)
  /opt/qcom/aistack/qairt/2.46.0.260424
  /opt/qcom/aistack/qairt/2.42.0.251225

交叉编译器:
  /usr/bin/aarch64-linux-gnu-g++          (系统级，gcc-13)

model-lib-generator:
  /opt/qcom/aistack/qairt/2.45.41.260507/bin/x86_64-linux-clang/qnn-model-lib-generator
```

### PC 上的本地工具

```
ADB:        C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe
Python:     C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe
QAIRT SDK:  C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626
Plink/Pscp: C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe  (用于 SSH)
工作目录:   C:\Users\RQILIN\Documents\Default Project
```

### 板子上的 QAIRT 部署位置

```
/data/qairt/
├── env.sh                          # 环境变量脚本
├── bin/                            # 命令行工具
│   ├── qnn-net-run                 # 推理工具
│   ├── qnn-platform-validator      # 硬件验证工具
│   └── ...
├── lib/                            # ARM 侧 runtime 库
│   ├── libQnnHtp.so               # HTP backend (核心!)
│   ├── libQnnCpu.so               # CPU backend
│   ├── libQnnHtpV73Stub.so        # HTP v73 stub (已加 CRC)
│   ├── libQnnHtpV73CalculatorStub.so
│   └── ...
└── lib/hexagon-v73/unsigned/      # DSP 侧 skel 库
    ├── libQnnHtpV73Skel.so        # HTP v73 skel (核心!)
    ├── libCalculator_skel.so      # calculator 测试
    ├── fastrpc_shell_unsigned_3   # ⚠️ 必须! DSP unsigned PD shell
    ├── libc++.so.1                # ⚠️ 必须! DSP C++ 运行时
    ├── libc++abi.so.1              # ⚠️ 必须! DSP C++ ABI
    └── ...
```

---

## 3. 第一步：准备 PC 环境

### 3.1 安装 ADB（如已有可跳过）

```powershell
# 检查是否已装
& "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe" version
```

### 3.2 安装 Python（如已有可跳过）

PC 上需要 Python 3.10（**不能用 3.11/3.12，QAIRT converter 只支持 3.10/3.12**）。

```powershell
# 检查
& "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe" --version
# 应输出: Python 3.10.x
```

### 3.3 安装 Python 依赖（一次性）

```powershell
$py = "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe"
& $py -m pip install "numpy<2" "onnx==1.16.2" pyyaml pandas packaging --no-warn-script-location
```

> ⚠️ **关键版本要求**:
> - `onnx==1.16.2`（新版 onnx 1.22 改了 API，converter 不兼容）
> - `numpy<2`（numpy 2.x 和 QAIRT 不兼容）
> - `pandas`（converter 的 arch_linter 需要）

### 3.4 配置 QAIRT converter 环境变量

每次在 PC 上跑 converter 前，必须设置:

```powershell
$sdk = "C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626"
$env:PATH = "$sdk\lib\x86_64-windows-msvc;$env:PATH"        # QNN 原生 DLL
$env:PYTHONPATH = "$sdk\lib\python"                          # qti.aisw 模块
$env:PYTHONIOENCODING = "utf-8"                              # 避免 GBK 编码错误
$env:PYTHONUTF8 = "1"
```

> ⚠️ **常见问题**: 不设 `PYTHONIOENCODING=utf-8` 会在中文 Windows 上报 `UnicodeEncodeError: 'gbk' codec`

### 3.5 下载 plink/pscp（SSH 工具，如已有可跳过）

```powershell
# 用于和 dev133 服务器通信
# 已下载到: C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe
# pscp.exe 在同目录
```

---

## 4. 第二步：部署 QAIRT runtime 到板子

> **如果板子已经部署过 QAIRT（`/data/qairt/` 存在），可跳过此步。**
> 运行验证命令确认:
> ```powershell
> & "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe" shell "ls /data/qairt/env.sh"
> ```

### 4.1 连接板子并获取 root

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"

# 确认设备连接
& $adb devices
# 应看到: 3ac4787  device

# ⚠️ 关键: 必须 adb root 获得完整权限!
& $adb root
# 输出: restarting adbd as root

# 等待重连
Start-Sleep -Seconds 5
& $adb wait-for-device

# 验证是 root
& $adb shell "id"
# 应输出: uid=0(root) ...
```

> ⚠️ **常见问题**: 不 `adb root` 的话，uid=2000(adb)，CapEff=0，所有 DSP 调用都会失败

### 4.2 推送 runtime 库（首次部署）

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$sdk = "C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626"

# 创建目录
& $adb shell "mkdir -p /data/qairt/lib /data/qairt/lib/hexagon-v73/unsigned /data/qairt/bin"

# 推送 ARM 侧 runtime 库
$armLibs = Get-ChildItem "$sdk\lib\aarch64-oe-linux-gcc11.2\*.so"
foreach ($lib in $armLibs) {
    & $adb push $lib.FullName "/data/qairt/lib/$($lib.Name)"
}

# 推送 DSP 侧 skel 库
$dspLibs = Get-ChildItem "$sdk\lib\hexagon-v73\unsigned\*"
foreach ($lib in $dspLibs) {
    & $adb push $lib.FullName "/data/qairt/lib/hexagon-v73/unsigned/$($lib.Name)"
}

# 推送命令行工具
$bins = @("qnn-net-run","qnn-platform-validator","qnn-profile-viewer",
          "qnn-throughput-net-run","qnn-context-binary-generator",
          "qnn-context-binary-utility","qairt-net-run","qairt-converter",
          "qairt-quantizer","qairt-dlc-prepare","qairt-dlc-info",
          "qtld-net-run","snpe-net-run","snpe-platform-validator")
foreach ($b in $bins) {
    $f = "$sdk\bin\aarch64-oe-linux-gcc11.2\$b"
    if (Test-Path $f) { & $adb push $f "/data/qairt/bin/$b" }
}
& $adb shell "chmod +x /data/qairt/bin/*"
```

### 4.3 ⚠️ 关键：复制 DSP 运行时依赖文件

**这是最容易漏的一步！** 这些文件板子上有，但不在 QAIRT 搜索路径里，必须手动复制:

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$dst = "/data/qairt/lib/hexagon-v73/unsigned"

# 1. fastrpc_shell_unsigned_3  (DSP 启动 unsigned PD 必需)
& $adb shell "cp /vendor/dsp/cdsp/fastrpc_shell_unsigned_3 $dst/"
& $adb shell "cp /vendor/dsp/cdsp/fastrpc_shell_3 $dst/"
& $adb shell "cp /vendor/dsp/cdsp1/fastrpc_shell_unsigned_4 $dst/"

# 2. libc++.so.1 + libc++abi.so.1  (DSP 侧 C++ 运行时，skel 库依赖)
& $adb shell "cp /vendor/dsp/cdsp/libc++.so.1 $dst/"
& $adb shell "cp /vendor/dsp/cdsp/libc++abi.so.1 $dst/"
```

> ⚠️ **不复制会报什么错**:
> - 缺 `fastrpc_shell_unsigned_3`: `apps_std_fopen FAIL(2)` / `dsp proc_init failed, err=2`
> - 缺 `libc++.so.1`: `dsp dlopen error: cannot open libc++.so.1, errno 67`

### 4.4 创建环境变量脚本

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$script = @"
#!/bin/sh
export QAIRT_ROOT=/data/qairt
export LD_LIBRARY_PATH=`$QAIRT_ROOT/lib:`$QAIRT_ROOT/lib/hexagon-v73/unsigned:`$LD_LIBRARY_PATH
export PATH=`$QAIRT_ROOT/bin:`$PATH
export ADSP_LIBRARY_PATH=`$QAIRT_ROOT/lib/hexagon-v73/unsigned
export CDSP_LIBRARY_PATH=`$QAIRT_ROOT/lib/hexagon-v73/unsigned
"@
$tmp = "C:\Users\RQILIN\AppData\Local\Temp\qairt_env.sh"
[System.IO.File]::WriteAllText($tmp, $script)
& $adb push $tmp /data/qairt/env.sh
& $adb shell "chmod +x /data/qairt/env.sh"
```

### 4.5 验证 HTP 硬件可用

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
& $adb shell ". /data/qairt/env.sh && qnn-platform-validator --backend dsp --testBackend 2>&1 | tail -15"
```

**期望输出**:
```
QNN is supported for backend DSP on the device.
  Backend Hardware  : Supported
  Backend Libraries  : Found
  Unit Test          : Passed      ← ✅ 成功
```

> **如果 Unit Test: Failed**, 见 [常见问题](#11-常见问题与排查)

---

## 5. 第三步：生成 ONNX 模型

### 方式 A：自己生成测试模型（用于学习）

创建 Python 脚本 `make_model.py`:

```python
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto

# 简单线性层: input(1x3x4) → MatMul(W:4x2) → Add(b:2) → output(1x3x2)
W = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.float32)
b = np.array([0.1, 0.2], dtype=np.float32)

inp = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4])
out = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2])
W_init = onnx.helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
b_init = onnx.helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())

nodes = [
    onnx.helper.make_node("MatMul", ["input","W"], ["tmp"]),
    onnx.helper.make_node("Add", ["tmp","b"], ["output"]),
]
graph = onnx.helper.make_graph(nodes, "simple_linear", [inp], [out], [W_init, b_init])
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
onnx.checker.check_model(model)
onnx.save(model, "simple_linear.onnx")
print("Saved simple_linear.onnx")

# 生成测试输入
x = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]], dtype=np.float32)
x.tofile("input_0_0.raw")
print("Saved input_0_0.raw")

# 保存期望输出用于验证
expected = np.matmul(x, W) + b
expected.tofile("expected_output.raw")
print("Expected output:", expected)
```

运行:
```powershell
$py = "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe"
& $py make_model.py
```

### 方式 B：使用现有 ONNX 模型

如果你已有 `.onnx` 文件，直接跳到第四步。只需准备好输入数据 `.raw` 文件和一个 `input_list.txt`。

**生成 .raw 输入文件**（如果模型需要预处理后的输入）:

```python
import numpy as np
# 示例: 1x3x224x224 float32 输入
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
x.tofile("input_0_0.raw")
```

**创建 input_list.txt**:
```
input_0_0.raw
```

### 方式 C：从 PyTorch 导出 ONNX

```python
import torch
model = MyModel().eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "mymodel.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=13)
```

> ⚠️ **opset_version**: 建议用 13，过高版本 converter 可能不认

---

## 6. 第四步：转换 ONNX → QNN（PC 上）

### 6.1 设置环境变量

```powershell
$py = "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe"
$sdk = "C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626"
$env:PATH = "$sdk\lib\x86_64-windows-msvc;$env:PATH"
$env:PYTHONPATH = "$sdk\lib\python"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$workdir = "C:\Users\RQILIN\Documents\Default Project"
```

### 6.2 创建转换脚本

创建 `convert_model.py`:

```python
import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

def main():
    from qti.aisw.converters import onnx as onnx_frontend
    from qti.aisw.converters.common.utils.converter_utils import log_error
    from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
    from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
    from qti.aisw.converters.common.arch_linter.arch_linter import ArchLinter
    from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
    from qti.aisw.converters.backend.qnn_quantizer import QnnQuantizer
    from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
    from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage

    class ArgParser(ArgParserWrapper):
        def __init__(self):
            super().__init__(formatter_class=CustomHelpFormatter,
                             conflict_handler='resolve',
                             parents=[onnx_frontend.OnnxConverterFrontend.ArgParser(),
                                      IROptimizations.ArgParser(),
                                      QnnQuantizer.ArgParser(),
                                      QnnConverterBackend.ArgParser(),
                                      ArchLinter.ArgParser(),
                                      GraphOptimizer.ArgParser()])

    # ⚠️ 修改这两个路径为你的模型
    INPUT_ONNX = r"C:\Users\RQILIN\Documents\Default Project\simple_linear.onnx"
    OUTPUT_PREFIX = r"C:\Users\RQILIN\Documents\Default Project\simple_linear"

    ap = ArgParser()
    args = ap.parse_args(["-i", INPUT_ONNX, "-o", OUTPUT_PREFIX])
    print("args parsed", flush=True)

    frontend = onnx_frontend.OnnxConverterFrontend(args,
                                                   custom_op_factory=QnnCustomOpFactory(),
                                                   validator=None)
    print("frontend created", flush=True)

    ir_graph = frontend.convert()
    print("convert done", flush=True)

    # QNN backend 优化标志
    args.perform_axes_to_spatial_first_order = True
    args.squash_box_decoder = True
    args.match_caffe_ssd_to_tf = True
    args.adjust_nms_features_dims = True
    args.extract_color_transform = True
    args.preprocess_roi_pool_inputs = True
    args.unroll_lstm_time_steps = True
    args.expand_gru_op_structure = True
    args.unroll_gru_time_steps = True
    args.inject_cast_for_gather = True
    args.force_prune_cast_ops = False
    args.align_matmul_ranks = True
    args.handle_gather_negative_indices = True

    optimizer = IROptimizations(args)
    optimized_graph = optimizer.optimize(ir_graph)
    print("optimized", flush=True)

    backend = QnnConverterBackend(args)
    backend.save(optimized_graph)
    print("=== CONVERT COMPLETE ===", flush=True)

if __name__ == '__main__':
    main()
```

> ⚠️ **必须用 `if __name__ == '__main__':`** 包裹，否则 Windows multiprocessing (shape inference) 会死锁

### 6.3 运行转换

```powershell
& $py -u "$workdir\convert_model.py" 2>&1
```

**成功输出**:
```
args parsed
frontend created
convert done
optimized
Model CPP saved at: ...simple_linear
Model BIN saved at: ...simple_linear.bin
=== CONVERT COMPLETE ===
```

**生成的文件**:
- `simple_linear`（.cpp 源码）
- `simple_linear.bin`（权重数据）
- `simple_linear_net.json`（网络结构）

### 6.4 可能遇到的问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'qti'` | PYTHONPATH 没设 | `$env:PYTHONPATH = "$sdk\lib\python"` |
| `UnicodeEncodeError: 'gbk' codec` | 中文 Windows 编码 | `$env:PYTHONIOENCODING="utf-8"` |
| `ImportError: libPyIrGraph310` | Python 版本不对 | 必须用 Python **3.10** |
| 卡住无输出 | multiprocessing 死锁 | 用 `if __name__=='__main__':` 包裹 |
| `onnx.version` 不存在 | onnx 版本太新 | `pip install onnx==1.16.2` |
| `ModuleNotFoundError: pandas/yaml/packaging` | 依赖没装 | `pip install pandas pyyaml packaging` |

---

## 7. 第五步：交叉编译 .so（dev133 服务器）

> QNN 生成的 `.cpp` 需要编译成 ARM `.so` 才能在板子上跑。PC 上没有 aarch64 交叉编译器，用 dev133。

### 7.1 上传 .cpp 和 .bin 到 dev133

```powershell
$plink = "C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe"
$pscp = $plink -replace "plink","pscp"
$fingerprint = "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"
$workdir = "C:\Users\RQILIN\Documents\Default Project"

# 上传
& $pscp -pw "Wangba521." -hostkey $fingerprint `
  "$workdir\simple_linear" rqilin@10.137.185.133:/home/rqilin/simple_linear.cpp

& $pscp -pw "Wangba521." -hostkey $fingerprint `
  "$workdir\simple_linear.bin" rqilin@10.137.185.133:/home/rqilin/simple_linear.bin
```

### 7.2 在 dev133 上编译

```powershell
$plink = "C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe"
$fingerprint = "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"

$cmd = @"
set -e
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.45.41.260507
export TARGET_PREFIX=aarch64-linux-gnu-
export SDKTARGETSYSROOT=/
cd /home/rqilin
`$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c simple_linear.cpp -b simple_linear.bin -o ./out -t aarch64-oe-linux-gcc11.2
find ./out -name '*.so' -ls
"@
$escaped = $cmd -replace '"','\"'
& $plink -ssh -batch -pw "Wangba521." -hostkey $fingerprint `
  rqilin@10.137.185.133 $escaped 2>&1 | Select-Object -Last 10
```

**成功输出**:
```
Target: aarch64-oe-linux-gcc11.2  Library: /home/rqilin/out/aarch64-oe-linux-gcc11.2/libsimple_linear.so
```

### 7.3 下载 .so 回 PC

```powershell
$pscp = "C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe" -replace "plink","pscp"
$fingerprint = "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"
$workdir = "C:\Users\RQILIN\Documents\Default Project"

& $pscp -pw "Wangba521." -hostkey $fingerprint `
  rqilin@10.137.185.133:/home/rqilin/out/aarch64-oe-linux-gcc11.2/libsimple_linear.so `
  "$workdir\libsimple_linear.so"
```

### 7.4 可能遇到的问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `Could not find compiler: ${TARGET_PREFIX}g++` | 环境变量没设 | `export TARGET_PREFIX=aarch64-linux-gnu-` |
| `Could not find sys root` | SDKTARGETSYSROOT 没设 | `export SDKTARGETSYSROOT=/` |
| SSH 连接超时 | 主机密钥没接受 | 第一次用 `echo y \| plink ...` 接受密钥 |
| keyboard-interactive 提示 | dev133 认证方式 | plink `-pw` 自动处理，忽略提示 |

---

## 8. 第六步：推到板子并运行 HTP 推理

### 8.1 推送文件到板子

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$workdir = "C:\Users\RQILIN\Documents\Default Project"

# 确保是 root
& $adb root
Start-Sleep -Seconds 3
& $adb wait-for-device

# 创建工作目录
& $adb shell "mkdir -p /data/qairt/demo"

# 推送 .so 模型
& $adb push "$workdir\libsimple_linear.so" /data/qairt/demo/

# 推送输入数据和 input_list
& $adb push "$workdir\input_0_0.raw" /data/qairt/demo/
& $adb shell "echo 'input_0_0.raw' > /data/qairt/demo/input_list.txt"
```

### 8.2 运行 HTP 推理

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
& $adb shell ". /data/qairt/env.sh && cd /data/qairt/demo && qnn-net-run --model libsimple_linear.so --backend libQnnHtp.so --input_list input_list.txt --output_dir output 2>&1"
```

**成功输出**:
```
qnn-net-run pid:xxxxx
qnn-net-run build version: v2.48.0.260626120635
Processing inference input(s):
input_0_0.raw
...
Starting stage: Graph Sequencing for Target   ← HTP 专有
Completed stage: Graph Sequencing for Target (700 us)
Starting stage: VTCM Allocation               ← HTP 专有
Completed stage: VTCM Allocation (41 us)
Starting stage: Parallelization Optimization  ← HTP 专有
Completed stage: Parallelization Optimization (22 us)
...
Executing Graphs
Finished Executing Graphs
```

### ⚡ 如何确认真的在 NPU 上推理（不是 CPU 假跑）

**对比 HTP vs CPU 的编译阶段**——这是决定性证据:

| 编译阶段 | HTP backend | CPU backend |
|---------|:-----------:|:-----------:|
| Graph Preparation Initializing | ✅ 有 | ❌ 没有 |
| Graph Optimizations | ✅ 有 | ❌ 没有 |
| **Graph Sequencing for Target** | ✅ 有 | ❌ 没有 |
| **VTCM Allocation** | ✅ 有 | ❌ 没有 |
| **Parallelization Optimization** | ✅ 有 | ❌ 没有 |

- **VTCM (Vector Tightly Coupled Memory)** = HTP 硬件专有的片上内存分配
- **Graph Sequencing for Target** = 为 HTP 硬件目标做指令调度
- **Parallelization Optimization** = HTP 多 HVX 线程并行优化

这三个阶段**只有 NPU 才出现**，CPU backend 根本没有。

**额外验证——看 DSP 侧日志**:
```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
& $adb shell "grep 'qnn-net-run\|fastrpc-rm' /data/log/DLT/DIM_Slog.log | tail -30"
```
会看到 `domain=3`（CDSP 域）、`cdsp0_cb`（CDSP0 上下文块）、`DMABUF`（DSP 内存缓冲）——都是 DSP/HTP 专属的通信记录，CPU backend 不会有。

### 8.3 可能遇到的问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `Device Creation failure` | 多种原因，看日志 | `grep 'qnn-net-run' /data/log/DLT/DIM_Slog.log \| tail -30` |
| `dsp proc_init failed, err=2` | `fastrpc_shell_unsigned_3` 缺失 | 见 [4.3](#43--关键复制-dsp-运行时依赖文件) |
| `cannot open libc++.so.1` | DSP C++ 运行时缺失 | 见 [4.3](#43--关键复制-dsp-运行时依赖文件) |
| `signature does not match` | signed PD 模式下用 unsigned skel | 确保用 unsigned PD (默认) |
| `Create From Binary failure` | context binary (.bin) 版本不匹配 | 用 .so 模式，不用 .bin |
| `Operation not permitted` | adb 不是 root | `adb root` |

### 8.4 查看 DSP 错误日志

推理失败时，看板子上的 DSP 日志:

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
& $adb shell "grep 'qnn-net-run' /data/log/DLT/DIM_Slog.log | tail -30"
```

---

## 9. 第七步：验证推理结果

### 9.1 取回输出

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$workdir = "C:\Users\RQILIN\Documents\Default Project"
New-Item -ItemType Directory -Path "$workdir\Result_0" -Force | Out-Null
& $adb pull /data/qairt/demo/output/Result_0/output.raw "$workdir\Result_0\output.raw"
```

### 9.2 对比输出

```powershell
$py = "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe"
$workdir = "C:\Users\RQILIN\Documents\Default Project"
& $py -c @"
import numpy as np
expected = np.fromfile(r'$workdir\expected_output.raw', dtype=np.float32)
actual = np.fromfile(r'$workdir\Result_0\output.raw', dtype=np.float32)
print('Expected:', expected)
print('Actual:  ', actual)
print('Max diff:', np.abs(actual - expected).max())
print('Match:', np.allclose(actual, expected, atol=1e-3))
"@
```

> **注意**: 浮点模型在 HTP 上可能有精度差异（特别是 MatMul 算子语义不同时）。如果数值差很大，检查 converter 的算子转换是否正确，或用 CPU backend 对比。

### 9.3 用 CPU backend 对比（可选）

CPU backend 不依赖 DSP，结果应该和 ONNX 完全一致:

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
& $adb shell ". /data/qairt/env.sh && cd /data/qairt/demo && qnn-net-run --model libsimple_linear.so --backend libQnnCpu.so --input_list input_list.txt --output_dir output_cpu 2>&1"
& $adb pull /data/qairt/demo/output_cpu/Result_0/output.raw "$workdir\Result_0\output_cpu.raw"
```

---

## 10. 完整一键脚本

> 以下脚本假设模型名为 `simple_linear`，按需修改。

```powershell
# ============================================================
# SA8255P 完整推理部署脚本
# ============================================================

# --- 配置 ---
$ModelName = "simple_linear"          # 模型名 (不含扩展名)
$ONNXFile = "$ModelName.onnx"
$workdir = "C:\Users\RQILIN\Documents\Default Project"
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$py = "C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe"
$sdk = "C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626"
$plink = "C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe"
$pscp = $plink -replace "plink","pscp"
$fingerprint = "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"
$dev133User = "rqilin"
$dev133Pass = "Wangba521."
$dev133IP = "10.137.185.133"

# --- Step 1: ONNX → QNN 转换 (PC) ---
Write-Output "=== Step 1: Convert ONNX → QNN ==="
$env:PATH = "$sdk\lib\x86_64-windows-msvc;$env:PATH"
$env:PYTHONPATH = "$sdk\lib\python"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
& $py -u "$workdir\convert_model.py" 2>&1 | Select-Object -Last 5

# --- Step 2: 上传到 dev133 编译 ---
Write-Output "=== Step 2: Upload to dev133 ==="
& $pscp -pw $dev133Pass -hostkey $fingerprint "$workdir\$ModelName" "${dev133User}@${dev133IP}:/home/${dev133User}/$ModelName.cpp"
& $pscp -pw $dev133Pass -hostkey $fingerprint "$workdir\$ModelName.bin" "${dev133User}@${dev133IP}:/home/${dev133User}/$ModelName.bin"

Write-Output "=== Step 3: Compile on dev133 ==="
$cmd = "set -e; export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.45.41.260507; export TARGET_PREFIX=aarch64-linux-gnu-; export SDKTARGETSYSROOT=/; cd /home/$dev133User; `$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator -c $ModelName.cpp -b $ModelName.bin -o ./out -t aarch64-oe-linux-gcc11.2 2>&1 | tail -3"
$escaped = $cmd -replace '"','\"'
& $plink -ssh -batch -pw $dev133Pass -hostkey $fingerprint "${dev133User}@${dev133IP}" $escaped 2>&1 | Select-Object -Last 5

# --- Step 4: 下载 .so ---
Write-Output "=== Step 4: Download .so ==="
& $pscp -pw $dev133Pass -hostkey $fingerprint "${dev133User}@${dev133IP}:/home/${dev133User}/out/aarch64-oe-linux-gcc11.2/lib$ModelName.so" "$workdir\lib$ModelName.so"

# --- Step 5: 推到板子 ---
Write-Output "=== Step 5: Push to board ==="
& $adb root; Start-Sleep 3; & $adb wait-for-device
& $adb shell "mkdir -p /data/qairt/demo"
& $adb push "$workdir\lib$ModelName.so" /data/qairt/demo/
& $adb push "$workdir\input_0_0.raw" /data/qairt/demo/
& $adb shell "echo 'input_0_0.raw' > /data/qairt/demo/input_list.txt"

# --- Step 6: HTP 推理 ---
Write-Output "=== Step 6: Run HTP inference ==="
& $adb shell ". /data/qairt/env.sh && cd /data/qairt/demo && qnn-net-run --model lib$ModelName.so --backend libQnnHtp.so --input_list input_list.txt --output_dir output 2>&1" | Select-Object -Last 10

# --- Step 7: 取回结果 ---
Write-Output "=== Step 7: Retrieve output ==="
New-Item -ItemType Directory -Path "$workdir\Result_0" -Force | Out-Null
& $adb pull /data/qairt/demo/output/Result_0/output.raw "$workdir\Result_0\output.raw"
Write-Output "Done! Output at: $workdir\Result_0\output.raw"
```

---

## 11. 常见问题与排查

### 11.1 platform-validator 失败

**现象**: `Unit Test: Failed`

**排查步骤**:

```powershell
$adb = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"

# 1. 确认是 root
& $adb root; Start-Sleep 3; & $adb wait-for-device
& $adb shell "id"   # 必须 uid=0(root)

# 2. 看完整错误日志
& $adb shell "grep 'qnn-platform-va' /data/log/DLT/DIM_Slog.log | tail -30"
```

**常见错误**:

| 日志关键词 | 原因 | 解决 |
|-----------|------|------|
| `fastrpc_shell_unsigned_3` + `FAIL(2)` | shell 文件不在搜索路径 | `cp /vendor/dsp/cdsp/fastrpc_shell_unsigned_3 /data/qairt/lib/hexagon-v73/unsigned/` |
| `cannot open libc++.so.1` | C++ 运行时缺失 | `cp /vendor/dsp/cdsp/libc++.so.1 /data/qairt/lib/hexagon-v73/unsigned/` |
| `lrmc: mq_open fails errno: 22` | 非 root + mqueue 限制 | `adb root` |
| `signature does not match` | signed PD 误用 | 确认用 unsigned skel |

### 11.2 qnn-net-run "Device Creation failure"

**第一步: 看 DSP 日志**:
```powershell
& $adb shell "grep 'qnn-net-run' /data/log/DLT/DIM_Slog.log | tail -30"
```

**常见原因**:
- `fastrpc_shell_unsigned_3` 缺失 → 见 [4.3](#43--关键复制-dsp-运行时依赖文件)
- `libc++.so.1` 缺失 → 见 [4.3](#43--关键复制-dsp-运行时依赖文件)
- 模型 .so 架构不匹配 → 确认用 `aarch64-oe-linux-gcc11.2` 编译

### 11.3 converter 卡住无输出

**原因**: Windows multiprocessing 死锁

**解决**: converter 脚本必须用 `if __name__ == '__main__':` 包裹 main 函数

### 11.4 converter 编码错误

```
UnicodeEncodeError: 'gbk' codec can't encode character
```

**解决**:
```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

### 11.5 dev133 SSH 连接问题

```powershell
# 第一次连接需要接受主机密钥
"y`n" | & $plink -ssh -pw "Wangba521." rqilin@10.137.185.133 "echo test"

# 之后用 -batch -hostkey 避免提示
& $plink -ssh -batch -pw "Wangba521." -hostkey "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c" rqilin@10.137.185.133 "command"
```

### 11.6 数值不匹配

HTP 推理结果和 ONNX 原始结果不一致:

1. **先用 CPU backend 对比** — CPU 结果应该和 ONNX 一致
2. **检查算子支持** — 某些 ONNX 算子在 QNN 里语义不同
3. **量化精度** — 如果用了 INT8，会有精度损失
4. **MatMul 语义** — ONNX MatMul 和 QNN MatMul 的维度处理可能不同

---

## 12. 关键路径速查表

### PC 上的关键路径

```
QAIRT SDK:        C:\Users\RQILIN\Downloads\qairt-v2.48\qairt\2.48.0.260626
  converter:        bin\x86_64-windows-msvc\qnn-onnx-converter
  QNN python:       lib\python\qti\aisw\
  Windows DLL:      lib\x86_64-windows-msvc\*.dll
  ARM skel:         lib\hexagon-v73\unsigned\

Python:           C:\Users\RQILIN\AppData\Local\Programs\Python310Standalone\python\python.exe
ADB:             C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe
Plink:            C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe
工作目录:         C:\Users\RQILIN\Documents\Default Project
```

### dev133 上的关键路径

```
QAIRT SDK:        /opt/qcom/aistack/qairt/2.45.41.260507
model-lib-gen:    bin/x86_64-linux-clang/qnn-model-lib-generator
交叉编译器:        /usr/bin/aarch64-linux-gnu-g++  (TARGET_PREFIX=aarch64-linux-gnu-)
```

### 板子上的关键路径

```
QAIRT 安装:       /data/qairt/
  env.sh:          /data/qairt/env.sh
  工具:            /data/qairt/bin/
  ARM 库:          /data/qairt/lib/
  DSP skel:        /data/qairt/lib/hexagon-v73/unsigned/
    ⚠️ fastrpc_shell_unsigned_3  (DSP PD shell)
    ⚠️ libc++.so.1               (C++ 运行时)
    ⚠️ libc++abi.so.1             (C++ ABI)

工作目录:         /data/qairt/demo/
  模型:            libxxx.so
  输入:            input_0_0.raw + input_list.txt
  输出:            output/Result_0/output.raw

系统 DSP 文件:    /vendor/dsp/cdsp/  (fastrpc_shell, libc++ 等来源)
DSP 日志:          /data/log/DLT/DIM_Slog.log
```

### 关键命令速查

```powershell
# adb root (每次重新连接后必须)
& $adb root; Start-Sleep 3; & $adb wait-for-device

# 验证 HTP
& $adb shell ". /data/qairt/env.sh && qnn-platform-validator --backend dsp --testBackend 2>&1 | tail -10"

# 运行推理 (HTP)
& $adb shell ". /data/qairt/env.sh && cd /data/qairt/demo && qnn-net-run --model libxxx.so --backend libQnnHtp.so --input_list input_list.txt --output_dir output"

# 运行推理 (CPU 对比)
& $adb shell ". /data/qairt/env.sh && cd /data/qairt/demo && qnn-net-run --model libxxx.so --backend libQnnCpu.so --input_list input_list.txt --output_dir output_cpu"

# 看 DSP 错误日志
& $adb shell "grep 'qnn-net-run' /data/log/DLT/DIM_Slog.log | tail -30"
```

---

## 附录: 整体流程图

```
                    ┌─────────────────────────┐
                    │   PyTorch / 原始模型     │
                    └────────────┬────────────┘
                                 │ torch.onnx.export()
                                 ▼
                    ┌─────────────────────────┐
                    │   ONNX 模型 (.onnx)     │  ← 也可以直接下载现成的
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   PC (Windows)           │
                    │   qnn-onnx-converter     │
                    │   + Python 3.10          │
                    │   + onnx==1.16.2         │
                    └────────────┬────────────┘
                                 │ 生成
                                 ▼
                    ┌─────────────────────────┐
                    │   QNN .cpp + .bin        │
                    └────────────┬────────────┘
                                 │ scp 上传
                                 ▼
                    ┌─────────────────────────┐
                    │   dev133 (Linux 服务器)  │
                    │   qnn-model-lib-generator│
                    │   + aarch64-linux-gnu-g++│
                    └────────────┬────────────┘
                                 │ 生成
                                 ▼
                    ┌─────────────────────────┐
                    │   libxxx.so (ARM .so)    │
                    └────────────┬────────────┘
                                 │ adb push
                                 ▼
    ┌─────────────────────────────────────────────────┐
    │   SA8255P 板子                                    │
    │   /data/qairt/demo/                               │
    │     libxxx.so  +  input_0_0.raw  +  input_list.txt│
    │                    │                              │
    │   source env.sh && qnn-net-run                    │
    │     --model libxxx.so                             │
    │     --backend libQnnHtp.so                        │
    │                    │                              │
    │                    ▼                              │
    │          ┌──────────────────────┐                 │
    │          │   HTP v73 (NPU)       │                 │
    │          │   Executing Graphs ✅ │                 │
    │          └──────────────────────┘                 │
    │                    │                              │
    │                    ▼                              │
    │          output/Result_0/output.raw               │
    └─────────────────────────────────────────────────┘
                                 │ adb pull
                                 ▼
                    ┌─────────────────────────┐
                    │   PC: 验证输出结果       │
                    └─────────────────────────┘
```

---

## 附录: 关键经验教训

1. **adb root 是第一步** — 不 root 的话所有 DSP 调用都失败
2. **fastrpc_shell_unsigned_3 必须复制** — 这是 DSP 启动 unsigned PD 的 shell，不在搜索路径里
3. **libc++.so.1 必须复制** — DSP 侧 skel 库依赖 C++ 运行时
4. **Python 必须 3.10** — QAIRT converter 的 C 扩展只编译了 3.10 和 3.12
5. **onnx 必须 1.16.2** — 新版改了 API
6. **converter 脚本要 if __name__=='__main__'** — 否则 Windows multiprocessing 死锁
7. **看 DSP 日志去 /data/log/DLT/DIM_Slog.log** — 不要猜，看实际错误
8. **板子自带的 nsp_test 在 adb 下跑不了** — 别被它误导
9. **dev133 有全套工具链** — 交叉编译 + model-lib-generator
10. **SDK 自带的 skel 已含 CRC** — 不需要额外签名，unsigned PD 模式可用
