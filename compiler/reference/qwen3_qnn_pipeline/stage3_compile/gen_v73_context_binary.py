#!/usr/bin/env python3
"""
===== Stage 3b: 生成 v73 Context Binary (实测验证) =====
机器: dev133 (编译机)
产出: v73 context binary (.bin), 3 个分片
耗时: ~5 min

用法: 在 dev133 上执行
    python3 gen_v73_context_binary.py
    # 或用 root:
    dzdo su -c 'bash -c "python3 /path/to/gen_v73_context_binary.py"'

前置:
    - Stage 3a 已完成 (DLC 已生成)
    - QNN SDK 已安装
    - DLC 文件可读 (如不可读, 用 dzdo su chmod a+r)

关键:
    使用最简 PerfSetting (只 soc_id + dsp_arch + pd_session)。
    官方 make_config_file 的复杂 PerfSetting 会导致
    "Failed to apply graph settings" (error 1002)。
"""
import json
import os
import subprocess
import sys

# ===== 配置 =====
QNN_SDK_ROOT = os.environ.get("QNN_SDK_ROOT", "/data01/rqilin/qnn-sdk")
PROJECT = os.environ.get("QWEN3_PROJECT", "/data01/rqilin/qwen3_llm_v2")

# DLC 输入目录 (Stage 3a 产出)
DLC_DIR = f"{PROJECT}/example2/host_linux/assets/artifacts/ar128-cl2048"

# context binary 输出目录
OUT_DIR = f"{PROJECT}/example2/host_linux/assets/artifacts/ar128-cl2048_v73"
CONF_DIR = f"{PROJECT}/example2/host_linux/assets/artifacts/ar128-cl2048_v73_conf"

# 模型参数
NUM_SPLITS = 3
ARN = 128
CL = 2048
SOC_ID = 43       # v73
DSP_ARCH = "v73"


def setup_env():
    """设置 QNN SDK 环境变量"""
    os.environ["QNN_SDK_ROOT"] = QNN_SDK_ROOT
    os.environ["PATH"] = (
        f"/usr/bin:/usr/local/bin:"
        f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang:"
        + os.environ.get("PATH", "")
    )
    os.environ["PYTHONPATH"] = (
        f"{QNN_SDK_ROOT}/lib/python:"
        f"{QNN_SDK_ROOT}/benchmarks/QNN"
    )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["HEXAGON_TOOLS_DIR"] = (
        f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang"
    )


def write_configs(split_idx, conf_dir):
    """写最简 PerfSetting 配置 (实测成功)"""
    graph_name = f"ar{ARN}-cl{CL}_{split_idx}_of_{NUM_SPLITS}"

    # HtpConfigFile: 指向 PerfSetting
    htp_config = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{conf_dir}/perf_{split_idx}.conf",
        }
    }
    with open(f"{conf_dir}/htp_{split_idx}.json", "w") as f:
        json.dump(htp_config, f, indent=4)

    # PerfSetting: 最简配置 (只 3 项, 多了会失败!)
    perf = {
        "devices": [{
            "soc_id": SOC_ID,
            "dsp_arch": DSP_ARCH,
            "pd_session": "unsigned",
        }]
    }
    with open(f"{conf_dir}/perf_{split_idx}.conf", "w") as f:
        json.dump(perf, f, indent=4)

    return graph_name


def gen_context_binary(split_idx, dlc_path, out_dir, conf_dir):
    """调用 qnn-context-binary-generator 生成单个分片"""
    graph_name = f"ar{ARN}-cl{CL}_{split_idx}_of_{NUM_SPLITS}"
    binary_name = f"ar{ARN}-cl{CL}_{split_idx}_of_{NUM_SPLITS}_v73.serialized"

    cmd = [
        "qnn-context-binary-generator",
        "--backend", "libQnnHtp.so",
        "--model", "libQnnModelDlc.so",
        "--input_output_tensor_mem_type", "memhandle",
        "--output_dir", out_dir,
        "--config_file", f"{conf_dir}/htp_{split_idx}.json",
        "--binary_file", binary_name,
        "--dlc_path", dlc_path,
    ]

    print(f"\n--- 生成 {split_idx}/{NUM_SPLITS} ---")
    print(f"DLC: {dlc_path}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # 打印最后几行输出
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  {line}")

    if result.stderr:
        print(f"  STDERR: {result.stderr[-200:]}")

    # 检查输出文件
    out_file = f"{out_dir}/{binary_name}.bin"
    if os.path.exists(out_file):
        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"  [OK] {out_file} ({size_mb:.0f} MB)")
        return True
    else:
        print(f"  [FAIL] 未生成输出文件")
        return False


def main():
    setup_env()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CONF_DIR, exist_ok=True)

    print("=" * 60)
    print("Stage 3b: 生成 v73 Context Binary")
    print(f"  DLC 目录: {DLC_DIR}")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  SOC_ID={SOC_ID}, DSP_ARCH={DSP_ARCH}")
    print(f"  分片数: {NUM_SPLITS}")
    print("=" * 60)

    # 检查 DLC 是否存在
    for i in range(1, NUM_SPLITS + 1):
        dlc = f"{DLC_DIR}/{i}_of_{NUM_SPLITS}/compiled_model/ar{ARN}-cl{CL}_{i}_of_{NUM_SPLITS}_quantized.dlc"
        if not os.path.exists(dlc):
            print(f"\n[错误] DLC 不存在: {dlc}")
            print("请先运行 Stage 3a: bash stage3_compile/run_example2.sh")
            sys.exit(1)
        if not os.access(dlc, os.R_OK):
            print(f"\n[错误] DLC 不可读: {dlc}")
            print("请用 root 修改权限: dzdo su -c \"chmod a+r <path>\"")
            sys.exit(1)

    all_ok = True
    for i in range(1, NUM_SPLITS + 1):
        dlc = f"{DLC_DIR}/{i}_of_{NUM_SPLITS}/compiled_model/ar{ARN}-cl{CL}_{i}_of_{NUM_SPLITS}_quantized.dlc"
        write_configs(i, CONF_DIR)
        if not gen_context_binary(i, dlc, OUT_DIR, CONF_DIR):
            all_ok = False

    # 汇总
    print("\n" + "=" * 60)
    print("输出文件:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".bin"):
            size = os.path.getsize(f"{OUT_DIR}/{f}")
            print(f"  {f}: {size / (1024*1024):.0f} MB")

    if all_ok:
        print("\n[成功] 全部分片生成完成")
        print("下一步: bash stage4_deploy/deploy_to_board.sh")
    else:
        print("\n[警告] 部分分片失败,请检查上面的输出")
        sys.exit(1)


if __name__ == "__main__":
    main()
