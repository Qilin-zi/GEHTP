# env.sh — GEHTP 五件套工具链路径收敛 (110 主家)
# =====================================================================
# 用法: source "$ROOT/env.sh"   (gehtp 脚本内已 source)
#
# 布局 (110):
#   宿主机 /disk1/toolchains/          → 容器内 /opt/toolchains/ (bind-mount 只读)
#   五件套:
#     android-ndk-r26c/               NDK   (aarch64 android 交叉编译)
#     Hexagon_SDK/6.6.0.0/            HEX   (Hexagon kernels 交叉编译)
#     qairt_2.48.40.260702/           SDK   (前端 ONNX→DLC→net.json)
#     qairt_2.40.1.251119/            Q     (model-lib 生成)
#     swiv_build_utility.py           SWIV  (skel/runner SWIV 签名)
# =====================================================================

# 自动检测: 容器内用 /opt/toolchains, 否则宿主机 /disk1/toolchains
if [ -d /opt/toolchains ]; then
    TC=/opt/toolchains
else
    TC=/disk1/toolchains
fi

export SDK="$TC/qairt_2.48.40.260702"      # 前端: qairt-converter / qairt-dlc-to-json
export PY=python3                            # 容器内 system py3.12 (numpy + QAIRT Py312 扩展)
export HEX="$TC/Hexagon_SDK/6.6.0.0"        # Hexagon SDK (kernels 交叉编译)
export HT="$HEX/tools/HEXAGON_Tools/19.0.07" # Hexagon Tools 19.0.07
export SWIV="$TC/swiv_build_utility.py"      # SWIV 签名工具
export Q="$TC/qairt_2.40.1.251119"          # model-lib 生成 (qnn-model-lib-generator)
export NDK="$TC/android-ndk-r26c"           # NDK (aarch64 android)

# device_run.sh 已按 ${VAR:-default} 读取这两个别名, 一并收敛
export HEXAGON_SDK_ROOT="$HEX"
export SWIV_TOOL="$SWIV"

export DEVROOT=/data/local/tmp/hvxhmx23     # 设备侧路径 (不变)
export GDIR="$DEVROOT/gehtp"
