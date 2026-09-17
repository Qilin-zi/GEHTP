# REQNN — Qualcomm libQnnHtp 逆向 + C++ 重实现

本目录是对 Qualcomm **`libQnnHtp.so`**（HTP Graph Compiler，host 端图编译器）的逆向工程与 C++ 重实现成果。代码内密集标注反汇编来源（地址 + 字节数），目标是还原其内部组织与编译管线。

## 目录结构

```
REQNN/
├── src/          # C++ 源码 (23 个 .cpp, 186 KB)
├── include/      # C++ 头文件 (18 个 .hpp, 75 KB)
├── tests/        # 单元测试与端到端测试
├── build/        # CMake 构建产物 (libhtp_core.a + test_ir.exe + test_e2e.exe)
├── CMakeLists.txt# 顶层构建脚本
└── reference/    # 参考资料: 真实 .so / 部署管线 / 文档
```

## 构建

```bash
# 工具链: MinGW g++ + CMake (路径见 build/CMakeFiles/*.make)
cmake -S REQNN -B REQNN/build
cmake --build REQNN/build
```

测试运行需把 MinGW 的 bin 目录加入 PATH（否则报 `0xC0000135 STATUS_DLL_NOT_FOUND`）：
```bash
$env:PATH = "<mingw64\bin>;" + $env:PATH
REQNN/build/test_ir.exe      # 10 项单元测试
REQNN/build/test_e2e.exe     # 21 步端到端 (建图->优化->序列化->反序列化->execute)
```

## 实现进度（已完成阶段）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 骨架 | 类层级/字段偏移/函数签名还原 (293 处反汇编注释) | ✅ |
| IR | GraphPrepare/OpDef/append_node/拓扑排序/DCE/CSE/const_prop | ✅ |
| 序列化 | 完整 .bin 往返 (config + IO + runlist ops + 结尾), 5104 字节含权重 | ✅ |
| const pool | 集中存储 + extent descriptor + 反序列化重建 | ✅ |
| block table | VTCM block_id -> 物理偏移映射 + 线性分配器 | ✅ |
| 分段 | 增量加载计划 (make_plan_for_deser_by_segments) | ✅ |
| op 参数 | OpDef::op_data blob 解析 + TypicalOp::params | ✅ |
| tiling | ConvTiler / MatMulTiler (读真实 OutputDef) | ✅ |
| 指令选择 | select_wrapper: HMX_Matrix / HVX_Vector / HVX_Scalar | ✅ |
| fusion | 8 条规则 (Conv+Relu->ConvActivations 等) + 重写框架 | ✅ |
| execute | host-side reference: Relu/Add/Softmax/Sigmoid 等 10+ op | ✅ |

## 仍存差距（诚实说明）

- **非真实 QNN .bin 二进制兼容**：用自定义 tag (TAG_OP_RECORD=0x4F50 等)，真实 `qnn-net-run` / 板子读不了
- **Conv/MatMul execute 是简化直通**，非真实卷积/矩阵乘
- **优化 pattern 规则**：8 条结构级融合，真实库有几十条带 op 语义的 matcher
- **DSP 执行**：无，这是 host 端编译器重实现

详见 `reference/` 下的真实 .so 与文档，以及各子目录 README。
