# src/api/ — C API 入口

对应真实 `hexagon_nn_env.cc` / `c_interface.cc`。对外暴露的编译接口。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `c_interface.cpp` | `hexagon_nn.c` | C 风格 API: append_node / append_const / prepare / serialize 等 |
| `hexagon_nn_env.cpp` | `hexagon_nn_env.cc` | HexagonNNEnv: num_nsps / soc_type (test 用 1 NSP, v75) |

## OpIoPtrs (hexagon_nn_env.hpp)

op 工厂的输入指针结构, 字段已默认初始化 (nullptr) 防垃圾值崩溃:
```
+0x00 graph_prepare
+0x28 opdef_ptr    <- generic_construct 从此取 op_data / output_def
```
