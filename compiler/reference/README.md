# reference/ — 参考资料

逆向工程所依据的真实 `.so`、板端部署管线与官方文档。这些是**只读参考**，不属于重实现源码。

## 子目录

| 目录 | 内容 |
|------|------|
| `skel_crc/` | 真实 Qualcomm `.so` 文件 (libQnnHtpV73* 等), 逆向来源 |
| `qwen3_qnn_pipeline/` | Qwen3 模型端到端上板管线 (5 阶段) |
| `docs/` | 官方文档: QAIRT 部署指南 / CDSP 调试 / LPAI 概览 / PDF 内容 |

## 说明

- `skel_crc/` 下的 `.so` 是真实库的二进制,反汇编注释中的 `@ 0x<addr>` 指向这些文件
- `qwen3_qnn_pipeline/` 是**另一条独立链路** (用真实 QNN SDK 上板),与 `src/` 重实现无依赖关系
- `docs/` 帮助理解 QNN 架构与 CRC / CDSP 调试流程
