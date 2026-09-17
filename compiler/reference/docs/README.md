# docs/ — 官方文档与参考资料

逆向工程参考的官方资料,帮助理解 QNN 架构、CRC 机制与 CDSP 调试。

## 文件

| 文件 | 内容 |
|------|------|
| `QAIRT_完整推理部署指南.md` | Qualcomm AI Runtime 完整推理部署指南 |
| `lpai_overview.txt` | QNN LPAI (Low Power AI) 后端概览 |
| `cdsp_debug.txt` | CDSP (Compute DSP) 调试相关 |
| `pdf_content.txt` | 两份 PDF 内容提取: <ul><li>Safe NSP CRC for Unsigned PD (SA8650P/SA8775P/SA8255P) — SWIV 工具用法</li><li>CDSP Debug on Automotive — 调试方法</li></ul> |
| `bin_format_analysis.md` | context binary (.bin) 格式分析 (大端, count+offset, 无 magic) |
| `schedule_analysis.md` | **schedule_for_alloc + serialize_op + make_dma_checkpoint_op 反汇编分析** (v2.48) |
| `schedule_for_alloc.disasm` | schedule_for_alloc @0x1302710 全量反汇编 (2608 行) |
| `deserialize_runlist.disasm` | deserialize_runlist @0xd33990 全量反汇编 |
| `context_binary_full_analysis.md` | **完整逆向分析报告 (模型→bin→19步执行表, 含可信度分级)** |
| `cross_validation_with_official_docs.md` | **与官方逆向文档 (libHtpPrepareDoc) 对照验证** |

## CRC 应用笔记要点

- unsigned PD 的动态 `.so` 必须嵌入 CRC, 否则加载失败:
  `CDSP0: ELF verification section header for CRC segment not found`
- 用 `python swiv_build_utility.py -i input.so -o out_crc.so` 加 CRC
- 主库 + 依赖库都要加 CRC (DSP 会逐一校验)
- signed PD **不要**加 CRC (会签名校验失败)
