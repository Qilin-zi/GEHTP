# qwen3_qnn_pipeline/ — Qwen3 端到端上板管线

**独立链路**: 用真实 QNN SDK 把 Qwen3 模型编译并部署到 HTP v73 板子执行。与 `src/` 重实现无依赖关系,是并行的工程参考。

## 架构

```
[HuggingFace] -> [dev133 编译机] -> [Android 板子]
   Stage 1        Stage 2/3          Stage 4/5
```

## 5 阶段

| 阶段 | 目录 | 机器 | 产出 | 耗时 |
|------|------|------|------|------|
| 1. 下载模型 | stage1_download | dev133 | safetensors | ~10 min |
| 2. 导出 ONNX | stage2_export | dev133 | ONNX + encodings + tokenizer | ~30 min |
| 3. 编译 | stage3_compile | dev133 | v73 context binary (.bin) | ~60 min |
| 4. 上板 | stage4_deploy | dev133->板子 | 板上 .bin + GenieX 配置 | ~5 min |
| 5. 推理 | stage5_infer | 板子 | 文本生成 | 实时 |

## 关键踩坑 (README.md 原文)

1. DLC 文件权限: root 拥有 `-rw-r-----`, 需 `dzdo su` 改 `chmod a+r`
2. PerfSetting 配置: 只用 `soc_id + dsp_arch + pd_session` 三项才能成功
3. v73 soc_id: 43 (v81 是 72)
4. GenieX dialog.type: `lade` (不是 `binary`)
5. 传输瓶颈: adb push 34 MB/s 最快

## 运行

```bash
cp env.example.sh env.sh  # 填密码、路径
bash stage1_download/download_model.sh
bash stage2_export/run_example1.sh
bash stage3_compile/run_example2.sh
python stage3_compile/gen_v73_context_binary.py  # 生成 v73 .bin
bash stage4_deploy/deploy_to_board.sh
bash stage5_infer/run_inference.sh
```
