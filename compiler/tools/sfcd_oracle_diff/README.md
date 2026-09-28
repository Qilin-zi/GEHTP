# SFCD 写侧 oracle 对拍工具（T1 验证件）

两棵树的写侧组装函数喂**同一输入集**，逐词 diff 输出流：

- `oracle_main.cpp` — 本地 REQNNFRAME oracle（`g4_sfcd_finalize_copies` + `g4_sfcd_assemble`）
- `g4_main.cpp` — 104 树（`g4_construct_sfcd`）

## 运行

```sh
ORACLE=/Users/apple/Documents/ge/REQNNFRAME/REQNN   # 本地 oracle 树根
clang++ -std=c++17 -O0 -I $ORACLE/include oracle_main.cpp \
        $ORACLE/src/vtcm/spillfill_g4.cpp -o oracle_sfcd
clang++ -std=c++17 -O0 -I ../../include g4_main.cpp \
        ../../src/vtcm/spillfill_g4.cpp -o g4_sfcd
diff <(./oracle_sfcd) <(./g4_sfcd) && echo IDENTICAL
```

不接入 CMake（依赖树外 oracle 路径，仅一次性验证用）。

## 验证记录（2026-09-26）

输入集：3 条 tcm 块记录（含相邻合并）+ 2 set_progress + 窄式/宽式/无 waits × ckpt 7/0。
结果：**IDENTICAL（25/28/21 词三变体逐词一致）**。

## 已知有界分歧（hint=0 奇偶区内无分歧）

1. **type nibble**：oracle `nib = hint==0?0:(hint>=6?2:1<<(hint-1))`（G4b 遗留，
   原文未打印）；104 硬编码 0（头注「禁止臆测补齐」）。hint=0（104 当前唯一
   可产出）双方一致。
2. **wait 分块**：oracle 单记录整包（n>0xFFFF 溢出 low16 污染 bit16）；
   104 按 0xFFFF/0x7FFF 分块（防御性扩展）。n≤0xFFFF 字节一致。
3. **混合窄/宽 wait**：oracle 一个向量全窄或全宽；104 每条 WAITFOR 记录独立
   判定 → 混合时多条记录（格式本身允许，oracle API 无法表达）。
4. **终结器**：104 `push_terminator` = {0, blob_off}（@0x1012460 反汇编锚点）；
   oracle 空 entries 走 `finalize_copies` 会产 (pool<<16)|0 —— oracle 模型
   该边自我不一致，104 采信反汇编。
