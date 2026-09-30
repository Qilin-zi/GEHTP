# 005 US1+US2 验收证据（2026-09-30, 设备 d0f1784）

## 单元/回归
- ctest **47/47**；test_opt_passes 0 failures（[13a-g] view_fold / [14a-e] eltwise+unwrap / 默认关反例）

## T004 普查（GEHTP_DUMP_VIEWS, 0.8B v2）
- PRE: ss=3507 reshape=4180 transpose=175; r1_ss_ss=1134 r2_reshape_absorb=1172 r3_tt_ident=0 ss_identity=6
- POST: r1=0 r2=0（视图链全消解; ss→2373, reshape→2684）

## opt-stats（最终默认配置）
- view_fold=2286/2286  fc_bias_fold=18/18  shape_fold=344/344
- eltwise_const_fold=54/0（真图无可合并同型对——90 命中全为 div 形）
- eltwise_div_to_mul=0/0（**默认关**，见下）
- before=26399 after=21121（**-5278 ops, -20%**）

## host 恒等门（权威精度门；host 输入不敏感为已知缺陷，A/B 自对自仍有效）
- P1（view_fold only）: /tmp/005_host_logits.f32.raw vs 基线 host_fcbias_logits_new.f32.raw
  **md5 相同（6153d61f…），cmp BYTE-IDENTICAL** → 视图折叠位级恒等 ✓
- P2（含 Div→Mul）: /tmp/005_p2_host_logits.f32.raw vs 同基线 **n_diff=0/7946240**
  （除数全为 2 的幂 → 1/c 精确，x*(1/c)==x/c）

## 设备门（d0f1784, scripts/gehtp run, input_p0）
| 产物 | ops | wall | 全零行 | NaN 行 |
|---|---|---|---|---|
| fcbias 基线(005 前) | 16438 | 39.5s | [0-15,18] | 无 |
| 005_p1 | 14152 | 31.48s | [0-15,18] | 无 |
| 005_p2(div 开) | 14152 | 31.47s | [0-15,18] | **[28-31]** |
| 005_final(默认) | 14152 | 31.49s | [0-15,18] | 无 |

- **SC-001 ✓✓**: ops -13.9%（门 -6%）; wall -20.3%（门 -4%）
  （连跨会话口径 33.9s 基线也降 7.2%）
- **SC-006 挂起（既有）**: 前 16 行全零 + 行 18 指纹与 005 前基线**逐行一致**;
  final vs p1（同效图）两次跑也差 37% 位 → 设备断链下位对拍无意义，精度以 host 门为准
- **Div→Mul 设备 NaN（新发现 → 处置）**: host 位级一致但设备开 div_to_mul 后行 28-31
  NaN、关掉即无 → 设备 EB DIVIDE/MULTIPLY 溢出口径不同。已拆独立 pass
  `eltwise_div_to_mul` **默认关**（--enable-pass eltwise_div_to_mul 可开）。
  教训：**host 位级一致不保证设备一致**（EB 两 opcode 执行路径不同）。

## 产物
- /tmp/005_final.bin (4077370544B) → $W/005_final.wtop + manifest
- $W/prof/005_{p1,p2,final}.device.txt; $W/out_005{p1,p2,final}_p0.f16.raw
