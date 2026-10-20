# T027 — 回合主线 merge（013 → origin/task/prof-wp）

日期 2026-10-19；worktree `013-usd`（base bca460a）。

## 合并拓扑

- 013 代码提交 `9871577`（US-D 形态 flag + T026 6-arity 删除），父 = bca460a。
- 主线 origin/task/prof-wp 已先被 graph_prepare 拆分（`29a931e`，refactor/graph-prepare 线，
  父同为 bca460a）快进。
- 两提交文件集**完全 disjoint**（013 触 op_matmul.cpp/wtop_ops.hpp/oplist_parse.h/oplist_exec.c；
  29a931e 触 graph_prepare*.cpp/.hpp/CMakeLists.txt）→ `recursive` 策略**零冲突自动合并**。

合并提交 `b776fcb`（merge/013-into-mainline，双亲 29a931e + 9871577）。

## 门 1：构建（host compiler 全量 + 设备 kernels）

- host `cmake --build compiler/build_linux`：**exit 0**，50 target 全编过，0 error（仅
  test_wtop_emit.cpp `-Wmissing-field-initializers` 警告，预存）。
- 设备 kernels（libhvxhmx_013usd.so）在 T026 已重编 + SWIV 签名 + 板上指纹核对，合并未触
  kernels 侧（graph_prepare 拆分是 compiler 侧），无需重编。

## 门 2：ctest（宿主）

**48/50 pass；2 失败 = test_wtop_emit + test_vtcm_auto，均为 VTCM 驻留 auto 域，预存非 013 引入。**

- `test_wtop_emit` 失败点 = 003 T022「TAG_MEM_PLAN ≥2 个 VTCM 驻留项 / 扫描定位并腐蚀」——
  驻留 auto 回关后 emit 产 0 驻留项，测试期望 ≥2。
- `test_vtcm_auto` 失败点 = a_auto/b_auto/c_auto「TEMPOFF 槽存在(驻留默认开)/vtcm_cap==…/≥1
  驻留项/布局段存在」——同根：auto 默认回关（[[gehtp-residency-pool-auto-line]] 的「auto 默认
  应回关」A/B 定谳，标量 gdn 内核 5.1× 净负）。
- 归因证据：013 diff `git diff --name-only bca460a 9871577` 只含 5 文件，**不含** ddr_offsets.cpp
  / wtop_emit.cpp 驻留段 / test_wtop_emit.cpp / test_vtcm_auto.cpp；29a931e 自己的收口笔记已记
  「ctest 48/50; 2 失败 = test_wtop_emit + test_vtcm_auto」——即这两红在 013 merge 之前已存在于
  主线。
- **013 相关测试全绿**：test_requant_w4a16（W4A16 供给侧）、test_wtop_emit 的 §9-arity 往返 +
  legacy 6 参等价断言（仅 TAG_MEM_PLAN 两断言红）。

## 门 3：三线终验（板 d0f1784，T026 已收）

| 线 | 锚 | 实测 | 裁决 |
|---|---|---|---|
| prefill（gdn-ON 默认） | exec 0.490s / md5 `2c09d8ba` | 0.489s / `2c09d8ba` | ✓ 位级同 |
| seq1 decode | exec 1.89s | 1.885s rc=0 | ✓ 无回退 |
| W4A16 合成 9-arity / 6-arity | `7654a8f6` / 硬拒 | `7654a8f6` / `w4a16 arity 6` 拒 | ✓ |

## 结论

013（US-D + T026）已零冲突并入主线；构建全绿；ctest 48/50 唯二红为 VTCM 驻留 auto 预存
（013 之前已在主线，非本回合引入）；三线板测位级同锚。可 push。
