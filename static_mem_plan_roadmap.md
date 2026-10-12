# 计划书：运行时重复搬运的静态化

> 配套文档：`static_mem_plan_gaps.md`（问题与目标）。本文档是**行动计划**，基于
> 2026-10-12 逐算子排查的量化证据。

---

## 一、背景与现状

编译器（host，逆向 libHtpPrepare）已算好内存规划（cp_solver 的 pin/remat/comp、
tcm_migration 优先级、carve footer 驻留区间），执行器（自研 skel，`oplist_exec.c`）
通过 wtop blob 消费。但逐算子排查发现：**搬运决策没有真正静态化**——显式搬运 op
（DMA/SPILL/FILL）已生效，但核心计算算子内部仍隐式搬，且与搬运 op 重复。

**排查结论**：

| 类别 | 算子 | 状态 |
|---|---|---|
| 只按地址算（达标） | RMSNORM/ADD/UNARY/BINARY 等标量算子 | ✓ |
| 显式搬运（达标） | DMA、SPILL、FILL | ✓ |
| act DMA 预置（达标） | MATMUL_W4A16 的 `WT_REF_VTCM_FLAG` 路径 | ✓ 唯一做对的搬运线 |
| **内部隐式搬（未达标）** | MATMUL_W4A16 restage、MATMUL_F16/CONV2D 运行时 pack | ✗ |
| **重复搬（未达标）** | PIN 与 MATMUL 各搬一次权重 | ✗ |

---

## 二、问题定义（量化）

| # | 问题 | 证据 | 量化损失 |
|---|---|---|---|
| E | 权重 crouton pack 运行时发生 | `exec_matmul_f16` 注释 | **20.6s/全图**（纯编译期可做掉的重活）|
| D | 权重每次 restage | `exec_matmul:647` 无条件 `cpu_to_vtcm` | 每次 op 重搬 3.2MB 权重（~32ms/op）|
| D2 | PIN 与 MATMUL 重复搬 | `xop_pin` + `exec_matmul:647` | 权重搬两次 |
| B | pin_cap 硬编码 | `oplist_exec.c:315,590` `wtcache_open(wc,4096)` | 编译器算的 pin 布局无效 |
| A | bias/atbl/otbl 按 len 猜 | `exec_matmul:554-558` | 正确性隐患 |
| C | carve 复用运行时判 | `exec_matmul:579` `shape_same` | 残留运行时决策 |

---

## 三、目标

编译器把内存决策**全部**定死进 blob（权重 crouton 布局、pin 决策与复用区间、pin_cap、
辅助面身份、carve 复用）；执行器退化为**纯查表 + 机械执行**——按 op 里的地址引用算，
零隐式搬、零重复搬、零按尺寸猜、零运行时 shape 判断。

---

## 四、分阶段计划（按性能收益排序）

### 阶段 1 — 权重 crouton pack 编译期化（P0，收益最大）

把 fp16/Q4_0 权重的 crouton 打包从执行器移到编译器。

```
编译器：weight_layout_pack.py / pack_oplist.cc 扩展——打包阶段直接产出 crouton 布局
        （复用已有 hmx_crouton_pos 公式 + 现成的 repack_q4_0_crouton 逻辑）
执行器：exec_matmul_f16 删 pack wgt 循环（:121）；exec_conv2d_hmx 删 pack wgt 循环（:878）
        改为直接读 blob 里已打包的 crouton 权重槽，mxmem 直接加载
```

**验收**：全图权重 pack 耗时 20.6s → 0（编译期一次性成本）；MATMUL_F16/CONV2D 运行时
无 crouton 重排；数值 byte-exact 不破。

### 阶段 2 — 消除 restage + pin 复用（P0）

让权重 pin 一次、跨 op 复用，执行器不再每次重搬。

```
编译器：cp_solver 已算 pin_slot；显式发射 OP_PIN（首个消费 op 之前），
        后续 op 的权重参数引用已 pin 的 VTCM 地址（复用 0x4000|temp / pin 面）
执行器：exec_matmul 的 :647-648 改为"该 slot 已 pin 且驻留区间覆盖当前 op → 跳过搬"
        xop_pin 从"记账"改为"真正 pin 一次"；PIN 与 MATMUL 不再重复搬
```

**验收**：同一权重多 op 复用只搬一次；restage 次数 == 编译期规划值（`wt_w3_report`
核对）；MATMUL 全路径 32ms 里的 3.2MB restage 成分归零。

### 阶段 3 — pin_cap 从 blob 读（P1，阶段 2 的使能）

```
编译器：memplan 已算 pin_cap → 写进 TEMPOFF 槽 reserve 高 16 位（"VTCM 池大小 KB"字段）
执行器：wtcache_open(wc, 4096) 两处（:315,590）改读 TEMPOFF 槽
```

**验收**：编译期 pin_cap 变更能传导到执行器；`4096` 硬编码删除。

### 阶段 4 — 辅助面身份显式化（P1，正确性）

```
编译器：OP_MATMUL_W4A16 加 3 个 arg → [act, w, bias_slot, atbl_slot, otbl_slot, out_t, M, K, N]
执行器：删 exec_matmul:554-558 按 len 猜，改读 args
```

**验收**：执行器零猜测；example 17/21 的 65536/65536 bit-exact 门保持。

### 阶段 5 — carve 复用静态化（P2，收尾）

```
编译器：编译期预判"哪些连续 op 形状相同可复用 carve"，写入 carve footer / 注解
执行器：删 exec_matmul:579 shape_same 运行时判断
```

**验收**：已知形状的图，执行器零运行时形状判断。

---

## 五、依赖与顺序

```
阶段1（独立，最大收益）
   │
   ▼
阶段2（依赖1：pin 进去的必须是 crouton 才有意义）
   │
   ▼
阶段3（依赖2：pin 复用的 pin_cap 要可配置）
   
阶段4（独立，正确性）   阶段5（独立，收尾）
```

建议顺序：**1 → 2 → 3 → 4 → 5**，其中 1+2 是性能主战场，4 是正确性，3/5 是配套收口。

---

## 六、验收总则

1. **回归不破**：每阶段后 `build_examples.sh` 全量回归保持 byte-exact
   （35 例 + 261 门；W4A16 256³ 的 65536/65536 门保持）。
2. **零运行时决策**：执行日志零 `fallback`、零按尺寸猜、零超出编译期规划的
   restage / crouton pack。
3. **量化可观测**：全图权重 pack 20.6s → 0；restage 次数 == 编译期规划值
   （作为回归门固化）。

---

## 七、待确认项

1. 编译器 `memplan` 算出的 pin_cap 是否已存进结构（只差序列化），还是仅打印未接出？
2. carve footer 现为 20B record，阶段 5 加 reuse 标志是扩展 record 还是另起注解槽？
3. `weight_layout_pack.py` 当前是否已有 crouton 输出路径（还是仅 4-bit packed）？
   阶段 1 需要确认复用 `repack_q4_0_crouton` 的程度。
