# revlibHtpPrepare 整体状态总结

- 日期: 2026-08-28
- 目标库: Qualcomm libHtpPrepare.so (QNN SDK 2.48.40.260702, 103MB)
- 主线: `REQNN/` (B) — 字节级重实现; 序列化层向真实 wire 对齐 (本会话主攻)

---

## 1. 项目背景与当前架构

**两条代码库**:

| 代码库 | 定位 | 状态 |
|---|---|---|
| `REQNNFRAME/src/` (A) | 旧实现 (M10 审计判"算法层自造") | 弃用 (仅 wire 常量被吸收) |
| `REQNNFRAME/REQNN/` (B) | 主线, 字节级重实现 | 活跃 |

**B 的构成** (合并决策 = B 做前端+IR+调度, 序列化对齐真实 wire):

```
JSON/ONNX → qnn_ir_loader (前端) → OpDef IR (字节级) → 优化(空壳) 
  → ST-Cut 调度 → 内存分配 (FancyAllocator) → 序列化 → .bin
```

**关键决策** (本会话确立):
1. **自研 skel** → opclass_hash 自定, 无需匹配 Qualcomm 运行时派生值。
2. **compute 不 copy** → 所有记录从 op/张量计算, 不嵌入 golden 字节。
3. **conv+add 最小范围** → 优化 pass 空壳化 (不影响产出)。

---

## 2. 整体开发状态

| # | Module | Key Assets | RE Progress | Impl. Progress | Combined |
|---|---|---|---|---|---|
| 1 | **Graph Parsing** (JSON→IR) | qnn_ir_loader + byte-exact OpDef + map container + do_append_node (4003B) | 90% | 85% | **~85%** |
| 2 | **Frontend Optimization** | do_prepare1 / opt passes (deliberately stubbed) | 60% | 15% | **~15%** |
| 3 | **Scheduling & Memory Mgmt** | ST-Cut full chain (M17-M37) + FancyAllocator + cp_solver | 85% | 75% | **~75%** |
| 4 | **Serialization → bin** | barrel/CO_AUX/OpSerHandle (byte-exact) + Section3/tail pending | 90% | 45% | **~50%** |
| 5 | **On-device (bring-up)** | CRC / skel consumption | 50% | 0% | **~0%** |

**整体综合 ≈ 50%** (RE 理解 ~75%, 代码实现 ~50%)。

---

## 3. 调度/内存的真实进度 (B 已做 ST-Cut, 非薄弱环节)

```
schedule_for_alloc @0x1302900 全链:
  M17-M30 解码 ✅ 逐指令贯通
  M31 重实现 ✅ st_cut.cpp ~1100 行
  M32 提案器指令级替换 ✅
  M36 create_supertiles ✅ (427 行)
  M37 phys_alloc_in_runlist ✅
  测试 7/7 ✅
+ FancyAllocator (1179 行) + cp_solver (1336 行) + dma/cost (~1200 行)
```

---

## 4. 序列化层的当前状态 (本会话主攻, 唯一缺口)

| 子项 | 状态 | 完成度 |
|---|---|---|
| barrel/pickle 容器 | ✅ 字节全等 (21428B) | 100% |
| CO_AUX 记录 | ✅ 字节全等 (33 word) | 100% |
| OpSerHandle + spcl_add 原语 | ✅ 实现+单测 | 100% |
| class-index 记录 | ✅ 实现+单测 | 100% |
| class_id 来源 | ✅ 破解 (aux_tag_word) | 100% |
| counters 机制 | ✅ 定位 (op 计数 per class, prescan_ops_func) | RE 90% |
| segment_descs | ⚠️ 机制定位, 字段待解 | RE 60% |
| OpRec (extra_info) | ⚠️ 架构清楚, conv+add extra_info 待 RE | RE 70% |
| tail (const_extent/mystery/identity) | ❌ 未 RE | 10% |
| CRC | ⚠️ 定位外部, 自定即可 | 10% |

---

## 5. 下一步目标: 全链闭环一个小模型

**目标定义**: 一个 conv+add (或等价小模型) 走通全链:

```
ONNX → qnn_ir_loader → OpDef → (优化空壳) → ST-Cut 调度 → FancyAllocator 
  → 真实 wire 序列化 (.bin) → CRC → 自研 skel 反序列化 → 设备执行 → 数值正确
```

验收标准: 同一 conv+add 输入, 我方 .bin 在自研 skel 上产出与 Qualcomm 工具链一致的数值。

---

## 6. 距离目标的 Gap

### 6.1 序列化对齐真实 wire (主体, 当前主攻)

| Gap | 现状 | 预估工作量 |
|---|---|---|
| Section3 class 表从图计算 | class_id 已破解, counters 机制已定位, 待组装实现 | 1 周 |
| segment_descs 从图计算 | tensor_serialize 字段待 RE+实现 | 1 周 |
| runlist OpRec 从 extra_info | 架构清楚, conv+add 的 extra_info 待 RE | 1 周 |
| tail (const_extent + mystery + identity) | 未 RE, 权重 const_extent 是关键 | 1 周 |

### 6.2 CRC (小)

自研 skel 场景下 CRC 自定 (或省略)。CRC 定位在外部 `qnn-context-binary-generator`, 不在 libHtpPrepare.so。**1-2 天**。

### 6.3 skel 消费 (反序列化侧)

自研 skel 需能反序列化我们的 .bin。反序列化侧已有镜像 RE (Deserializer/Deserz), 但实现待做。**1 周**。

### 6.4 上板数值验证

设备 52f67807。依赖 skel 就绪 + 设备可用。**1 周**。

### 6.5 Gap 汇总

| 类别 | 预估 | 依赖 |
|---|---|---|
| Section3 + segment_descs + OpRec 从图计算 | 2-3 周 | RE 已 70-90% |
| tail | 1 周 | 未 RE |
| CRC | 1-2 天 | 自定 |
| skel 反序列化 | 1 周 | 镜像 RE 已做 |
| 上板验证 | 1 周 | skel + 设备 |

**总 gap ≈ 4-6 周 (1 人专注)**, 其中序列化从图计算 (Section3/segment_descs/OpRec/tail) 占大头 (~4 周)。

---

## 7. 关键结论

1. **B 是近乎完整的重实现** — 图解析、ST-Cut 调度、内存管理均已 RE + 重实现 + 测试 (M12-M37)。
2. **唯一缺口是序列化** — B 用自定义 tag, 需对齐真实 wire。这正是本会话在做的事。
3. **序列化 RE 已近完成** (架构全清, class_id 破解, counters/segment_descs/OpRec 机制定位), **实现完成一半** (容器/CO_AUX 字节级验证)。
4. **距离"全链闭环一个小模型"约 4-6 周**, 关键路径 = 序列化从图计算 (Section3/segment_descs/OpRec/tail) + CRC + skel + 上板。

---

## 8. RE 文档索引 (本会话产出)

| 文档 | 内容 |
|---|---|
| M21 | opname 哈希 + string_tag_t + opclass_hash 自定契约 |
| M22 | class-index 记录 + class_id=aux_tag_word 破解 + CO_AUX 纠正 |
| M23 | OpSerHandle + spcl_add_* 原语 |
| M24 | 序列化完整 pipeline 流程图 + 类成员 + Section3 布局 |
| M26 | Section3 差分分析 (382 golden 关联 shape) |
| M27 | serialize_op 架构 (OpRec=extra_info) + prescan counter 机制 |
