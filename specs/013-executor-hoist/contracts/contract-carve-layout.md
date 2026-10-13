# Contract: carve 布局序列化（C 段）

**契约主体**: carve footer 扩展（blob 级，承接 003 carve footer 语义）

## 记录格式

```
[n_layouts u32]
每条: {layout_id u32, M u32, K u32, N u32, n0 u32,
       off_wt u32, off_bias u32, off_act u32, off_out u32,  // VTCM arena 内偏移
       arena_hi u32}
每 op 注解: layout_id 引用
```

- 去重键 = (M,K,N,n0)；同键 op 共享 layout_id（0.8B 预计 ~几十条布局）。
- 旧 carve footer 字段（op_index/off/size/residency 窗口）保留——C 段为超集扩展，双源对拍哨兵继续生效。

## 装载期对拍（哨兵搬家，不删）

1. 装载时对每个 layout 重算 dc_w4_carve / w4_carve_need 闭式结果，与 emit 偏移逐字段比对；任何漂移 → 拒绝装载（硬报错，防拼接/异版工具产出的静默错地址）。
2. 既有 carve↔TEMPOFF 双向存在性对拍（003 T013）不变。

## 运行时热路径

- W4A16 matmul 执行 = `e.wt = arena_base + off_wt`（四指针直赋）+ arena_hi 查 emit 值；**无** carve_live_top 扫描、无 dc_arena_init、无 dc_w4_carve 调用。

## 验收断言

1. w4 线位级对拍 A 末态。
2. matmul 族逐 op 时长 op_ts 前后对照（量化收益证据）。
3. 手工破坏 emit 偏移的 blob → 装载期拒绝（哨兵有效性测试）。
