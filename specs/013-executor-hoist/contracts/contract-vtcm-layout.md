# Contract: VTCM 静态布局表（A 段）

**契约主体**: TEMPOFF 槽载荷扩展段（blob 级）

## 载荷格式（现有 [cap|reserve|pin_cap|n] 头 + temp 表之后追加）

```
[布局段 magic u32][version u32]
region_pin      {off u32, size u32}
region_arena    {off u32, size u32}
region_resident {off u32, size u32}
```

- region 数组语义：VTCM 基址相对偏移，三区互斥；resident 即现有尾倒划驻留池（vtcm_cap 口径并入本表，旧字段保留兼容读）。

## 编译期预算门（BudgetReport 口径）

```
resident_pool + hmx_arena ≤ VTCM_BUDGET (16MB)
```

超限 = 编译期硬报错（带逐区诊断），禁止落到运行时。

## 运行时消费不变量

1. 装载期：wtcache 按 region 表一次分区；此后**零运行时布局决策**。
2. `hmx_runtime_setup(base, size)`：0/NULL = 全量（兼容旧义）；A 段后 matmul/conv 路径一律传有界 region_arena。
3. 探戈禁令：全程不得出现 hmx_runtime_setup ↔ wtcache/驻留池 teardown/re-acquire 互斥回收（op_ts + trace 断言）。
4. flush：装载布局后全 VTCM flush 一次；re-carve 仅 arena 区段 flush；驻留写 flush 保留（正确性）。

## 分域规则（008 GATE3 铁律）

- HMX 消费的 temp → VTCM（region_resident）；HVX/scalar 消费的 temp → DDR（TEMPOFF 静态区）。
- 归属由编译期 dataflow 定，运行时只查表。

## 验收断言

1. f16 默认线输出位级（md5 f87461e2）；gdn 线 cos ≥0.999930。
2. op_ts 实证全 VTCM flush 次数 = 1。
3. 三线（默认/gdn/w4）对拍 B 末态位级/门不回退。
