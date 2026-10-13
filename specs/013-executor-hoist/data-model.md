# Data Model: 013-executor-hoist

**Created**: 2026-10-13

## 1. W4A16 供给槽组（B）

| 字段 | 说明 | 校验 |
|---|---|---|
| packed_w slot | W4 紧凑权重 | len == K*N/2；128B 对齐 |
| atbl slot | K 向 scale 表 | len == 8*(K/32)*4；i*M*8 布局 |
| otbl slot | N 向 scale 表 | len == 8*(N/32)*4；K≠N 分列契约 |
| folded_bias slot | 折叠偏置 | len == (N/32)*512 |

关系：被 MatmulW4a16Op v9 直接引用——packed_w→args[1]，bias/atbl/otbl→args[6..8]（见 §2）；由 requant_w4a16 从 Q4_0 GGUF 单次生成。校验规则 = emit 契约哨兵（长度互斥无歧义，沿用 W4A16 紧凑 0.5 B/elem 判别式）。

## 2. MatmulW4a16Op（B，opcode 28）

| args 位 | v9（正典） | v6（legacy，只读兼容） |
|---|---|---|
| 0-5 | act_ref, w_slot, out_temp, M, K, N | 同 |
| 6-8 | bias_slot, atbl_slot, otbl_slot | —（运行时长度匹配扫描） |

状态转移：v6 路径只保留到 D 段后删除；v9 为唯一新发射形态。

## 3. VTCM 静态布局表（A，TEMPOFF 槽扩展段）

| 字段 | 说明 |
|---|---|
| region[3] | {pin, carve_arena, resident_pool} 各 {offset, size} |
| domain_map | temp→消费引擎分域（HMX→VTCM / HVX,scalar→DDR） |
| budget_check | resident_pool + hmx_arena ≤ 16MB（编译期硬门，BudgetReport 口径） |

不变量：三区互斥不重叠；arena 上界随活跃 carve 记录下压的旧运行时逻辑由编译期 per-op arena_hi（见 §4）取代。

## 4. Carve 布局记录（C）

| 字段 | 说明 |
|---|---|
| layout_id | 去重共享键 = hash(M,K,N,n0) |
| offsets | e.wt / e.bias / e.act / e.out 在 VTCM arena 内偏移 |
| arena_hi | 该布局活跃期的 arena 上界（承接 003 carve footer residency 窗口语义） |

状态转移：装载期一次重算 dc_w4_carve 对拍 emit 表（漂移→拒绝装载）；运行期只读。

## 5. 形态 flag（D）

| 载体 | 内容 |
|---|---|
| broadcast op flags | form ∈ {1 整块, 2 行循环, 3 strided→计划标量} |
| matmul 注解 | planned_engine 直接编码 GEMV（M<32 事实编译期已知） |
| matmul op flags | Q4_0 槽判别位（替代 len==640*tiles 探针） |

## 6. 三线基线锚（回归对拍实体）

| 线 | 锚 |
|---|---|
| 默认/gdn | cos 0.999930 / top1 32/32 / exec ~2.77s |
| f16 非融合 | 输出 md5 f87461e2 / exec 5.99s / blob md5 3f857926 |
| 007 q4 | cos 0.993484 / exec 25.7s / 权重 476.5MB |

校验规则：每阶段收口对拍；cos 门一律设锚值之下（cos-min 严格 >= 假 FAIL 坑）。
