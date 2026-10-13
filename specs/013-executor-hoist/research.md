# Research: 013-executor-hoist

**Created**: 2026-10-13 | 来源：执行器审计（1013，主线 734986b）+ 既有记忆锚

## R1 三槽走 op args（arity 6→9）还是 per-op footer

- **Decision**: 走 op args（arity 9）。op_spec.inc 增可选 `legacy_arity` 列（默认 0=无），OP_MATMUL_W4A16 legacy_arity=6；parse 双接受，执行器按 n_args 分派。
- **Rationale**: args 是算子契约的本位（011 注册表单点纪律）；≤16 参无需 footer 溢出（GDN footer 是 44B 几何超 16 参才用的先例）。legacy_arity 列保持"散落表"不回潮。
- **Alternatives**: per-op footer（拒——契约藏进旁路，执行器仍要查找）；直接拒绝旧 blob（拒——板上共享库须同时服务 6/9 两代 blob，009/010 已有混跑现实）。

## R2 blob 版本号是否 bump

- **Decision**: 不 bump WT_BLOB_VER（维持 v3）。9-arity 由 n_args 自识别（opcode 28 的 arity 校验放宽为 {9, legacy 6}）。
- **Rationale**: ver 门是解析语义变化的粗闸；本次 op 表布局不变（变的是单 opcode 的合法参数数），n_args 双接受即可，避免 v4 全生态配对成本。
- **Alternatives**: bump v4（拒——设备库/板上 blob/runner 三方配对成本高，收益仅是"非法 6 参早期暴露"，可由 legacy_arity 显式列替代）。

## R3 requant 生成器形态与落点

- **Decision**: 新建 `compiler/tools/requant_w4a16.cc`，复用 007 供给侧的 gguf_wtlink GGUF 读取层；产物 = 打包权重槽 + atbl/otbl（i*M*8 布局，K≠N 分列——已验证契约）+ folded_bias；小矩阵对拍单测先行。
- **Rationale**: 010 US2 定谳该生成器"从未落地 = blocker"；atbl/otbl 布局与 otbl K≠N 扫描 bug 均已在 010 回合修定，契约直接引用。
- **Alternatives**: 在 op_matmul.cpp emit 时即时 requant（拒——权重变换属编译期工具职责，混进 emit 使路径不可单测）。

## R4 VTCM 布局表载体

- **Decision**: 扩展 TEMPOFF 槽载荷（现有 [cap|reserve|pin_cap|n] 头后追加布局段），不新开 footer。
- **Rationale**: TEMPOFF 已是"编译期内存事实进 blob"的既有通道，且 carve↔TEMPOFF 双源对拍哨兵（003 T013）已建在装载期——布局表进同一记录流可复用整套对拍机制。
- **Alternatives**: 新 magic footer（拒——第四个 footer 探测点，装载期解析复杂度+1，无新收益）。

## R5 hmx_runtime_setup 有界区域 API

- **Decision**: setup 增区域参数（base+size，0/NULL=全量兼容旧义）；wtcache attach 共享 scratch 模式（003 T014 已存在）作为分区语义的运行时底座。
- **Rationale**: 现状"持全量 VTCM 从不释放"是三向探戈的根；保留全量默认值使既有调用点零扰动（17_w4a16_gemm 等示例不改即编译）。
- **Alternatives**: 全局改双 API（拒——示例与旧路径全要动，爆炸面大）。

## R6 flush 收敛的正确性边界

- **Decision**: 装载布局后全 VTCM flush 一次；re-carve 事件的 flush 降为 **carve arena 区段 flush**；驻留写（temp_get VTCM 路径）的 flush 保留（正确性所需，008 上板实证）。
- **Rationale**: 一致性要求的是"写后清"，不是"全量清"；wtcache_open memset 留 dirty 零行的坑（008 dualdomain 根因）由装载期一次全量覆盖。fence.c HMX+DDR 面不动（tiling 硬前置，归 012/后续）。
- **Alternatives**: 全删 flush（拒——HMX 直写与 CPU 读的可见性会破，008 已实证数值不一致）。

## R7 carve 布局去重键与序列化内容

- **Decision**: 去重键 = (M,K,N,n0) 四元组（形状+列区间唯一确定 carve 结果）；每布局序列化 {layout_id, offsets(e.wt/bias/act/out), arena_hi}；op 经注解/新槽引用 layout_id。
- **Rationale**: 0.8B w4 全图 151 层形状高度重复（同层同形），去重后布局表 ~几十条；per-op arena_hi 与 003 carve footer 的 residency 窗口语义对齐。
- **Alternatives**: 逐 op 全量序列化（拒——151 条重复记录，浪费且无信息增量）。

## R8 D 段前置与 reconcile 面

- **Decision**: D 开工硬前置 = 012 落地（exec_matmul_f16_impl 分发段同区）；B/A/C 期间在独立 worktree（/disk1/gehtp-wt/013-executor-hoist，off 734986b）开发，回合 merge 时 reconcile 012 的 HMX 会话外提段。
- **Rationale**: 主树 oplist_exec.c 有 012 未提交 WIP（HMX 会话外提+act 驻留）；D 的 GEMV 门改动正落其分发段。
- **Alternatives**: 等 012 全落地再开 013（拒——B 是编译器为主，无文件冲突，串行浪费）。

## R9 w4 线性能门校准（1.5× / ≤17.1s）

- **Decision**: 硬门 = exec ≤17.1s（较 25.7s 锚 ≥1.5×）；stretch = 5.99s（f16 锚量级），不作门。
- **Rationale**: 25.7s 锚的 matmul 族 15.2s 主项是 Q4 dequant 标量（007 T017 结论）；W4A16 原生 HMX 消除 dequant，保守估计总墙钟砍 1/3。007 冻结门 20s 未达的前车之鉴——stretch 不进门。
- **Alternatives**: 门设 5.99s（拒——不可达硬门会像 007 冻结门一样终局 FAIL）；门设"任意下降"（拒——无区分度）。
