# Research: 005-graph-identity-passes

Phase 0 结论。每项给 Decision / Rationale / Alternatives。证据均为 2026-09-30 代码与实测（0.8B v2，qwen35_08b_v2.fcbias 链）。

## R1: Eltwise_Binary 操作码映射（P2 前置）

- **Decision**: 采用 QNN 2.48 数字枚举直读（extract_eltwise_extra 已是此口径）。已确证：**0=ADD**（pass_fc_bias 的 is_add_op 按 0 判）、**13=MULTIPLY**、8=NEG（源码注释）。net.json 普查分布：13×1509、0×1291、2×36、18×36、3×1。
- **Rationale**: "2->13" 链 36 条 = 除后乘（RMSNorm 型缩放），正是 Div→Mul + 同型合并的靶子；"13->13"×60、"0->0"×47 是纯同型链。
- **Alternatives**: 按字符串名匹配（旧形态）——extract 已回退支持，不作为主口径。
- **遗留**: 2/18 的确切枚举名（推测 2=DIV 或 SUB、18=POW/MAX）——T007 普查时以 `scalar_str` 交叉验证一次即可，不阻塞设计（同型合并只信同枚举值相等，语义名仅日志用）。

## R2: StridedSlice 参数源与代数输入（P1 前置）

- **Decision**: 代数输入 = scalar 四掩码（begin_mask/end_mask/new_axes_mask/shrink_axes，int 位掩码）+ tensor_param const **"ranges"**（rank×3 个 i32：begin/end/strides），经 `find_param_const(gp, opdef, "ranges")` + `read_param_u32` 读取——与 extract_strided_slice_extra 同源。
- **Rationale**: graph_prepare.cpp:2041 起的 extract 已验证该数据源在 prepare 阶段可读；pass 与 extract 同视角，无新序列化需求。
- **Alternatives**: 从 output_def 反推区间——信息不足（无法区分 stride 与负向），否决。
- **风险**: ranges 的三元组布局（按轴交错 vs 三块连续）需 T002 首个单测用已知图锚定（extract 只透传 offset，布局契约在设备 exec_slice 消费端——读 oplist_exec 的 slice 执行段确认，T002 第一步）。

## R3: Reshape 吸收的安全性判据（P1）

- **Decision**: 仅当 Reshape 是"纯 1 维增删"（元素数不变且非 1 维的维序不变）且**单消费者**时，SS 直接索引底层张量；一般 Reshape（跨维重排 = 隐含数据搬移）不吸收。
- **Rationale**: 纯 1 维增删不改变线性内存布局（SS 按行主序索引同一缓冲），位级恒等可静态证明；跨维重排需要 SS 改写为带 stride 视图，当前发射器无此通路。
- **Alternatives**: 通用 reshape-into-slice（引入 strided 视图 IR）——超出"零设备语义改动"纪律，否决（真库 stridedslice_opts 有此能力，留作后续）。

## R4: MatMul bias 推广的折叠集与 HMX 门（P3）

- **Decision**: 3a 沿用 FC 的 HMX 形状门（M/K/N 全 %32==0 不折）+ 排除 batched（flags&4，设备硬错"matmul_f16_bias 不支持 batched"）。42 个 MatMul→Add 候选多为注意力系 32 对齐 batched 形——**3a 预期折叠 0-5 个，正确性优先，性能收益归 3b**。
- **Rationale**: exec_matmul_f16_impl（oplist_exec.c:1805 起）`if (bias) want_hmx = 0`——v1 bias 仅标量路径；折叠 32 对齐形 = 踢下 HMX，倒退 30×（US1 实测 HMX 4.24s vs 标量 219s）。发射门 op_matmul.cpp:40 硬编码 FullyConnected，但 OP_MATMUL_F16_BIAS(36) 通路通用（含 W4A16 硬错保护）——放行是 1 行。
- **Alternatives**: 一步到位撤 HMX 门——无设备验证的内核改动违反"恒等+零设备语义"纪律，拆 3b。
- **3b 技术口径**: GEMM 出 crouton 后、unpack 写回前，VTCM 内 f32 累加器逐列加 bias（与标量路径同口径，oplist_exec.c:1094 起 host 参考已按列加）；合成 blob 对拍标量路径（cos ≥ 0.9999、逐列 ≤ 2 ULP f16）后撤 `want_hmx=0`。

## R5: 视图链真实基数口径（P1 目标校准）

- **Decision**: net.json 源图普查（SS∘SS 1158、SS←Reshape 1184、identity 仅 6、Reshape∘Reshape 0）只作**上界参考**；SC-001 的真实基数以 T001 的 prepare-后 dump 为准（const 注入/DCE 后图不同）。
- **Rationale**: net.json 是 loader 前；pass 看到的是 inject_htp_prepare_inputs + 既有 pass 跑完的图。T001 产出修正目标，避免 M1 验收时口径打架。
- **Alternatives**: 直接信 net.json 口径——历史教训（host_run 输入不敏感那类"想当然"）否决。

## R6: 收益模型（为什么删 op 就省时）

- **Decision**: 视图 op 的设备成本以 dispatch 为主（0.8B 实测 SS 均 1.9ms，而其张量多为 [3,1,32] 级小片——带宽解释不了），故"删 op ≈ 省 1.9ms"线性模型成立。
- **Rationale**: op_timing code=15 n=3507 total 6.69s；同类 code=11（Reshape 混合）0.9ms/op。删 1000 个视图 op ≈ 1.9s ≈ 4.8% wall。
- **Alternatives**: 内核侧批处理相邻视图 op（设备改动）——违反纪律，且 003 的 runlist 优化另有盘子。

## R7: 共享文件协调（过程约束）

- **Decision**: graph_prepare.cpp（注册点）、op_matmul.cpp（发射门）、oplist_exec.c（3b）均 003 会话在途文件。每个提交前 `git status` + 精确暂存（e4d1dbb 先例）；3b 的 oplist_exec.c 改动安排在 003 T014/T015 落地后的空窗，或走私有名隔离（-DGEHTP_JOB_PATH + 私有 lib）。
- **Rationale**: 跨会话协调表（memory/gehtp-cross-session-coordination）实锤 4 会话共树；job.txt 已被并发覆盖过一次。
- **Alternatives**: 开独立 worktree（004 先例）——本 feature 的 P1/P2/3a 改动面小且与 003 无同段冲突，不值得分裂；3b 视冲突情况再定。
