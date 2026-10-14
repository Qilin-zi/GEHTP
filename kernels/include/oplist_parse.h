/* oplist_parse.h — blob v1 格式常量 + 解析 API (host/设备同源, 唯一真值)
 *
 * 布局 (v1, 全部字段小端):
 *   off 0  : MAGIC "WTOP" (4B)
 *   off 4  : u16 ver      (= WT_BLOB_VER)
 *   off 6  : u16 endian_chk (= WT_ENDIAN_CHK, 读出不等即拒)
 *   off 8  : u32 n_slots
 *   off 12 : u32 n_ops
 *   off 16 : slot[n_slots] × 16B { u32 len; u32 count; u32 offset; u32 addr }
 *            offset 相对 weight 区起点, 128B 对齐; addr 恒 0 (设备 pin 后回填)
 *   接    : op[n_ops] { u16 opcode; u16 n_args; u32 args[n_args] }
 *   接    : weight 区 (起点 = op 表结束后首个 128B 对齐偏移; slot 间 pad 128B)
 *
 * opcode 语义 (args 只放 slot id / temp id / 维度, 不放指针):
 *   temp 位置的 arg 可用 (0x8000|slot_id) 引用 slot —— 输入注入(首层
 *   transpose 读输入 slot 0, 引擎按位 0x8000 区分 temp/slot 空间)。
 *   OP_NOP          = 0  : []
 *   OP_MATMUL_W4A16 = 1  : [act_slot, w_slot, out_temp, M, K, N]
 *   OP_RMSNORM_F16  = 2  : [x_temp, w_slot, y_temp, n]
 *   OP_PIN          = 3  : [slot]
 *   OP_SILU_F16     = 4  : [x_temp, y_temp, n_elem]   (V2.3 U16)
 *   OP_IM2COL       = 5  : [act_temp, out_temp, H, W, C, kh, kw, ph, pw,
 *                           sh, sw, y0, x0, th, tw]   (GEHTP 阶段8 契约)
 *   OP_CONV2D_F16   = 6  : [act_temp, w_slot, bias_slot, out_temp, M, K, N,
 *                           out_y0, out_x0, out_H, out_W, co0, co_n]
 *   OP_ADD_F16      = 7  : [a_temp, b_temp, out_temp, n_elem]
 *   OP_SPILL        = 8  : [src_temp, pool_slot, pool_offset, n_elem]
 *   OP_FILL         = 9  : [pool_slot, pool_offset, dst_temp, n_elem]
 *   OP_TRANSPOSE_F16= 10 : [src_temp, out_temp, H, W, C, perm_u32]
 *                          (perm: 每轴 1 字节, N 恒 1 契约)
 */
#ifndef OPLIST_PARSE_H
#define OPLIST_PARSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WT_BLOB_VER 1u
/* v2 (C4, 2026-09-17): slot 线记录 16B → 24B, offset u32 → u64
 * {len u32, count u32, offset_lo u32, offset_hi u32, addr u32, reserved u32}。
 * 动因: minicpm f16 全模型 blob 5.15GB 撞 v1 的 u32 字节偏移 4GB 天花板
 * (emit 自校验 max_slot_end=4309975296 > 0xFFFFFFFF 实锤; 旧树 split stageA/B
 * 即此限制的历史绕行)。v2 寻址上限 u64, reserved 留 dtype/flags 等扩展。
 * emit 仅在权重区(内联或外置)逼近 4GB 时写 v2, 小模型继续写 v1
 * (存量 byte-exact 回归门零影响)。内存侧 wt_slot.offset 恒 u64 字节。 */
#define WT_BLOB_VER_2 2u
#define WT_SLOT_SIZE_V2 24u
/* v3 (003 compiler-driven scheduling, 2026-09-29): 逐算子引擎注解 footer。
 * 编译器给每个 op 指派计算单元 + 回落策略, 运行期照做 (FR-001..004)。
 * footer = op 表之后、weight 区 128B 对齐之前, n_ops 条 4B (u32 打包):
 *   bit 0-1  engine           0=scalar 1=HVX 2=HMX 3=reserved (对齐 wt_engine)
 *   bit 2    fallback_allowed 0=禁止回落(硬报错) 1=允许回落
 *   bit 3-4  fallback_engine  允许回落时回落到的单元 (0/1/2)
 *   bit 5    carve_reuse      1=复用上一 carve 0=逐 op carve (010 US5)
 *   bit 6-31 保留 (0)
 * ver<=2 无此 footer → 运行期走 legacy 自决 (FR-009)。v3 恒 24B slot 记录
 * (ver>=2 单规则), 与 v2 大权重路径共享 slot 线格式; 小模型多付 12B/slot。 */
#define WT_BLOB_VER_3 3u
#define WT_ENDIAN_CHK 0x1234u
#define WT_SLOT_SIZE 16u
#define WT_WEIGHT_ALIGN 128u
/* M2 容量: 千节点图(0.8B ~6K op)需大表; 设备侧 wt_parse/exec 同步 */
#define WT_MAX_SLOTS 4096u
#define WT_MAX_OPS 65536u
#define WT_MAX_ARGS 16u
/* 外部槽标记 (Level 1 运行期输入注入, GEHTP 阶段9):
 * slot.addr == EXT_IN  → 该 slot 数据在 wt_exec_run_io 的 in_ptr (blob 不固化)
 * slot.addr == EXT_OUT → 输出 temp 拷贝目标由 run_io 的 out_ptr 提供 */
#define WT_SLOT_EXT_IN  0xFFFFFFFFu
#define WT_SLOT_EXT_OUT 0xFFFFFFFEu
/* 第7步阶段一: 静态 temp 偏移表槽(wtop_emit 打包, wt_exec_run 消费)
 * 槽数据 = [cap u32][reserve u32][n u32][n × {temp_id u32, offset u32,
 * size u32}]。表内 temp 静态定址, 表外 temp 回落池尾预留区 bump。 */
#define WT_SLOT_TEMPOFF  0xFFFFFFFDu
/* 路线B(4B 外置权重): 权重 slot 的 len/offset 是描述符不内嵌 blob,
 * slot.addr == EXT_WGT → 该 slot 数据在外部权重区(run_io 的 wgt_ptr 基址
 * + slot.offset)。blob 只带元数据, 19GB 级权重独立文件传输/驻留。
 * 与 EXT_IN 并列: EXT_IN=运行时输入, EXT_WGT=外部权重文件。 */
#define WT_SLOT_EXT_WGT 0xFFFFFFFCu
/* 第7步阶段二: VTCM 驻留编码(0x4000|temp_id, 与 0x8000|slot 同族不冲突)
 * 驻留张量 = VTCM 静态偏移表内的 temp; 用时由引擎放 VTCM 基址+偏移。
 * TEMPOFF 槽 reserve 字段拆两段(u32): [低 16 位=表外 bump 预留 KB] |
 * [高 16 位=VTCM 池大小 KB](0=无 VTCM 驻留)。 */
#define WT_REF_VTCM_FLAG 0x4000u
/* 013 A 段 (T013/T016): TEMPOFF 槽尾随 VTCM 静态布局段 (R4 载体)。
 * 槽数据 = [cap][reserve][pin_cap][n][n × {tid,off,sz}] 之后追加:
 * [magic u32][ver u32][region_pin{off,size}][region_arena{off,size}]
 * [region_resident{off,size}] —— 三区互斥、VTCM 基址相对偏移; resident
 * 尾倒划 (off = budget - pool_rounded)。旧运行期只认前 16+n*12 字节, 忽略
 * 尾随段 (len ≥ 16+n*12 容忍); 新运行期按 magic 判存在、按表一次分区。 */
#define WT_TEMPOFF_LAYOUT_MAGIC 0x5654434Cu  /* 'VTCL' */
#define WT_TEMPOFF_LAYOUT_VER   1u
#define WT_TEMPOFF_LAYOUT_BYTES 32u           /* magic+ver+3×{off,size} = 8 u32 */
/* 004 decode P2 (FR-004): 命名持久状态表槽。slot.addr == STATE →
 * 槽数据 = [n u32][n × {bytes u32}] (n ≤ WT_MAX_STATES), 每项一块
 * DDR 缓冲 (memalign 128, 零初始化)。状态生命期 = 进程级持久:
 * 跨 wt_exec_run_io 存活, wt_exec_shutdown 不回收 (persistent decode
 * 语义), 新序列用 wt_exec_state_reset 清零。名称留在 emit 侧
 * manifest json (wtop.manifest.json), blob 内是紧凑 idx。
 * op 引用编码 0x6000|state_idx (平行 0x8000|slot / 0x4000|temp;
 * idx ≤ 0x1FFF)。KV 状态 = kvc ring 布局 (K面|V面 per slot);
 * SSM 状态 = f32 面 (conv win / delta S), 解释权在 op。 */
#define WT_SLOT_STATE 0xFFFFFFFBu
#define WT_REF_STATE_FLAG 0x6000u
#define WT_MAX_STATES 64u
/* T018 (004 decode): KV op 的 pos/n_kv 哨兵 —— 状态内自增计数器驱动
 * (图静态、位置动态的折中)。值域避开全部 WT_SLOT_*(FF..FB)。 */
#define WT_KV_POS_AUTO 0xFFFFFFFAu
/* P5: 引擎面引用编码(0xC000|surface_id), 仅 OP_DMA 的 src/dst 可用。
 * 0=e.act, 1=e.out, 2=e.wt, 3=e.bias。 */
#define WT_REF_ENG_FLAG 0xC000u
/* v3 VTCM carve footer: [magic u32][count u32][count × 20B record].
 * The header makes the optional region distinguishable from alignment padding. */
#define WT_CARVE_MAGIC 0x43524156u /* "VARC" little-endian */
#define WT_CARVE_REC_SIZE 20u
/* 009 GDN 块融合 (2026-10-10): 可选 per-op 几何尾挂 footer, 位于 carve footer
 * 之后、weight 128B 对齐之前:
 *   [magic u32 = WT_GDN_EXTRA_MAGIC][count u32][count × {op_index u32, 40B extra}]
 * 40B extra = 10×u32 (struct wt_gdn_extra), 字段序与 compiler ExtraGdnBlock
 * (op_extra.hpp) 逐一对应: t/ntok/c/h/d/d_conv/chunk/vtcm_budget_bytes/
 * state_idx/contract_ver。仅含 OP_GDN_BLOCK 的 blob 写此项; 无融合层则此处
 * 是 128B 对齐填充(全 0), magic 探测永不误判(同 carve 逻辑)。 */
#define WT_GDN_EXTRA_MAGIC 0x584E4447u /* "GNDX" little-endian */
#define WT_GDN_EXTRA_WORDS 10u
#define WT_GDN_EXTRA_BYTES (WT_GDN_EXTRA_WORDS * 4u) /* 40 */
#define WT_GDN_EXTRA_REC   (4u + WT_GDN_EXTRA_BYTES) /* op_index + 40B = 44 */
/* 013 C 段 (T018): 可选 carve 布局 footer, 位于 gdn footer 之后、weight
 * 128B 对齐之前:
 *   [magic u32][n_layouts u32][n_layouts × 40B][n_refs u32]
 *   [n_refs × {op_index u32, layout_id u32}]
 * 40B record = 10×u32 {layout_id, M, K, N, n0, off_wt, off_bias, off_act,
 * off_out, arena_hi} —— dc_w4_carve 分配序列的闭式镜像 (arena 相对偏移;
 * atbl/otbl/mask/extra 偏移由 M/K/N 闭式推导, 不上线)。dedup 键 =
 * (M,K,N,n0); layout_id = 记录下标。执行器装载期独立重算闭式对拍 (漂移 =
 * 拒载) 并展开为绝对指针, W4A16 热路径退化为指针直赋。仅 has_layout
 * (VTCL 在场) 时消费; 仅含 OP_MATMUL_W4A16 的 blob 写此项; magic 探测
 * 同 carve/gdn 逻辑 (对齐填充全 0 永不误判)。 */
#define WT_CARVE_LAYOUT_MAGIC 0x4C565243u /* "CRVL" little-endian */
#define WT_CL_REC_SIZE 40u
#define WT_CL_REF_REC 8u
#define WT_MAX_CL 256u        /* 执行器布局表容量 (emit 超 = footer 仍写全) */
#define WT_MAX_CL_REF 1024u   /* 执行器 op→layout 引用表容量 */
#define WT_ENG_ACT 0u
#define WT_ENG_OUT 1u
#define WT_ENG_WT 2u
#define WT_ENG_BIAS 3u
/* 不做该侧 cache/fence 操作。合法通道值取 fence.h 的 FC_*。 */
#define WT_DMA_FENCE_NONE 0xFFu
/* 003 T015: OP_DMA 的 src/dst 另可用 0x4000|temp (VTCM 驻留, 含编译期
 * staging) 与 0x8000|slot (含 EXT_IN 注入) —— DDR↔VTCM 预置搬移的发射
 * 形态 (exec_dma 的 dma_ref_resolve 解析两侧)。W4A16 act 预置 = 主发射面:
 * OP_DMA {src=act 源, dst=0x4000|stg, M*K*2, 0,0, flags=1, FF, FF} 紧邻
 * OP_MATMUL_W4A16 {0x4000|stg, w_裸slot, out, M,K,N}。 */

/* ── opcode 枚举 ─────────────────────────────────────────────────────────
 * 各 opcode 参数契约 (arity / 参数身份 slot_check / 语义名 name / kind) 的
 * 机器可读真源在 op_spec.inc 的 wt_op_spec[] (011 单点注册表)。下列注释已
 * 降级为人类导读: 括号内 arg 列表为简写, subtype 枚举 / 行为注记(尚未机器化
 * 的部分)仍以本注释为准。 */
enum {
    OP_NOP = 0,
    OP_MATMUL_W4A16 = 1,
    OP_RMSNORM_F16 = 2,
    OP_PIN = 3,
    OP_SILU_F16 = 4,
    /* GEHTP 阶段8/9 契约(conv2d+add 流水线) */
    OP_IM2COL = 5,
    OP_CONV2D_F16 = 6,
    OP_ADD_F16 = 7,
    OP_SPILL = 8,
    OP_FILL = 9,
    OP_TRANSPOSE_F16 = 10,
    /* M2/M3 契约(D6 清单, QNN op 语义; 0-10 不动, 向后兼容) */
    OP_UNARY_F16 = 11,      /* [x_t,y_t,n,subtype]   subtype: 0=NEG 1=EXP 2=SQRT 3=RSQRT 4=LOG 5=ABS 6=SIN 7=COS; (Neuron) 8=SIGMOID 9=TANH 10=GELU 11=RELU 12=SWISH 13=SOFTPLUS(稳定化, host ops.cpp 同款) 14=COPY; 15=ISNAN(bool 0/1) 16=ISINF(bool 0/1) */
    OP_BINARY_F16 = 12,     /* [a_t,b_t,y_t,n,subtype] subtype: 0=ADD 1=SUB 2=MUL 3=DIV 4=EQ(bool 0/1); 5=AND(bool 0/1); 8=SELECT 三元(另 4 参形态) */
    OP_SOFTMAX_F16 = 13,    /* [x_t,y_t,rows,n]       rows = 行数(每行 n 元素) */
    OP_CONCAT_F16 = 14,     /* [in_t0..7,out_t,axis,n_segments,n_elems,size0..3] arity 16
                                (每段 axis 维尺寸显式; ≤4 段, 0.8B 实测 2-3 段) */
    OP_STRIDED_SLICE_F16 = 15, /* [x_t,y_t,n_out,rank,b0..2,e0..2,s0..2,d0..2]
                                arity 16: rank≤3 通用切片, d=输入 dims(rank4 且
                                dim0=1 由 emit 降 rank 并取轴 1..3) */
    OP_SPLIT_F16 = 16,      /* [x_t,out_t0..7,axis,n_segments,size0..3,split_index]
                                arity 16 (副本 op 取第 split_index 段写 out_t0) */
    OP_REDUCE_F16 = 17,     /* [x_t,y_t,n,axis,subtype,dim0..3] arity 9
                                n=输入元素数, dims=输入形状(rank≤4; rank>4 时发射侧
                                归一化为 axis=0 + 轴起后缀, leading 维由 outer 展平吸收)
                                subtype: 0=SUM 1=AVG 2=MAX 3=MIN; 单轴 strided 归约,
                                axis/subtype 越界硬错误 (禁 0xFFFFFFFF 哨兵野读) */
    OP_CUMSUM_F32 = 18,     /* [x_t,y_t,rows,n,axis,exclusive,reverse] f32 保持 */
    OP_CONV1D_SSM_F16 = 19, /* [x_t,w_s,y_t,seq,C,k]    depthwise causal conv(SSM);
                               y[t]=Σ_j w[j]·x[t-k+1+j] (纯卷积, 无激活 — SiLU 由图中
                               独立 Sigmoid/Mul 节点分解, 别再套 silu); 内存 tap j=0 配
                               最老样本; x 有效行 = seq-k+1, 窗外隐式零 (QNN pad flush) */
    OP_GATHER_F16 = 20,     /* [tbl,idx,out,n,packed,td0..4,xd0..4] arity 15
                                packed = axis | xrank<<8 | irank<<16 (均紧凑形口径);
                                host ops.cpp Gather 坐标分解同款: od=td 把 axis 换成 xd,
                                idx 块查 tbl[axis], 余维直通, r 越界出零;
                                idx int32 (const 原样槽/输入槽), tbl 可 temp(GDN state 回读) */
    OP_ARGMAX_F16 = 21,     /* [x_t,out_t,n] */
    OP_KV_APPEND_F16 = 22,  /* [k_t,v_t,state_ref,pos] (004 P2 实义化)
                               k/v 各 [kv_heads*head_dim] f16 (ref: temp/
                               0x8000|slot); state_ref=0x6000|idx; pos 为
                               绝对位置 —— 写 ring 槽 pos%nslots (kvcache.h);
                               T018: pos==WT_KV_POS_AUTO → 取状态内自增计数器
                               (append 后 +1; 首步=0), 计数器由
                               wt_exec_state_reset 清零, 满 nslots=硬错 */
    OP_KV_GATHER_F16 = 23,  /* [state_ref,kT_out,v_out,n_kv,n_heads,kv_heads,
                               head_dim] arity 7 (004 P2 实义化)。读位置
                               [0,n_kv) 出 GQA repeat_interleave 展开的双面:
                               kT_out [n_heads, head_dim, n_kv] (per-head Kᵀ,
                               供 batched scores matmul 的 W=[K,N]),
                               v_out [n_heads, n_kv, head_dim] (供 probs·V)。
                               T018: n_kv==WT_KV_POS_AUTO → 读计数器当前值
                               (append 后调用 = 已含本 token); kT_out/v_out
                               为 0 = 该面跳过 (KT/V 拆两节点单输出形态)。
                               槽 miss (posmap 不符) = 硬错误, 状态 bug 必炸 */
    OP_MATMUL_F16 = 24,     /* [a_ref,w_s,out_t,M,K,N,flags] f16×f16→f16 (f32 累加)
                               flags bit0=转置 a(存[K,M]) bit1=转置 w(存[N,K]);
                               W4A16 留给 Q4_0 打包权重 (float 图 GEMM) */
    OP_RMSNORM2_F16 = 25,   /* [x_ref,w_s,b_s,y_t,n] 通用 RMSNorm: 纯 f16 面直读
                               (无 crouton); n=总元素, 行宽=w 槽长/2, m=n/行宽;
                               b_s 为 bias 槽(无 bias 时 zero dummy 槽); eps=1e-6
                               (opcode 2 保留 conv 管线 crouton 契约不动) */
    OP_BROADCAST_F16 = 26,  /* [b_ref,y_t,n,b_elems,in_d0..3,out_d0..3] numpy 广播
                               (右对齐逐轴: in==1 的轴扩到 out, 缺失轴视 1);
                               emit 在二元 op 前把广播操作数物化为全尺寸 temp,
                               保持 OP_ADD/OP_BINARY 纯元素语义不动 */
    OP_TRANSPOSE_GEN_F16 = 27, /* [x_ref,y_t,rank,d0..d3,perm4B] 通用 N-D C 序转置
                               (rank 2/3/4; dims 为输入形状; opcode 10 保留
                               conv 管线 4-D NCHW 契约不动) */
    OP_SCATTER_ND_F16 = 28, /* [data_ref,idx_s,upd_ref,out_t,n_out,rank,d0..d4,K,n_idx,block]
                               arity 14 (A3② 真语义; 恒等拷贝=数值死刑, docs/A3 判决)
                               data/upd f16, idx i32 (0x8000|slot 或 temp 引用);
                               out=data 拷贝后按 n_idx 组 K 维坐标写 block 块 */
    /* P5: True DMA runlist operator (arity=8, args: [src,dst,bytes,src_off,dst_off,flags,fence_src,fence_dst]) */
    OP_DMA = 33,
    /* M4.5: Pad 真物化 (2026-09-24, 0.8B GDN 门控链修复).
       恒等拷贝降级在 temp 复用下把 pad 区留成垃圾, CumSum 累加后全链毒化。
       形态门: 单维增长 (前导维折 outer, 尾维折 inner), 常量填充。
       [x_t,y_t,outer,in_inner,out_inner,pad_before_elems,pad_val_f16]
       每 outer 块: 先填 pad_before_elems 个 pad_val, 拷 in_inner 个 f16,
       再填 (out_inner-pad_before-in_inner) 个 pad_val。 */
    OP_PAD_F16 = 34,
    /* 002 VTCM 统一 arena (2026-09-28): matmul 编译期 N 分块。
       [a_ref,w_s,out_t,M,K,N_full,n0,N_chunk,flags] arity 9
       与 OP_MATMUL_F16 同源: f16×f16→f16 (f32 累加), flags 同 bit 布局;
       N_full=整 op 列数, n0=本 chunk 列基址, N_chunk=本 chunk 列数(32×)。
       act/wgt/out 三引用跨 chunk 共享, 仅 n0/N_chunk 两标量不同;
       逐字节恒等五纪律见 spec FR-013 (每 tile clracc / 重装 bias 页 /
       k 升序 / 回落全有或全无 / 同引擎)。 */
    OP_MATMUL_F16_SPLIT = 35,
    /* FC bias 折叠 (pass_fc_bias, 2026-09-29): 与 OP_MATMUL_F16 同源 + 尾部
     * 追加 bias 槽引用。 [a_ref,w_s,out_t,M,K,N,flags,bias_s] arity 8
     * bias_s = 0x8000|slot (N 元素 f16 按列加); flags 同 bit 布局。
     * 设备 v1 仅标量路径应用 bias (f32 累加器按列加); HMX 形状由编译期
     * pass 门挡住 (M/K/N 全 32 倍数不折), 故本 opcode 实际只遇标量形。 */
    OP_MATMUL_F16_BIAS = 36,
    /* 004 decode P2 (2026-10-08): 状态化步进 op。占用 29/30 (两树均空闲;
       主线 003/005 已占 33-36, 勿撞)。 */
    OP_CONV1D_SSM_STEP_F16 = 29, /* [x_ref, w_s, y_t, state_ref, C, d_conv]
                               arity 6。单 token 步进 conv (状态机 B1):
                               y = silu(Σ_j w[j]·win[j] + w[dc-1]·x),
                               win = state 内 [(d_conv-1)*C] f32 滑窗
                               (win[0] 最老); 步进后窗口推入 x。
                               !!! 含 SiLU —— 与 op 19 (纯卷积, SiLU 图内
                               分解) 不同, decode 图发射端不再接 Sigmoid/Mul。
                               w 槽 [d_conv, C] f16 同 op 19 布局。 */
    OP_DELTA_CHUNK_F16 = 30,     /* [k_ref,v_ref,q_ref,beta_ref,g_ref,y_t,
                               state_ref,c,ntok,H,D,q_raw] arity 12 (T018)。
                               状态机 B2 gated delta rule chunk 更新:
                               k/v/q/beta/g 输入, y 输出; 缓冲布局
                               [H][ntok][D] (k/v/q/y) 与 [H][ntok] (beta/g);
                               状态 S = [H*D*D] f32 外置原地更新;
                               c = 本 chunk token 数 (decode 步 c=1);
                               q_raw=0: q 的 1/√D 缩放在 kernel 内;
                               q_raw=1: q 原样 (0.8B decode 图 chunk 机器
                               契约无此 scale, q̂ 直喂)。 */
    /* 009 GDN 块融合 (2026-10-10): linear_attn 整块单发射 op。
       [xqkv_t, conv_w_s, beta_t, g_t, y_t] arity 5。
       几何 (T/ntok/C/H/D/d_conv/chunk/vtcm_budget_bytes/state_idx/
       contract_ver) 全走 serialized_extra (ExtraGdnBlock 尾挂
       [u32 len][bytes]), 不占 args (args≤16 硬约束下几何外置)。
       ref 四套编码: 0x8000|slot 权重槽 (仅 idx1) / 裸 temp id /
       0x4000|temp VTCM 驻留 / 0x6000|idx 状态槽 (US5 decode 步形,
       prefill 态 state_idx=0xFFFFFFFF 瞬态 carve 哨兵, 在 extra 不在 args)。
       xqkv_t = in_proj 出值 f16 [ntok, 3C]; conv_w_s = [d_conv, C] f16
       同 op19 布局; beta/g_t = [C] 门控向量; y_t = 块出口 f16。 */
    OP_GDN_BLOCK_F16 = 37,
};

struct wt_slot {
    uint32_t len;
    uint32_t count;
    uint64_t offset;   /* 内存侧恒为字节偏移; 线格式 v1=u32 字节, v2=u64(24B 记录) */
    uint32_t addr;
};

struct wt_op {
    uint16_t opcode;
    uint16_t n_args;
    uint32_t args[WT_MAX_ARGS];
};

/* 009 GDN 块融合: OP_GDN_BLOCK 的几何 (ExtraGdnBlock 设备侧同构镜像)。
 * 字段序与 compiler ExtraGdnBlock 逐一对应 (勿动); 全 u32 小端。 */
struct wt_gdn_extra {
    uint32_t t;                 /* 序列全长 (prefill = 输入 token 数) */
    uint32_t ntok;              /* 本块 token 跨度 (prefill 单发射 = T) */
    uint32_t c;                 /* 本 chunk 实际 token 数 (≤ GDN_C_MAX=64) */
    uint32_t h;                 /* 头数 (生产 16) */
    uint32_t d;                 /* 头维 (生产 128; conv 通道 3HD) */
    uint32_t d_conv;            /* conv 窗长 (生产 4) */
    uint32_t chunk;             /* delta 名义分块 (生产 32) */
    uint32_t vtcm_budget_bytes; /* 闭式预算 (128 对齐, FR-004) */
    uint32_t state_idx;         /* US5 decode 状态槽 idx; prefill=0xFFFFFFFF */
    uint32_t contract_ver;      /* numerics-contract.md 版本 */
};

struct wt_blob {
    uint16_t ver;
    uint32_t n_slots;
    uint32_t n_ops;
    uint32_t weight_off;        /* weight 区在 blob 内的绝对偏移 */
    struct wt_slot slots[WT_MAX_SLOTS];
    struct wt_op ops[WT_MAX_OPS];
    const uint8_t* weight_base; /* 指向输入缓冲内 weight 区 */
    size_t weight_bytes;        /* weight 区可用字节数 */
    const uint8_t* annot_base;  /* v3: 引擎注解表 (n_ops 条 u32); ver<3 = NULL */
    const uint8_t* carve_base;  /* v3: optional VTCM carve footer records */
    uint32_t n_carve;           /* number of entries in carve_base */
    const uint8_t* gdn_base;    /* 009: optional per-op GDN 几何 footer (records) */
    uint32_t n_gdn;             /* number of gdn extra records */
    const uint8_t* cl_base;     /* 013 C 段: optional carve 布局 footer (40B/条) */
    uint32_t n_cl;              /* 布局条数 */
    const uint8_t* cl_ref_base; /* op→layout 引用对区 (8B/条 {op_index, layout_id}) */
    uint32_t n_cl_ref;
};

/* 算子输入契约单点注册表: enum wt_arg_kind + struct wt_arg_spec/wt_op_spec +
 * static const wt_op_spec[]。下标 = opcode, arity/身份/语义的唯一事实源
 * (011 op-contract-registry)。位置须在 OP_* enum 与 WT_MAX_ARGS 之后。 */
#include "op_spec.inc"

enum {
    WT_OK = 0,
    WT_ERR_SHORT = -1,          /* 缓冲小于 header */
    WT_ERR_MAGIC = -2,
    WT_ERR_VER = -3,
    WT_ERR_ENDIAN = -4,
    WT_ERR_NSLOTS = -5,
    WT_ERR_NOPS = -6,
    WT_ERR_SLOTS_OVERRUN = -7,  /* slot 表越出缓冲 */
    WT_ERR_WEIGHT_OVERRUN = -8, /* slot.offset+len 越出 weight 区 */
    WT_ERR_OP_TRUNC = -9,       /* op 表截断 */
    WT_ERR_OPCODE = -10,        /* 未知 opcode */
    WT_ERR_ARITY = -11,         /* n_args 与 opcode 不符 */
    WT_ERR_BAD_REF = -12,       /* args 引用不存在的 slot */
    WT_ERR_ALIGN = -13,         /* slot.offset 非 128B 对齐 */
};

/* 解析 + 全部边界检查。返回 WT_OK 或负错误码; out 仅在 WT_OK 时有效。 */
int wt_parse(const uint8_t* buf, size_t size, struct wt_blob* out);

const char* wt_err_str(int rc);

/* 供 builder/测试复用的小端读取 */
static inline uint32_t wt_rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint16_t wt_rd_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/* v3 引擎注解 footer 打包/解包 (host emit 打包 / 设备 exec 解包共用) */
static inline uint32_t wt_annot_pack(uint32_t engine, uint32_t fallback_allowed,
                                     uint32_t fallback_engine) {
    return (engine & 0x3u) | ((fallback_allowed & 1u) << 2) |
           ((fallback_engine & 0x3u) << 3);
}
static inline uint32_t wt_annot_engine(uint32_t a) { return a & 0x3u; }
static inline uint32_t wt_annot_fb_allowed(uint32_t a) { return (a >> 2) & 1u; }
static inline uint32_t wt_annot_fb_engine(uint32_t a) { return (a >> 3) & 0x3u; }
/* 契约 C (010 US5): op_annot bit5 = carve 复用标志。emit 侧在 wt_annot_pack 后置位;
 * exec 侧读 bit5 决定跳过 re-carve (见 oplist_exec.c exec_matmul)。 */
static inline uint32_t wt_annot_set_reuse(uint32_t* a, int reuse) {
    if (reuse) { *a |= (1u << 5); } else { *a &= ~(1u << 5); }
    return *a;
}
static inline uint32_t wt_annot_reuse(uint32_t a) { return (a >> 5) & 1u; }

#ifdef __cplusplus
}
#endif

#endif /* OPLIST_PARSE_H */
