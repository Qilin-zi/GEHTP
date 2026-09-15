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
/* 第7步阶段二: VTCM 驻留编码(0x4000|temp_id, 与 0x8000|slot 同族不冲突)
 * 驻留张量 = VTCM 静态偏移表内的 temp; 用时由引擎放 VTCM 基址+偏移。
 * TEMPOFF 槽 reserve 字段拆两段(u32): [低 16 位=表外 bump 预留 KB] |
 * [高 16 位=VTCM 池大小 KB](0=无 VTCM 驻留)。 */
#define WT_REF_VTCM_FLAG 0x4000u

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
    OP_UNARY_F16 = 11,      /* [x_t,y_t,n,subtype]   subtype: 0=NEG 1=EXP 2=SQRT 3=RSQRT 4=LOG 5=ABS 6=SIN 7=COS; (Neuron) 8=SIGMOID 9=TANH 10=GELU 11=RELU 12=SWISH */
    OP_BINARY_F16 = 12,     /* [a_t,b_t,y_t,n,subtype] subtype: 0=ADD 1=SUB 2=MUL 3=DIV */
    OP_SOFTMAX_F16 = 13,    /* [x_t,y_t,rows,n]       rows = 行数(每行 n 元素) */
    OP_CONCAT_F16 = 14,     /* [in_t0..7,out_t,axis,n_segments,n_elems,size0..3] arity 16
                                (每段 axis 维尺寸显式; ≤4 段, 0.8B 实测 2-3 段) */
    OP_STRIDED_SLICE_F16 = 15, /* [x_t,y_t,n_out,rank,b0..2,e0..2,s0..2,d0..2]
                                arity 16: rank≤3 通用切片, d=输入 dims(rank4 且
                                dim0=1 由 emit 降 rank 并取轴 1..3) */
    OP_SPLIT_F16 = 16,      /* [x_t,out_t0..7,axis,n_segments,size0..3,split_index]
                                arity 16 (副本 op 取第 split_index 段写 out_t0) */
    OP_REDUCE_F16 = 17,     /* [x_t,y_t,n,axis,subtype,dim0..3] arity 9
                                n=输入元素数, dims=输入形状(rank≤4) */
    OP_CUMSUM_F32 = 18,     /* [x_t,y_t,rows,n,axis,exclusive,reverse] f32 保持 */
    OP_CONV1D_SSM_F16 = 19, /* [x_t,w_s,y_t,seq,C,k]    depthwise causal conv+SiLU(SSM) */
    OP_GATHER_F16 = 20,     /* [table_s,idx_s,out_t,n,row_bytes] idx_s 为 int32 槽 */
    OP_ARGMAX_F16 = 21,     /* [x_t,out_t,n] */
    OP_KV_APPEND_F16 = 22,  /* [k_t,v_t,state_s,pos]    (M5 decode 用) */
    OP_KV_GATHER_F16 = 23,  /* [state_s,q_t,pos,n_kv,head_dim] */
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
};

/* 每个 opcode 的参数个数 (下标 = opcode) */
#define WT_ARITY_NOP 0
#define WT_ARITY_MATMUL 6
#define WT_ARITY_RMSNORM 4
#define WT_ARITY_PIN 1
#define WT_ARITY_SILU 3
#define WT_ARITY_IM2COL 15
#define WT_ARITY_CONV2D_F16 13
#define WT_ARITY_ADD_F16 4
#define WT_ARITY_SPILL 4
#define WT_ARITY_FILL 4
#define WT_ARITY_TRANSPOSE_F16 6
#define WT_ARITY_UNARY_F16 4
#define WT_ARITY_BINARY_F16 5
#define WT_ARITY_SOFTMAX_F16 4
#define WT_ARITY_CONCAT_F16 16
#define WT_ARITY_STRIDED_SLICE_F16 16
#define WT_ARITY_SPLIT_F16 16
#define WT_ARITY_REDUCE_F16 9
#define WT_ARITY_CUMSUM_F32 7
#define WT_ARITY_CONV1D_SSM_F16 6
#define WT_ARITY_GATHER_F16 5
#define WT_ARITY_ARGMAX_F16 3
#define WT_ARITY_KV_APPEND_F16 4
#define WT_ARITY_KV_GATHER_F16 5
#define WT_ARITY_MATMUL_F16 7
#define WT_ARITY_RMSNORM2_F16 5
#define WT_ARITY_BROADCAST_F16 12
#define WT_ARITY_TRANSPOSE_GEN_F16 8

struct wt_slot {
    uint32_t len;
    uint32_t count;
    uint32_t offset;
    uint32_t addr;
};

struct wt_op {
    uint16_t opcode;
    uint16_t n_args;
    uint32_t args[WT_MAX_ARGS];
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
};

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

#ifdef __cplusplus
}
#endif

#endif /* OPLIST_PARSE_H */
