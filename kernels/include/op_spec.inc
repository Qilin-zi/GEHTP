/* op_spec.inc — 算子输入契约单点注册表 (唯一事实源)
 *
 * 由 oplist_parse.h 在 struct wt_blob 之后、错误码 enum 之前 #include。
 * 下标 = opcode (0..37; 31/32 为保留空洞, arity = -1 哨兵)。
 *
 * 字段来源 (迁移前, 6 处散落 → 此处单点):
 *   arity      ← #define WT_ARITY_* (旧 oplist_parse.h:265-300) + arity_of() switch
 *   slot_check ← arg_is_slot() switch (旧 oplist_parse.c:53-88), 逐值转录(不从 kind 推导)
 *   name/kind  ← enum 语义注释 (旧 oplist_parse.h:133-262), 机器化
 *
 * kind 分类规则 (语义, 非行为):
 *   恒槽权重/常量   → WT_ARG_SLOT   (w_s / bias_s / pool_slot / conv_w_s ...)
 *   输入激活/引用    → WT_ARG_REF    (槽或 temp, 运行时按 0x8000 位判; a_ref/x_t 等)
 *   输出/临时       → WT_ARG_TEMP   (out_t / y_t / dst_temp / out_t0..7 ...)
 *   外置状态槽      → WT_ARG_STATE  (state_ref, 0x6000|idx)
 *   标量 dim/flag   → WT_ARG_IMM
 *   打包位域标量    → WT_ARG_IMM_PACKED (Gather packed / matmul flags)
 *
 * 关键不变量:
 *   - slot_check 与 kind 无推导关系 (research §2: out_t=false 而 state_ref=true 反例),
 *     故 slot_check 逐值转录, 独立于 kind。
 *   - 迁移后 arity_of()/arg_is_slot() 改读本表, 返回值与迁移前逐值等价。
 *   - 本表是只读编译期常量, 无运行时状态。
 *
 * 注: WT_ARG_UNCLASSIFIED=0 是「未分类」哨兵, 仅供覆盖率单测 (T011) 断言
 *     每个合法 arg (idx < arity) 都有非零 kind (即已分类)。故真实 kind 从 1 起。
 */

enum wt_arg_kind {
    WT_ARG_UNCLASSIFIED = 0,   /* 哨兵: 未分类 (合法 arg 不得留此值) */
    WT_ARG_SLOT = 1,           /* 恒槽权重/常量   0x8000|slot 或裸 slot id */
    WT_ARG_REF = 2,            /* 槽或 temp       裸 id 或 0x8000|slot(运行时按 flag) */
    WT_ARG_TEMP = 3,           /* 输出/临时       裸 id, 可 0x4000 VTCM */
    WT_ARG_STATE = 4,          /* 外置状态槽      0x6000|idx */
    WT_ARG_IMM = 5,            /* 标量            裸 u32 */
    WT_ARG_IMM_PACKED = 6,     /* 打包位域标量    裸 u32 (axis|xrank<<8|irank<<16 等) */
};

struct wt_arg_spec {
    const char *name;       /* 语义名, 如 "act"/"w"/"out"/"M"/"bias" */
    uint8_t     kind;       /* 上述 enum */
    int8_t      slot_check; /* 逐值转录 arg_is_slot() 布尔 (0/1) */
};

struct wt_op_spec {
    int8_t            arity;        /* ≤ WT_MAX_ARGS(16); -1 = 空洞/无效哨兵 */
    struct wt_arg_spec args[WT_MAX_ARGS];
    int8_t            legacy_arity; /* 013: 旧 arity(默认 0=无); OP_MATMUL_W4A16=6 */
};

/* 下标 = opcode (0..37, 位置序)。仅前 arity 个 args 有效, 其余零初始化。
 * slot_check 逐值转录自 arg_is_slot() switch (含 default:0 语义)。 */
#if defined(__GNUC__)
#define WT_OP_SPEC_UNUSED __attribute__((unused))
#else
#define WT_OP_SPEC_UNUSED
#endif
static const struct wt_op_spec wt_op_spec[] WT_OP_SPEC_UNUSED = {
    /* 0  OP_NOP: [] */
    { 0, { { NULL, WT_ARG_UNCLASSIFIED, 0 } } },

    /* 1  OP_MATMUL_W4A16: [act, w, out, M, K, N, bias, atbl, otbl] (legacy 6) */
    { 9, {
        { "act",  WT_ARG_REF,  1 },
        { "w",    WT_ARG_SLOT, 1 },
        { "out",  WT_ARG_TEMP, 0 },
        { "M",    WT_ARG_IMM,  0 },
        { "K",    WT_ARG_IMM,  0 },
        { "N",    WT_ARG_IMM,  0 },
        { "bias", WT_ARG_SLOT, 1 },
        { "atbl", WT_ARG_SLOT, 1 },
        { "otbl", WT_ARG_SLOT, 1 },
    }, 6 },

    /* 2  OP_RMSNORM_F16: [x, w, y, n] */
    { 4, {
        { "x", WT_ARG_TEMP, 0 },
        { "w", WT_ARG_SLOT, 1 },
        { "y", WT_ARG_TEMP, 0 },
        { "n", WT_ARG_IMM,  0 },
    } },

    /* 3  OP_PIN: [slot] */
    { 1, {
        { "slot", WT_ARG_SLOT, 1 },
    } },

    /* 4  OP_SILU_F16: [x, y, n] */
    { 3, {
        { "x", WT_ARG_TEMP, 0 },
        { "y", WT_ARG_TEMP, 0 },
        { "n", WT_ARG_IMM,  0 },
    } },

    /* 5  OP_IM2COL: [act, out, H, W, C, kh, kw, ph, pw, sh, sw, y0, x0, th, tw] */
    { 15, {
        { "act", WT_ARG_TEMP, 0 },
        { "out", WT_ARG_TEMP, 0 },
        { "H",   WT_ARG_IMM,  0 },
        { "W",   WT_ARG_IMM,  0 },
        { "C",   WT_ARG_IMM,  0 },
        { "kh",  WT_ARG_IMM,  0 },
        { "kw",  WT_ARG_IMM,  0 },
        { "ph",  WT_ARG_IMM,  0 },
        { "pw",  WT_ARG_IMM,  0 },
        { "sh",  WT_ARG_IMM,  0 },
        { "sw",  WT_ARG_IMM,  0 },
        { "y0",  WT_ARG_IMM,  0 },
        { "x0",  WT_ARG_IMM,  0 },
        { "th",  WT_ARG_IMM,  0 },
        { "tw",  WT_ARG_IMM,  0 },
    } },

    /* 6  OP_CONV2D_F16: [act, w, bias, out, M, K, N, out_y0, out_x0, out_H, out_W, co0, co_n] */
    { 13, {
        { "act",    WT_ARG_TEMP, 0 },
        { "w",      WT_ARG_SLOT, 1 },
        { "bias",   WT_ARG_SLOT, 1 },
        { "out",    WT_ARG_TEMP, 0 },
        { "M",      WT_ARG_IMM,  0 },
        { "K",      WT_ARG_IMM,  0 },
        { "N",      WT_ARG_IMM,  0 },
        { "out_y0", WT_ARG_IMM,  0 },
        { "out_x0", WT_ARG_IMM,  0 },
        { "out_H",  WT_ARG_IMM,  0 },
        { "out_W",  WT_ARG_IMM,  0 },
        { "co0",    WT_ARG_IMM,  0 },
        { "co_n",   WT_ARG_IMM,  0 },
    } },

    /* 7  OP_ADD_F16: [a, b, out, n] */
    { 4, {
        { "a",   WT_ARG_TEMP, 0 },
        { "b",   WT_ARG_TEMP, 0 },
        { "out", WT_ARG_TEMP, 0 },
        { "n",   WT_ARG_IMM,  0 },
    } },

    /* 8  OP_SPILL: [src, pool, pool_off, n] */
    { 4, {
        { "src",      WT_ARG_TEMP, 0 },
        { "pool",     WT_ARG_SLOT, 1 },
        { "pool_off", WT_ARG_IMM,  0 },
        { "n",        WT_ARG_IMM,  0 },
    } },

    /* 9  OP_FILL: [pool, pool_off, dst, n] */
    { 4, {
        { "pool",     WT_ARG_SLOT, 1 },
        { "pool_off", WT_ARG_IMM,  0 },
        { "dst",      WT_ARG_TEMP, 0 },
        { "n",        WT_ARG_IMM,  0 },
    } },

    /* 10 OP_TRANSPOSE_F16: [src, out, H, W, C, perm] */
    { 6, {
        { "src",  WT_ARG_TEMP, 0 },
        { "out",  WT_ARG_TEMP, 0 },
        { "H",    WT_ARG_IMM,  0 },
        { "W",    WT_ARG_IMM,  0 },
        { "C",    WT_ARG_IMM,  0 },
        { "perm", WT_ARG_IMM,  0 },
    } },

    /* 11 OP_UNARY_F16: [x, y, n, subtype] */
    { 4, {
        { "x",       WT_ARG_REF,  1 },
        { "y",       WT_ARG_TEMP, 0 },
        { "n",       WT_ARG_IMM,  0 },
        { "subtype", WT_ARG_IMM,  0 },
    } },

    /* 12 OP_BINARY_F16: [a, b, y, n, subtype] */
    { 5, {
        { "a",       WT_ARG_REF,  1 },
        { "b",       WT_ARG_REF,  1 },
        { "y",       WT_ARG_TEMP, 0 },
        { "n",       WT_ARG_IMM,  0 },
        { "subtype", WT_ARG_IMM,  0 },
    } },

    /* 13 OP_SOFTMAX_F16: [x, y, rows, n] */
    { 4, {
        { "x",    WT_ARG_REF,  1 },
        { "y",    WT_ARG_TEMP, 0 },
        { "rows", WT_ARG_IMM,  0 },
        { "n",    WT_ARG_IMM,  0 },
    } },

    /* 14 OP_CONCAT_F16: [in0..7, out, axis, n_segments, n_elems, size0..3] */
    { 16, {
        { "in0",        WT_ARG_REF,  1 },
        { "in1",        WT_ARG_REF,  1 },
        { "in2",        WT_ARG_REF,  1 },
        { "in3",        WT_ARG_REF,  1 },
        { "in4",        WT_ARG_REF,  1 },
        { "in5",        WT_ARG_REF,  1 },
        { "in6",        WT_ARG_REF,  1 },
        { "in7",        WT_ARG_REF,  1 },
        { "out",        WT_ARG_TEMP, 0 },
        { "axis",       WT_ARG_IMM,  0 },
        { "n_segments", WT_ARG_IMM,  0 },
        { "n_elems",    WT_ARG_IMM,  0 },
        { "size0",      WT_ARG_IMM,  0 },
        { "size1",      WT_ARG_IMM,  0 },
        { "size2",      WT_ARG_IMM,  0 },
        { "size3",      WT_ARG_IMM,  0 },
    } },

    /* 15 OP_STRIDED_SLICE_F16: [x, y, n_out, rank, b0..2, e0..2, s0..2, d0..2] */
    { 16, {
        { "x",     WT_ARG_REF,  1 },
        { "y",     WT_ARG_TEMP, 0 },
        { "n_out", WT_ARG_IMM,  0 },
        { "rank",  WT_ARG_IMM,  0 },
        { "b0",    WT_ARG_IMM,  0 },
        { "b1",    WT_ARG_IMM,  0 },
        { "b2",    WT_ARG_IMM,  0 },
        { "e0",    WT_ARG_IMM,  0 },
        { "e1",    WT_ARG_IMM,  0 },
        { "e2",    WT_ARG_IMM,  0 },
        { "s0",    WT_ARG_IMM,  0 },
        { "s1",    WT_ARG_IMM,  0 },
        { "s2",    WT_ARG_IMM,  0 },
        { "d0",    WT_ARG_IMM,  0 },
        { "d1",    WT_ARG_IMM,  0 },
        { "d2",    WT_ARG_IMM,  0 },
    } },

    /* 16 OP_SPLIT_F16: [x, out0..7, axis, n_segments, size0..3, split_index] */
    { 16, {
        { "x",           WT_ARG_REF,  1 },
        { "out0",        WT_ARG_TEMP, 0 },
        { "out1",        WT_ARG_TEMP, 0 },
        { "out2",        WT_ARG_TEMP, 0 },
        { "out3",        WT_ARG_TEMP, 0 },
        { "out4",        WT_ARG_TEMP, 0 },
        { "out5",        WT_ARG_TEMP, 0 },
        { "out6",        WT_ARG_TEMP, 0 },
        { "out7",        WT_ARG_TEMP, 0 },
        { "axis",        WT_ARG_IMM,  0 },
        { "n_segments",  WT_ARG_IMM,  0 },
        { "size0",       WT_ARG_IMM,  0 },
        { "size1",       WT_ARG_IMM,  0 },
        { "size2",       WT_ARG_IMM,  0 },
        { "size3",       WT_ARG_IMM,  0 },
        { "split_index", WT_ARG_IMM,  0 },
    } },

    /* 17 OP_REDUCE_F16: [x, y, n, axis, subtype, dim0..3] */
    { 9, {
        { "x",       WT_ARG_REF,  1 },
        { "y",       WT_ARG_TEMP, 0 },
        { "n",       WT_ARG_IMM,  0 },
        { "axis",    WT_ARG_IMM,  0 },
        { "subtype", WT_ARG_IMM,  0 },
        { "dim0",    WT_ARG_IMM,  0 },
        { "dim1",    WT_ARG_IMM,  0 },
        { "dim2",    WT_ARG_IMM,  0 },
        { "dim3",    WT_ARG_IMM,  0 },
    } },

    /* 18 OP_CUMSUM_F32: [x, y, rows, n, axis, exclusive, reverse] */
    { 7, {
        { "x",         WT_ARG_REF,  1 },
        { "y",         WT_ARG_TEMP, 0 },
        { "rows",      WT_ARG_IMM,  0 },
        { "n",         WT_ARG_IMM,  0 },
        { "axis",      WT_ARG_IMM,  0 },
        { "exclusive", WT_ARG_IMM,  0 },
        { "reverse",   WT_ARG_IMM,  0 },
    } },

    /* 19 OP_CONV1D_SSM_F16: [x, w, y, seq, C, k] */
    { 6, {
        { "x",   WT_ARG_REF,  1 },
        { "w",   WT_ARG_SLOT, 1 },
        { "y",   WT_ARG_TEMP, 0 },
        { "seq", WT_ARG_IMM,  0 },
        { "C",   WT_ARG_IMM,  0 },
        { "k",   WT_ARG_IMM,  0 },
    } },

    /* 20 OP_GATHER_F16: [tbl, idx, out, n, packed, td0..4, xd0..4] */
    { 15, {
        { "tbl",    WT_ARG_REF,        1 },
        { "idx",    WT_ARG_REF,        1 },
        { "out",    WT_ARG_TEMP,       0 },
        { "n",      WT_ARG_IMM,        0 },
        { "packed", WT_ARG_IMM_PACKED, 0 },
        { "td0",    WT_ARG_IMM,        0 },
        { "td1",    WT_ARG_IMM,        0 },
        { "td2",    WT_ARG_IMM,        0 },
        { "td3",    WT_ARG_IMM,        0 },
        { "td4",    WT_ARG_IMM,        0 },
        { "xd0",    WT_ARG_IMM,        0 },
        { "xd1",    WT_ARG_IMM,        0 },
        { "xd2",    WT_ARG_IMM,        0 },
        { "xd3",    WT_ARG_IMM,        0 },
        { "xd4",    WT_ARG_IMM,        0 },
    } },

    /* 21 OP_ARGMAX_F16: [x, out, n] */
    { 3, {
        { "x",   WT_ARG_REF,  1 },
        { "out", WT_ARG_TEMP, 0 },
        { "n",   WT_ARG_IMM,  0 },
    } },

    /* 22 OP_KV_APPEND_F16: [k, v, state, pos] */
    { 4, {
        { "k",     WT_ARG_REF,   1 },
        { "v",     WT_ARG_REF,   1 },
        { "state", WT_ARG_STATE, 1 },
        { "pos",   WT_ARG_IMM,   0 },
    } },

    /* 23 OP_KV_GATHER_F16: [state, kT_out, v_out, n_kv, n_heads, kv_heads, head_dim]
     * 注意: arg_is_slot() 对 idx1(kT_out, 输出) 返回 true —— 逐值转录, 不从 kind 推导 */
    { 7, {
        { "state",    WT_ARG_STATE, 1 },
        { "kT_out",   WT_ARG_TEMP,  1 },
        { "v_out",    WT_ARG_TEMP,  0 },
        { "n_kv",     WT_ARG_IMM,   0 },
        { "n_heads",  WT_ARG_IMM,   0 },
        { "kv_heads", WT_ARG_IMM,   0 },
        { "head_dim", WT_ARG_IMM,   0 },
    } },

    /* 24 OP_MATMUL_F16: [a, w, out, M, K, N, flags] */
    { 7, {
        { "a",     WT_ARG_REF,        1 },
        { "w",     WT_ARG_SLOT,       1 },
        { "out",   WT_ARG_TEMP,       0 },
        { "M",     WT_ARG_IMM,        0 },
        { "K",     WT_ARG_IMM,        0 },
        { "N",     WT_ARG_IMM,        0 },
        { "flags", WT_ARG_IMM_PACKED, 0 },
    } },

    /* 25 OP_RMSNORM2_F16: [x, w, bias, y, n] */
    { 5, {
        { "x",    WT_ARG_REF,  0 },
        { "w",    WT_ARG_SLOT, 1 },
        { "bias", WT_ARG_SLOT, 1 },
        { "y",    WT_ARG_TEMP, 0 },
        { "n",    WT_ARG_IMM,  0 },
    } },

    /* 26 OP_BROADCAST_F16: [b, y, n, b_elems, in_d0..3, out_d0..3] */
    { 12, {
        { "b",       WT_ARG_REF,  1 },
        { "y",       WT_ARG_TEMP, 0 },
        { "n",       WT_ARG_IMM,  0 },
        { "b_elems", WT_ARG_IMM,  0 },
        { "in_d0",   WT_ARG_IMM,  0 },
        { "in_d1",   WT_ARG_IMM,  0 },
        { "in_d2",   WT_ARG_IMM,  0 },
        { "in_d3",   WT_ARG_IMM,  0 },
        { "out_d0",  WT_ARG_IMM,  0 },
        { "out_d1",  WT_ARG_IMM,  0 },
        { "out_d2",  WT_ARG_IMM,  0 },
        { "out_d3",  WT_ARG_IMM,  0 },
    } },

    /* 27 OP_TRANSPOSE_GEN_F16: [x, y, rank, d0..3, perm] */
    { 8, {
        { "x",    WT_ARG_REF,  1 },
        { "y",    WT_ARG_TEMP, 0 },
        { "rank", WT_ARG_IMM,  0 },
        { "d0",   WT_ARG_IMM,  0 },
        { "d1",   WT_ARG_IMM,  0 },
        { "d2",   WT_ARG_IMM,  0 },
        { "d3",   WT_ARG_IMM,  0 },
        { "perm", WT_ARG_IMM,  0 },
    } },

    /* 28 OP_SCATTER_ND_F16: [data, idx, upd, out, n_out, rank, d0..4, K, n_idx, block] */
    { 14, {
        { "data",  WT_ARG_REF,  1 },
        { "idx",   WT_ARG_REF,  1 },
        { "upd",   WT_ARG_REF,  1 },
        { "out",   WT_ARG_TEMP, 0 },
        { "n_out", WT_ARG_IMM,  0 },
        { "rank",  WT_ARG_IMM,  0 },
        { "d0",    WT_ARG_IMM,  0 },
        { "d1",    WT_ARG_IMM,  0 },
        { "d2",    WT_ARG_IMM,  0 },
        { "d3",    WT_ARG_IMM,  0 },
        { "d4",    WT_ARG_IMM,  0 },
        { "K",     WT_ARG_IMM,  0 },
        { "n_idx", WT_ARG_IMM,  0 },
        { "block", WT_ARG_IMM,  0 },
    } },

    /* 29 OP_CONV1D_SSM_STEP_F16: [x, w, y, state, C, d_conv] */
    { 6, {
        { "x",      WT_ARG_REF,   1 },
        { "w",      WT_ARG_SLOT,  1 },
        { "y",      WT_ARG_TEMP,  0 },
        { "state",  WT_ARG_STATE, 0 },
        { "C",      WT_ARG_IMM,   0 },
        { "d_conv", WT_ARG_IMM,   0 },
    } },

    /* 30 OP_DELTA_CHUNK_F16: [k, v, q, beta, g, y, state, c, ntok, H, D, q_raw] */
    { 12, {
        { "k",     WT_ARG_REF,   1 },
        { "v",     WT_ARG_REF,   1 },
        { "q",     WT_ARG_REF,   1 },
        { "beta",  WT_ARG_REF,   1 },
        { "g",     WT_ARG_REF,   1 },
        { "y",     WT_ARG_TEMP,  0 },
        { "state", WT_ARG_STATE, 0 },
        { "c",     WT_ARG_IMM,   0 },
        { "ntok",  WT_ARG_IMM,   0 },
        { "H",     WT_ARG_IMM,   0 },
        { "D",     WT_ARG_IMM,   0 },
        { "q_raw", WT_ARG_IMM,   0 },
    } },

    /* 31 保留空洞 (arity = -1 哨兵) */
    { -1, { { NULL, WT_ARG_UNCLASSIFIED, 0 } } },

    /* 32 保留空洞 (arity = -1 哨兵) */
    { -1, { { NULL, WT_ARG_UNCLASSIFIED, 0 } } },

    /* 33 OP_DMA: [src, dst, bytes, src_off, dst_off, flags, fence_src, fence_dst] */
    { 8, {
        { "src",       WT_ARG_REF,  1 },
        { "dst",       WT_ARG_REF,  1 },
        { "bytes",     WT_ARG_IMM,  0 },
        { "src_off",   WT_ARG_IMM,  0 },
        { "dst_off",   WT_ARG_IMM,  0 },
        { "flags",     WT_ARG_IMM,  0 },
        { "fence_src", WT_ARG_IMM,  0 },
        { "fence_dst", WT_ARG_IMM,  0 },
    } },

    /* 34 OP_PAD_F16: [x, y, outer, in_inner, out_inner, pad_before_elems, pad_val_f16] */
    { 7, {
        { "x",                WT_ARG_REF,  1 },
        { "y",                WT_ARG_TEMP, 0 },
        { "outer",            WT_ARG_IMM,  0 },
        { "in_inner",         WT_ARG_IMM,  0 },
        { "out_inner",        WT_ARG_IMM,  0 },
        { "pad_before_elems", WT_ARG_IMM,  0 },
        { "pad_val_f16",      WT_ARG_IMM,  0 },
    } },

    /* 35 OP_MATMUL_F16_SPLIT: [a, w, out, M, K, N_full, n0, N_chunk, flags] */
    { 9, {
        { "a",       WT_ARG_REF,        1 },
        { "w",       WT_ARG_SLOT,       1 },
        { "out",     WT_ARG_TEMP,       0 },
        { "M",       WT_ARG_IMM,        0 },
        { "K",       WT_ARG_IMM,        0 },
        { "N_full",  WT_ARG_IMM,        0 },
        { "n0",      WT_ARG_IMM,        0 },
        { "N_chunk", WT_ARG_IMM,        0 },
        { "flags",   WT_ARG_IMM_PACKED, 0 },
    } },

    /* 36 OP_MATMUL_F16_BIAS: [a, w, out, M, K, N, flags, bias] */
    { 8, {
        { "a",     WT_ARG_REF,        1 },
        { "w",     WT_ARG_SLOT,       1 },
        { "out",   WT_ARG_TEMP,       0 },
        { "M",     WT_ARG_IMM,        0 },
        { "K",     WT_ARG_IMM,        0 },
        { "N",     WT_ARG_IMM,        0 },
        { "flags", WT_ARG_IMM_PACKED, 0 },
        { "bias",  WT_ARG_SLOT,       1 },
    } },

    /* 37 OP_GDN_BLOCK_F16: [xqkv, conv_w, beta, g, y] */
    { 5, {
        { "xqkv",   WT_ARG_REF,  0 },
        { "conv_w", WT_ARG_SLOT, 1 },
        { "beta",   WT_ARG_REF,  0 },
        { "g",      WT_ARG_REF,  0 },
        { "y",      WT_ARG_TEMP, 0 },
    } },
};
