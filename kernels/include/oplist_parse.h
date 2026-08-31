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
 *   OP_NOP          = 0  : []
 *   OP_MATMUL_W4A16 = 1  : [act_slot, w_slot, out_temp, M, K, N]
 *   OP_RMSNORM_F16  = 2  : [x_temp, w_slot, y_temp, n]
 *   OP_PIN          = 3  : [slot]
 *   OP_SILU_F16     = 4  : [x_temp, y_temp, n_elem]   (V2.3 U16)
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
#define WT_MAX_SLOTS 64u
#define WT_MAX_OPS 256u
#define WT_MAX_ARGS 8u

enum {
    OP_NOP = 0,
    OP_MATMUL_W4A16 = 1,
    OP_RMSNORM_F16 = 2,
    OP_PIN = 3,
    OP_SILU_F16 = 4,
};

/* 每个 opcode 的参数个数 (下标 = opcode) */
#define WT_ARITY_NOP 0
#define WT_ARITY_MATMUL 6
#define WT_ARITY_RMSNORM 4
#define WT_ARITY_PIN 1
#define WT_ARITY_SILU 3

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
