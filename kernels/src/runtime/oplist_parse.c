/* oplist_parse.c — 平台无关纯 C 解析器 (host 单测 + 设备实跑同一份源码) */
#include <string.h>

#include "oplist_parse.h"

static uint32_t align_up(uint32_t x, uint32_t a) {
    return a * ((x + a - 1u) / a);
}

static int arity_of(uint16_t opcode) {
    switch (opcode) {
    case OP_NOP: return WT_ARITY_NOP;
    case OP_MATMUL_W4A16: return WT_ARITY_MATMUL;
    case OP_RMSNORM_F16: return WT_ARITY_RMSNORM;
    case OP_PIN: return WT_ARITY_PIN;
    case OP_SILU_F16: return WT_ARITY_SILU;
    case OP_IM2COL: return WT_ARITY_IM2COL;
    case OP_CONV2D_F16: return WT_ARITY_CONV2D_F16;
    case OP_ADD_F16: return WT_ARITY_ADD_F16;
    case OP_SPILL: return WT_ARITY_SPILL;
    case OP_FILL: return WT_ARITY_FILL;
    case OP_TRANSPOSE_F16: return WT_ARITY_TRANSPOSE_F16;
    default: return -1;
    }
}

/* 该 arg 下标是否是 slot 引用 */
static int arg_is_slot(uint16_t opcode, uint16_t idx) {
    switch (opcode) {
    case OP_MATMUL_W4A16: return idx == 0 || idx == 1;
    case OP_RMSNORM_F16: return idx == 1;
    case OP_PIN: return idx == 0;
    case OP_CONV2D_F16: return idx == 1 || idx == 2;
    case OP_SPILL: return idx == 1;
    case OP_FILL: return idx == 0;
    default: return 0;
    }
}

int wt_parse(const uint8_t* buf, size_t size, struct wt_blob* out) {
    if (!buf || !out || size < 16u) return WT_ERR_SHORT;
    if (buf[0] != 'W' || buf[1] != 'T' || buf[2] != 'O' || buf[3] != 'P')
        return WT_ERR_MAGIC;
    out->ver = wt_rd_u16(buf + 4);
    if (out->ver != WT_BLOB_VER) return WT_ERR_VER;
    if (wt_rd_u16(buf + 6) != WT_ENDIAN_CHK) return WT_ERR_ENDIAN;
    out->n_slots = wt_rd_u32(buf + 8);
    out->n_ops = wt_rd_u32(buf + 12);
    if (out->n_slots == 0 || out->n_slots > WT_MAX_SLOTS) return WT_ERR_NSLOTS;
    if (out->n_ops == 0 || out->n_ops > WT_MAX_OPS) return WT_ERR_NOPS;

    size_t need = 16u + (size_t)out->n_slots * WT_SLOT_SIZE;
    if (size < need) return WT_ERR_SLOTS_OVERRUN;
    for (uint32_t i = 0; i < out->n_slots; i++) {
        const uint8_t* s = buf + 16u + (size_t)i * WT_SLOT_SIZE;
        out->slots[i].len = wt_rd_u32(s);
        out->slots[i].count = wt_rd_u32(s + 4);
        out->slots[i].offset = wt_rd_u32(s + 8);
        out->slots[i].addr = wt_rd_u32(s + 12);
        if (out->slots[i].len == 0) return WT_ERR_WEIGHT_OVERRUN;
        if (out->slots[i].offset % WT_WEIGHT_ALIGN != 0) return WT_ERR_ALIGN;
        if ((uint64_t)out->slots[i].offset + out->slots[i].len > 0xFFFFFFFFull)
            return WT_ERR_WEIGHT_OVERRUN;
    }

    /* op 表 */
    size_t p = need;
    for (uint32_t i = 0; i < out->n_ops; i++) {
        if (size < p + 4u) return WT_ERR_OP_TRUNC;
        uint16_t opcode = wt_rd_u16(buf + p);
        uint16_t n_args = wt_rd_u16(buf + p + 2);
        int ar = arity_of(opcode);
        if (ar < 0) return WT_ERR_OPCODE;
        if (n_args != (uint16_t)ar) return WT_ERR_ARITY;
        if (size < p + 4u + (size_t)n_args * 4u) return WT_ERR_OP_TRUNC;
        out->ops[i].opcode = opcode;
        out->ops[i].n_args = n_args;
        for (uint16_t a = 0; a < n_args; a++)
            out->ops[i].args[a] = wt_rd_u32(buf + p + 4u + (size_t)a * 4u);
        p += 4u + (size_t)n_args * 4u;
    }

    out->weight_off = align_up((uint32_t)p, WT_WEIGHT_ALIGN);
    if ((size_t)out->weight_off > size) return WT_ERR_WEIGHT_OVERRUN;
    out->weight_base = buf + out->weight_off;
    out->weight_bytes = size - out->weight_off;

    for (uint32_t i = 0; i < out->n_slots; i++) {
        uint64_t end = (uint64_t)out->slots[i].offset + out->slots[i].len;
        if (end > out->weight_bytes) return WT_ERR_WEIGHT_OVERRUN;
    }
    for (uint32_t i = 0; i < out->n_ops; i++) {
        const struct wt_op* op = &out->ops[i];
        for (uint16_t a = 0; a < op->n_args; a++)
            if (arg_is_slot(op->opcode, a) && op->args[a] >= out->n_slots)
                return WT_ERR_BAD_REF;
    }
    return WT_OK;
}

const char* wt_err_str(int rc) {
    switch (rc) {
    case WT_OK: return "OK";
    case WT_ERR_SHORT: return "buffer shorter than header";
    case WT_ERR_MAGIC: return "bad magic";
    case WT_ERR_VER: return "unsupported version";
    case WT_ERR_ENDIAN: return "endian check failed";
    case WT_ERR_NSLOTS: return "n_slots out of range";
    case WT_ERR_NOPS: return "n_ops out of range";
    case WT_ERR_SLOTS_OVERRUN: return "slot table overruns buffer";
    case WT_ERR_WEIGHT_OVERRUN: return "slot offset+len overruns weight region";
    case WT_ERR_OP_TRUNC: return "op table truncated";
    case WT_ERR_OPCODE: return "unknown opcode";
    case WT_ERR_ARITY: return "n_args mismatch for opcode";
    case WT_ERR_BAD_REF: return "op arg references missing slot";
    case WT_ERR_ALIGN: return "slot offset not 128B aligned";
    default: return "unknown";
    }
}
