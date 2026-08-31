/* bledger.c — U19 Buffer Ledger 数据流溯源审计 (见 include/bledger.h) */
#include "bledger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bl_init(struct bledger* b) {
    if (!b) return BL_ERR_PARAM;
    memset(b, 0, sizeof(*b));
    return BL_OK;
}

int bl_register(struct bledger* b, uint32_t slot, void* base, uint32_t bytes,
                uint32_t row_bytes, const char* name) {
    if (!b || !base || row_bytes == 0 || bytes == 0) return BL_ERR_PARAM;
    if (bytes % row_bytes != 0) return BL_ERR_PARAM;
    if (slot >= BL_MAX_BUFS) return BL_ERR_RANGE;
    if (b->buf[slot].rows) return BL_ERR_PARAM;          /* 重复注册 */
    struct bl_buf* f = &b->buf[slot];
    f->n_rows = bytes / row_bytes;
    f->rows = (struct bl_row*)calloc(f->n_rows, sizeof(struct bl_row));
    if (!f->rows) return BL_ERR_PARAM;
    f->base = (uint8_t*)base;
    f->bytes = bytes;
    f->row_bytes = row_bytes;
    if (name) {
        strncpy(f->name, name, sizeof(f->name) - 1);
        f->name[sizeof(f->name) - 1] = 0;
    } else f->name[0] = 0;
    f->n_writes = f->n_reads = f->n_breaks = 0;
    if (slot + 1 > b->n_bufs) b->n_bufs = slot + 1;
    return BL_OK;
}

int bl_write(struct bledger* b, uint32_t slot, uint16_t writer,
             uint32_t row0, uint32_t nrows, uint16_t qtag) {
    if (!b || writer == BL_ANY_WRITER) return BL_ERR_PARAM;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return BL_ERR_NOSUCH;
    struct bl_buf* f = &b->buf[slot];
    if (nrows == 0 || row0 >= f->n_rows || nrows > f->n_rows - row0)
        return BL_ERR_RANGE;
    b->seq++;
    for (uint32_t i = 0; i < nrows; i++) {
        struct bl_row* r = &f->rows[row0 + i];
        if (r->state == BL_ST_WRITTEN && r->writer != writer)
            b->n_double_write++;                     /* 3.6 未读异写覆写 */
        r->seq = b->seq;
        r->writer = writer;
        r->qtag = qtag;
        r->state = BL_ST_WRITTEN;
    }
    f->n_writes++;
    return BL_OK;
}

int bl_canary(struct bledger* b, uint32_t slot) {
    if (!b) return BL_ERR_PARAM;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return BL_ERR_NOSUCH;
    struct bl_buf* f = &b->buf[slot];
    memset(f->base, 0xAA, f->bytes);
    for (uint32_t i = 0; i < f->n_rows; i++)
        f->rows[i].state = BL_ST_CANARY;             /* seq/writer 保留供取证 */
    return BL_OK;
}

int bl_release(struct bledger* b, uint32_t slot) {
    if (!b) return BL_ERR_PARAM;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return BL_ERR_NOSUCH;
    struct bl_buf* f = &b->buf[slot];
    memset(f->base, 0xAA, f->bytes);                 /* §8.4 归还前 canary */
    for (uint32_t i = 0; i < f->n_rows; i++)
        f->rows[i].state = BL_ST_RELEASED;
    return BL_OK;
}

int bl_expect(struct bledger* b, uint32_t slot, uint32_t row,
              uint16_t writer_filter, uint16_t qtag) {
    if (!b) return BL_ERR_PARAM;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return BL_ERR_NOSUCH;
    struct bl_buf* f = &b->buf[slot];
    if (row >= f->n_rows) return BL_ERR_RANGE;
    f->rows[row].has_expect = 1;
    f->rows[row].exp_writer = writer_filter;
    f->rows[row].exp_qtag = qtag;
    return BL_OK;
}

int bl_verify(struct bledger* b, uint32_t slot, uint32_t row) {
    if (!b) return BL_ERR_PARAM;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return BL_ERR_NOSUCH;
    struct bl_buf* f = &b->buf[slot];
    if (row >= f->n_rows) return BL_ERR_RANGE;
    struct bl_row* r = &f->rows[row];
    int rc;
    switch (r->state) {
    case BL_ST_UNWRITTEN: rc = BL_ERR_NEVER;    break;   /* 3.1/3.2 断链 */
    case BL_ST_CANARY:    rc = BL_ERR_CANARY;   break;   /* E3 未初始化读 */
    case BL_ST_RELEASED:  rc = BL_ERR_RELEASED; break;   /* 3.4 归还后消费 */
    default:
        rc = BL_OK;
        if (r->has_expect && r->exp_writer != BL_ANY_WRITER
            && r->writer != r->exp_writer) rc = BL_ERR_WRITER;      /* H3 */
        else if (r->has_expect && r->exp_qtag != BL_NO_QTAG
            && r->qtag != r->exp_qtag) rc = BL_ERR_QTAG;            /* 3.5 */
        break;
    }
    if (rc != BL_OK) { f->n_breaks++; return rc; }
    r->state = BL_ST_READ;
    f->n_reads++;
    return BL_OK;
}

uint32_t bl_seq(const struct bledger* b) { return b ? b->seq : 0u; }

uint32_t bl_timeline(const struct bledger* b, uint32_t slot,
                     char* out, uint32_t cap) {
    if (!b || !out || cap == 0) return 0;
    if (slot >= b->n_bufs || !b->buf[slot].rows) return 0;
    const struct bl_buf* f = &b->buf[slot];
    char* p = out;
    uint32_t left = cap, n;
    n = (uint32_t)snprintf(p, left, "buf '%s' base=%p rows=%u rb=%u\n",
                           f->name, (void*)f->base, f->n_rows, f->row_bytes);
    p += n; if (n >= left) return cap; left -= n;
    n = (uint32_t)snprintf(p, left, "writes=%u reads=%u breaks=%u double=%u seq=%u\n",
                           f->n_writes, f->n_reads, f->n_breaks,
                           b->n_double_write, b->seq);
    p += n; if (n >= left) return cap; left -= n;
    for (uint32_t i = 0; i < f->n_rows; i++) {
        const struct bl_row* r = &f->rows[i];
        if (r->seq == 0 && r->state == BL_ST_UNWRITTEN) continue;
        n = (uint32_t)snprintf(p, left, "r%-3u w=%u seq=%u st=%u q=%u\n",
                               i, r->writer, r->seq, r->state, r->qtag);
        p += n; if (n >= left) return cap; left -= n;
    }
    return (uint32_t)(p - out);
}

void bl_close(struct bledger* b) {
    if (!b) return;
    for (uint32_t i = 0; i < BL_MAX_BUFS; i++)
        if (b->buf[i].rows) {
            free(b->buf[i].rows);
            b->buf[i].rows = 0;
        }
    memset(b, 0, sizeof(*b));
}
