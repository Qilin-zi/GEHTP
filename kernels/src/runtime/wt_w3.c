/* wt_w3.c — W3 解析报告 (host inspect 与设备 wt_main 共用, 输出行逐字节一致) */
#include <stdio.h>
#include <string.h>

#include "oplist_parse.h"
#include "wt_sha256.h"

void wt_w3_report(const char* blob_name, const uint8_t* buf, size_t size,
                  const struct wt_blob* w,
                  void (*emit)(const char* line, void* ud), void* ud) {
    char line[1024], sha[65], wsha[65];
    wt_sha256_hex(buf, size, sha);
    wt_sha256_hex(w->weight_base, w->weight_bytes, wsha);
    snprintf(line, sizeof(line),
             "{\"t\":\"W3\",\"mode\":\"parse\",\"blob\":\"%s\",\"ver\":%u,\"endian_chk\":%u,"
             "\"n_slots\":%u,\"n_ops\":%u,\"blob_bytes\":%zu,\"blob_sha256\":\"%s\","
             "\"weight_off\":%u,\"weight_bytes\":%zu,\"weight_sha256\":\"%s\"}",
             blob_name, w->ver, WT_ENDIAN_CHK, w->n_slots, w->n_ops, size, sha,
             w->weight_off, w->weight_bytes, wsha);
    emit(line, ud);
    for (uint32_t i = 0; i < w->n_slots; i++) {
        wt_sha256_hex(w->weight_base + w->slots[i].offset, w->slots[i].len, sha);
        snprintf(line, sizeof(line),
                 "{\"t\":\"W3\",\"slot\":%u,\"len\":%u,\"count\":%u,\"offset\":%u,\"addr\":%u,"
                 "\"sha256\":\"%s\"}",
                 i, w->slots[i].len, w->slots[i].count, w->slots[i].offset,
                 w->slots[i].addr, sha);
        emit(line, ud);
    }
    for (uint32_t i = 0; i < w->n_ops; i++) {
        char args[200] = "";   /* n_args=0 时必须空串, 否则 %s 读未初始化栈 */
        int off = 0;
        for (uint16_t a = 0; a < w->ops[i].n_args; a++)
            off += snprintf(args + off, sizeof(args) - off, "%s%u",
                            a ? "," : "", w->ops[i].args[a]);
        snprintf(line, sizeof(line),
                 "{\"t\":\"W3\",\"op\":%u,\"opcode\":%u,\"n_args\":%u,\"args\":[%s]}",
                 i, w->ops[i].opcode, w->ops[i].n_args, args);
        emit(line, ud);
    }
    emit("{\"t\":\"W3\",\"verdict\":\"PASS\"}", ud);
}
