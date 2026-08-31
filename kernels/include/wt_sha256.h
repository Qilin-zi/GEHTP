#ifndef WT_SHA256_H
#define WT_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FIPS 180-4 SHA-256, 标量实现 (host/设备同源; 与 python hashlib 对拍判据) */
typedef struct {
    uint32_t h[8];
    uint64_t len;
    uint8_t block[64];
    size_t fill;
} wt_sha256_ctx;

void wt_sha256_init(wt_sha256_ctx* c);
void wt_sha256_update(wt_sha256_ctx* c, const void* data, size_t n);
void wt_sha256_final(wt_sha256_ctx* c, uint8_t out[32]);

/* 一次性便捷接口; hex 写入 out (65 字节含 NUL), 返回 out */
const char* wt_sha256_hex(const void* data, size_t n, char out[65]);

#ifdef __cplusplus
}
#endif

#endif /* WT_SHA256_H */
