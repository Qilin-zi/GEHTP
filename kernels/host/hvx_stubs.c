/* hvx_stubs.c — host 构建的 HVX 内核空桩 (门控默认关, host 验证不触达) */
#include <stdint.h>
void hvhx_v2_add_f16(uint16_t* y, const uint16_t* a, const uint16_t* b, uint32_t n) { (void)y; (void)a; (void)b; (void)n; }
void gehtp_hvx_mul_f16(uint16_t* y, const uint16_t* a, const uint16_t* b, uint32_t n) { (void)y; (void)a; (void)b; (void)n; }
int  gehtp_hvx_unary_f16(uint16_t* y, const uint16_t* x, uint32_t n, uint32_t subtype, uint16_t* sa, uint16_t* sb) { (void)y; (void)x; (void)n; (void)subtype; (void)sa; (void)sb; return -1; }
void gehtp_hvx_silu_f16(uint16_t* y, const uint16_t* x, uint32_t n, uint16_t* sa) { (void)y; (void)x; (void)n; (void)sa; }
void gehtp_hvx_softmax_f16(uint16_t* y, const uint16_t* x, uint32_t rows, uint32_t n, uint16_t* sa) { (void)y; (void)x; (void)rows; (void)n; (void)sa; }
void gehtp_hvx_rmsnorm_mul_f16(uint16_t* y, const uint16_t* x, const uint16_t* w, uint32_t rows, uint32_t n, float eps, uint16_t* sa) { (void)y; (void)x; (void)w; (void)rows; (void)n; (void)eps; (void)sa; }
