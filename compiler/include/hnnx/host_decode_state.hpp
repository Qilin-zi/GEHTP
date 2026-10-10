// host_decode_state.hpp — 004 T018: decode 图持久状态的 host 参考执行镜像
//
// 与设备侧 g_exec 状态表 (oplist_exec.c) 同口径: 42 项 =
//   0..5   KV    (count 自增计数器 + K/V 面 [KV_MAX][KV_ELEMS] f32)
//   6..23  conv  (滑窗 [(DC-1)*C] f32, win[0] 最老)
//   24..41 delta (S [H*D*D] f32)
// 生命期 = 进程级静态; 多步 host 验证用 GEHTP_HOST_STATE=<path> 萅盘文件
// (execute_host 首装末存, 存在即续跑, 删除文件 = 新序列)。
// 数值内核 = gdn_ref.c ref_conv_step/ref_delta_token 同式 (f32)。
#ifndef HNNX_HOST_DECODE_STATE_HPP
#define HNNX_HOST_DECODE_STATE_HPP

#include <cstdint>

namespace hnnx {

struct HostDecodeState {
    static constexpr int N_KV = 6, KV_ELEMS = 512, KV_MAX = 64;
    static constexpr int N_CONV = 18, CONV_C = 6144, CONV_DC = 4;
    static constexpr int N_DELTA = 18, GDN_H = 16, GDN_D = 128;

    uint32_t kv_count[N_KV] = {};
    float kv_k[N_KV][KV_MAX][KV_ELEMS] = {};
    float kv_v[N_KV][KV_MAX][KV_ELEMS] = {};
    float conv_win[N_CONV][(CONV_DC - 1) * CONV_C] = {};
    float delta_s[N_DELTA][GDN_H * GDN_D * GDN_D] = {};
};

void host_decode_state_reset();
/* 0 = 装载成功; 1 = 文件不存在(保持零态); -1 = 文件坏 */
int  host_decode_state_load(const char* path);
int  host_decode_state_save(const char* path);

/* op 语义 (与设备 exec_kv_append / exec_conv1d_ssm_step / exec_delta_chunk 对齐) */
bool host_decode_kv_append(int idx, const float* k, const float* v, int elems);
/* face: 0 = kT [nh][hd][max_pos], 1 = v [nh][max_pos][hd]; 尾部零填 */
void host_decode_kv_gather(int idx, int face, int n_heads, int kv_heads, int hd,
                           int max_pos, float* out);
/* y = silu(w·win + w[dc-1]·x); 窗推进 (ref_conv_step 同式) */
void host_decode_conv_step(int idx, int c, int dc, const float* w,
                           const float* x, float* y);
/* gated delta rule 单步 (ref_delta_token 同式; q_raw=1 无 1/√D) */
void host_decode_delta_step(int idx, int h, int d, int q_raw,
                            const float* k, const float* v, const float* q,
                            const float* beta, const float* g, float* y);

} // namespace hnnx

#endif
