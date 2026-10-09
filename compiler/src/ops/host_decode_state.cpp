// host_decode_state.cpp — 004 T018: decode 持久状态 host 镜像 + 数值内核
#include "hnnx/host_decode_state.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace hnnx {

static HostDecodeState g_state;

void host_decode_state_reset() { g_state = HostDecodeState{}; }

int host_decode_state_load(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return 1;
    char magic[4];
    uint32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "HDS1", 4)
              && std::fread(&n, 4, 1, f) == 1
              && n == (uint32_t)(HostDecodeState::N_KV + HostDecodeState::N_CONV
                                 + HostDecodeState::N_DELTA);
    if (ok) ok = std::fread(g_state.kv_count, 4, HostDecodeState::N_KV, f)
                     == HostDecodeState::N_KV;
    if (ok) ok = std::fread(g_state.kv_k, 4,
                            sizeof(g_state.kv_k) / 4, f) * 4 == sizeof(g_state.kv_k);
    if (ok) ok = std::fread(g_state.kv_v, 4,
                            sizeof(g_state.kv_v) / 4, f) * 4 == sizeof(g_state.kv_v);
    if (ok) ok = std::fread(g_state.conv_win, 4,
                            sizeof(g_state.conv_win) / 4, f) * 4 == sizeof(g_state.conv_win);
    if (ok) ok = std::fread(g_state.delta_s, 4,
                            sizeof(g_state.delta_s) / 4, f) * 4 == sizeof(g_state.delta_s);
    std::fclose(f);
    if (!ok) {
        host_decode_state_reset();
        return -1;
    }
    return 0;
}

int host_decode_state_save(const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return -1;
    uint32_t n = HostDecodeState::N_KV + HostDecodeState::N_CONV + HostDecodeState::N_DELTA;
    bool ok = std::fwrite("HDS1", 1, 4, f) == 4 && std::fwrite(&n, 4, 1, f) == 1
              && std::fwrite(g_state.kv_count, 4, HostDecodeState::N_KV, f)
                     == (size_t)HostDecodeState::N_KV
              && std::fwrite(g_state.kv_k, 4, sizeof(g_state.kv_k) / 4, f) * 4
                     == sizeof(g_state.kv_k)
              && std::fwrite(g_state.kv_v, 4, sizeof(g_state.kv_v) / 4, f) * 4
                     == sizeof(g_state.kv_v)
              && std::fwrite(g_state.conv_win, 4, sizeof(g_state.conv_win) / 4, f) * 4
                     == sizeof(g_state.conv_win)
              && std::fwrite(g_state.delta_s, 4, sizeof(g_state.delta_s) / 4, f) * 4
                     == sizeof(g_state.delta_s);
    std::fclose(f);
    return ok ? 0 : -1;
}

bool host_decode_kv_append(int idx, const float* k, const float* v, int elems) {
    if (idx < 0 || idx >= HostDecodeState::N_KV || elems != HostDecodeState::KV_ELEMS)
        return false;
    uint32_t c = g_state.kv_count[idx];
    if (c >= HostDecodeState::KV_MAX) return false;
    std::memcpy(g_state.kv_k[idx][c], k, (size_t)elems * 4);
    std::memcpy(g_state.kv_v[idx][c], v, (size_t)elems * 4);
    g_state.kv_count[idx] = c + 1;
    return true;
}

void host_decode_kv_gather(int idx, int face, int n_heads, int kv_heads, int hd,
                           int max_pos, float* out) {
    /* GQA repeat_interleave: head h 用 kv 面 h/rep; 输出定宽 max_pos:
     * kT=[nh][hd][max_pos], v=[nh][max_pos][hd]; 尾部 [n_kv, max_pos) 零填
     * (mask 面 -inf 湮灭尾位, 但 NaN 位型会穿 +(-inf), 必须清零) */
    const size_t total = (size_t)n_heads * hd * max_pos;
    std::memset(out, 0, total * 4);
    if (idx < 0 || idx >= HostDecodeState::N_KV) return;
    uint32_t nkv = g_state.kv_count[idx];
    if (nkv > (uint32_t)max_pos) nkv = (uint32_t)max_pos;
    uint32_t rep = n_heads / kv_heads;
    for (uint32_t p = 0; p < nkv; p++) {
        const float* kf = g_state.kv_k[idx][p];
        const float* vf = g_state.kv_v[idx][p];
        for (int h = 0; h < n_heads; h++) {
            const float* ksrc = kf + (h / rep) * hd;
            const float* vsrc = vf + (h / rep) * hd;
            if (face == 0) {
                float* kd = out + (size_t)h * hd * max_pos;
                for (int d = 0; d < hd; d++) kd[(size_t)d * max_pos + p] = ksrc[d];
            } else {
                float* vd = out + (size_t)h * max_pos * hd;
                std::memcpy(vd + (size_t)p * hd, vsrc, (size_t)hd * 4);
            }
        }
    }
}

static inline float host_silu(float x) { return x / (1.0f + std::exp(-x)); }

void host_decode_conv_step(int idx, int c, int dc, const float* w,
                           const float* x, float* y) {
    float* win = g_state.conv_win[idx];
    for (int ch = 0; ch < c; ch++) {
        float s = 0.0f;
        for (int i = 0; i < dc - 1; i++) s += w[(size_t)i * c + ch] * win[(size_t)i * c + ch];
        s += w[(size_t)(dc - 1) * c + ch] * x[ch];
        y[ch] = host_silu(s);
    }
    int w_n = dc - 1;
    std::memmove(win, win + c, (size_t)(w_n - 1) * c * 4);
    std::memcpy(win + (size_t)(w_n - 1) * c, x, (size_t)c * 4);
}

void host_decode_delta_step(int idx, int h, int d, int q_raw,
                            const float* k, const float* v, const float* q,
                            const float* beta, const float* g, float* y) {
    float scale = q_raw ? 1.0f : 1.0f / std::sqrt((float)d);
    for (int hh = 0; hh < h; hh++) {
        float* S = g_state.delta_s[idx] + (size_t)hh * d * d;
        const float* kh = k + (size_t)hh * d, *vh = v + (size_t)hh * d, *qh = q + (size_t)hh * d;
        float a = std::exp(g[hh]);
        /* S *= a; yk = S k */
        for (int j = 0; j < d; j++) {
            float* row = S + (size_t)j * d, s = 0.0f;
            for (int n = 0; n < d; n++) { row[n] *= a; s += row[n] * kh[n]; }
            /* yk[j] 暂存进 y (第二遍覆写) */
            y[(size_t)hh * d + j] = s;
        }
        /* S += β(v−yk)⊗k; y = scale·S q */
        for (int j = 0; j < d; j++) {
            float dj = beta[hh] * (vh[j] - y[(size_t)hh * d + j]);
            float* row = S + (size_t)j * d, s = 0.0f;
            for (int n = 0; n < d; n++) { row[n] += dj * kh[n]; s += row[n] * qh[n]; }
            y[(size_t)hh * d + j] = s * scale;
        }
    }
}

} // namespace hnnx
