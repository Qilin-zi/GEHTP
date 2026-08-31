// test_conv_add_host: conv2d+add 契约模型的 op 注册 + host 参考执行对拍(阶段 3)
//
// 图结构 == test_models/conv_add 的 net.json 实测结构(NCHW I/O + 内部 NHWC):
//   Input(1) [1,32,32,32] NCHW
//     -> Transpose(3) perm[0,2,3,1] -> X_nhwc [1,32,32,32]
//     -> Conv2d(6) inputs[X_nhwc, W(4), B(5)], W=[3,3,32,32] (Kh,Kw,Cin,Cout)
//     -> Eltwise_Binary(7) inputs[Y(6), X_nhwc(3)]  (残差)
//     -> Transpose(9) perm[0,3,1,2] -> Z NCHW
//     -> Output(10)
// 验收:
//   1. prepare 后 Conv2d/Eltwise_Binary/Transpose 均存活(注册生效)
//   2. execute_host 输出 vs 测试内 NCHW 全图参考 (1e-5 相对)
//   3. execute_host 输出 vs 独立金标 Z_ort.f32.raw(onnxruntime NCHW 输出, 1e-4 相对)
// 数据来自 test_models/conv_add/{X,W,B}.f32.raw + Z_ort.f32.raw (gen_all.sh 生成);
// 缺失则 SKIP。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool load_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz > 0 ? (size_t)sz : 0);
    if (sz > 0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static OutputDef make_od4(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    OutputDef od{};
    od.rank = 4;
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    return od;
}

// 参考: NCHW -> NHWC 转置 (perm [0,2,3,1])
static void nchw_to_nhwc(const float* x, float* y, size_t C = 32) {
    for (size_t n = 0; n < 1; n++)
        for (size_t c = 0; c < C; c++)
            for (size_t h = 0; h < C; h++)
                for (size_t w = 0; w < C; w++)
                    y[(h * C + w) * C + c] = x[((n * C + c) * C + h) * C + w];
}
static void nhwc_to_nchw(const float* x, float* y, size_t C = 32) {
    for (size_t n = 0; n < 1; n++)
        for (size_t c = 0; c < C; c++)
            for (size_t h = 0; h < C; h++)
                for (size_t w = 0; w < C; w++)
                    y[((n * C + c) * C + h) * C + w] = x[(h * C + w) * C + c];
}

// NHWC 直算参考: X [1,H,W,C], W [Kh,Kw,Cin,Cout], B [Cout] (3x3 s1 same pad)
static void conv_ref(const float* x, const float* w, const float* b,
                     float* y, size_t C = 32) {
    const size_t Kh = 3, Kw = 3, pad = 1;
    for (size_t h = 0; h < C; h++)
        for (size_t wd = 0; wd < C; wd++)
            for (size_t co = 0; co < C; co++) {
                float acc = b[co];
                for (size_t kh = 0; kh < Kh; kh++)
                    for (size_t kw = 0; kw < Kw; kw++) {
                        long hi = (long)(h + kh) - pad;
                        long wi = (long)(wd + kw) - pad;
                        if (hi < 0 || hi >= (long)C || wi < 0 || wi >= (long)C) continue;
                        for (size_t ci = 0; ci < C; ci++) {
                            acc += x[(hi * C + wi) * C + ci]
                                 * w[((kh * Kw + kw) * C + ci) * C + co];
                        }
                    }
                y[(h * C + wd) * C + co] = acc;
            }
}

int main() {
    std::printf("=== conv2d+add host reference test ===\n\n");

    const std::string dir = std::string(HNNX_GEHTP_TEST_DATA_DIR) + "/conv_add/";
    std::vector<uint8_t> xb, wb, bb, orb;
    bool have = load_file(dir + "X.f32.raw", xb)
             && load_file(dir + "W.f32.raw", wb)
             && load_file(dir + "B.f32.raw", bb)
             && load_file(dir + "Z_ort.f32.raw", orb);
    if (!have) {
        std::printf("SKIP: conv_add 数据缺失 (先跑 test_models/conv_add/gen_all.sh)\n");
        return 0;
    }
    const size_t N = 32 * 32 * 32;  // 单张 32x32x32
    const float* X = reinterpret_cast<const float*>(xb.data());
    const float* B = reinterpret_cast<const float*>(bb.data());
    const float* ZORT = reinterpret_cast<const float*>(orb.data());

    // W.f32.raw 为 OIHW(ONNX 语义); QNN IR 布局 = [kh,kw,ci,co](params.bin 实测,
    // 阶段1字节级核对)。转置: Wq[((kh*3+kw)*32+ci)*32+co] = W[((co*32+ci)*3+kh)*3+kw]
    const float* Wraw = reinterpret_cast<const float*>(wb.data());
    std::vector<float> Wq(32 * 32 * 3 * 3);
    for (size_t co = 0; co < 32; co++)
        for (size_t ci = 0; ci < 32; ci++)
            for (size_t kh = 0; kh < 3; kh++)
                for (size_t kw = 0; kw < 3; kw++)
                    Wq[((kh * 3 + kw) * 32 + ci) * 32 + co] =
                        Wraw[((co * 32 + ci) * 3 + kh) * 3 + kw];
    const float* W = Wq.data();

    register_all_ops();

    GraphPrepare gp;
    auto od_nchw = make_od4(1, 32, 32, 32);
    gp.append_node("Input", 1, nullptr, 0, &od_nchw, 1, nullptr);

    // perm consts (int32)
    const int32_t perm_in[4] = {0, 2, 3, 1};   // NCHW -> NHWC
    const int32_t perm_out[4] = {0, 3, 1, 2};  // NHWC -> NCHW
    OutputDef perm_od{};
    perm_od.rank = 1;
    perm_od.dims[0] = 4;
    perm_od.element_size = 4;
    perm_od.dtype = static_cast<uint32_t>(DType::Int32);
    gp.append_const_node(2, perm_od,
                         reinterpret_cast<const uint8_t*>(perm_in), sizeof(perm_in));

    InputDef t1in[2] = {{1, 0}, {2, 0}};
    gp.append_node("Transpose", 3, t1in, 2, &od_nchw, 1, nullptr);

    auto w_od = make_od4(3, 3, 32, 32);
    gp.append_const_node(4, w_od,
                         reinterpret_cast<const uint8_t*>(Wq.data()),
                         Wq.size() * sizeof(float));

    OutputDef b_od{};
    b_od.rank = 1;
    b_od.dims[0] = 32;
    b_od.element_size = 4;
    b_od.dtype = static_cast<uint32_t>(DType::Float32);
    gp.append_const_node(5, b_od, bb.data(), bb.size());

    InputDef cin[3] = {{3, 0}, {4, 0}, {5, 0}};
    gp.append_node("Conv2d", 6, cin, 3, &od_nchw, 1, nullptr);

    InputDef ain[2] = {{6, 0}, {3, 0}};  // Y + X_nhwc 残差
    gp.append_node("Eltwise_Binary", 7, ain, 2, &od_nchw, 1, nullptr);

    gp.append_const_node(8, perm_od,
                         reinterpret_cast<const uint8_t*>(perm_out), sizeof(perm_out));
    InputDef t2in[2] = {{7, 0}, {8, 0}};
    gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);

    InputDef oin[1] = {{9, 0}};
    gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);

    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    GraphStatus st = gp.prepare(env);
    CHECK(st == GraphStatus::Success, "prepare");

    // 注册生效断言
    CHECK(gp.get_op_at(6) != nullptr && !gp.get_op_at(6)->is_dead(),
          "Conv2d survives prepare (registration active)");
    CHECK(gp.get_op_at(7) != nullptr && !gp.get_op_at(7)->is_dead(),
          "Eltwise_Binary survives prepare (registration active)");
    CHECK(gp.get_op_at(3) != nullptr && !gp.get_op_at(3)->is_dead(),
          "Transpose(in) survives prepare");

    // 执行
    std::vector<float> xvec(X, X + N);
    auto r = gp.execute_host(xvec);
    CHECK(r.ok && r.output.size() == N, "execute_host");

    // 测试内参考: NCHW -> NHWC -> conv -> +残差 -> NCHW
    std::vector<float> Xn(N), Y(N), Zn(N), Zref(N);
    nchw_to_nhwc(X, Xn.data());
    conv_ref(Xn.data(), W, B, Y.data());
    for (size_t i = 0; i < N; i++) Zn[i] = Y[i] + Xn[i];
    nhwc_to_nchw(Zn.data(), Zref.data());

    double maxrel = 0.0;
    for (size_t i = 0; i < N; i++) {
        double d = std::fabs((double)r.output[i] - Zref[i]);
        double scale = std::fabs((double)Zref[i]);
        if (scale < 1.0) scale = 1.0;
        double rr = d / scale;
        if (rr > maxrel) maxrel = rr;
    }
    std::printf("  host-vs-inline-ref maxrel=%.3e\n", maxrel);
    CHECK(maxrel < 1e-5, "execute_host == inline NCHW graph reference (1e-5 rel)");

    double maxrel_ort = 0.0;
    for (size_t i = 0; i < N; i++) {
        double d = std::fabs((double)r.output[i] - ZORT[i]);
        double scale = std::fabs((double)ZORT[i]);
        if (scale < 1.0) scale = 1.0;
        double rr = d / scale;
        if (rr > maxrel_ort) maxrel_ort = rr;
    }
    std::printf("  host-vs-ORT maxrel=%.3e\n", maxrel_ort);
    CHECK(maxrel_ort < 1e-4, "execute_host == onnxruntime golden (1e-4 rel)");

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
