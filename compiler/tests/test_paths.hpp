#pragma once
// 测试数据路径便携化: Windows 原开发机 / Linux(GEHTP 迁移后)
// CMake 定义: HNNX_TEST_DATA_DIR=compiler/test_models, HNNX_REF_DIR=compiler/reference
#ifdef _WIN32
#define TP_T2_DIR    "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\test_models\\transformer2"
#define TP_REF_DIR   "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\reference\\"
#define TP_SL_CTXBIN TP_REF_DIR "simple_linear_context.bin"
#else
#define TP_T2_DIR    HNNX_TEST_DATA_DIR "/transformer2"
#define TP_REF_DIR   HNNX_REF_DIR "/"
#define TP_SL_CTXBIN HNNX_REF_DIR "/simple_linear_context.bin"
#endif

#define TP_T2_BIN     TP_T2_DIR "/transformer2.bin"
#define TP_T2_CPP     TP_T2_DIR "/transformer2.cpp"
#define TP_T2_NETJSON TP_T2_DIR "/transformer2_net.json"
#define TP_T2_BEFORE  TP_T2_DIR "/transformer2_before_graph.json"
#define TP_T2_AFTER   TP_T2_DIR "/transformer2_after_graph.json"
