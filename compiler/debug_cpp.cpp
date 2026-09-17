#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
int main() {
    using namespace hnnx;
    std::ifstream f("C:/Users/RQILIN/Documents/Default Project/mlib_build/jni/simple_linear.cpp");
    std::stringstream buf; buf << f.rdbuf();
    std::string src = buf.str();
    auto funcs = QnnIRLoader::extract_functions(src, "addTensor_");
    std::cout << "addTensor_ funcs: " << funcs.size() << "\n";
    for (size_t i = 0; i < funcs.size() && i < 3; ++i) {
        std::cout << "  [" << i << "] first 120: " << funcs[i].substr(0, 120) << "\n";
    }
    auto nfuncs = QnnIRLoader::extract_functions(src, "addNode_");
    std::cout << "addNode_ funcs: " << nfuncs.size() << "\n";
    for (size_t i = 0; i < nfuncs.size() && i < 3; ++i) {
        std::cout << "  [" << i << "] first 80: " << nfuncs[i].substr(0, 80) << "\n";
    }
    return 0;
}
