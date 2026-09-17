// ============================================================================
// M36c 反汇编保真实现 —— 见 include/hnnx/vtcm/tile_shape_m36c.hpp 头注
// 证据基线: audit_verify/reports/M36c_tile_shape_subsystem_disasm.md (2026-08-28)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// 未逐指令确认处一律注明"遗留/记录级"，禁止臆测补齐。
// ============================================================================
#include "hnnx/vtcm/tile_shape_m36c.hpp"
#include <cstdio>

namespace hnnx {

std::string m36c_invalid_name_error(const std::string& name) {
    // fmt@0x55ba472: '%s:888::ERROR:invalid name in TILE_SHAPE or TILE_SIZE
    //                specification for %s'（file=tiling_registration.cc@0x55b9f70）
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "tiling_registration.cc:888::ERROR:invalid name in TILE_SHAPE or "
                  "TILE_SIZE specification for %s",
                  name.c_str());
    return buf;
}

// ---- 统一名字解析模板（六访问器同构）--------------------------------------
// "*"@0x57d37ee → 基准对象 [0x138f95f-0x138f970]
// 名字表线性 strcmp 扫 [0x138f976-0x138f9a6]；命中 i → 基准+0x30 表第 i 项
// OpRef 解引用 [0x138f9d3-0x138f9ed]；miss → 888 ERROR [0x138f9af-0x138f9cb]
// 记录级: 真码 miss 后以"表尾下标"继续取 OpRef（越界语义未证），此处返回 nullptr。
M36cOpDef* M36cTileShapeBase::resolve(const std::string& name) {
    if (name == "*") return base;                        // [0x138f970 je]
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name)                            // strcmp==0 [0x138f99d]
            return i < 16 ? oprefs[i] : nullptr;
    }
    log += m36c_invalid_name_error(name);                // [0x138f9cb qnndsp_log]
    log += "\n";
    return nullptr;                                      // 记录级（见表注）
}

// OPTION_INT(tag) [0x138fa00]：ctx 选项表(+0x54d0)查找 0x10f6590
//   miss → GetLogPriorityLevel>0 时 WARNING [0x138fa30-0x138fa5b]，返回缺省 0（记录级）
int64_t M36cTileShapeBase::option_int(const std::string& tag) {
    auto it = ctx->options.find(tag);
    if (it != ctx->options.end()) return it->second;
    log += "tiling_registration.cc:WARNING:no option " + tag + "\n"; // fmt@0x55ba4bd 半证
    return 0;
}

// OPTION_UINT(tag) [0x138fa80]：
//   "tcm_size"            → nn_os_vtcm_get_hardware_size() [0x138fab0]
//   "tcm_size_for_tiling" → GraphPrepare::get_vtcm_tile_size() [0x138fad3]
//   其余 → 同 OPTION_INT 查表 [0x138fada-0x138faf0]
uint32_t M36cTileShapeBase::option_uint(const std::string& tag) {
    if (tag == "tcm_size") return ctx->hw_vtcm_size;             // 串@0x469d1db
    if (tag == "tcm_size_for_tiling") return ctx->vtcm_tile_size; // 串@0x39ae79d
    return static_cast<uint32_t>(option_int(tag));
}

// gen_perf_Shape(w,x,y,z) [0x1391d40]（sret TinyVector）
M36cTinyVector M36cTileShapeBase::gen_perf_shape(uint32_t w, uint32_t x, uint32_t y,
                                                 uint32_t z) const {
    M36cTinyVector r;
    r.count = 4;                                          // [0x1391d5e/0x1391d76 movl $4]
    if (minimize_tiling()) return r;                      // ctx+0x5554>0 → 全零 [0x1391d51-0x1391d64]
    r.v[0] = w;                                           // edx→+4  [0x1391d6c]
    r.v[1] = x;                                           // ecx→+8  [0x1391d6f]
    r.v[2] = y;                                           // r8d→+0xc [0x1391d72]
    r.v[3] = z;                                           // r9d→+0x10 [0x1391d76]
    return r;
}

// crouton/flat/weights(char const*, TV) 共尾 [0x1391770-0x1391827 等]
//   解析 → conditionally_validate_single_quant（记录级直通）→ dtype=+0x4c [0x1391819]
//   → 布局构造（遗留）：留痕 (layout, dtype, TV)
M36cTileShapeBase::BuiltShape M36cTileShapeBase::build(const char* layout,
                                                       const std::string& name,
                                                       const M36cTinyVector& tv) {
    BuiltShape b;
    b.layout = layout;
    b.dims = tv;
    M36cOpDef* od = resolve(name);                       // [0x139177f-0x1391811]
    if (od == nullptr) return b;                         // miss（888 已记）
    b.resolved = true;
    b.dtype = od->dtype;                                 // [0x1391819 movl 0x4c]
    return b;
}

// GraphOptInfo::declare_tiling_rule @0x138f8b0（160B）
int m36c_declare_tiling_rule(M36cTileShapeBase& tsb, uint32_t id, const std::string& name,
                             M36cOpDef& holder, std::vector<M36cTilingRule>& registry) {
    // 注册门: holder+0xd 的 0x48 位 ≠0 → 不注册直接返回 [0x138f8dc-0x138f8e0]
    if (holder.flags_0xd & 0x48) return 0;
    // 形状函数指针 = *([[holder+0x18]]+8)（std::function 目标）[0x138f8c2-0x138f8ca]
    //   记录级: 用 holder 地址充当指针身份
    M36cTilingRule rec;
    rec.shape_fn = &holder;                              // 留痕替身（真实取链在 tsb.ctx 侧，遗留）
    rec.id = id;                                         // +0x08 [0x138f8f5]
    rec.holder = &holder;                                // +0x10 [0x138f8fd]
    rec.name = name;                                     // +0x18 [0x138f905]
    registry.push_back(rec);                             // 全局向量 0x6248428 [0x138f8f2-0x138f914]
    (void)tsb;
    return 0;                                            // [0x138f944 xorl eax]
}

} // namespace hnnx
