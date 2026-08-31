#pragma once
// ============================================================================
// M36c 反汇编保真实现：tile 子系统（tiler.h DSL / tiling::TileShapeBase）
// 证据: audit_verify/reports/M36c_tile_shape_subsystem_disasm.md
//       audit_verify/asm/f3/M36c_tileshape_accessors.asm / _builders.asm
// 规则: 每个逻辑段标注 [0x地址]；未在指令级确认的部分显式标注"遗留/记录级"。
//
// 一句话语义: tile 形状用"按名取参"DSL 描述——DEF_TILE_PROPERTIES 规则给张量
// 起角色名，TileShapeBase 以 ("*" 通配 | 名字表线性扫) 解析到 OpDef，再用
// crouton/flat/weights 构造 ≤8 维形状；数值选项走 OPTION_INT/UINT 键查表，
// "tcm_size"/"tcm_size_for_tiling" 两个键有硬件事务特例。
// ============================================================================
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hnnx {

// TinyVector<uint,8>：+0 计数、+4 起元素（gen_perf_Shape 写法证实 [0x1391d65-0x1391d7a]）
struct M36cTinyVector {
    uint32_t count = 0;             // +0
    uint32_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // +4..+0x20
    bool operator==(const M36cTinyVector& o) const {
        if (count != o.count) return false;
        for (uint32_t i = 0; i < count && i < 8; ++i)
            if (v[i] != o.v[i]) return false;
        return true;
    }
};

// OpDef 记录级替身（TileShapeBase 触及的字段）
struct M36cOpDef {
    uint8_t flags_0xd = 0;          // +0xd（注册门 0x48 位 [0x138f8dc]）
    std::vector<uint32_t> shape;    // +0x48 起形状（SAME_SHAPE 比较 [0x1390449]）
    uint32_t dtype = 0;             // +0x4c（crouton 取作 DType [0x1391819]）
    // +0x18 上下文指针 / +0x30 OpRef 表：模型里由 M36cCtx 与 oprefs 承担
};

// 基准对象的 +0x18 上下文（记录级：真码是 AnyGraphContext，此处只放被读字段）
struct M36cCtx {
    std::map<std::string, int64_t> options; // ctx+0x54d0 的选项表 [0x138fa23/0x138fada]
    int flag_5554 = 0;              // ctx+0x5554（minimize_tiling [0x1391d27]）
    uint32_t hw_vtcm_size = 0;      // nn_os_vtcm_get_hardware_size 的模型替身
    uint32_t vtcm_tile_size = 0;    // GraphPrepare::get_vtcm_tile_size 的模型替身
};

// tiling::TileShapeBase 记录级替身
struct M36cTileShapeBase {
    M36cOpDef* base = nullptr;      // +0x00 基准对象 [0x138f9d6]
    std::vector<std::string> names; // +0x08/+0x10 名字表（8B/项 strcmp 扫 [0x138f990]）
    M36cOpDef* oprefs[16] = {};     // 基准+0x30 的 OpRef 表（按名下标解引用 [0x138f9d9-0x138f9e1]）
    M36cCtx* ctx = nullptr;         // 基准+0x18 上下文 [0x138f9dd]
    std::string log;                // qnndsp_log 留痕（真码走 GetLogPriorityLevel 门）

    // 名字→OpDef（统一解析模板；miss 时 888 ERROR 且按"表尾"续走——真码语义 [0x138f9af-0x138f9d0]）
    M36cOpDef* resolve(const std::string& name);
    // get(char const*) [0x138f950]
    M36cOpDef* get(const std::string& name) { return resolve(name); }
    // OPTION_INT(tag) [0x138fa00]：查 ctx 选项表；查不到 WARNING，返回 0（记录级缺省）
    int64_t option_int(const std::string& tag);
    // OPTION_UINT(tag) [0x138fa80]：tcm_size / tcm_size_for_tiling 特例 + 查表
    uint32_t option_uint(const std::string& tag);
    // minimize_tiling() [0x1391d20]
    bool minimize_tiling() const { return ctx && ctx->flag_5554 > 0; }
    // gen_perf_Shape(w,x,y,z) [0x1391d40]：minimize → {4,0,0,0,0}；否则 {4,w,x,y,z}
    M36cTinyVector gen_perf_shape(uint32_t w, uint32_t x, uint32_t y, uint32_t z) const;
    // crouton/flat/weights(name, TV) 共尾：解析→(记录级)quant 校验→dtype=+0x4c→布局构造
    // [0x1391770-0x1391827]；布局构造体本身是遗留，返回 (名字, dtype, TV) 留痕
    struct BuiltShape {
        std::string layout;         // "crouton"/"flat"/"weights"
        uint32_t dtype;             // OpDef+0x4c
        M36cTinyVector dims;
        bool resolved = false;      // false = 名字 miss（888 已记 log）
    };
    BuiltShape build(const char* layout, const std::string& name, const M36cTinyVector& tv);
};

// declare_tiling_rule @0x138f8b0 的注册表记录（0x20B/项 [0x138f8e2-0x138f914]）
struct M36cTilingRule {
    void* shape_fn = nullptr;       // +0x00：*([[holder+0x18]]+8) [0x138f8c2-0x138f8ca]
    uint32_t id = 0;                // +0x08：参数1(uint)
    M36cOpDef* holder = nullptr;    // +0x10：参数3
    std::string name;               // +0x18：参数2
};

// GraphOptInfo::declare_tiling_rule 记录级替身（返回 0 [0x138f944]）
// gate: holder->flags_0xd & 0x48 → 不注册 [0x138f8dc]
int m36c_declare_tiling_rule(M36cTileShapeBase& tsb, uint32_t id, const std::string& name,
                             M36cOpDef& holder, std::vector<M36cTilingRule>& registry);

// 888/206 号错误文本（fmt@0x55ba472 / 0x55b9f70 区，tiling_registration.cc）
std::string m36c_invalid_name_error(const std::string& name);

} // namespace hnnx
