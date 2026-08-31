// ============================================================================
// 张量配置层 — tensor_definitions.h 精确移植 (Ldefs / Tdefs), M34-C3
//
// 证据:
//   · LAYOUTDEF/TENSORDEF 展开结构与参数序 = SDK tensor_definitions.h:44-129 原文;
//     .so 侧 19 个 Ldefs × ~60 Tdefs 实例化 (M20 §2 表; ChunkSizes 反汇编直证)。
//   · Interface_t 选择: dtype_traits<DTYPE>::is_quant → ScaleOffsetInterface,
//     否则 PlainInterface (TENSORDEF_MC 原文; QuantUint8_TCM 用 Flat_8+TCM → "FB",
//     QUint8Crouton 用 Crouton_8 → "cB", PlainFloat 用 Flat_32 → "ff")。
//   · typetag 字符串 = 序列化元素编码名 (tensor_definitions.h TENSORDEF 清单原文)。
//   · M34 三代表实例: PlainFloat / QUint8Crouton / QuantUint8_TCM (本里程碑显式
//     实例化; 其余 Tdefs 配置在此仅定义, 按需实例化)。
// ============================================================================
#pragma once

#include <cstdint>
#include <type_traits>

#include "../ir/tensor_base.hpp" // hnnx::MemoryClass + 全局 ::DType (经 op_def.hpp)
#include "interface.hpp"         // PlainInterface / ScaleOffsetInterface
#include "tensor_layouts.hpp"    // 布局族 + Ldefs::stype_for
// 注: 不含 ir/types.hpp —— 其 GCP 旧世界 hnnx::OpDef_Const 与 op_def.hpp 的
// hnnx::OpDef_Const 重定义冲突; ::DType 已由 tensor_base.hpp 传递。

// ---------------------------------------------------------------------------
// dtype_traits<DType> — element_type / is_quant / is_float / is_signed
// (dtype.h 语义; element_type 仅按 sizeof 匹配 storage, 与 .so elbytes 表一致)
// ---------------------------------------------------------------------------
template <::DType DT> struct dtype_traits;
#define REQNN_DTYPE_TRAIT(DT, CPT, QUANT, FLOAT)                                                          \
    template <> struct dtype_traits<::DType::DT> {                                                        \
        using element_type = CPT;                                                                         \
        static constexpr bool is_quant = QUANT;                                                           \
        static constexpr bool is_float = FLOAT;                                                           \
    };
REQNN_DTYPE_TRAIT(QUInt8, uint8_t, true, false)
REQNN_DTYPE_TRAIT(QInt8, int8_t, true, false)
REQNN_DTYPE_TRAIT(QUInt16, uint16_t, true, false)
REQNN_DTYPE_TRAIT(QInt16, int16_t, true, false)
REQNN_DTYPE_TRAIT(QInt32, int32_t, true, false)
REQNN_DTYPE_TRAIT(Float32, float, false, true)
REQNN_DTYPE_TRAIT(Float16, uint16_t, false, true) // 半精度位模式以 16B 载体存储
REQNN_DTYPE_TRAIT(BFloat16, uint16_t, false, true)
REQNN_DTYPE_TRAIT(Int32, int32_t, false, false)
REQNN_DTYPE_TRAIT(Int64, int64_t, false, false)
#undef REQNN_DTYPE_TRAIT

// ---------------------------------------------------------------------------
// LAYOUTDEF (tensor_definitions.h:44-54 原文)
// ---------------------------------------------------------------------------
#define LAYOUTDEF(NAME, ELBYTES, LAYOUT, PAD)                                                             \
    namespace Ldefs {                                                                                     \
    struct NAME {                                                                                         \
        using Tlayout = LAYOUT;                                                                           \
        using storage_type = Ldefs::stype_for<ELBYTES>::type;                                             \
        static constexpr unsigned Rank = Tlayout::Rank;                                                   \
        using Pad_t = PAD<Rank>;                                                                          \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                      \
        static constexpr bool is_indirect = is_chunked;                                                   \
    };                                                                                                    \
    }
#define LAYOUTDEF_CONTIG(NAME, ELBYTES, LAYOUT, PAD)                                                      \
    namespace Ldefs {                                                                                     \
    struct NAME {                                                                                         \
        using Tlayout = LAYOUT;                                                                           \
        using storage_type = Ldefs::stype_for<ELBYTES>::type;                                             \
        static constexpr unsigned Rank = Tlayout::Rank;                                                   \
        using Pad_t = PAD<Rank>;                                                                          \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                      \
        static constexpr bool is_indirect = false;                                                        \
    };                                                                                                    \
    }

// 布局清单 (tensor_definitions.h:132-156 原文)
LAYOUTDEF(Flat_8, 1, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_8, 1, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_16, 2, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_16, 2, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_32, 4, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_32, 4, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat6D_32, 4, R6FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_64, 8, R4FlatMemoryLayout, NoPadding)

LAYOUTDEF(Crouton_8, 1, R4CroutonLayout, Padding)
LAYOUTDEF(Crouton_16, 2, R4Crouton2Layout, Padding)
LAYOUTDEF(Crouton_16_DeepAR4, 2, R4DeepAR4_16bLayout, Padding)
LAYOUTDEF(Crouton_16_DeepAR8, 2, R4DeepAR8_16bLayout, Padding)
LAYOUTDEF(Crouton_32, 4, R4Crouton4Layout, Padding)
LAYOUTDEF(Crouton4x1_8, 1, R4Crouton4x1Layout, Padding)
LAYOUTDEF(Crouton2x2_8, 1, R4Crouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_8, 1, R4WideCroutonLayout, Padding)
LAYOUTDEF(WideCrouton2x2_8, 1, R4WideCrouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_32, 4, R4WideCrouton4Layout, Padding)

LAYOUTDEF(R4Singular_8, 1, R4SingularMemoryLayout, NoPadding)
LAYOUTDEF(R4Singular_16, 2, R4SingularMemoryLayout, NoPadding)

// ---------------------------------------------------------------------------
// TENSORDEF / TENSORDEF_MC (tensor_definitions.h:107-129 原文)
// ---------------------------------------------------------------------------
#define TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, MCLASS, ENCODENAME)                                         \
    namespace Tdefs {                                                                                     \
    struct NAME {                                                                                         \
        using Lconfig = Ldefs::LAYOUTNAME;                                                                \
        using Tlayout = Lconfig::Tlayout;                                                                 \
        using storage_type = Lconfig::storage_type;                                                       \
        using element_type = dtype_traits<DTYPE>::element_type;                                           \
        static_assert(sizeof(element_type) == sizeof(storage_type), "layout has wrong element size");     \
        using Interface_t = std::conditional_t<dtype_traits<DTYPE>::is_quant,                             \
                                               ::ScaleOffsetInterface<element_type>,                      \
                                               ::PlainInterface<element_type>>;                           \
        static constexpr size_t Rank = Lconfig::Rank;                                                     \
        using Pad_t = Lconfig::Pad_t;                                                                     \
        static constexpr bool is_chunked = Lconfig::is_chunked;                                           \
        static constexpr bool is_indirect = Lconfig::is_indirect;                                         \
        static constexpr hnnx::MemoryClass memclass = MCLASS;                                             \
        static constexpr const char *typetag = ENCODENAME;                                                \
    };                                                                                                    \
    }

#define TENSORDEF(NAME, LAYOUTNAME, DTYPE, ENCODENAME)                                                    \
    TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, hnnx::MemoryClass::Default, ENCODENAME)

// M34 三代表 (显式实例化见 src/tens/tensor_concrete.cpp):
TENSORDEF(PlainFloat, Flat_32, DType::Float32, "ff")
TENSORDEF(QUint8Crouton, Crouton_8, DType::QUInt8, "cB")
TENSORDEF_MC(QuantUint8_TCM, Flat_8, DType::QUInt8, hnnx::MemoryClass::TCM, "FB")
// 对照组 (contig 量化, Flat_8):
TENSORDEF(QuantUint8, Flat_8, DType::QUInt8, "fB")
