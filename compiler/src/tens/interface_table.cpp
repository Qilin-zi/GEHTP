// ifc_method_table — 逐字节复刻 @0x5ec7fd0 (0x210 字节) + 16 个 thunk
// 语义精确复刻 (thunk @0xdcd040..0xdcd38e, 局部符号 → 本 TU 匿名命名空间)。
// 表中 0 = 空槽 (read_float/write_float 对 QUInt8/QUInt16/Float32/Int32/Int64
// 为 0; ifc_hash/ifc_compare 对非量化型为 0) —— 原样拷贝, 非省略。
#include "hnnx/tens/interface.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// thunk 群 (全部 static —— .so 中为内部链接局部符号)
// ---------------------------------------------------------------------------
namespace {

// 0xdcd040 — read_float(UNKNOWN): xorps xmm0,xmm0; ret
float read_float_unknown(Interface const *, void const *) noexcept
{
    return 0.0f;
}

// 0xdcd050 — write_float(UNKNOWN): ret (无条件空操作)
void write_float_unknown(Interface const *, void *, float) noexcept
{
}

// 0xdcd060 — get_qparms(plain): movq GOT(&null_parms),%rax; ret
Interface::qparms const *get_qparms_null(Interface const *) noexcept
{
    return &Interface::null_parms;
}

// 0xdcd070 — get_qparms(quant): leaq 0x4(%rdi),%rax; ret
Interface::qparms const *get_qparms_quant(Interface const *ifc) noexcept
{
    return reinterpret_cast<Interface::qparms const *>(reinterpret_cast<unsigned char const *>(ifc) + 4);
}

// 0xdcd080 — ifc_hash(quant):
//   ebx = (i32)ifc->offset * 0x10661 (imull 截断);
//   eax = bit_cast<u32>(ifc->scale); eax += eax; eax ^= ebx; ret
uint32_t ifc_hash_quant(Interface const *ifc) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    uint32_t scale_bits;
    memcpy(&scale_bits, &q->scale, 4);
    return (scale_bits * 2u) ^ (uint32_t(q->offset) * 0x10661u);
}

// 0xdcd0e0 — ifc_compare(quant): a=rdi, b=rsi
//   a.offset != b.offset → (a.offset >= b.offset) ? 1 : -1   (setge)
//   a.scale == b.scale (且非 NaN) → 0
//   否则 (含 NaN): ucomiss b,a; setbe → (b <= a 或 NaN) ? 1 : -1
int ifc_compare_quant(Interface const *a, Interface const *b) noexcept
{
    auto const *qa = get_qparms_quant(a);
    auto const *qb = get_qparms_quant(b);
    if (qa->offset != qb->offset) return qa->offset >= qb->offset ? 1 : -1;
    if (qa->scale == qb->scale) {
        // ucomiss 等且非 NaN → 0 (NaN 时 ZF=0 → 走下一分支)
        if (!std::isnan(qa->scale)) return 0;
    }
    // ucomiss %xmm0(a),%xmm1(b) 即比较 b:a; setbe = CF|ZF (含 unordered=1)
    int const be = std::isunordered(qa->scale, qb->scale) || (qb->scale <= qa->scale);
    return be ? 1 : -1;
}

// ---- QInt16 @0xdcd120/0xdcd130 ----
float read_float_qint16(Interface const *ifc, void const *p) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    return float(int32_t(*static_cast<int16_t const *>(p)) - q->offset) * q->scale;
}
void write_float_qint16(Interface const *ifc, void *p, float f) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    float r = std::nearbyintf(f * q->scale_recip + float(q->offset));
    r = std::fmin(r, 2147483520.0f);  // 0x39b3180
    r = std::fmax(r, -2147483648.0f); // 0x39b3184
    r = std::fmax(r, -32768.0f);      // 0x399db90
    r = std::fmin(r, 32767.0f);       // 0x39b3188
    *static_cast<int16_t *>(p) = int16_t(int32_t(r));
}

// ---- QInt32 @0xdcd180/0xdcd1a0 ----
//  read: cvtsi2ssl (%rsi) — 无符号 32 位源; cvtsi2ssl 0x4(%rdi) — offset;
//        subss; mulss 0x8(%rdi) (scale)
float read_float_qint32(Interface const *ifc, void const *p) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    float const v = float(*static_cast<uint32_t const *>(p)); // cvtsi2ssl (%rsi)
    float const off = float(q->offset);                       // cvtsi2ssl 0x4(%rdi)
    return (v - off) * q->scale;
}
void write_float_qint32(Interface const *ifc, void *p, float f) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    float r = std::nearbyintf(f * q->scale_recip + float(q->offset));
    r = std::fmin(r, 2147483520.0f);
    r = std::fmax(r, -2147483648.0f);
    *static_cast<uint32_t *>(p) = uint32_t(int32_t(r));
}

// ---- QInt8 @0xdcd1d0/0xdcd1e0 ----
float read_float_qint8(Interface const *ifc, void const *p) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    return float(int32_t(*static_cast<int8_t const *>(p)) - q->offset) * q->scale;
}
void write_float_qint8(Interface const *ifc, void *p, float f) noexcept
{
    auto const *q = get_qparms_quant(ifc);
    float r = std::nearbyintf(f * q->scale_recip + float(q->offset));
    r = std::fmin(r, 2147483520.0f);
    r = std::fmax(r, -2147483648.0f);
    r = std::fmax(r, -128.0f); // 0x39b318c
    r = std::fmin(r, 127.0f);  // 0x39b3190
    *static_cast<int8_t *>(p) = int8_t(int32_t(r));
}

// ---- Float16 @0xdcd230/0xdcd290 —— 逐指令位级转录, 无 IEEE 语义解释 ----
float read_float_f16(Interface const *, void const *p) noexcept
{
    uint32_t const ecx0 = *static_cast<uint16_t const *>(p); // movzwl (%rsi)
    uint32_t eax = (ecx0 << 16) & 0x80000000u;               // 符号位
    uint32_t const mag = ecx0 & 0x7fffu;
    if (mag == 0) return __builtin_bit_cast(float, eax); // je 0xdcd27e
    if (mag <= 0x3ffu) {                                 // 次正规 (cmp 0x3ff; ja)
        eax |= 0x33800000u;
        return __builtin_bit_cast(float, eax) * float(mag); // mulss
    }
    // 规格化/inf/nan: 重偏置 +13; mag >= 0x7c00 时强置指教位
    uint32_t norm = (mag << 13) + 0x38000000u;
    uint32_t const sp = (mag < 0x7c00u) ? norm : (norm | 0x7f800000u); // cmovbl
    return __builtin_bit_cast(float, eax | sp);
}
void write_float_f16(Interface const *, void *p, float f) noexcept
{
    uint32_t const x = __builtin_bit_cast(uint32_t, f);       // movd xmm0→r8d
    uint32_t edi = x & 0x7fffffffu;                           // 幅值
    uint32_t const eax_sign = (x >> 16) & 0x8000u;            // 半精度符号位
    uint16_t out;
    if (edi < 0x38800000u) {                                  // jb 0xdcd2dd
        if (edi < 0x33000001u) {                              // jb 0xdcd31c
            out = uint16_t(eax_sign);                         // 下溢 → ±0
        } else {                                              // 次正规窗
            uint32_t const e = edi >> 23;                     // dil
            uint32_t const m = (x & 0x7fffffu) | 0x800000u;   // r8d
            uint32_t const cl = 0x7eu - e;                    // 0x7e - dil
            uint32_t const half = (1u << cl) >> 1;
            uint32_t edx = m ^ 0xff7fffffu;
            edx >>= cl;
            edx &= 1u;
            edx = 0u - edx;
            edx += m;
            edx += 0x800000u;
            edx += half;
            edx >>= cl;
            out = uint16_t(eax_sign | edx);
        }
    } else if (edi < 0x47800001u || edi < 0x7f800001u) {      // 规格化窗
        if (edi >= 0x47800001u) edi = 0x47800000u;            // 溢出钳到 inf 基值
        uint32_t ecx = (~edi >> 13) & 1u;
        ecx = 0u - ecx;
        ecx += edi;
        ecx += 0x1000u;
        ecx >>= 13;
        ecx += 0xfffe4000u;
        out = uint16_t(eax_sign | ecx);
    } else {                                                  // NaN (ax >= 0x7f800001)
        uint32_t const r8 = (x >> 13) & 0x1ffu;
        out = uint16_t(eax_sign | r8 | 0x7e00u);
    }
    *static_cast<uint16_t *>(p) = out;
}

// ---- BFloat16 @0xdcd350/0xdcd360 ----
float read_float_bf16(Interface const *, void const *p) noexcept
{
    uint32_t const b = uint32_t(*static_cast<uint16_t const *>(p)) << 16;
    return __builtin_bit_cast(float, b);
}
void write_float_bf16(Interface const *, void *p, float f) noexcept
{
    uint32_t const x = __builtin_bit_cast(uint32_t, f);
    uint16_t out;
    if ((x & 0x7f80ffffu) > 0x7f800000u) {
        out = 0x7fa0; // canonical NaN
    } else {
        // 就近舍入: + lsb + 0x7fff 后截断高 16
        out = uint16_t((x + ((x >> 16) & 1u) + 0x7fffu) >> 16);
    }
    *static_cast<uint16_t *>(p) = out;
}

} // namespace

// ---------------------------------------------------------------------------
// Interface::null_parms @0x39b5ea4: {offset=0, scale=1.0f, scale_recip=1.0f}
// ---------------------------------------------------------------------------
Interface::qparms const Interface::null_parms = {0, 1.0f, 1.0f};

// ---------------------------------------------------------------------------
// get_refobj @0xdcd390 精确复刻 (告警路径以日志级别 0 恒静默 —— 构造不失败)
// ---------------------------------------------------------------------------
hnnx::InterfaceRef Interface::get_refobj() const
{
    unsigned const dt_raw = dtinfo.dtype;
    unsigned const dt = dt_raw < 0xb ? dt_raw : 0;
    hnnx::intfc_methods const &e = hnnx::ifc_method_table[dt];
    if (dt_raw != e.exemplar.get_dt_info().dtype || dtinfo.elbytes != e.exemplar.get_dt_info().elbytes) {
        // .so: qnndsp_log(0, ...) 类型失配告警 (不影响返回值)
    }
    return hnnx::InterfaceRef{&e, this};
}

// ---------------------------------------------------------------------------
// ifc_method_table — 值域逐字节对表 @0x5ec7fd0
//   e0  UNKNOWN  {0,0,0}      read=040 write=050 qparms=060 hash=0    cmp=0
//   e1  QUInt8   {1,1,1}      read=0   write=0   qparms=070 hash=080  cmp=0e0
//   e2  QUInt16  {2,2,1}      read=0   write=0   qparms=070 hash=080  cmp=0e0
//   e3  QInt16   {2,3,5}      read=120 write=130 qparms=070 hash=080  cmp=0e0
//   e4  Float32  {4,4,6}      read=0   write=0   qparms=060 hash=0    cmp=0
//   e5  Int32    {4,5,4}      read=0   write=0   qparms=060 hash=0    cmp=0
//   e6  QInt32   {4,6,5}      read=180 write=1a0 qparms=070 hash=080  cmp=0e0
//   e7  QInt8    {1,7,5}      read=1d0 write=1e0 qparms=070 hash=080  cmp=0e0
//   e8  Float16  {2,8,6}      read=230 write=290 qparms=060 hash=0    cmp=0
//   e9  Int64    {8,9,4}      read=0   write=0   qparms=060 hash=0    cmp=0
//   e10 BFloat16 {2,10,6}     read=350 write=360 qparms=060 hash=0    cmp=0
// ---------------------------------------------------------------------------
namespace hnnx {
ifc_method_table_t const ifc_method_table = {{
        // e0 UNKNOWN
        intfc_methods{IfcExemplar{}, read_float_unknown, write_float_unknown, get_qparms_null, nullptr, nullptr},
        // e1 QUInt8 (read/write 空 —— 原样)
        intfc_methods{IfcExemplar{dtype_info{1u, 1u, 1u, 0u, 0u}}, nullptr, nullptr, get_qparms_quant,
                      ifc_hash_quant, ifc_compare_quant},
        // e2 QUInt16
        intfc_methods{IfcExemplar{dtype_info{2u, 2u, 1u, 0u, 0u}}, nullptr, nullptr, get_qparms_quant,
                      ifc_hash_quant, ifc_compare_quant},
        // e3 QInt16
        intfc_methods{IfcExemplar{dtype_info{2u, 3u, 1u, 0u, 1u}}, read_float_qint16, write_float_qint16,
                      get_qparms_quant, ifc_hash_quant, ifc_compare_quant},
        // e4 Float32
        intfc_methods{IfcExemplar{dtype_info{4u, 4u, 0u, 1u, 0u}}, nullptr, nullptr, get_qparms_null, nullptr,
                      nullptr},
        // e5 Int32
        intfc_methods{IfcExemplar{dtype_info{4u, 5u, 0u, 0u, 1u}}, nullptr, nullptr, get_qparms_null, nullptr,
                      nullptr},
        // e6 QInt32
        intfc_methods{IfcExemplar{dtype_info{4u, 6u, 1u, 0u, 1u}}, read_float_qint32, write_float_qint32,
                      get_qparms_quant, ifc_hash_quant, ifc_compare_quant},
        // e7 QInt8
        intfc_methods{IfcExemplar{dtype_info{1u, 7u, 1u, 0u, 1u}}, read_float_qint8, write_float_qint8,
                      get_qparms_quant, ifc_hash_quant, ifc_compare_quant},
        // e8 Float16
        intfc_methods{IfcExemplar{dtype_info{2u, 8u, 0u, 1u, 0u}}, read_float_f16, write_float_f16,
                      get_qparms_null, nullptr, nullptr},
        // e9 Int64
        intfc_methods{IfcExemplar{dtype_info{8u, 9u, 0u, 0u, 1u}}, nullptr, nullptr, get_qparms_null, nullptr,
                      nullptr},
        // e10 BFloat16
        intfc_methods{IfcExemplar{dtype_info{2u, 10u, 0u, 1u, 0u}}, read_float_bf16, write_float_bf16,
                      get_qparms_null, nullptr, nullptr},
}};
} // namespace hnnx
