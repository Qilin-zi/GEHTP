// hnnx::Deserz::deserialize_shared_obj_func @0xcfd490 — 逐指令复刻 (M34)
//
// 共享对象表: *(Deserz+0x50) (= full_deser/共享上下文) 的 +0xd8..+0xe8 处
//   vector<void*> {begin@+0xd8, end@+0xe0, cap@+0xe8}。
// 身份词 id (u32): idx = id & 0x7fffffff; 表项 = begin[idx-1]。
//   · idx-1 >= 表长 (无符号, idx==0 亦然) → 新对象: 追加空表项,
//     返回 {existing=0, new_where=&新表项}; 容量不足时扩容
//     operator new(旧字节数+0x4000), 旧数据拷至新缓冲区头部, 释放旧缓冲
//     (0xcfd531..0xcfd5a3)。
//   · 既有表项且 id 符号位置位 → 表项清 0, 返回 {0, &表项} (0xcfd5c6)。
//   · 既有表项非空 → 返回 {*表项, 0} (0xcfd50a..0xcfd516)。
//   · 既有表项为空 (对象构造中, 前向引用) → 段修正登记 (0xcfd5f2..0xcfd780,
//     SUPPFIXUP_CAT_sharedobj=5 打包 {Δid<<12|槽偏移}, fixbuf 计数 <0x40,
//     满 0x3f 调 0xcf6c70 冲刷)。该冲刷子系统 (Deserializer 侧) 未解码,
//     本实现显式抛错并标注地址 —— 不做任何未证实的语义替代。
#include "hnnx/serialize/deserz.hpp"

#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>

namespace hnnx {

Deserz::shared_obj_lookup Deserz::deserialize_shared_obj_func(void const **where)
{
    (void)where; // 仅登记用途 (待修正路径); 既有/新路径不读
    // ---- 读身份词: 游标 +0x68 / 上限 +0x70 / 越界 fill_buffer (虚槽 +0x10) ----
    unsigned char *cur = this->read_cursor();
    if (cur >= this->read_end()) cur = this->refill();
    uint32_t const id = *reinterpret_cast<uint32_t const *>(cur);
    this->set_read_cursor(cur + 4);

    uint32_t const idx = id & 0x7fffffffu;

    // ---- 共享表 ----
    char *const fd = static_cast<char *>(this->shared_ctx());
    void **const vb = *reinterpret_cast<void ***>(fd + 0xd8);
    void **const ve = *reinterpret_cast<void ***>(fd + 0xe0);
    void **const vc = *reinterpret_cast<void ***>(fd + 0xe8);

    if (idx - 1u < static_cast<uint32_t>(ve - vb)) { // 无符号: idx==0 → 巨值 → 走追加
        void **const entry = vb + (idx - 1u);
        if (static_cast<int32_t>(id) < 0) { // 0xcfd5c6: 符号位 → 表项作废重定义
            *entry = nullptr;
            return {nullptr, entry};
        }
        if (*entry != nullptr) return {*entry, nullptr}; // 既有对象
        // 0xcfd5f2: 表项空 (构造中前向引用) → 段修正登记, 依赖未解码的
        // 冲刷例程 0xcf6c70 —— 显式失败而非臆造行为。
        throw std::runtime_error(
                "REQNN deserialize_shared_obj_func: pending-fixup path not reimplemented "
                "(.so 0xcfd5f2..0xcfd780, flush 0xcf6c70)");
    }

    // ---- 新对象: 追加空表项 (容量不足先扩容) ----
    if (ve == vc) {
        size_t const old_bytes = size_t(reinterpret_cast<char *>(ve) - reinterpret_cast<char *>(vb));
        char *const nb = static_cast<char *>(::operator new(old_bytes + 0x4000));
        if (old_bytes != 0) memcpy(nb, vb, old_bytes);
        char *const old_buf = reinterpret_cast<char *>(vb);
        void **const nvb = reinterpret_cast<void **>(nb);
        void **const nve = reinterpret_cast<void **>(nb + old_bytes);
        void **const nvc = reinterpret_cast<void **>(nb + old_bytes + 0x4000);
        *reinterpret_cast<void ***>(fd + 0xd8) = nvb;
        *reinterpret_cast<void ***>(fd + 0xe0) = nve;
        *reinterpret_cast<void ***>(fd + 0xe8) = nvc;
        if (vb != nullptr) ::operator delete(old_buf); // 0xcfd596: 旧缓冲非空才释放
        void **const slot = nve;
        *slot = nullptr;
        *reinterpret_cast<void ***>(fd + 0xe0) = slot + 1;
        return {nullptr, slot};
    }
    *ve = nullptr;
    *reinterpret_cast<void ***>(fd + 0xe0) = ve + 1;
    return {nullptr, ve};
}

} // namespace hnnx
