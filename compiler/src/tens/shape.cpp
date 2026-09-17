// Shape<1..6> canonical/crated/deserialize — 逐指令复刻 (M34-C2)
//
// 反汇编源 (libHtpPrepare.so; 本文件全部区域已逐字重核):
//   canonical_shape(Graph&,Shape const&)   R1..R6 @0x12f91a0/0x12f8e20/0x12f8a40/
//                                           0x12f7cf0/0x12f8110/0x12f8590
//   canonical_shape(…,ShapeFlag)  六秩      R1 @0x12f9390 … R6 @0x12f8880
//   canonical_shape(…,OutputDef)  六秩      R1 @0x12f93e0 … R6 @0x12f8900
//   crated_shape                  六秩      R1 @0x12f9110 … R6 @0x12f84d0
//   deserialize                   六秩      R1 @0xd93720 … R6 @0xd94d10
//   BST 插入助手 (七秩共享静态, ICF 于 Shape<1> 符号域)  @0x12f9630
//   BST 再平衡 (libc++ __tree_balance_after_insert)     @0x868c50
//   serialize @0x12f5940 / get_shape_info @0x12f7020 —— 序列化族, 本 TU 不涉
#include "hnnx/tens/shape.hpp"

#include <cstring>
#include <new>

#include "hnnx/ir/tensor_base.hpp"   // hnnx::Crate / hnnx::ShapeFlag
#include "hnnx/serialize/deserz.hpp" // hnnx::Deserz / throw_dcrate_seg_overflow

// ---------------------------------------------------------------------------
// Graph 钩子 (::Graph 完整类属 M36 —— 此处按调用点偏移直读)
//   · 形映射: Graph+0x53b0+0x18(Rank-1) (R1..R6 调用点 0x12f92d2/0x12f8f82/
//     0x12f8bc2/0x12f7e7e/0x12f82e2/0x12f8782 偏移直证)
//   · graph_crate 内联体 (canonical @0x12f7eda / crated @0x12f7c58 同文):
//     graph+0xd0 或 *(graph+0xc8)+0x150, 择取于字节 graph+0x68b8
// ---------------------------------------------------------------------------
namespace {

constexpr size_t graph_off_shape_bst = 0x53b0; // R1 multimap; 步长 0x18
constexpr size_t graph_off_crate_selector = 0x68b8;
constexpr size_t graph_off_crate_embedded = 0xd0;
constexpr size_t graph_off_crate_ext_ptr = 0xc8; // 解引用后 +0x150

hnnx::Crate *graph_crate(::Graph &g) noexcept
{
    char *const p = reinterpret_cast<char *>(&g);
    if (p[graph_off_crate_selector] != 0) // cmpb $0,0x68b8(%rdi); cmovne
        return reinterpret_cast<hnnx::Crate *>(p + graph_off_crate_embedded);
    return reinterpret_cast<hnnx::Crate *>(*reinterpret_cast<char **>(p + graph_off_crate_ext_ptr) + 0x150);
}

// ---------------------------------------------------------------------------
// 规范化 BST —— libc++ __tree 布局
//   · 节点: operator new(0x30) @0x12f964a {left,right,parent,black@0x18,
//     key u32@0x20, value@0x28}
//   · 容器 0x18 {begin@0, end.__left_@+8, size@+0x10}; end 节点地址 = 容器+8
//     (canonical R4: 容器@+0x53f8, descent 之 r15 = graph+0x5400 = end 直证)
// ---------------------------------------------------------------------------
struct ShapeNode {
    ShapeNode *left;   // +0x00
    ShapeNode *right;  // +0x08
    ShapeNode *parent; // +0x10
    bool black;        // +0x18 (__is_black_)
    uint32_t key;      // +0x20 (哈希)
    void *value;       // +0x28 (Shape<Rank> const*)
};

struct ShapeMap {
    ShapeNode *begin_node; // +0x00 最左节点
    ShapeNode *root_slot;  // +0x08 end 节点 __left_ = 根 (故 end = 容器+8)
    size_t size;           // +0x10
};
static_assert(sizeof(ShapeMap) == 0x18);

// ---- 旋转 (0x868c50 内联四份; 以下两体与各份字段序逐条一致) ----
// rotate_right(g): g->left 上提 (0x868d7e..0x868db5)
void tree_rotate_right(ShapeNode *g) noexcept
{
    ShapeNode *const piv = g->left;   // rcx
    ShapeNode *const t = piv->right;  // rdx
    g->left = t;
    if (t != nullptr) t->parent = g;
    ShapeNode *const gg = g->parent; // rsi
    piv->parent = gg;
    size_t const i = (gg->left != g) ? 1 : 0; // setne
    (&gg->left)[i] = piv;
    piv->right = g;
    g->parent = piv;
}

// rotate_left(g): g->right 上提 (0x868d3b..0x868d72; LR 预旋 0x868cb8 同体)
void tree_rotate_left(ShapeNode *g) noexcept
{
    ShapeNode *const piv = g->right; // rcx
    ShapeNode *const t = piv->left;  // rdx
    g->right = t;
    if (t != nullptr) t->parent = g;
    ShapeNode *const gg = g->parent; // rsi
    piv->parent = gg;
    size_t const i = (gg->left != g) ? 1 : 0;
    (&gg->left)[i] = piv;
    piv->left = g;
    g->parent = piv;
}

// ---- 再平衡 @0x868c50; 首参 = *(容器+8) = 根 (调用点 0x12f975e 直证:
//      根涂黑判据 x==root, 非红叔循环的父色读取从未触及 end) ----
void tree_balance_after_insert(ShapeNode *root, ShapeNode *x) noexcept
{
    x->black = (x == root);        // 0x868c50 sete
    if (x == root) return;         // jne 0x868c77
    for (;;) {
        ShapeNode *const p = x->parent; // rcx @0x868c77
        if (p->black) return;           // 0x868c7b
        ShapeNode *const g = p->parent; // rax
        if (g->left == p) {             // 0x868c88: 父为左孩子
            ShapeNode *const u = g->right; // rdx @0x868ca0
            if (u != nullptr && !u->black) { // 红叔重染色 0x868c60
                p->black = true;
                x = g;
                x->black = (x == root);
                u->black = true;
                if (x == root) return; // 0x868c75 je
                continue;
            }
            ShapeNode *top; // 共尾 0x868d76 的旋后子树根
            if (p->left == x) { // LL: 0x868d73 (rdx=p)
                top = p;
            } else { // LR: 0x868cb8 先左旋 p (x 上提)
                tree_rotate_left(p);
                top = x;
            }
            top->black = true;  // 0x868d76
            g->black = false;   // 0x868d7a
            tree_rotate_right(g); // 0x868d7e
            return;
        }
        // 父为右孩子 (叔 = g->left)
        ShapeNode *const u = g->left; // rdx
        if (u != nullptr && !u->black) { // 红叔重染色 0x868c60
            p->black = true;
            x = g;
            x->black = (x == root);
            u->black = true;
            if (x == root) return;
            continue;
        }
        ShapeNode *top;
        if (p->left == x) { // RL: 0x868cff 先右旋 p (x 上提)
            tree_rotate_right(p);
            top = x;
        } else { // RR: 0x868cfa (x=p)
            top = p;
        }
        top->black = true; // 0x868d33
        g->black = false;  // 0x868d37
        tree_rotate_left(g); // 0x868d3b
        return;
    }
}

// ---- 插入助手 @0x12f9630 (提示式多重插入; 相等键回落路径不同侧!) ----
ShapeNode *tree_insert(ShapeMap *map, ShapeNode *hint, uint32_t key, void *value) noexcept
{
    ShapeNode *const node = static_cast<ShapeNode *>(::operator new(0x30));
    node->key = key;   // 0x12f9652/0x12f9656
    node->value = value; // 0x12f965a/0x12f965e

    ShapeNode *const end = reinterpret_cast<ShapeNode *>(&map->root_slot); // rcx = map+8
    ShapeNode *parent; // rcx
    ShapeNode **link;  // rdx

    if (hint == end || hint->key >= key) goto branch_a; // 0x12f9662/0x12f966b

    { // hint->key < key (0x12f9670): 全树下潜, 相等键走左 (0x12f968f jae→left)
        ShapeNode *n = map->root_slot;
        if (n == nullptr) { // 0x12f9676 → 0x12f96d9 根挂
            parent = end;
            link = reinterpret_cast<ShapeNode **>(end); // &root_slot (偏移 0 技巧)
            goto attach;
        }
        for (;;) {
            ShapeNode *c;
            if (n->key >= key) { // 0x12f968f jae → 0x12f9680
                c = n->left;
                if (c == nullptr) { // 0x12f9779: rcx=rdx=n, rdx=n (&left)
                    parent = n;
                    link = reinterpret_cast<ShapeNode **>(n);
                    goto attach;
                }
            } else { // 0x12f9694
                c = n->right;
                if (c == nullptr) { // 0x12f969d: rdx = n+8, rcx = n
                    parent = n;
                    link = &n->right;
                    goto attach;
                }
            }
            n = c; // 0x12f968c
        }
    }

branch_a : { // hint == end 或 hint->key >= key (0x12f96a9)
    ShapeNode *const left = hint->left; // r8
    ShapeNode *rdi = hint;              // 0x12f96ac
    if (map->begin_node == hint) goto a_attach; // 0x12f96af (begin 无左孩子)
    if (left != nullptr) {
        // 0x12f96b9: hint 左子树最右 (中序前驱); key < 前驱 → 重下潜
        ShapeNode *rm = left;
        while (rm->right != nullptr) rm = rm->right; // 0x12f96c0
        rdi = rm;
        if (key < rm->key) goto full_descent; // 0x12f96cc jb
        goto a_attach;
    }
    // hint->left == 0 (0x12f9702): 上行至首个右孩子转折, rdi = 其父 = 中序前驱
    {
        ShapeNode **plink = &hint->parent;                  // rdx = rbx+0x10
        ShapeNode *cur = hint->parent;                      // rdi
        if (cur->left == hint) {                            // 0x12f970a je→0x12f9710
            for (;;) {
                cur = *plink;                               // 0x12f9710 (首趟重读)
                plink = &cur->parent;
                if (cur->parent->left != cur) break;        // 0x12f971b jne 出
            }
        }
        rdi = *plink;              // 0x12f9720 (非循环入口: = hint->parent)
        if (key < rdi->key) goto full_descent; // 0x12f9723 jb → 0x12f96d1
        goto a_attach;
    }

a_attach: // 0x12f9728: r8==0 (cmove) → 挂 hint 左; 否则挂 rdi 右
    link = &rdi->right;
    if (left == nullptr) {
        parent = hint;
        link = reinterpret_cast<ShapeNode **>(hint); // &hint->left (偏移 0)
    } else {
        parent = rdi;
    }
    goto attach;
}

full_descent: { // 0x12f96d1: lower_bound 失效后的全树下潜 —— 相等键走右!
    ShapeNode *n = map->root_slot;
    if (n == nullptr) { // 0x12f96d9
        parent = end;
        link = reinterpret_cast<ShapeNode **>(end);
        goto attach;
    }
    for (;;) {
        ShapeNode *c;
        if (key >= n->key) { // 0x12f96f0 jae → 0x12f96e0 右
            c = n->right;
            if (c == nullptr) { // 0x12f9781: rcx=rdx=n, rdx=n+8
                parent = n;
                link = &n->right;
                goto attach;
            }
        } else { // 0x12f96f5 左
            c = n->left;
            if (c == nullptr) { // 0x12f96fd: rcx=rdx=n, rdx=n
                parent = n;
                link = reinterpret_cast<ShapeNode **>(n);
                goto attach;
            }
        }
        n = c; // 0x12f96ed
    }
}

attach: { // 0x12f973a
    node->left = nullptr;   // movups xor (l=r=0)
    node->right = nullptr;
    node->parent = parent;  // 0x12f9741
    *link = node;           // 0x12f9745
    // begin 更新 0x12f9748: begin->left 非空 → begin = begin->left
    if (map->begin_node->left != nullptr) map->begin_node = map->begin_node->left;
    tree_balance_after_insert(map->root_slot, node); // 0x12f975e (挂接后取根)
    map->size += 1;                                   // 0x12f9767
    return node;
}
}

// ---- 中序后继 (canonical 内联 0x12f7e29..0x12f7e70) ----
ShapeNode *tree_successor(ShapeNode *n) noexcept
{
    if (n->right != nullptr) { // 0x12f7e29: 右子树最左
        ShapeNode *m = n->right;
        while (m->left != nullptr) m = m->left; // 0x12f7e40
        return m;
    }
    ShapeNode *p = n->parent;   // 0x12f7e50
    if (p->left == n) return p; // 自身为左孩子 → 父即后继
    for (;;) {                  // 0x12f7e60: 沿右孩子链上行
        ShapeNode *const g = p->parent;
        if (g->left == p) return g; // p 为左孩子 → g 即后继 (可为 end)
        p = g;
    }
}

// ---- 逐字 FNV (R4 @0x12f7d0f..0x12f7db7 直译; 末字只加不乘) ----
constexpr uint32_t fnv_mult = 0x12401D1u;
constexpr uint32_t rank_salt[6] = {0x48D26249u, 0x18619B8Au, 0x3179DF4Bu,
                                   0x8065DD8Cu, 0x0B93B1DDu, 0xC2C91AEEu};

template <size_t Rank> uint32_t shape_hash(Shape<Rank> const *shp) noexcept
{
    constexpr unsigned nwords = (8u + 17u * Rank + 3u) / 4u; // ⌈shplen/4⌉
    uint32_t const *const w = reinterpret_cast<uint32_t const *>(shp);
    uint32_t h = w[0] * fnv_mult;
    for (unsigned k = 1; k + 1 < nwords; ++k) h = (h + w[k]) * fnv_mult;
    return h + w[nwords - 1] + rank_salt[Rank - 1];
}

template <size_t Rank> ShapeMap *shape_map_at(::Graph &g) noexcept
{
    return reinterpret_cast<ShapeMap *>(reinterpret_cast<char *>(&g) + graph_off_shape_bst + 0x18 * (Rank - 1));
}

// canonical 尾部分配 (0x12f7eda..0x12f7f51): 记录尺寸 = sizeof = align8(shplen)
template <size_t Rank> void *crate_copy_shape(::Graph &g, Shape<Rank> const *shp) noexcept
{
    hnnx::Crate *const crate = graph_crate(g);
    auto const r = crate->add_record_slot(sizeof(Shape<Rank>), 8);
    void *const rec = r.slot; // sret+0x10 即槽字段 (其值为记录地址)
    memcpy(rec, shp, sizeof(Shape<Rank>)); // 0x50 字节 movups ×5
    if (r.status >= 0) crate->bump_record_count(); // 0x12f7f45/0x12f7f4c
    return rec;
}

} // namespace

// ---------------------------------------------------------------------------
// canonical_shape(Graph&, Shape const&) — R4 @0x12f7cf0 (六秩同构)
// ---------------------------------------------------------------------------
template <size_t Rank>
Shape<Rank> const *Shape<Rank>::canonical_shape(::Graph &graph, Shape const &shp_ref)
{
    Shape const *const shp = &shp_ref;
    uint32_t const hash = shape_hash<Rank>(shp); // 0x12f7d0f

    ShapeMap *const map = shape_map_at<Rank>(graph);
    ShapeNode *const end = reinterpret_cast<ShapeNode *>(&map->root_slot);

    // ---- 下行查找 (0x12f7dc1..0x12f7df3): cand = 最后一个 key>=hash 者 ----
    ShapeNode *cand = end; // rbx
    ShapeNode *n = map->root_slot;
    if (n != nullptr) {
        do {
            int const c = (n->key < hash) ? 1 : 0; // setb
            if (n->key >= hash) cand = n;          // cmovae
            n = c ? n->right : n->left;
        } while (n != nullptr);
    }

    // ---- 哈希相等链 (0x12f7e08..0x12f7e70): 首个 memcmp 命中即取 ----
    ShapeNode *c2 = cand;
    ShapeNode *hit = nullptr;
    while (c2 != end && c2->key == hash) {
        void const *const val = c2->value; // 0x12f7e0d (memcmp 直用, 无空检)
        if (memcmp(shp, val, shp->shplen()) == 0) { // 0x12f7e21 je
            hit = c2;
            break;
        }
        c2 = tree_successor(c2); // 后继; 可达 end
    }

    void **vslot;
    if (hit != nullptr) {
        vslot = &hit->value; // 0x12f7ed1: rbp = rbx+0x28
        if (hit->value != nullptr) return static_cast<Shape const *>(hit->value); // 0x12f7eaa
    } else {
        // 插入 (0x12f7e75..0x12f7e92): 值 = null, 惰性填充
        ShapeNode *const hint = c2; // r15: 链尽=end / 键不符=cand (0x12f7e72)
        ShapeNode *const node = tree_insert(map, hint, hash, nullptr);
        vslot = &node->value; // 0x12f7e9a/0x12f7ea1
        // node->value == 必为 null (刚传)
    }

    void *const rec = crate_copy_shape<Rank>(graph, shp); // 0x12f7eda
    *vslot = rec;                                         // 0x12f7f51
    return static_cast<Shape const *>(rec);
}

// ---------------------------------------------------------------------------
// canonical_shape(…, ShapeFlag) — R4 @0x12f7f70: 整形栈拷贝 + flags16 覆写
// ---------------------------------------------------------------------------
template <size_t Rank>
Shape<Rank> const *Shape<Rank>::canonical_shape(::Graph &graph, Shape const &shp_ref, hnnx::ShapeFlag const flags)
{
    Shape tmp;
    memcpy(&tmp, &shp_ref, shp_ref.shplen()); // 0x4c 字节 (尾部填充不拷)
    tmp.flags = uint16_t(flags);              // 0x12f7fad movw %dx
    return canonical_shape(graph, tmp);
}

// ---------------------------------------------------------------------------
// canonical_shape(…, OutputDef) — R4 @0x12f7fe0: dims[i]=max_dims[i]=max_sizes[i]
// (R1 @0x12f93fa 读 def+8; R5 @0x12f845a..0x12f8482 读 def+8..+0x28 互证)
// ---------------------------------------------------------------------------
template <size_t Rank>
Shape<Rank> const *Shape<Rank>::canonical_shape(::Graph &graph, ::OutputDef const &def)
{
    Shape tmp; // flags qword = 0, pad = 0 (ctor)
    for (size_t i = 0; i < Rank; ++i) {
        tmp.dims[i] = def.max_sizes[i];     // 栈 +8..+8+8R
        tmp.max_dims[i] = def.max_sizes[i]; // 栈 +0x28.. (R4 dump 0x12f801e..0x12f802d)
    }
    return canonical_shape(graph, tmp);
}

// ---------------------------------------------------------------------------
// crated_shape — R4 @0x12f7c40: 不查重, 直接 crate 新建
// ---------------------------------------------------------------------------
template <size_t Rank> Shape<Rank> const *Shape<Rank>::crated_shape(::Graph &graph, Shape const &shp_ref)
{
    hnnx::Crate *const crate = graph_crate(graph); // 0x12f7c58
    auto const r = crate->add_record_slot(sizeof(Shape<Rank>), 8);
    void *const rec = r.slot;
    memcpy(rec, &shp_ref, sizeof(Shape<Rank>)); // movups ×5
    if (r.status >= 0) crate->bump_record_count(); // 0x12f7cc4
    return static_cast<Shape const *>(rec);
}

// ---------------------------------------------------------------------------
// deserialize — R4 @0xd94110 (六秩同构; 位格式见 shape.hpp 头注)
// ---------------------------------------------------------------------------
template <size_t Rank>
Shape<Rank> const *Shape<Rank>::deserialize(hnnx::Deserz &dctx, Shape const **slot)
{
    // 共享对象去重 (0xd9412f): rdx==0 → 既有 rax; 否则 rdx = 新对象写入槽
    auto const so = dctx.deserialize_shared_obj_func(reinterpret_cast<void const **>(slot));
    if (so.new_where == nullptr) return static_cast<Shape const *>(so.existing); // 0xd94134

    // ---- 读词内联 (游标 +0x68 / 上限 +0x70 / 越界虚槽 +0x10) ----
    auto read_word = [&dctx]() noexcept -> uint32_t {
        unsigned char *cur = dctx.read_cursor();
        if (cur >= dctx.read_end()) cur = dctx.refill();
        uint32_t const w = *reinterpret_cast<uint32_t const *>(cur);
        dctx.set_read_cursor(cur + 4);
        return w;
    };

    // ---- 首字 (0xd9415e..0xd941ec): 0xcccc 前缀 → flags16 ----
    uint32_t flags16 = 0;
    uint32_t const w0 = read_word();
    uint32_t cw;
    if ((w0 & 0xffff0000u) == 0xcccc0000u) {
        flags16 = w0; // 存整词, 填充时仅取低 16 (movw)
        cw = read_word();
    } else {
        cw = w0;
    }

    // ---- 逐轴 (0xd94191 起, 轴 g 占 cw 位 4g) ----
    Shape tmp; // 全零初始化 (flags 填充/pad 与 .so 写零语义一致)
    for (unsigned g = 0; g < Rank; ++g) {
        uint32_t const mode = (cw >> (4 * g)) & 3u;
        if (mode != 0) {
            uint32_t const w = read_word();
            if (mode == 1) { // {dim:16, Δ:8@16..23, pad:8@24..31}; max = dim+Δ
                tmp.dims[g] = w & 0xffffu;
                tmp.max_dims[g] = (w & 0xffffu) + ((w >> 16) & 0xffu);
                tmp.pad[g] = uint8_t(w >> 24);
            } else if (mode == 2) { // {dim:24, pad:8}; max = dim
                tmp.dims[g] = w & 0xffffffu;
                tmp.max_dims[g] = w & 0xffffffu;
                tmp.pad[g] = uint8_t(w >> 24);
            } else { // mode 3: 全宽 dim = max, pad = 0
                tmp.dims[g] = w;
                tmp.max_dims[g] = w;
            }
        } else { // mode 0: {1,1,0} 无字 (0xd941ee)
            tmp.dims[g] = 1;
            tmp.max_dims[g] = 1;
        }
        if (cw & (4u << (4 * g))) { // max 覆盖字: 全宽 (0xd94225)
            uint32_t const w = read_word();
            tmp.max_dims[g] = w;
        }
        if (cw & (8u << (4 * g))) { // pad 覆盖字: 低字节 (0xd94248)
            uint32_t const w = read_word();
            tmp.pad[g] = uint8_t(w);
        }
    }
    tmp.flags = uint16_t(flags16); // flags 8B = flags16|0 (0xd94552..0xd94560)

    // ---- 构造 (0xd9452a): DCrate 快路 / crate 慢路 ----
    void *rec;
    unsigned char *next = dctx.scratch_ptr(); // *(dctx+0x20)
    if (next != nullptr) {
        unsigned char *const p = reinterpret_cast<unsigned char *>(
                (reinterpret_cast<uintptr_t>(next) + 7u) & ~uintptr_t(7u)); // 0xd94533
        unsigned char *const after = p + sizeof(Shape);
        if (after > dctx.scratch_end()) hnnx::throw_dcrate_seg_overflow(); // 0xd94543 ja
        dctx.set_scratch_ptr(after); // 0xd94549 (快路不 bump 计数)
        if (p == nullptr) goto slow;  // 0xd9454d (nextp∈{1..7} 之防)
        rec = p;
    } else {
    slow: // 0xd945a7: cratep(+0x30) 记录槽
        hnnx::Crate *const crate = dctx.scratch_crate();
        auto const r = crate->add_record_slot(sizeof(Shape<Rank>), 8);
        rec = r.slot;
        if (r.status >= 0) crate->bump_record_count(); // 0xd9463c/0xd94648
    }
    memcpy(rec, &tmp, tmp.shplen()); // 恰 shplen 字节 (0xd94552..0xd9459f)

    *so.new_where = rec;   // 0xd9464d/0xd94652
    return static_cast<Shape const *>(rec);
}

// ---------------------------------------------------------------------------
// 显式实例化 (与 .so dynsym 30 符号一一对应)
// ---------------------------------------------------------------------------
#define REQNN_SHAPE_INSTANTIATE(R)                                                                                     \
    template Shape<R> const *Shape<R>::canonical_shape(::Graph &, Shape<R> const &);                                   \
    template Shape<R> const *Shape<R>::canonical_shape(::Graph &, Shape<R> const &, hnnx::ShapeFlag const);            \
    template Shape<R> const *Shape<R>::canonical_shape(::Graph &, ::OutputDef const &);                                \
    template Shape<R> const *Shape<R>::crated_shape(::Graph &, Shape<R> const &);                                      \
    template Shape<R> const *Shape<R>::deserialize(hnnx::Deserz &, Shape<R> const **);

REQNN_SHAPE_INSTANTIATE(1)
REQNN_SHAPE_INSTANTIATE(2)
REQNN_SHAPE_INSTANTIATE(3)
REQNN_SHAPE_INSTANTIATE(4)
REQNN_SHAPE_INSTANTIATE(5)
REQNN_SHAPE_INSTANTIATE(6)
