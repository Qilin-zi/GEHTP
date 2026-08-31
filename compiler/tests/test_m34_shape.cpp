// test_m34_shape.cpp — M34-C2 Shape<Rank> 验证
//
// 覆盖: 布局断言 / 逐字 FNV 哈希抽查 (python 独立转录常量) / canonical_shape
// BST 插入-去重-红黑不变量 (逐插校验) / 等哈希异内容碰撞对 / 六秩容器 /
// OutputDef·ShapeFlag 重载 / crated_shape 不查重 / deserialize 位格式
// (mode0/1/2/3 + max/pad 覆盖 + 0xcccc 前缀 + 既有对象 + 负 id + DCrate
// 快慢两路)。
#define HNNX_SER_PRECISE 1
#include "hnnx/tens/shape.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hnnx/ir/op_def.hpp"
#include "hnnx/serialize/deserz.hpp"
#include "hnnx/serialize/serializer.hpp"

static int failures = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                       \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// 伪 Graph: 过渡 crate bump_ctl 位于 crate+0x48 (crate_records.cpp 约定)
// ---------------------------------------------------------------------------
alignas(16) static unsigned char gbuf[0x7000];
static unsigned char crate_arena[0x10000];
static unsigned char dcrate_arena[0x2000];

static ::Graph &graph()
{
    return *reinterpret_cast<::Graph *>(gbuf);
}
static unsigned char *embedded_crate()
{
    return gbuf + 0xd0;
}
struct TNode {
    TNode *l, *r, *up;
    bool black;
    uint32_t key;
    void *value;
};
struct TMap {
    TNode *begin;
    TNode *root; // end.__left_
    size_t size;
};
static void init_graph()
{
    std::memset(gbuf, 0, sizeof gbuf);
    std::memset(crate_arena, 0, sizeof crate_arena);
    gbuf[0x68b8] = 1; // selector → 内嵌 crate
    // 空树不变量: begin == end (Graph ctor 语义; 全零会令插入助手上行路径解引用 end->parent)
    for (unsigned r = 1; r <= 6; ++r) {
        TMap *const m = reinterpret_cast<TMap *>(gbuf + 0x53b0 + 0x18 * (r - 1));
        m->begin = reinterpret_cast<TNode *>(&m->root);
    }
    uint64_t *ctl = reinterpret_cast<uint64_t *>(embedded_crate() + 0x48);
    ctl[0] = reinterpret_cast<uint64_t>(crate_arena);
    ctl[1] = reinterpret_cast<uint64_t>(crate_arena + sizeof crate_arena);
}
static uint64_t record_count()
{
    uint64_t v;
    std::memcpy(&v, embedded_crate() + 0x40, 8);
    return v;
}

// 哈希确定性: 源对象尾部填充清零 (crate 记录语义)
template <size_t R> static Shape<R> &stable(Shape<R> &s)
{
    std::memset(reinterpret_cast<char *>(&s) + s.shplen(), 0, sizeof(s) - s.shplen());
    return s;
}

// ---------------------------------------------------------------------------
// 树布局镜像 (shape.cpp 内部 0x30 节点 / 0x18 容器)
// ---------------------------------------------------------------------------
static TMap *map_of(unsigned rank)
{
    return reinterpret_cast<TMap *>(gbuf + 0x53b0 + 0x18 * (rank - 1));
}

static int chk_sub(TNode *n, uint32_t *prev_key, bool *first)
{
    if (n == nullptr) return 1;
    int const hl = chk_sub(n->l, prev_key, first);
    if (*first || n->key >= *prev_key) {
        *prev_key = n->key;
        *first = false;
    } else {
        std::fprintf(stderr, "FAIL: 中序键回退\n");
        ++failures;
    }
    if (n->l && n->l->up != n) { std::fprintf(stderr, "FAIL: 左父指针\n"); ++failures; }
    if (n->r && n->r->up != n) { std::fprintf(stderr, "FAIL: 右父指针\n"); ++failures; }
    if (!n->black) {
        if (n->l && !n->l->black) { std::fprintf(stderr, "FAIL: 红红(左)\n"); ++failures; }
        if (n->r && !n->r->black) { std::fprintf(stderr, "FAIL: 红红(右)\n"); ++failures; }
    }
    int const hr = chk_sub(n->r, prev_key, first);
    if (hl != hr) { std::fprintf(stderr, "FAIL: 黑高\n"); ++failures; }
    return hl + (n->black ? 1 : 0);
}
static size_t count_sub(TNode *n)
{
    return n ? 1 + count_sub(n->l) + count_sub(n->r) : 0;
}
static void validate_map(unsigned rank)
{
    TMap *const m = map_of(rank);
    TNode *const end = reinterpret_cast<TNode *>(&m->root);
    if (m->root == nullptr) {
        CHECK(m->size == 0);
        CHECK(m->begin == end);
        return;
    }
    CHECK(m->root->up == end);
    CHECK(m->root->black);
    uint32_t pk = 0;
    bool first = true;
    chk_sub(m->root, &pk, &first);
    CHECK(m->size == count_sub(m->root));
    TNode *lm = m->root;
    while (lm->l != nullptr) lm = lm->l;
    CHECK(m->begin == lm);
}

// ---------------------------------------------------------------------------
// 哈希独立转录 (python 常量交叉核对)
// ---------------------------------------------------------------------------
static uint32_t ref_hash(void const *p, unsigned words, uint32_t salt)
{
    uint32_t const *w = static_cast<uint32_t const *>(p);
    uint32_t h = w[0] * 0x12401D1u;
    for (unsigned k = 1; k + 1 < words; ++k) h = (h + w[k]) * 0x12401D1u;
    return h + w[words - 1] + salt;
}

int main()
{
    // ---- 布局 ----
    static_assert(sizeof(Shape<1>) == 32 && sizeof(Shape<2>) == 48 && sizeof(Shape<3>) == 64);
    static_assert(sizeof(Shape<4>) == 80 && sizeof(Shape<5>) == 96 && sizeof(Shape<6>) == 112);
    static_assert(sizeof(hnnx::ShapeFlags) + 16 * 4 + 4 == 0x4c);

    // ---- 哈希抽查 (python: R1 dims=[5] → 0x8028d3b3; R4 {1,2,3,4} → 0xba6a37a0)
    {
        Shape<1> s1(std::array<size_t, 1>{5});
        stable(s1);
        CHECK(ref_hash(&s1, 7, 0x48D26249u) == 0x8028d3b3u);
        Shape<4> s4(std::array<size_t, 4>{1, 2, 3, 4});
        stable(s4);
        CHECK(ref_hash(&s4, 19, 0x8065DD8Cu) == 0xba6a37a0u);
    }

    // ---- canonical: 逐插校验 + 去重 ----
    init_graph();
    {
        Shape<4> arr[70];
        Shape<4> const *ptrs[70];
        uint32_t lcg = 12345;
        for (int i = 0; i < 70; ++i) {
            std::array<size_t, 4> d;
            for (int k = 0; k < 4; ++k) {
                lcg = lcg * 1103515245u + 12345u;
                d[k] = 1 + (lcg >> 16) % 97;
            }
            arr[i] = Shape<4>(d);
            stable(arr[i]);
            ptrs[i] = Shape<4>::canonical_shape(graph(), arr[i]);
            validate_map(4);
        }
        CHECK(map_of(4)->size == 70);
        uint64_t rc0 = record_count();
        for (int i = 0; i < 70; ++i) {
            Shape<4> const *const p = Shape<4>::canonical_shape(graph(), arr[i]);
            CHECK(p == ptrs[i]); // 去重命中, 不再分配
        }
        CHECK(record_count() == rc0);   // 无新记录
        CHECK(map_of(4)->size == 70);   // 树不变
    }

    // ---- 等哈希碰撞对 (python: Δw4 = −Δw2·M²) ----
    {
        init_graph();
        Shape<4> a(std::array<size_t, 4>{1, 2, 3, 4});
        Shape<4> b(std::array<size_t, 4>{2, 3677664097u, 3, 4});
        stable(a);
        stable(b);
        CHECK(ref_hash(&a, 19, 0x8065DD8Cu) == ref_hash(&b, 19, 0x8065DD8Cu)); // 前提: 同哈希
        Shape<4> const *pa = Shape<4>::canonical_shape(graph(), a);
        Shape<4> const *pb = Shape<4>::canonical_shape(graph(), b);
        CHECK(pa != pb); // memcmp 不同 → 双节点同键
        CHECK(map_of(4)->size == 2);
        validate_map(4);
        CHECK(Shape<4>::canonical_shape(graph(), a) == pa);
        CHECK(Shape<4>::canonical_shape(graph(), b) == pb); // 链上各自命中
        CHECK(map_of(4)->size == 2);
    }

    // ---- 六秩容器 ----
    {
        init_graph();
        for (unsigned r = 1; r <= 6; ++r) {
            for (unsigned i = 0; i < 8; ++i) {
                switch (r) { // 每秩构造同秩形并规范化
                case 1: {
                    Shape<1> s(std::array<size_t, 1>{i + 1});
                    stable(s);
                    Shape<1>::canonical_shape(graph(), s);
                    break;
                }
                case 2: {
                    Shape<2> s(std::array<size_t, 2>{i + 1, 3});
                    stable(s);
                    Shape<2>::canonical_shape(graph(), s);
                    break;
                }
                case 3: {
                    Shape<3> s(std::array<size_t, 3>{i + 1, 3, 5});
                    stable(s);
                    Shape<3>::canonical_shape(graph(), s);
                    break;
                }
                case 4: {
                    Shape<4> s(std::array<size_t, 4>{i + 1, 3, 5, 7});
                    stable(s);
                    Shape<4>::canonical_shape(graph(), s);
                    break;
                }
                case 5: {
                    Shape<5> s(std::array<size_t, 5>{i + 1, 3, 5, 7, 9});
                    stable(s);
                    Shape<5>::canonical_shape(graph(), s);
                    break;
                }
                case 6: {
                    Shape<6> s(std::array<size_t, 6>{i + 1, 3, 5, 7, 9, 11});
                    stable(s);
                    Shape<6>::canonical_shape(graph(), s);
                    break;
                }
                }
            }
            CHECK(map_of(r)->size == 8);
            validate_map(r);
        }
        // 各秩容器互不串扰
        for (unsigned r = 1; r <= 6; ++r) CHECK(map_of(r)->size == 8);
    }

    // ---- OutputDef 重载: dims=max=max_sizes, pad=0, flags=0 ----
    {
        init_graph();
        OutputDef def;
        std::memset(&def, 0, sizeof def);
        def.max_sizes[0] = 2;
        def.max_sizes[1] = 3;
        def.max_sizes[2] = 4;
        def.max_sizes[3] = 5;
        Shape<4> const *pd = Shape<4>::canonical_shape(graph(), def);
        Shape<4> manual(std::array<size_t, 4>{2, 3, 4, 5});
        stable(manual);
        Shape<4> const *pm = Shape<4>::canonical_shape(graph(), manual);
        CHECK(pd == pm); // 与手工等形同规范化点
        CHECK(std::memcmp(pd->dims.data(), manual.dims.data(), 32) == 0);
        CHECK(pd->pad[0] == 0 && pd->pad[3] == 0);
        CHECK(pd->flags == 0);
    }

    // ---- ShapeFlag 重载: 整形拷贝 + flags16 覆写 ----
    {
        init_graph();
        Shape<4> base(std::array<size_t, 4>{7, 8, 9, 10});
        stable(base);
        Shape<4> const *p0 = Shape<4>::canonical_shape(graph(), base);
        Shape<4> const *p1 = Shape<4>::canonical_shape(graph(), base, hnnx::ShapeFlag::constant);
        CHECK(p0 != p1);
        CHECK(p1->flags == 1);
        CHECK(std::memcmp(p1->dims.data(), base.dims.data(), 32) == 0);
        CHECK(Shape<4>::canonical_shape(graph(), base, hnnx::ShapeFlag::constant) == p1); // 旗标形自身去重
    }

    // ---- crated_shape: 不查重 ----
    {
        init_graph();
        Shape<4> s(std::array<size_t, 4>{4, 4, 4, 4});
        stable(s);
        Shape<4> const *c1 = Shape<4>::crated_shape(graph(), s);
        Shape<4> const *c2 = Shape<4>::crated_shape(graph(), s);
        CHECK(c1 != c2);
        CHECK(record_count() == 2);
        CHECK(std::memcmp(c1, &s, s.shplen()) == 0);
        CHECK(map_of(4)->size == 0); // 不触树
    }

    // ---- deserialize ----
    {
        // 载荷 (0xd941b3..0xd9425e 逐指令: 覆盖字与 mode 字按轴交错):
        //   id=1 | cw=0x3285 | 轴0 mode1 字 0x02030005 | 轴0 max 覆盖 77
        //   | 轴1 pad 覆盖 9 | 轴2 mode2 字 0x07112233 | 轴3 mode3 字 0x100
        //   | 复用 id=1 字 (二次调用消费该字后早退)
        // cw 位序: 每 4 位一轴 {mode:2, max:1, pad:1}:
        //   轴0 0b101=5 (mode1+max覆盖), 轴1 0b1000=8 (mode0+pad覆盖),
        //   轴2 0x2 (mode2), 轴3 0x3 (mode3)
        //   → 5 | 8<<4 | 2<<8 | 3<<12 = 0x3285 ✓
        uint32_t const data[] = {0x1u,  0x3285u,     0x02030005u, 77u,
                                 9u,    0x07112233u, 0x100u,      0x1u};
        unsigned char const *const dp = reinterpret_cast<unsigned char const *>(data);

        // -- DCrate 快路 --
        init_graph();
        {
            hnnx::Deserializer deser(reinterpret_cast<char const *>(dp), sizeof data, &graph());
            deser.setup_dcrate_out(dcrate_arena, sizeof dcrate_arena);
            Shape<4> const *slot = nullptr;
            Shape<4> const *const p = Shape<4>::deserialize(deser, &slot);
            CHECK(p >= reinterpret_cast<Shape<4> const *>(dcrate_arena)
                  && p < reinterpret_cast<Shape<4> const *>(dcrate_arena + sizeof dcrate_arena));
            CHECK(p->dims[0] == 5 && p->dims[1] == 1 && p->dims[2] == 0x112233 && p->dims[3] == 0x100);
            CHECK(p->max_dims[0] == 77 && p->max_dims[1] == 1 && p->max_dims[2] == 0x112233 && p->max_dims[3] == 0x100);
            CHECK(p->pad[0] == 2 && p->pad[1] == 9 && p->pad[2] == 7 && p->pad[3] == 0);
            CHECK(p->flags == 0);
            CHECK(record_count() == 0); // 快路不 bump
            // 同 id 再反序列化 → 既有对象; id 字仍被消费 (0xcfd4b2 无条件读)
            unsigned char const *cur0 = deser.read_cursor();
            Shape<4> const *const p2 = Shape<4>::deserialize(deser, &slot);
            CHECK(p2 == p);
            CHECK(deser.read_cursor() == cur0 + 4);
        }

        // -- 0xcccc 前缀 flags=1 + crate 慢路 --
        init_graph();
        {
            uint32_t const data2[] = {0x2u, 0xcccc0001u, 0x3285u, 0x02030005u, 77u, 9u, 0x07112233u, 0x100u};
            hnnx::Deserializer deser(reinterpret_cast<char const *>(data2), sizeof data2, &graph());
            deser.dcrate()->set_crate(*reinterpret_cast<hnnx::Crate *>(embedded_crate()));
            Shape<4> const *slot = nullptr;
            Shape<4> const *const p = Shape<4>::deserialize(deser, &slot);
            CHECK(p->flags == 1);
            CHECK(p->dims[0] == 5 && p->max_dims[0] == 77 && p->pad[1] == 9);
            CHECK(record_count() == 1); // 慢路 status≥0 → bump
            // 负 id (bit31): 表项作废重定义 → 新对象
            uint32_t const data3[] = {0x80000002u, 0xcccc0002u, 0x0000u}; // 全轴 mode0
            hnnx::Deserializer d3(reinterpret_cast<char const *>(data3), sizeof data3, &graph());
            d3.dcrate()->set_crate(*reinterpret_cast<hnnx::Crate *>(embedded_crate()));
            Shape<4> const *slot3 = nullptr;
            Shape<4> const *const p3 = Shape<4>::deserialize(d3, &slot3);
            CHECK(p3 != p);
            CHECK(p3->flags == 2);
            CHECK(p3->dims[0] == 1 && p3->dims[3] == 1 && p3->max_dims[3] == 1 && p3->pad[3] == 0);
        }

        // -- DCrate 段溢出 --
        {
            hnnx::Deserializer deser(reinterpret_cast<char const *>(dp), sizeof data, &graph());
            static unsigned char tiny[16];
            deser.setup_dcrate_out(tiny, sizeof tiny);
            Shape<4> const *slot = nullptr;
            bool threw = false;
            try {
                Shape<4>::deserialize(deser, &slot);
            } catch (hnnx::dcrate_seg_overflow_error const &) {
                threw = true;
            }
            CHECK(threw);
        }
    }

    if (failures == 0) {
        std::printf("test_m34_shape: ALL PASS\n");
        return 0;
    }
    std::printf("test_m34_shape: %d FAILURES\n", failures);
    return 1;
}
