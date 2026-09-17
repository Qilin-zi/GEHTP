// ============================================================================
// G4 反汇编保真实现 —— 见 include/hnnx/vtcm/spillfill_g4.hpp 头注
// 证据基线: audit_verify/reports/G4_spillfill_disasm.md (2026-08-31)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// grdep 重写簇 / 0x129f0b0 设计主体 / SFCD 写侧 / type nibble 语义为 G4b 遗留，
// 禁止臆测补齐。
// ============================================================================
#include "hnnx/vtcm/spillfill_g4.hpp"
#include <cstdio>
#include <stdexcept>

namespace hnnx {

// ============================================================================
// §A SFCD dump（内联区域 0x1013a90，容器 = serialize_blob_epilogue@0xFDB330）
// ============================================================================
namespace {

// 头行/各记录行的 %s 前缀: verbose → "        #"（8 空格+#，@0x4626db5），
// 非 verbose → 空串 [0x1013ab3 前的 cmov 选择]
const char* g4_prefix(bool verbose) { return verbose ? "        #" : ""; }

// 箭头: fill → "->"@0x55a4d41，spill → "<-"@0x4626f8f [0x1013b37-0x1013b48 cmovneq]
const char* g4_arrow(bool is_fill) { return is_fill ? "->" : "<-"; }

// 记录行 fmt 选择: is_fill → "%s     %d fill from mempool %d, offset 0x%x\n"@0x4626ef1
//                 否则    → "%s     %d spill to mempool %d, offset 0x%x\n"@0x4626f1e
// [0x1013b51-0x1013b5f cmovneq]
const char* g4_block_rec_fmt(bool is_fill) { return is_fill ? "fill from" : "spill to"; }

} // namespace

std::string g4_err_2095(uint32_t bad_hdr) {
    // "%s:2095::ERROR:Bad SFCD record header %08X\n" @0x4626e9e，TU@0x46269f1，tag 空@0x4628a0e
    char buf[128];
    std::snprintf(buf, sizeof buf,
                  "grdep_spillfill.cc:2095::ERROR:Bad SFCD record header %08x",
                  bad_hdr);
    return buf;
}

std::string g4_err_2095_file(uint32_t bad_hdr) {
    // "%s   !!!!  Bad SFCD record header %08X" @0x4626eca [0x1013f20]
    char buf[128];
    std::snprintf(buf, sizeof buf, "   !!!!  Bad SFCD record header %08x", bad_hdr);
    return buf;
}

std::string g4_dump_sfcd(const uint32_t* sfcd, bool is_fill, bool verbose,
                         G4SfcdWalk* walk) {
    std::string out;
    char buf[256];
    const char* pre = g4_prefix(verbose);
    const char* arrow = g4_arrow(is_fill);

    if (walk) *walk = G4SfcdWalk{};

    // ---- 头部 3 词 ----
    const uint32_t total_bytes = sfcd[0];                     // [+0] [0x1013ab0]
    const int32_t  ckpt_idx    = (int32_t)sfcd[1];            // [+4] [0x1013af7]
    const uint32_t rec_count   = sfcd[2] & 0xffffffu;         // [+8]&0xFFFFFF [0x1013b12]
    if (walk) {
        walk->hdr.total_bytes = total_bytes;
        walk->hdr.checkpoint_index = ckpt_idx;
        walk->hdr.record_count = rec_count;
    }

    // 头行 "%s---- SFCD for %s @ %p, %d bytes\n" @0x4626dbf（标签 spill@0x55a6706/fill@0x39a98f0）
    std::snprintf(buf, sizeof buf, "%s---- SFCD for %s @ %p, %d bytes\n",
                  pre, is_fill ? "fill" : "spill", (const void*)sfcd, (int)total_bytes);
    out += buf;

    // checkpoint_index 非 0 才打印 "%s    checkpoint_index = %d\n" @0x4626de2
    if (ckpt_idx != 0) {                                      // [0x1013af7-0x1013b0d]
        std::snprintf(buf, sizeof buf, "%s    checkpoint_index = %d\n", pre, ckpt_idx);
        out += buf;
    }

    // 零记录: "%s    ?? zero records??\n" @0x4626dff → 直接返回（fprintf 尾调用）
    if (rec_count == 0) {                                     // [0x1013b12-0x1013b1e]
        std::snprintf(buf, sizeof buf, "%s    ?? zero records??\n", pre);
        out += buf;
        if (walk) walk->zero_records = true;
        return out;                                           // [0x1013ec3-0x1013ed1]
    }

    // ---- 外层记录循环 [0x1013b80-0x1013e2f]，r15 = 游标 ----
    const uint32_t* cur = sfcd + 3;                           // 记录自 +0xc [0x1013b33]
    uint32_t blocks_total = 0;                                // [rsp+0x14] [0x1013b6f]

    for (uint32_t i = 0; i < rec_count; ++i) {
        const uint32_t w0 = *cur;                             // [0x1013b80]
        if ((int32_t)w0 >= 0) {
            // ==== (a) tcm 块记录 ====
            const uint32_t pool     = w0 >> 16;               // [0x1013b9a shrl $0x10]
            const uint32_t nblocks  = w0 & 0xffffu;           // [0x1013ba1]
            const uint32_t blob_off = cur[1];                 // w1 [0x1013b93]
            // "%s     %d fill from/spill to mempool %d, offset 0x%x\n"
            std::snprintf(buf, sizeof buf, "%s     %u %s mempool %u, offset 0x%x\n",
                          pre, nblocks, g4_block_rec_fmt(is_fill), pool, blob_off);
            out += buf;
            cur += 2;                                         // [0x1013bc1 addq $8]
            if (nblocks == 0) continue;                       // 合法空记录 [0x1013bcc je]

            uint32_t mp_cur = blob_off;                       // r13d = w1 起步
            for (uint32_t b = 0; b < nblocks; ++b) {
                const uint32_t w   = cur[0];
                const uint32_t len = cur[1];
                const uint32_t tcm_off = w & ~0xfu;           // [0x1013bfe andl $-0x10]
                const uint32_t type    = w & 0xfu;            // [0x1013c0b andl $0xf]
                cur += 2;
                blocks_total += len;                          // [0x1013c05]
                if (verbose) {
                    // 元组 "        (%d, 0x%x, 0x%x, 0x%x, 0x%x),  #" @0x4626f4a
                    // 5 参 = (pool, mp_cur, tcm_off, len, type)，type 走栈传参 [0x1013c0e]
                    std::snprintf(buf, sizeof buf,
                                  "        (%u, 0x%x, 0x%x, 0x%x, 0x%x),  #",
                                  pool, mp_cur, tcm_off, len, type);
                    out += buf; out += "\n";
                }
                // 行 "0x%X %s tcm 0x%X  %d bytes" @0x4626f73（fmt 无换行，行尾由模型补）
                std::snprintf(buf, sizeof buf, "0x%X %s tcm 0x%X  %u bytes",
                              mp_cur, arrow, tcm_off, len);
                out += buf; out += "\n";
                // 非 verbose 紧凑循环先 fputs 8 空格 @0x46250f7 [0x1013c70-0x1013cc2]——
                // 模型并入行前缀（记录级取舍，不影响断言面）
                // 游标步进: mp_cur += (len + 0x3f) & ~0x3f  [0x1013c58-0x1013c5e]
                mp_cur += g4_round_up_64(len);
            }
        } else if ((w0 >> 24) == 0x81) {
            // ==== (c) set_progress [0x1013cd6 cmpb $0x81 → 0x1013dd5] ====
            const uint32_t mb_idx = w0 & 0xffffffu;
            const uint32_t val    = cur[1];
            cur += 2;
            if (verbose) {
                // "        ('set_progress', %d, %d)," @0x4626e59
                std::snprintf(buf, sizeof buf, "        ('set_progress', %u, %u),",
                              mb_idx, val);
                out += buf; out += "\n";
            }
            // "%s     set_progress: mb[%d] := %d\n" @0x4626e7b
            std::snprintf(buf, sizeof buf, "%s     set_progress: mb[%u] := %d\n",
                          pre, mb_idx, (int32_t)val);
            out += buf;
        } else if ((w0 >> 24) == 0x80) {
            // ==== (b) wait_for_progress [0x1013cde cmpb $0x80 → 0x1013ce6] ====
            ++cur;                                            // r15 += 4
            if (w0 & 0x10000u) {                              // bit16 对形态 [0x1013d01]
                const uint32_t n = (w0 & 0xffffu) >> 1;       // [0x1013cf7 shrl]
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t mb_idx = cur[0], val = cur[1];
                    cur += 2;
                    if (verbose) {
                        // "        ('wait_for_progress', %d, %d)," @0x4626e18
                        std::snprintf(buf, sizeof buf,
                                      "        ('wait_for_progress', %u, %u),",
                                      mb_idx, val);
                        out += buf; out += "\n";
                    }
                    // "%s wait for mb[%d] >= %d" @0x4626e3f
                    std::snprintf(buf, sizeof buf, "%s wait for mb[%u] >= %d",
                                  pre, mb_idx, (int32_t)val);
                    out += buf; out += "\n";
                }
            } else {                                          // 单词形态
                const uint32_t n = w0 & 0xffffu;
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t w = *cur++;
                    const uint32_t mb_idx = w & 0xffffffu;    // 24 位 id
                    const uint32_t val    = w >> 24;          // 8 位 val
                    if (verbose) {
                        std::snprintf(buf, sizeof buf,
                                      "        ('wait_for_progress', %u, %u),",
                                      mb_idx, val);
                        out += buf; out += "\n";
                    }
                    std::snprintf(buf, sizeof buf, "%s wait for mb[%u] >= %d",
                                  pre, mb_idx, (int32_t)val);
                    out += buf; out += "\n";
                }
            }
        } else {
            // ==== 非法头: 双重打印后立即返回 [0x1013eff-0x1013f42] ====
            if (walk) walk->errors.push_back(g4_err_2095(w0));
            out += g4_err_2095(w0); out += "\n";
            out += pre; out += g4_err_2095_file(w0); out += "\n";
            if (walk) walk->cursor_bytes = (uint64_t)((const unsigned char*)cur -
                                                      (const unsigned char*)sfcd);
            return out;                                       // 无汇总行、无尾校验
        }
    }

    // ---- 汇总行 "%s    ---> total 0x%X (%d) bytes\n" @0x4626f92 [0x1013e73-0x1013e8e]
    std::snprintf(buf, sizeof buf, "%s    ---> total 0x%X (%u) bytes\n",
                  pre, blocks_total, blocks_total);
    out += buf;

    // ---- 尾校验: 游标 − 基址 != [+0] + 4 → 坏长度 [0x1013e93-0x1013ea3] ----
    const uint64_t cursor_bytes = (uint64_t)((const unsigned char*)cur -
                                             (const unsigned char*)sfcd);
    if (walk) {
        walk->cursor_bytes = cursor_bytes;
        walk->blocks_total = blocks_total;
    }
    if (cursor_bytes != (uint64_t)total_bytes + 4u) {
        const uint64_t words = (cursor_bytes - ((uint64_t)total_bytes + 4u)) >> 2;
        // sarq $2 [0x1013ed6]；"%s    ** Bad length: %zd words in all" @0x4626fb4
        if (walk) { walk->bad_length = true; walk->bad_length_words = words; }
        std::snprintf(buf, sizeof buf, "%s    ** Bad length: %zd words in all",
                      pre, (size_t)words);
        out += buf; out += "\n";
    }
    return out;
}

// ============================================================================
// §B slc 序列化侧（0x1294570）
// ============================================================================
std::vector<G4TaggedField> g4_serialize_slc_area(const G4SlcArea& area) {
    std::vector<G4TaggedField> f;
    auto add_key = [&f](const std::string& k) {
        G4TaggedField t;
        t.key = k;
        f.push_back(t);
        return &f.back();   // 仅注释用：键写出时 OR G4_STRING_TAG [0x1294605]
    };
    (void)add_key;

    auto emit_u = [&f](const char* key, uint64_t v, int tag) {
        G4TaggedField t;
        t.key = key; t.uval = v; t.value_tag = tag;
        f.push_back(t);
    };
    auto emit_s = [&f](const char* key, const std::string& v) {
        G4TaggedField t;
        t.key = key; t.sval = v; t.is_string = true;
        f.push_back(t);
    };
    auto emit_b = [&f](const char* key, bool v) {
        G4TaggedField t;
        t.key = key; t.uval = v ? G4_TAG_BOOL_TRUE : G4_TAG_BOOL_FALSE;
        t.value_tag = (int)(v ? G4_TAG_BOOL_TRUE : G4_TAG_BOOL_FALSE); // movw $0xa; sbbw $0
        f.push_back(t);
    };

    // 区域名 "SLC_spillfill_area_" + ostream 十六进制 "0x%llx" [0x12948d7-0x1294920]
    {
        char name[64];
        std::snprintf(name, sizeof name, "SLC_spillfill_area_0x%llx",
                      (unsigned long long)area.area_id);
        emit_s("__area_name__", name);            // 名字键（键串 OR string_tag）
    }
    emit_u("runlist_idx", (uint64_t)(int64_t)area.runlist_idx, 0);   // 值标签未核（G4b）
    emit_u("memgroup_tags", 0, 0);                // 值编码未核 → 占位
    emit_b("is_fill", area.is_fill);              // [0x1294b6b-0x1294ba9]
    emit_b("is_multi_nsp", area.is_multi_nsp);    // [0x1294bb0-0x1294bee]
    emit_u("dma_checkpoint", (uint64_t)(int64_t)area.dma_checkpoint,
           (int)G4_TAG_I32_DMA_CHECKPOINT);       // tag 0xe [0x1294bf5-0x1294c35]
    emit_u("nsp_id", (uint64_t)(int64_t)area.nsp_id, 0);
    emit_u("records", area.records.size(), 0);

    // 逐记录（0x40 步长 [0x1294cba]），rec_type 三路分派 [0x1294cd7/0x1294cec/0x1294d14]
    for (const auto& r : area.records) {
        switch (r.rec_type) {
        case G4_REC_SPILLFILL:
            emit_s("rec_type", "spillfill");      // 值是字符串（已核）
            emit_u("ddr_pool", r.ddr_pool, 0);
            emit_u("sf_offset", r.sf_offset, 0);
            emit_u("total_copy", r.total_copy, 0);
            for (const auto& c : r.copies) {      // 12B 元素向量 @+0x20
                emit_u("tcm_offset", c.tcm_offset, 0);
                emit_u("copy_len", c.copy_len, 0);
                emit_u("cache_hints", c.cache_hints, 0);
            }
            break;
        case G4_REC_WAITFOR:
            emit_u("rec_type", G4_REC_WAITFOR, 0); // 值形态未核（G4b）
            for (const auto& p : r.pairs) {
                emit_u("mb_idx", p.mb_idx, 0);
                emit_u("wait_for_val", p.val, 0);
            }
            break;
        case G4_REC_SETPROGRESS:
            emit_u("rec_type", G4_REC_SETPROGRESS, 0);
            emit_u("mb_idx", r.mb_idx, 0);
            emit_u("value_to_set", r.value_to_set, 0);
            break;
        default:
            break;                                // 其它值直接跳过 [0x129533e]
        }
    }
    return f;
}

// ============================================================================
// §C 分配面
// ============================================================================
uint64_t g4_set_spillfill_size(G4FancyAllocator& fa, uint32_t sizes[3]) {
    uint64_t total = 0;
    for (int i = 0; i < 3; ++i) {                 // 三槽全展开 [0xf4cd09/+0xf4cd2d/+0xf4cd53]
        if (sizes[i] == 0) continue;              // [0xf4cd0b]
        sizes[i] = g4_round_up_64k(sizes[i]);     // 原地 64K 取整 [0xf4cd14-0xf4cd1e]
        total += (uint64_t)sizes[i] + 0x10000u;   // 每槽多加 64KB [0xf4cd21-0xf4cd27]
    }
    if (total > 0xffffff00u)                      // [0xf4cd75 ja]
        throw std::runtime_error("oversize mem pool");   // @0x461b127
    // allocate_new_pool(align=0x10000, total, pool_id=2) [0xf4cd89-0xf4cd9c]
    G4PoolDesc pd;
    pd.size = (uint32_t)total;
    fa.pools.push_back(pd);                       // 池表（步长 0x30）追加
    fa.spillfill_pool = fa.pools.size();          // +0xa0 句柄（模型 = 下标+1）
    fa.slot_sizes[0] = sizes[0];                  // +0xa8 [0xf4cda1]
    fa.slot_sizes[1] = sizes[1];                  // +0xac
    fa.slot_sizes[2] = sizes[2];                  // +0xb0 [0xf4cdc1]
    return total;
}

uint64_t g4_set_spillfill_shared_size(G4FancyAllocator& fa, uint64_t size,
                                      bool env_far_enabled, uint32_t env_far_mb) {
    if (size == 0) return 0;                      // [0xf4ce34]
    fa.shared_size_290 = size;                    // +0x290 [0xf4ce43]
    const uint64_t sz = g4_round_up_64k((uint32_t)size);   // [0xf4ce4a-0xf4ce51]
    G4PoolDesc pd;                                // allocate_new_pool(...,2,...) [0xf4ce5f]
    pd.size = (uint32_t)sz;                       // +0x10 [0xf4ce8c]
    pd.flags |= G4_POOL_SHARED;                   // bit0 [0xf4ce98 orl $1]
    if (env_far_enabled &&                        // env byte@+0x6143 [0xf4cea0]
        sz >= ((uint64_t)env_far_mb << 20))       // u32@0x6148 << 20 [0xf4ceb3 shll $0x14]
        pd.flags |= G4_POOL_FAR;                  // bit4 [0xf4cec2 orl $0x10]
    fa.pools.push_back(pd);
    return sz;
}

bool g4_is_shared_spillfill(uint32_t pool, uint16_t flags, bool env_shared_set) {
    return pool == G4_SPILLFILL_POOL_ID           // [0xd8b552 cmpl $0x2]
        && (flags & G4_POOL_SHARED) == 0          // [0xd8b556 testb $1 → 要求未置]
        && env_shared_set;                        // env +0x5c9c [0xd8b55e-0xd8b567]
}

bool g4_can_mempool_be_far(const G4PoolDesc& pd, bool env_far_enabled,
                           uint32_t env_far_mb) {
    return env_far_enabled                        // 谓词版同判据 [0xf4cee0-0xf4cefe]
        && (uint64_t)pd.size >= ((uint64_t)env_far_mb << 20);
}

// ---- grdep 三槽填法（调用点 0x1010c09-0x1010cc5）----
void g4_fill_slots_multi(const uint32_t peaks[3], uint32_t out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = peaks[i] ? g4_round_up_64k(peaks[i]) : 0;   // [0x1010c1e-0x1010c54]
}

void g4_fill_slots_single(const uint32_t peaks[3], uint32_t out[3]) {
    out[0] = out[1] = out[2] = 0;
    if (peaks[0] == 0 && peaks[1] == 0 && peaks[2] == 0)
        return;                                   // 三峰值和 0 → 不分配 [0x1010c64]
    // argmax: 峰值2 严格大于 max(峰值0,峰值1) → 槽 2；
    // 否则 (峰值1 > 峰值0) ? 1 : 0（并列归 0）[0x1010c6f-0x1010c8d]
    uint32_t slot;
    if (peaks[2] > (peaks[0] > peaks[1] ? peaks[0] : peaks[1]))
        slot = 2;
    else
        slot = (peaks[1] > peaks[0]) ? 1 : 0;
    out[slot] = g4_round_up_64k(peaks[slot]);     // 仅获胜槽 [0x1010c90-0x1010caa]
}

// ============================================================================
// §D 桩与检查点 op
// ============================================================================
int g4_insert_spill_fill(std::string& log) {
    // qnndsp_log(0, fmt@0x462adf1, "insert_spillfill.cc"@0x462ae1f, ""@0x4628a0e)
    // [0x106d7d1-0x106d7ea]；return -1 [0x106d7ef movl $-1]
    log += "insert_spillfill.cc:17::ERROR:insert_spill_fill not supported\n";
    return -1;
}

G4CheckpointOp g4_make_dma_checkpoint_op(uint64_t graph, uint64_t opid,
                                         uint32_t mb_idx, bool is_set) {
    // new 0x18B op；vtable 二选一 [0xd95af2 testl %ebp / 0xd95b04 / 0xd95b2a]
    G4CheckpointOp op;
    op.vtable = is_set ? 0x5ec2488ull : 0x5ec2568ull;
    op.mb_idx = mb_idx;                           // +0x8 [0xd95b1x]
    op.value  = 0;                                // +0x10 构造时恒 0
    op.opid   = opid;
    op.owned  = false;                            // insert_op 第 3 参 [0xd95b57]
    (void)graph;
    return op;
}

} // namespace hnnx
