/*
 * 36_absoak — GENERIC_A/B fuzz+soak (问题书 §7.3 fuzz 与 soak 交付物)
 * =====================================================================
 * Part A: ring soak —— 随机轨迹 (xorshift32 种子 0xC0FFEE 可复现),
 *         门铃 ≥ 1e6, 每 97 消息注入一次 F2 门铃丢失 (re-kick 必须救回),
 *         零 FATAL / 零 invariant 违例 / 零数据错误。
 * Part B: btrack/bflush/dcache fuzz —— 200k 事件混合三写路径/convert/
 *         boundary/F-B1 绕钩子注入, 影子审计 (refcp 逐字节比对) 抓注入,
 *         置疑后永久 FULL, dc 统计对账。
 */
#include "hvxhmx_v23.h"
#include "example_util.h"
static uint32_t xorshift_state = 0xC0FFEE;
static uint32_t xorshift32(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; xorshift_state = x; return x;
}
static void flush_full_cb(void) { }
static void flush_range_inval_cb(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
static int convert_xor_cb(const uint8_t *src, uint64_t src_size, uint8_t *dst, uint64_t dst_cap, uint64_t *dst_size_out, void *user) {
    (void)user; if (dst_cap < src_size) return -1;
    for (uint64_t i = 0; i < src_size; i++) dst[i] = src[i] ^ 0xA5u;
    *dst_size_out = src_size; return 0;
}
static int convert_rev_cb(const uint8_t *src, uint64_t src_size, uint8_t *dst, uint64_t dst_cap, uint64_t *dst_size_out, void *user) {
    (void)user; if (dst_cap < src_size) return -1;
    for (uint64_t i = 0; i < src_size; i++) dst[i] = src[src_size - 1 - i];
    *dst_size_out = src_size; return 0;
}
static uint8_t ddr_src[8 * 64 * 1024];
static uint8_t slots[4 * 512 * 1024];
static uint8_t space[4 * 1024 * 1024];
static uint8_t refcp[4 * 1024 * 1024];

int main(void) {
    ex_open_result("36_absoak");
    /* ---- Part A ---- */
    uint64_t total_doorbells = 0, law_bad = 0, inv_bad = 0, inj_f2 = 0, rescue_fail = 0;
    uint64_t data_bad = 0;
    for (int i = 0; i < 8; i++) ex_fill_u8(ddr_src + i * 64 * 1024, 64 * 1024, i + 1, 251);
    uint32_t n_msg = 0;
    while (total_doorbells < 1000000 && n_msg < 30000) {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, 4, RING_MAX_DESCS_SLOT, 512 * 1024, slots);
        uint32_t ng = 8 + (xorshift32() % 57);
        uint32_t serial_us = 200 + (xorshift32() % 800);
        for (uint32_t i = 0; i < ng; i++) {
            uint64_t src = (uint64_t)(uintptr_t)(ddr_src + (i % 8) * 64 * 1024);
            if (ring_enqueue(r, src, 4096, 1, 4096, 0) != 0) { law_bad++; break; }
        }
        ring_on_serial_phase(r, serial_us);
        sim_advance((double)serial_us);
        for (uint32_t i = 0; i < ng; i++) {
            if (i == 2 && (n_msg % 97) == 0) { sim_adapter_fault(SIM_FAULT_F2_DOORBELL_LOST); inj_f2++; }
            uint64_t slot_out = 0;
            ring_need_result nr = ring_need(r, i, 60000, &slot_out);
            if (nr != RING_READY) { if (i == 2 && (n_msg % 97) == 0) rescue_fail++; law_bad++; break; }
            if ((i % 16) == 0) {
                if (memcmp((const void *)(uintptr_t)slot_out, ddr_src + (i % 8) * 64 * 1024, 4096) != 0) data_bad++;
            }
            ring_release(r, i);
        }
        if (ring_check_invariants(r) != 0) inv_bad++;
        sim_report rep; sim_adapter_report(&rep);
        if (rep.law != 0) law_bad++;
        ring_stats rs; ring_get_stats(r, &rs);
        total_doorbells += rs.n_doorbells;
        ring_destroy(r);
        n_msg++;
    }
    ex_check("A soak 门铃≥1e6", total_doorbells < 1000000, 0);
    ex_check("A soak 零FATAL零invariant", law_bad != 0 || inv_bad != 0, 0);
    ex_check("A soak F2 注入全救回", inj_f2 == 0 || rescue_fail != 0, 0);
    ex_check("A soak 数据零错误", data_bad != 0, 0);
    ex_log("A: doorbells=%llu msgs=%u law=%llu inv=%llu inj=%llu rescue_fail=%llu data_bad=%llu",
           (unsigned long long)total_doorbells, n_msg, (unsigned long long)law_bad,
           (unsigned long long)inv_bad, (unsigned long long)inj_f2, (unsigned long long)rescue_fail,
           (unsigned long long)data_bad);
    /* ---- Part B ---- */
    btrack_ctx *bt = bt_create(4 * 1024 * 1024, 64, BT_MODE_ATOMIC);
    bflush_ctx *bf = bf_create(bt, flush_full_cb, flush_range_inval_cb, 50000, 1000);
    dcache_ctx *dc = dc_create(bt, 32, 4 * 1024 * 1024, 0);
    uint64_t reg_bad = 0;
    dc_format fmt_xor = { 1, 1, convert_xor_cb, NULL };
    dc_format fmt_rev = { 2, 1, convert_rev_cb, NULL };
    if (dc_register_format(dc, &fmt_xor) != 0) reg_bad++;
    if (dc_register_format(dc, &fmt_rev) != 0) reg_bad++;
    uint32_t buf_ids[8];
    for (int i = 0; i < 8; i++) bt_register_buffer(bt, (uint64_t)i * 256 * 1024, 256 * 1024, &buf_ids[i]);
    for (int i = 0; i < 8; i++) { ex_fill_u8(space + i * 256 * 1024, 256 * 1024, i + 10, 251); }
    memcpy(refcp, space, 4 * 1024 * 1024);
    uint64_t event_count = 0, inj_b1 = 0, audit_catch = 0, unexplained_diff = 0;
    uint64_t b3_bad = 0, n_hit_obs = 0, n_miss_obs = 0, n_retry_obs = 0;
    while (event_count < 200000) {
        uint32_t op = xorshift32() % 100;
        uint32_t bidx = xorshift32() % 8;
        uint64_t base = (uint64_t)bidx * 256 * 1024;
        uint64_t off = base + (uint64_t)(xorshift32() % (256 * 1024 / 64)) * 64;
        uint64_t room = 256 * 1024 - (off - base);
        uint64_t smax = room < 8192 ? room : 8192;
        uint64_t size = 64 + (xorshift32() % (smax - 63));
        size &= ~63ULL; if (!size) size = 64;
        if (op < 50) {
            bt_mark_cpu_write(bt, off, size);
            for (uint64_t a = off; a < off + size; a += 64) { space[a] ^= 0x5A; refcp[a] = space[a]; }
        } else if (op < 65) {
            uint64_t tok = 0;
            if (bt_mark_dma_write(bt, off, size, &tok) == 0) {
                for (uint64_t a = off; a < off + size; a += 64) { space[a] ^= 0x3C; refcp[a] = space[a]; }
                bt_dma_complete(bt, tok);
            }
        } else if (op < 75) {
            bt_mark_peer_write(bt, off, size);
            for (uint64_t a = off; a < off + size; a += 64) { space[a] ^= 0x77; refcp[a] = space[a]; }
        } else if (op < 85) {
            const uint8_t *out = NULL; uint64_t out_size = 0;
            int rc = dc_get_or_convert(dc, buf_ids[bidx], space + bidx * 256 * 1024, 256 * 1024,
                                       (uint32_t)(1 + xorshift32() % 2), &out, &out_size);
            if (rc == 0) n_miss_obs++; else if (rc == 1) n_hit_obs++; else n_retry_obs++;
        } else if (op < 95) {
            bflush_report rep;
            bf_boundary(bf, &rep);
            if (bt_is_suspect(bt) && rep.used_full != 1) b3_bad++;
        } else {
            for (uint64_t a = off; a < off + size; a += 64) space[a] ^= 0x99;
            inj_b1++;
        }
        event_count++;
        if ((event_count % 4096) == 0) {
            if (memcmp(space, refcp, 4 * 1024 * 1024) != 0) {
                if (!bt_is_suspect(bt)) { audit_catch++; bt_flag_suspect(bt); }
                if (inj_b1 == 0) unexplained_diff++;
                memcpy(refcp, space, 4 * 1024 * 1024);
            }
        }
    }
    {
        bflush_report rep;
        bf_boundary(bf, &rep);
        if (bt_is_suspect(bt) && rep.used_full != 1) b3_bad++;
    }
    dc_stats dcs; dc_get_stats(dc, &dcs);
    ex_check("B fuzz 事件≥200k", event_count < 200000, 0);
    ex_check("B F-B1 注入发生", inj_b1 == 0, 0);
    ex_check("B 影子审计捕获", audit_catch == 0, 0);
    ex_check("B 置疑后永久FULL", b3_bad != 0, 0);
    ex_check("B 统计对账", dcs.hits != n_hit_obs || dcs.misses != n_miss_obs || reg_bad != 0, 0);
    ex_check("B 无审计外差异", unexplained_diff != 0, 0);
    ex_log("B: ev=%llu inj=%llu catch=%llu b3bad=%llu hit=%llu miss=%llu retry=%llu reg_bad=%llu unexpl=%llu",
           (unsigned long long)event_count, (unsigned long long)inj_b1, (unsigned long long)audit_catch,
           (unsigned long long)b3_bad, (unsigned long long)n_hit_obs, (unsigned long long)n_miss_obs,
           (unsigned long long)n_retry_obs, (unsigned long long)reg_bad, (unsigned long long)unexplained_diff);
    dc_destroy(dc); bf_destroy(bf); bt_destroy(bt);
    return ex_summary();
}
