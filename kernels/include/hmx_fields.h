/*
 * hmx_fields.h — V81 HMX mxmem Rt/Sp 字段编码表
 * =====================================================================
 * 来源: recon_hmx_representative.c (10 kernel 反汇编验证) +
 *       HMX_全部189算子精解报告.md + v81docs HMX PRM (80-N2040-62).
 *
 * HMX mxmem 指令 (activation.hf / weight.hf / mxmem:after.hf 等) 的
 * 第二寄存器 Rt 编码了几何 (spatial mask, channel stop, deep, dilate).
 *
 * 通用形式 (activation Rt, PRM Table 1):
 *   bit  4..0 : channel_stop  (默认 31 = 全 32 通道)
 *   bit  9..5 : SM subgroup    (0b11100 = 标准 4×8 spatial)
 *   bit 21..11: 几何扩展 (dY dilate 或 dC deep crouton 计数)
 *
 * 对齐检测: bitsclr(addr, 0x783) == 0 表示 2KB 对齐 (走 aligned 路径).
 */
#ifndef HVXHMX_HMX_FIELDS_H
#define HVXHMX_HMX_FIELDS_H

/* ============== 对齐常量 ============== */
#define HMX_ALIGN_2KB_MASK    0x783u    /* bitsclr(addr, this)==0 → 2KB aligned */
#define HMX_ALIGN_2KB         0x7FFu    /* dW = 2048-1 (满 crouton 范围) */
#define HMX_ALIGN_256B        0xFFu

/* ============== Spatial mask (bit 9..5 of act_rt) ============== */
/* YYYXX 编码: 行数×列数. 0b11100 = 标准 deep 卷积 4 行 × 8 列. */
#define HMX_SP_MASK           0b11100u
#define HMX_SP_LINEAR_1x32    0x10u     /* 1×32 linear (1x1 GEMM) */

/* ============== Activation Rt 编码 (PRM Table 1) ============== */
/* ACT_RT_BASIC: SM=0b11100, channel_stop=31, 无 dilate/deep.
 * 这是 recon kernel 1/2/4/6/8 通用的基础 act_rt. */
#define HMX_ACT_RT_BASIC      0x3FCu

/* ACT_RT_FULL: 全范围 (0x7FF), 配合 :deep 修饰符 (Phase0 验证路径). */
#define HMX_ACT_RT_FULL       0x7FFu

/* DY stride 标志 (N>1 路径, recon kernel 1 路径 A1/C1): BASIC | 0x70 */
#define HMX_ACT_RT_DY         (HMX_ACT_RT_BASIC | 0x70u)

/* ============== Dilate 编码 (bit 21..11 of act_rt) ============== */
/* dilate_rate 编码进 act_rt bit 21..11: act_rt = BASIC | (dilate << 11). */
static inline unsigned int hmx_act_rt_dilate(unsigned int basic, unsigned int dilate_rate)
{
    return basic | (dilate_rate << 11);
}

/* ============== Deep 编码 (bit 21..11 of act_rt, deep crouton) ============== */
/* deep: K>32 跨多 crouton 累加. dC = (K/32)-1 编码进 act_rt bit 21..11.
 * 需配合 activation_hf_mxmem_RR_deep 变体 (recon kernel 5). */
static inline unsigned int hmx_act_rt_deep(unsigned int basic, unsigned int K)
{
    unsigned int dC = (K / 32u) - 1u;
    return basic | (dC << 11);
}

/* ============== Weight Rt 编码 (PRM Table 2) ============== */
/* WGT_RT_2KB: dW=2047 (满 fp16 crouton 2KB 范围). recon 全 kernel 通用. */
#define HMX_WGT_RT_2KB        0x7FFu

/* wgt_rt for variable K (deep): dW = K*2 - 1 字节 (recon kernel 5). */
static inline unsigned int hmx_wgt_rt_bytes(unsigned int byte_len)
{
    return byte_len - 1u;
}

/* ============== 对齐检测辅助 ============== */
/* 返回 1 若 addr 2KB 对齐 (HMX aligned 路径). */
static inline int hmx_addr_aligned_2kb(const void *addr)
{
    return (((unsigned int)(uintptr_t)addr) & HMX_ALIGN_2KB_MASK) == 0u;
}

/* ============== Writeback 变体 (config bit4 = ReLU) ============== */
/* after.hf  (无 ReLU): Q6_mxmem_AR_after_hf(out, 0)
 * after:pos.hf (ReLU): Q6_mxmem_AR_after_pos_hf(out, 0)
 * before.hf (unaligned 前缀写): Q6_mxmem_AR_before_hf(out, 0) */

#endif /* HVXHMX_HMX_FIELDS_H */
