#!/usr/bin/env python3
"""emit_gdn_synth.py — 009 T022 合成单层 wtop 发射器（004 手写 wtop 发射器法）

手组 v3 blob（不经过 hnnx 编译器），含单个 OP_GDN_BLOCK_F16=37 融合块 op，
几何走 gdn 尾挂 footer（ExtraGdnBlock 十元组），输入 xqkv 经 EXT_IN 槽注入，
conv_w/beta/g 作内联权重槽。供设备 42_gehtp_runner 实跑 + check_3way 对拍
T011 host 参考（ab_standalone A 侧分立链）。

线格式（全小端，与 kernels/include/oplist_parse.h 一致）:
  header 16B: "WTOP" + u16 ver=3 + u16 endian 0x1234 + u32 n_slots + u32 n_ops
  slot[n] × 24B: {len, count, off_lo, off_hi, addr, rsv}
  op[n]: {u16 opcode, u16 n_args, args u32×n_args}
  annot footer: n_ops × u32 (engine 注解; 标量=0)
  gdn footer: [magic 0x584E4447][count][count × {op_index + 10×u32}]
  weight 区: 128B 对齐后内联权重

用法:
  emit_gdn_synth.py fused <T> <indir> <out.wtop>
"""
import argparse
import struct
import sys

H, D, DC, CK = 16, 128, 4, 32
DI = 3 * H * D   # 6144
HD = H * D       # 2048
EXT_IN = 0xFFFFFFFF
GDN_MAGIC = 0x584E4447
WT_GDN_EXTRA_REC = 44

# 生产形闭式预算（state H·D·D·4 + chunk 缓冲 Σ，128 对齐）——exec 走 malloc 不消费
VTCM_BUDGET = 3024896


def align128(x):
    return (x + 127) & ~127


def slot(blob, length, count, off, addr):
    """追加 24B slot 记录 (v2/v3)。"""
    blob += struct.pack("<IIIII", length, count, off & 0xFFFFFFFF, off >> 32, addr)
    blob += struct.pack("<I", 0)  # reserved
    return blob


def emit_fused(T, indir, outpath):
    xqkv = open(f"{indir}/in_xqkv.f16.raw", "rb").read()
    cw = open(f"{indir}/in_conv_w.f16.raw", "rb").read()
    beta = open(f"{indir}/in_beta.f16.raw", "rb").read()
    g = open(f"{indir}/in_g.f16.raw", "rb").read()
    assert len(xqkv) == T * DI * 2, f"xqkv {len(xqkv)} != {T*DI*2}"
    assert len(cw) == DC * DI * 2
    assert len(beta) == T * H * 2
    assert len(g) == T * H * 2

    off_cw = 0
    off_beta = align128(len(cw))
    off_g = align128(off_beta + len(beta))

    b = bytearray()
    b += b"WTOP"
    b += struct.pack("<H", 3)          # ver=3 (gdn footer 仅 v3 解析)
    b += struct.pack("<H", 0x1234)     # endian chk
    b += struct.pack("<I", 4)          # n_slots
    b += struct.pack("<I", 1)          # n_ops

    # slot0 = EXT_IN xqkv（len 仅作校验用, 注入后 data 取 g_ext_in）
    slot(b, 128, 64, 0, EXT_IN)
    # slot1 = conv_w 内联权重
    slot(b, len(cw), DC * DI, off_cw, 0)
    # slot2 = beta
    slot(b, len(beta), T * H, off_beta, 0)
    # slot3 = g
    slot(b, len(g), T * H, off_g, 0)

    # op0: GDN_BLOCK_F16=37, arity=5 [xqkv, conv_w, beta, g, y]
    b += struct.pack("<HH", 37, 5)
    for a in (0x8000 | 0, 0x8000 | 1, 0x8000 | 2, 0x8000 | 3, 0):
        b += struct.pack("<I", a)

    # annot footer: 1 × u32 = 0 (标量引擎; 当前 gdn_block_f16 标量 malloc)
    b += struct.pack("<I", 0)

    # gdn footer: [magic][count=1][op_index + 10×u32]
    b += struct.pack("<II", GDN_MAGIC, 1)
    b += struct.pack("<I", 0)           # op_index (全局 op 序号)
    b += struct.pack("<I", T)           # t
    b += struct.pack("<I", T)           # ntok
    b += struct.pack("<I", CK)          # c (chunk 实际 token 数 = min(T, chunk))
    b += struct.pack("<I", H)           # h
    b += struct.pack("<I", D)           # d
    b += struct.pack("<I", DC)          # d_conv
    b += struct.pack("<I", CK)          # chunk
    b += struct.pack("<I", VTCM_BUDGET) # vtcm_budget_bytes
    b += struct.pack("<I", 0xFFFFFFFF)  # state_idx (prefill 瞬态)
    b += struct.pack("<I", 1)           # contract_ver

    # weight 区 128B 对齐
    while len(b) % 128:
        b += b"\x00"
    w = bytearray()
    w += cw
    while len(w) < off_beta:
        w += b"\x00"
    w += beta
    while len(w) < off_g:
        w += b"\x00"
    w += g
    b += w

    open(outpath, "wb").write(b)
    print(f"[emit] fused T={T} slots=4 ops=1 bytes={len(b)} -> {outpath}")


def emit_discrete(T, indir, outpath):
    """分立序列 blob（对照 T022(b) 计时）: conv(19)+sigmoid(8)+mul(2)+
    split×3(16)+transpose×3(10)+delta(30)+transpose(10) = 11 op。
    与 fused 同输入 xqkv/conv_w/beta/g, 同输出 y[T][HD] f16（token-major）。
    delta 需状态槽（addr=WT_SLOT_STATE, 表=[n=1][H·D·D·4 字节]），
    state_ref=0x6000|0。全 op annot=0（标量引擎，避免 engine deviation）。
    out_temp=10（末 transpose 的 y_t）。"""
    xqkv = open(f"{indir}/in_xqkv.f16.raw", "rb").read()
    cw = open(f"{indir}/in_conv_w.f16.raw", "rb").read()
    beta = open(f"{indir}/in_beta.f16.raw", "rb").read()
    g = open(f"{indir}/in_g.f16.raw", "rb").read()
    assert len(xqkv) == T * DI * 2, f"xqkv {len(xqkv)} != {T*DI*2}"
    assert len(cw) == DC * DI * 2
    assert len(beta) == T * H * 2
    assert len(g) == T * H * 2

    # beta/g 输入是 [T][H] (token-major, ab_standalone 口径), 但 op 30
    # delta_chunk_f16 读 [H][ntok] (头-major, bh=h*ntok+t)。fused op 37 读
    # [T][H] (beta[(t)*H+h])。分立序列须 [T][H]→[H][T] 重排对齐 fused。
    def th_to_ht(data):
        v = struct.unpack(f"<{T*H}H", data)
        out = [0] * (T * H)
        for t in range(T):
            for h in range(H):
                out[h * T + t] = v[t * H + h]
        return struct.pack(f"<{T*H}H", *out)
    beta = th_to_ht(beta)
    g = th_to_ht(g)

    STATE_ADDR = 0xFFFFFFFB
    STATE_BYTES = H * D * D * 4  # 16*128*128*4 = 1048576

    off_cw = 0
    off_beta = align128(len(cw))
    off_g = align128(off_beta + len(beta))
    off_state = align128(off_g + len(g))

    b = bytearray()
    b += b"WTOP"
    b += struct.pack("<H", 3)          # ver=3
    b += struct.pack("<H", 0x1234)
    b += struct.pack("<I", 5)          # n_slots
    b += struct.pack("<I", 11)         # n_ops

    # slot0 = EXT_IN xqkv; slot1 = conv_w; slot2 = beta; slot3 = g; slot4 = STATE
    slot(b, 128, 64, 0, EXT_IN)
    slot(b, len(cw), DC * DI, off_cw, 0)
    slot(b, len(beta), T * H, off_beta, 0)
    slot(b, len(g), T * H, off_g, 0)
    slot(b, 8, 2, off_state, STATE_ADDR)   # 状态表 [n=1][bytes]

    # 通用 op 追加: [opcode u16][n_args u16][args u32×n]
    bb = [b]
    def op(code, args):
        bb[0] += struct.pack("<HH", code, len(args))
        for a in args:
            bb[0] += struct.pack("<I", a)

    # perm=0x03010200 → p=[0,2,1,3] 即 swap dim1(T)↔dim2(H): [T][H][D]→[H][T][D]
    # (0x00030201=p[1,2,3,0] 是 batch 后置的恒等, 曾致 cos=0.31 布局错)
    perm_swap = 0x03010200
    # 1. conv(19): xqkv → yc[T][DI]   (temp0)
    #    w_s 走 slot_ptr(裸槽号), 非 0x8000|slot (exec_conv1d_ssm 直取 slot_ptr(b,w_s))
    op(19, [0x8000 | 0, 1, 0, T, DI, DC])
    # 2. sigmoid(11,sub=8): yc → sig (temp1)
    op(11, [0, 1, T * DI, 8])
    # 3. mul(12,sub=2): yc*sig → yc_silu (temp2)
    op(12, [0, 1, 2, T * DI, 2])
    # 4-6. split(16): yc_silu → q_raw/k_raw/v_raw (temp3/4/5); n_seg=3, sizes=[HD,HD,HD]
    for seg in range(3):
        args = [2, 3 + seg] + [0] * 7 + [0, 3, HD, HD, HD, 0, seg]
        op(16, args)
    # 7-9. transpose(10): q/k/v [T][H][D] → [H][T][D] (temp6/7/8)
    for seg in range(3):
        op(10, [3 + seg, 6 + seg, T, H, D, perm_swap])
    # 10. delta(30): [k,v,q,beta,g,y,state,c,ntok,H,D] → y[H][T][D] (temp9)
    op(30, [7, 8, 6, 0x8000 | 2, 0x8000 | 3, 9, 0x6000 | 0, T, T, H, D])
    # 11. transpose(10): y [H][T][D] → [T][HD] (temp10)
    op(10, [9, 10, H, T, D, perm_swap])

    # annot footer: 11 × u32 = 0 (标量)
    for _ in range(11):
        b += struct.pack("<I", 0)

    # weight 区 128B 对齐 + 内联权重 + 状态表
    while len(b) % 128:
        b += b"\x00"
    w = bytearray()
    w += cw
    while len(w) < off_beta:
        w += b"\x00"
    w += beta
    while len(w) < off_g:
        w += b"\x00"
    w += g
    while len(w) < off_state:
        w += b"\x00"
    w += struct.pack("<II", 1, STATE_BYTES)   # 状态表 [n=1][bytes]
    b += w

    open(outpath, "wb").write(b)
    print(f"[emit] discrete T={T} slots=5 ops=11 bytes={len(b)} -> {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fused", "discrete"])
    ap.add_argument("T", type=int)
    ap.add_argument("indir")
    ap.add_argument("out")
    args = ap.parse_args()
    if args.mode == "fused":
        emit_fused(args.T, args.indir, args.out)
    else:
        emit_discrete(args.T, args.indir, args.out)


if __name__ == "__main__":
    sys.exit(main())
