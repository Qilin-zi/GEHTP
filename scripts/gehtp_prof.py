#!/usr/bin/env python3
"""gehtp_prof.py — GEHTP 整网性能归因报表 (PROF 战役 W-P1)

输入三件套:
  --wtop     model.wtop          (runlist + slot 表; blob v1/v2 双格式)
  --manifest model.wtop.manifest.json  (缺省 = wtop 同名后缀; v2 带 op_names)
  --op-ts    prof/x.op_ts.bin    (设备逐 op 时间戳对; 缺省尝试 optrace.txt 回退)

归因链: 编译期全静态内存规划 ⇒ 每 op 字节数可纯静态推导 (无需硬件计数器);
        join 设备实测 µs ⇒ 有效带宽 / 算密 / roofline 落点 / 算·搬·缝隙三分账。

标定常量 (唯一权威行纪律, 源: V81HexSim CURRENT_STATUS 2026-08-20 器件留痕):
  PEAK_FLOPS_F16 = 12.34e12  HMX f16 器件实测 (12.29 model vs 12.34 device)
  PEAK_DRAM_BW   = 59.9e9    UserDMA 边际律 B/s (P3 per-op 律, 860cyc 固定另计)
  注: htpacc 部署曾测 HMX 20447 GFLOPS (不同口径/配置), 与本表 12.34T 并存待考;
      可用 --peak-flops 覆盖。VTCM 内带宽 109 GB/s/线程 (vtcm_bank_probe)。

用法:
  gehtp_prof.py --wtop m.wtop --op-ts prof/m.op_ts.bin [--top 20]
                [--trace-json prof/m.trace.json] [--csv prof/m.csv]
                [--device-txt prof/m.device.txt] [--optrace prof/m.optrace.txt]
"""
import argparse
import json
import struct
import sys

# ---------------------------------------------------------------- 标定常量
PEAK_FLOPS_F16 = 12.34e12   # HMX f16 器件实测 TFLOPS (V81HexSim 权威行)
PEAK_DRAM_BW = 59.9e9       # UserDMA 边际律 B/s (P3 器件律)
VTCM_BW_PER_THREAD = 109e9  # VTCM 内带宽/线程 (vtcm_bank_probe 2026-08-20)
SMALL_OP_US = 20.0          # 小 op 判阈 (派发/延迟主导)

# ---------------------------------------------------------------- opcode 表
# 与 kernels/include/oplist_parse.h 契约一一对应 (单一真源; 改了那边要同步这边)
OP_NAMES = {
    0: "NOP", 1: "MATMUL_W4A16", 2: "RMSNORM_F16", 3: "PIN", 4: "SILU_F16",
    5: "IM2COL", 6: "CONV2D_F16", 7: "ADD_F16", 8: "SPILL", 9: "FILL",
    10: "TRANSPOSE_F16", 11: "UNARY_F16", 12: "BINARY_F16", 13: "SOFTMAX_F16",
    14: "CONCAT_F16", 15: "STRIDED_SLICE_F16", 16: "SPLIT_F16",
    17: "REDUCE_F16", 18: "CUMSUM_F32", 19: "CONV1D_SSM_F16", 20: "GATHER_F16",
    21: "ARGMAX_F16", 22: "KV_APPEND_F16", 23: "KV_GATHER_F16",
    24: "MATMUL_F16", 25: "RMSNORM2_F16", 26: "BROADCAST_F16",
    27: "TRANSPOSE_GEN_F16", 28: "SCATTER_ND_F16",
}
# 搬运类 op (三分账的"搬"): 纯数据移动/重排, 零 FLOPs
MOVE_OPS = {3, 5, 8, 9, 10, 14, 15, 16, 20, 26, 27}
# W-P3: 设备侧 enum wt_engine 同序 (oplist_exec.h)
ENGINE_NAMES = {0: "scalar", 1: "HVX", 2: "HMX"}


class OpRec:
    __slots__ = ("idx", "opcode", "args", "name", "in_b", "w_b", "out_b",
                 "pool_b", "flops", "vtcm", "note", "start_us", "dur_us",
                 "dma_us", "dma_bytes", "engine")

    def __init__(self, idx, opcode, args):
        self.idx = idx
        self.opcode = opcode
        self.args = args
        self.name = "?"
        self.in_b = self.w_b = self.out_b = self.pool_b = None
        self.flops = 0
        self.vtcm = False
        self.note = ""
        self.start_us = self.dur_us = None
        self.dma_us = self.dma_bytes = None
        self.engine = 0

    @property
    def total_b(self):
        parts = [b for b in (self.in_b, self.w_b, self.out_b, self.pool_b)
                 if b is not None]
        return sum(parts) if parts else None


# ---------------------------------------------------------------- blob 解析
def parse_wtop(path):
    """解析 .wtop → (ver, slots[(len,count,off_bytes,addr)], ops[(opcode,args)]).
    blob v1: slot 线记录 16B {len,count,off u32 字节,addr};
         v2 (C4): 24B {len,count,off_lo,off_hi,addr,reserved}, offset=u64 字节
         (>4GB 模型; 唯一真源 = kernels/include/oplist_parse.h WT_SLOT_SIZE_V2)。
    流式读: 只取 header+slot 表+op 表 (大模型 blob 5GB+, 不整读权重区)。"""
    with open(path, "rb") as f:
        hdr = f.read(16)
        if len(hdr) < 16 or hdr[:4] != b"WTOP":
            raise SystemExit(f"bad magic: {path}")
        ver, endian = struct.unpack_from("<HH", hdr, 4)
        if endian != 0x1234:
            raise SystemExit(f"bad endian chk: {path}")
        if ver not in (1, 2):
            raise SystemExit(f"unsupported blob ver {ver} (v1/v2 已知)")
        n_slots, n_ops = struct.unpack_from("<II", hdr, 8)
        slot_rec = 24 if ver >= 2 else 16
        slots = []
        stab = f.read(n_slots * slot_rec)
        if len(stab) < n_slots * slot_rec:
            raise SystemExit("slot 表截断")
        for i in range(n_slots):
            base = stab[i * slot_rec:(i + 1) * slot_rec]
            if ver >= 2:
                ln, cnt, off_lo, off_hi, addr, _rsv = struct.unpack("<6I", base)
                off_b = off_lo | (off_hi << 32)
            else:
                ln, cnt, off_b, addr = struct.unpack("<4I", base)
            slots.append((ln, cnt, off_b, addr))
        ops = []
        for _ in range(n_ops):
            oh = f.read(4)
            if len(oh) < 4:
                raise SystemExit("op 表截断")
            opcode, n_args = struct.unpack("<HH", oh)
            abuf = f.read(4 * n_args)
            if len(abuf) < 4 * n_args:
                raise SystemExit("op args 截断")
            ops.append((opcode, list(struct.unpack("<%dI" % n_args, abuf))))
    return ver, slots, ops


def parse_op_ts(path):
    """op_ts.bin (WTS1): magic|ver|n_ops|pad|wall_us u64|n×(start,dur[,dma_us,dma_bytes,engine]).
    ver=1: 每 op 8B {start,dur}; ver=2: 每 op 20B 加 dma_us/dma_bytes/engine。
    统一返回 5 元组 (start,dur,dma_us,dma_bytes,engine), ver=1 后三项补 0。"""
    data = open(path, "rb").read()
    if data[:4] != b"WTS1":
        raise SystemExit(f"bad op_ts magic: {path}")
    ver, n_ops = struct.unpack_from("<II", data, 4)
    if ver not in (1, 2):
        raise SystemExit(f"unsupported op_ts ver {ver}")
    (wall_us,) = struct.unpack_from("<Q", data, 16)
    ts = []
    off = 24
    for _ in range(n_ops):
        if ver == 1:
            s, d = struct.unpack_from("<II", data, off); off += 8
            ts.append((s, d, 0, 0, 0))
        else:
            s, d, dus, db, eng = struct.unpack_from("<IIIII", data, off); off += 20
            ts.append((s, d, dus, db, eng))
    return wall_us, ts


def parse_optrace(path):
    """回退路径: optrace.txt 的 'opN code=C rc=R us=D' 行 → [(None, dur)]。
    无 start (timeline 不可得); 索引按出现序 = run_range 的 ii (单段=全局序)。"""
    ts = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("op") or " us=" not in line:
                continue
            try:
                us = int(line.rsplit(" us=", 1)[1])
                ts.append((None, us if us >= 0 else None))
            except ValueError:
                continue
    return None, ts


# ------------------------------------------------------------ 字节/FLOPs 推导
def _slot_len(slots, sid):
    sid = sid & 0x7FFF  # 0x8000 编码位剥除 (权重槽直连场景防御)
    return slots[sid][0] if sid < len(slots) else None


def derive_bytes(op, slots):
    """按 oplist_parse.h 契约推 per-op 字节/FLOPs。推不出 → None (报表标
    'unknown', 不阻塞)。in/w/out/pool 分列; DRAM/VTCM 归属由引用编码注记。"""
    a = op.args
    c = op.opcode
    def need(n):
        return len(a) >= n
    if c == 0:    # NOP
        op.in_b = op.w_b = op.out_b = 0
    elif c == 1 and need(6):   # MATMUL_W4A16 [act,w,out,M,K,N]
        M, K, N = a[3], a[4], a[5]
        op.in_b, op.w_b, op.out_b = M * K * 2, _slot_len(slots, a[1]), M * N * 2
        op.flops = 2 * M * N * K
        op.note = "w=W4 packed(含 scale)"
    elif c == 2 and need(4):   # RMSNORM_F16 [x,w,y,n]
        op.in_b, op.w_b, op.out_b = a[3] * 2, _slot_len(slots, a[1]), a[3] * 2
    elif c == 3 and need(1):   # PIN [slot] — 引擎 slot→VTCM 搬运
        op.in_b = _slot_len(slots, a[0])
        op.out_b = 0
        op.note = "slot→VTCM"
    elif c == 4 and need(3):   # SILU [x,y,n]
        op.in_b, op.out_b = a[2] * 2, a[2] * 2
    elif c == 5 and need(15):  # IM2COL [act,out,H,W,C,kh,kw,ph,pw,sh,sw,y0,x0,th,tw]
        H, W, C, kh, kw, th, tw = a[2], a[3], a[4], a[5], a[6], a[13], a[14]
        op.in_b = H * W * C * 2
        op.out_b = th * tw * kh * kw * C * 2
        op.note = "out=patch 展开(≈)"
    elif c == 6 and need(13):  # CONV2D_F16 [act,w,b,out,M,K,N,...]
        M, K, N = a[4], a[5], a[6]
        op.in_b, op.w_b, op.out_b = M * K * 2, _slot_len(slots, a[1]), M * N * 2
        op.flops = 2 * M * N * K
        op.note = "act=im2col 面"
    elif c == 7 and need(4):   # ADD [a,b,out,n]
        op.in_b, op.out_b = a[3] * 4, a[3] * 2   # 两操作数各 n*2
    elif c == 8 and need(4):   # SPILL [src,pool,off,n] temp→池
        op.pool_b = a[3] * 2
        op.note = "temp→DDR池"
    elif c == 9 and need(4):   # FILL [pool,off,dst,n] 池→temp
        op.pool_b = a[3] * 2
        op.note = "DDR池→temp"
    elif c == 10 and need(6):  # TRANSPOSE_F16 [src,out,H,W,C,perm]
        n = a[2] * a[3] * a[4]
        op.in_b, op.out_b = n * 2, n * 2
    elif c == 11 and need(4):  # UNARY [x,y,n,sub]
        op.in_b, op.out_b = a[2] * 2, a[2] * 2
    elif c == 12 and need(5):  # BINARY [a,b,y,n,sub]
        op.in_b, op.out_b = a[3] * 4, a[3] * 2
    elif c == 13 and need(4):  # SOFTMAX [x,y,rows,n]
        n = a[2] * a[3]
        op.in_b, op.out_b = n * 2, n * 2
    elif c == 14 and need(12):  # CONCAT [in0..7,out,axis,nseg,n_elems,sz0..3]
        op.in_b, op.out_b = a[11] * 2, a[11] * 2
    elif c == 15 and need(3):  # STRIDED_SLICE [x,y,n_out,...]
        op.in_b, op.out_b = a[2] * 2, a[2] * 2
        op.note = "in 按 n_out 近似(读≤输入)"
    elif c == 16 and need(16):  # SPLIT [x,out0..7,axis,nseg,sz0..3,idx]
        seg = a[11 + min(a[15], 3)] if a[15] <= 3 else None
        op.in_b = op.out_b = (seg * 2) if seg else None
        op.note = "副本 op 取本段(≈)"
    elif c == 17 and need(9):  # REDUCE [x,y,n,axis,sub,d0..3]
        n, ax = a[2], a[3]
        dims = a[5:9]
        op.in_b = n * 2
        op.out_b = (n // dims[ax]) * 2 if ax < 4 and dims[ax] else None
    elif c == 18 and need(7):  # CUMSUM_F32 [x,y,rows,n,axis,excl,rev]
        n = a[2] * a[3]
        op.in_b, op.out_b = n * 4, n * 4
        op.note = "f32"
    elif c == 19 and need(6):  # CONV1D_SSM [x,w,y,seq,C,k]
        seq, C, k = a[3], a[4], a[5]
        op.in_b, op.w_b, op.out_b = seq * C * 2, _slot_len(slots, a[1]), seq * C * 2
        op.flops = 2 * seq * C * k
    elif c == 20 and need(5):  # GATHER [tbl,idx,out,n,row_bytes]
        op.in_b = a[3] * a[4]
        op.w_b = _slot_len(slots, a[1])
        op.out_b = a[3] * a[4]
        op.note = "in=聚集行(≈)"
    elif c == 21 and need(3):  # ARGMAX [x,out,n]
        op.in_b, op.out_b = a[2] * 2, 4
    elif c in (22, 23):        # KV_APPEND/KV_GATHER — 设备侧未实现 (unimpl)
        op.note = "unimpl"
    elif c == 24 and need(7):  # MATMUL_F16 [a,w,out,M,K,N,flags]
        M, K, N = a[3], a[4], a[5]
        op.in_b, op.w_b, op.out_b = M * K * 2, _slot_len(slots, a[1]), M * N * 2
        op.flops = 2 * M * N * K
    elif c == 25 and need(5):  # RMSNORM2 [x,w,b,y,n]
        op.in_b, op.w_b, op.out_b = a[4] * 2, _slot_len(slots, a[1]), a[4] * 2
    elif c == 26 and need(12):  # BROADCAST [b,y,n,b_elems,...]
        op.in_b, op.out_b = a[3] * 2, a[2] * 2
    elif c == 27 and need(8):  # TRANSPOSE_GEN [x,y,rank,d0..3,perm]
        n = 1
        for d in a[3:3 + min(a[2], 4)]:
            n *= d
        op.in_b, op.out_b = n * 2, n * 2
    elif c == 28 and need(14):  # SCATTER_ND [data,idx,upd,out,n_out,rank,d0..4,K,n_idx,block]
        n_out, K, n_idx, block = a[4], a[11], a[12], a[13]
        op.in_b = n_out * 2 + n_idx * block * 2   # data 全读 + upd 块
        op.w_b = _slot_len(slots, a[1])            # idx (i32 槽)
        op.out_b = n_out * 2
        op.note = f"idx={n_idx}×K{K} block={block}(≈)"
    # VTCM 驻留引用注记 (0x4000|temp 编码, 任一操作数命中即标)
    op.vtcm = any((x & 0xC000) == 0x4000 for x in a)


def verdict(op, peak_flops, peak_bw):
    """roofline 落点一句话判词。"""
    if op.dur_us is None:
        return "-"
    if op.opcode in MOVE_OPS:
        tb = op.total_b
        if tb and op.dur_us > 0:
            bw = tb / (op.dur_us * 1e-6)
            return f"搬运 bw={bw / 1e9:.1f}GB/s ({bw / peak_bw * 100:.0f}%峰值)"
        return "搬运"
    if op.dur_us < SMALL_OP_US:
        return "小op(派发/延迟)"
    tb = op.total_b
    if op.flops and tb:
        ai = op.flops / tb
        balance = peak_flops / peak_bw
        return ("算力bound" if ai >= balance else "带宽bound") + f" ai={ai:.0f}"
    if op.flops:
        return f"算 op flops={op.flops / 1e6:.1f}M (bytes 未知)"
    return "elem/控制"


# ---------------------------------------------------------------- 报表
def fmt_b(b):
    if b is None:
        return "?"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.1f}{u}" if u != "B" else f"{int(b)}B"
        b /= 1024
    return f"{b:.1f}GB"


def main():
    ap = argparse.ArgumentParser(description="GEHTP 整网性能归因报表")
    ap.add_argument("--wtop", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--op-ts", default=None, help="op_ts.bin (优先)")
    ap.add_argument("--optrace", default=None, help="optrace.txt (回退, 无 start)")
    ap.add_argument("--device-txt", default=None, help="设备结果 txt ([run] 分段)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--trace-json", default=None, help="chrome-trace JSON 输出")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--peak-flops", type=float, default=PEAK_FLOPS_F16)
    ap.add_argument("--peak-bw", type=float, default=PEAK_DRAM_BW)
    ap.add_argument("--strict", action="store_true",
                    help="op_ts 与 blob 长度不齐时报错退出 (默认仅 warn)")
    args = ap.parse_args()

    ver, slots, raw_ops = parse_wtop(args.wtop)
    manifest_path = args.manifest or (args.wtop + ".manifest.json")
    man = {}
    try:
        man = json.load(open(manifest_path))
    except FileNotFoundError:
        print(f"warn: manifest 缺 {manifest_path} (op_names 不可用)", file=sys.stderr)

    op_ids = man.get("op_ids", [])
    op_names = man.get("op_names", [])   # manifest v2; v1 无此字段
    if man and "op_names" not in man:
        print("warn: manifest v1 无 op_names (重编译得 v2)", file=sys.stderr)

    ops = []
    for i, (opcode, oargs) in enumerate(raw_ops):
        rec = OpRec(i, opcode, oargs)
        if i < len(op_names):
            rec.name = op_names[i]
        elif i < len(op_ids):
            rec.name = f"op_id={op_ids[i]}"
        derive_bytes(rec, slots)
        ops.append(rec)

    # ---- join 时间
    wall_us = None
    ts = None
    if args.op_ts:
        wall_us, ts = parse_op_ts(args.op_ts)
    elif args.optrace:
        wall_us, ts = parse_optrace(args.optrace)
    if ts:
        if len(ts) != len(ops):
            msg = (f"ts n={len(ts)} != blob n_ops={len(ops)} — op_ts.bin 陈腐"
                   f"(上轮别模型/别会话)或未重跑; join 按 min={min(len(ts), len(ops))}")
            if args.strict:
                raise SystemExit(f"strict: {msg}  (跑前 gehtp run 已清设备侧 op_ts.bin; "
                                 f"若仍不齐 = 该 blob 未在本轮重跑)")
            print(f"warn: {msg}", file=sys.stderr)
        for rec, (s, d, dus, db, eng) in zip(ops, ts):
            rec.start_us, rec.dur_us = s, d
            rec.dma_us, rec.dma_bytes, rec.engine = dus, db, eng

    # ---- [run] 分段 (可选)
    if args.device_txt:
        try:
            for line in open(args.device_txt, errors="replace"):
                if line.startswith("[run] seg_us"):
                    print("== run 分段 (设备) ==")
                    print(line.strip())
                    print()
                    break
        except FileNotFoundError:
            print(f"warn: device txt 缺 {args.device_txt}", file=sys.stderr)

    # ---- 总账
    n = len(ops)
    sum_dur = sum(r.dur_us for r in ops if r.dur_us is not None)
    have_dur = sum(1 for r in ops if r.dur_us is not None)
    print(f"== GEHTP prof: {args.wtop} ==")
    print(f"blob_ver={ver} ops={n} ts_ops={have_dur}", end="")
    if wall_us:
        gap = wall_us - sum_dur
        print(f" wall={wall_us / 1e3:.1f}ms Σdur={sum_dur / 1e3:.1f}ms "
              f"缝隙={gap / 1e3:.1f}ms ({gap / max(wall_us, 1) * 100:.1f}%)")
    else:
        print(f" Σdur={sum_dur / 1e3:.1f}ms (无 wall — optrace 回退或无 ts)")
    # 三分账: 算 (flops>0) / 搬 (MOVE_OPS) / 其他+缝隙
    t_calc = sum(r.dur_us or 0 for r in ops if r.flops > 0)
    t_move = sum(r.dur_us or 0 for r in ops if r.opcode in MOVE_OPS)
    t_other = sum_dur - t_calc - t_move
    denom = wall_us or sum_dur or 1
    print(f"三分账: 算 {t_calc / denom * 100:.1f}% | 搬 {t_move / denom * 100:.1f}% "
          f"| 其他 {t_other / denom * 100:.1f}% | 缝隙 "
          f"{(denom - sum_dur) / denom * 100:.1f}%")
    print(f"标定: peak_flops={args.peak_flops / 1e12:.2f}TFLOPS f16 "
          f"peak_bw={args.peak_bw / 1e9:.1f}GB/s "
          f"(源: V81HexSim 器件权威行; --peak-* 可覆盖)")
    # W-P3: 真 UserDMA 搬运实测 (只含 dc_dma_once 的 DRAM↔VTCM; CPU 拷贝不计)
    t_dma = sum(r.dma_us or 0 for r in ops)
    b_dma = sum(r.dma_bytes or 0 for r in ops)
    if t_dma > 0:
        print(f"DMA 搬运实测: Σdma_us={t_dma / 1e3:.2f}ms Σdma={fmt_b(b_dma)} "
              f"有效带宽={b_dma / (t_dma * 1e-6) / 1e9:.1f}GB/s "
              f"({b_dma / (t_dma * 1e-6) / args.peak_bw * 100:.0f}%峰值)")
    else:
        print("DMA 搬运实测: Σdma_us=0 (本 blob 无 W4A16 matmul 的 UserDMA 搬运)")
    print()

    # ---- top N
    ranked = sorted((r for r in ops if r.dur_us is not None),
                    key=lambda r: -r.dur_us)
    print(f"== top {args.top} by dur ==")
    print(f"{'idx':>6} {'dur_us':>9} {'dma_us':>8} {'%wall':>6} {'bytes':>12} {'GB/s':>7} "
          f"{'GFLOP/s':>8} {'opcode':<18} {'eng':<7} {'name':<40} verdict")
    dw = denom
    for r in ranked[:args.top]:
        tb = r.total_b
        bw = f"{tb / (r.dur_us * 1e-6) / 1e9:.1f}" if tb and r.dur_us else "-"
        gf = f"{r.flops / (r.dur_us * 1e-6) / 1e9:.1f}" if r.flops and r.dur_us else "-"
        vt = " [vtcm]" if r.vtcm else ""
        dma_s = f"{r.dma_us:>8}" if r.dma_us is not None else "       -"
        eng = ENGINE_NAMES.get(r.engine, f"e{r.engine}")
        v = verdict(r, args.peak_flops, args.peak_bw)
        if r.dma_us:
            v += f" dma={r.dma_us}us ({r.dma_bytes / (r.dma_us * 1e-6) / 1e9:.1f}GB/s)"
        print(f"{r.idx:>6} {r.dur_us:>9} {dma_s} {r.dur_us / dw * 100:>5.1f}% "
              f"{fmt_b(tb):>12} {bw:>7} {gf:>8} "
              f"{OP_NAMES.get(r.opcode, f'OP{r.opcode}'):<18} {eng:<7} "
              f"{(r.name[:40] + vt):<40} {v}")
    print()

    # ---- per-opcode 聚合
    print("== per-opcode 聚合 ==")
    print(f"{'opcode':<18} {'n':>5} {'total_us':>10} {'%wall':>6} {'avg_us':>9} "
          f"{'max_us':>9} {'tot_bytes':>12}")
    agg = {}
    for r in ops:
        a = agg.setdefault(r.opcode, [0, 0, 0, 0])
        a[0] += 1
        if r.dur_us is not None:
            a[1] += r.dur_us
            a[2] = max(a[2], r.dur_us)
        tb = r.total_b
        if tb:
            a[3] += tb
    for c, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        avg = a[1] / a[0] if a[0] else 0
        print(f"{OP_NAMES.get(c, f'OP{c}'):<18} {a[0]:>5} {a[1]:>10} "
              f"{a[1] / dw * 100:>5.1f}% {avg:>9.1f} {a[2]:>9} {fmt_b(a[3]):>12}")
    print()

    # ---- chrome trace JSON
    if args.trace_json:
        evs = []
        cur = 0
        for r in ops:
            if r.dur_us is None:
                continue
            s = r.start_us if r.start_us is not None else cur
            evs.append({"name": f"{r.idx}:{r.name}", "cat": OP_NAMES.get(r.opcode, "?"),
                        "ph": "X", "ts": s, "dur": r.dur_us, "pid": 1, "tid": 1,
                        "args": {"opcode": r.opcode, "bytes": r.total_b,
                                 "flops": r.flops, "engine": r.engine,
                                 "dma_us": r.dma_us}})
            cur = s + r.dur_us
        json.dump({"traceEvents": evs,
                   "displayTimeUnit": "ms"}, open(args.trace_json, "w"))
        print(f"trace-json: {args.trace_json} (Perfetto/ui.perfetto.dev 打开)")

    # ---- CSV
    if args.csv:
        import csv as csvmod
        with open(args.csv, "w", newline="") as f:
            wcsv = csvmod.writer(f)
            wcsv.writerow(["idx", "name", "opcode", "opcode_name", "start_us",
                           "dur_us", "dma_us", "dma_bytes", "engine", "in_b",
                           "w_b", "out_b", "pool_b", "flops", "vtcm", "note"])
            for r in ops:
                wcsv.writerow([r.idx, r.name, r.opcode,
                               OP_NAMES.get(r.opcode, "?"), r.start_us, r.dur_us,
                               r.dma_us, r.dma_bytes,
                               ENGINE_NAMES.get(r.engine, r.engine),
                               r.in_b, r.w_b, r.out_b, r.pool_b, r.flops,
                               int(r.vtcm), r.note])
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
