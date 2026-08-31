#!/usr/bin/env python3
"""Phase E (T-E1/T-E3) reference weight/bias packer — E-A experiment result.

Locates and byte-verifies ALL 95 conv weight tensors + 95 bias tensors of the
InceptionV3 golden serialized.bin against the raw converter output tar
(inception_v3.bin). Result (2026-08-14): 95/95 weights + 95/95 biases matched
with unique transform rules; the weights+bias extent [0x2b000, 0x2df5000) is
100% accounted for (data + 256B-aligned bias blobs + zero padding).

Transform rules (Q3 answers):
  weights: FP16 values preserved; crouton interleave-32 tile repack.
    - tile = 32 rows (cin) x 32 cols (cout), rows AND cols zero-padded to 32
    - interleave32(row-pair a,b): out[2c]=a[c], out[2c+1]=b[c]  (2048B/tile)
    - emit order: for n0 (cout banks of 32): for k0 (cin banks of 32):
                  for (h,kw) in tap order T: tile32(...)
    - tap orders (per conv class):
        T1 (default, stride-1, all kernel shapes):
            [(h,kw) for h in range(KH) for kw in range(KW-1,-1,-1)]
        T2 (stride-2 3x3, SpaceToDepth path — members 52/58/148/150):
            (h%2,kw%2) parity classes row-major; h asc, kw desc within class
        T5 (5x5 stride-1 — members 22/36/50):
            [(h,kw) for h in range(5) for kw in (4,3,2,1)] + [(h,0) for h in range(5)]
        natural (member 0, inconv [3,3,3,32]): K-major flat rows, pad row count
            to even, no tap reordering (28x32 rows = 1792B)
  bias: fp16 value v -> u32 (v << 16), i.e. fp16 in upper halfword.
    - stride-2 convs: chunked per 128 channels (128 vals = 0x200B), chunks
      interleaved after each 128-output-channel weight bank (52: after banks
      128/256/384; 150: banks coalesced after 256 + tail after end).
    - all others: whole blob (COUT*4 B), aligned up to 256B, packed in a
      scheduler-ordered block (0x11bce00..) or the tail region (0x2decc00..).

Container anchors (Q1/Q4 answers):
  CONST_EXTENT descriptor @abs 0x2a000 (48B used): magic 0x71c43c9b,
    w1=0x01000040 (hdr_len=1, desc=0x40*64B), 1 extent [align=0xc(4KB),
    off=0x40*64, len=0xb7280*64], 1 mempool [3,1,0,0xb7248].
    pickle SIZE[4] = desc_len/64 + extent_len/64 = 0xb72c0  (R-09 closed)
  extent = [0x2b000, 0x2df5000); first 0x100 B holds a lone 0x3f800000 (fp32
    1.0) prologue; first weight tensor (member 148) starts 0x2b100.

Usage: python3 tools/weight_layout_pack.py [--golden PATH] [--tar PATH] [--net PATH]
Exit 0 iff every tensor verifies.
"""
import argparse, json, struct, sys

T2 = []
for rh in (0, 1):
    for rw in (0, 1):
        T2 += [(h, kw) for h in range(3) if h % 2 == rh
               for kw in range(2, -1, -1) if kw % 2 == rw]
T5 = [(h, kw) for h in range(5) for kw in (4, 3, 2, 1)] + [(h, 0) for h in range(5)]
S2_MEMBERS = {52, 58, 148, 150}   # stride-2 3x3 convs (Mixed_5a/6a/7a branches)
NATURAL_MEMBERS = {0}             # inconv

def interleave32(rows):
    out = bytearray()
    if len(rows) % 2:
        rows = rows + [[0] * 32]
    for m in range(len(rows) // 2):
        a, b = rows[m * 2], rows[m * 2 + 1]
        for c in range(32):
            out += struct.pack('<HH', a[c], b[c])
    return bytes(out)

def tile32(v, KW, CIN, COUT, h, kw, n0, k0):
    nn = min(n0 + 32, COUT)
    padc = 32 - (nn - n0)
    rows = []
    for cin in range(k0, min(k0 + 32, CIN)):
        base = ((h * KW + kw) * CIN + cin) * COUT
        rows.append(list(v[base + n0:base + nn]) + [0] * padc)
    while len(rows) < 32:
        rows.append([0] * 32)
    return interleave32(rows)

def pack_natural(v):
    nrows = (len(v) + 31) // 32
    if nrows % 2:
        nrows += 1
    rows = list(v) + [0] * (32 * nrows - len(v))
    return interleave32([rows[r * 32:(r + 1) * 32] for r in range(nrows)])

def tap_order(member_idx, KH, KW):
    if (KH, KW) == (5, 5):
        return T5
    if member_idx in S2_MEMBERS and KH == 3 and KW == 3:
        return T2
    return [(h, kw) for h in range(KH) for kw in range(KW - 1, -1, -1)]

def pack_weights(v, KH, KW, CIN, COUT, member_idx, n0_only=None):
    if member_idx in NATURAL_MEMBERS:
        return pack_natural(v)
    T = tap_order(member_idx, KH, KW)
    out = bytearray()
    banks = [n0_only] if n0_only is not None else range(0, COUT, 32)
    for n0 in banks:
        for k0 in range(0, CIN, 32):
            for (h, kw) in T:
                out += tile32(v, KW, CIN, COUT, h, kw, n0, k0)
    return bytes(out)

def pack_bias(hv):
    return b''.join(struct.pack('<I', x << 16) for x in hv)

def parse_tar(buf):
    members, o = [], 0
    while o + 512 <= len(buf):
        if buf[o:o + 512] == b'\0' * 512:
            break
        size = int(buf[o + 124:o + 136].rstrip(b'\0 ').decode() or '0', 8)
        members.append((buf[o:o + 100].rstrip(b'\0').decode(), o + 512, size))
        o += 512 + ((size + 511) // 512) * 512
    return members

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--golden', default='/disk2/qnn_example/InceptionV3/serialized/inception_v3_v81_serialized.bin')
    ap.add_argument('--tar', default='/disk2/qnn_example/InceptionV3/output/inception_v3.bin')
    ap.add_argument('--net', default='/disk2/qnn_example/InceptionV3/output/inception_v3_net.json')
    a = ap.parse_args()
    g = open(a.golden, 'rb').read()
    w = open(a.tar, 'rb').read()
    tens = json.load(open(a.net))['graph']['tensors']
    members = parse_tar(w)
    ok = miss = 0
    for i, (name, doff, sz) in enumerate(members):
        key = name[:-4]
        dims = tens[key]['dims']
        if i % 2 == 0:
            KH, KW, CIN, COUT = dims
            v = list(struct.unpack('<%dH' % (sz // 2), w[doff:doff + sz]))
            blob = pack_weights(v, KH, KW, CIN, COUT, i)
            p = g.find(blob, 0x2a000)
            good = p >= 0 and g.find(blob, p + 1) < 0
            tag = 'weight'
            if not good and i in S2_MEMBERS:
                # bias chunks interleave between 128-channel weight banks for
                # stride-2 convs; verify per (n0) chunk instead
                tag = 'weight-chunked'
                good = all(g.find(pack_weights(v, KH, KW, CIN, COUT, i, n0)) >= 0
                           for n0 in range(0, COUT, 32))
                p = g.find(pack_weights(v, KH, KW, CIN, COUT, i, 0))
        else:
            hv = list(struct.unpack('<%dH' % (sz // 2), w[doff:doff + sz]))
            blob = pack_bias(hv)
            p = g.find(blob)
            good = p >= 0
            tag = 'bias'
            if i - 1 in S2_MEMBERS:
                tag = 'bias-chunked'
                chunks = [pack_bias(hv[c:c + 128]) for c in range(0, len(hv), 128)]
                good = all(g.find(c) >= 0 for c in chunks)
                p = g.find(chunks[0])
        if good:
            ok += 1
        else:
            miss += 1
            print('MISS %3d %s %s %s' % (i, tag, key, dims), file=sys.stderr)
    print('%s: %d ok, %d miss' % (a.golden, ok, miss))
    return 0 if miss == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
