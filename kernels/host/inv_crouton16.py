"""Inverse of pack_a16_crouton16_row4_surface (and native_hmxa_compact_activation_surface,
which is byte-identical to crouton16_row4).

Forward pack (from prepare_owned_inputs.py:133):
  packed = empty(M*K)
  out = 0
  for row4_phase in 0..8:
    for kt in 0..(K//32):
      k_base = kt*32
      for m32_group in 0..(M//32):
        for row_pair in 0..2:
          row0 = m32_group*32 + row4_phase*4 + row_pair*2
          row1 = row0 + 1
          for col in k_base..k_base+32:
            packed[out]   = raw_mk[row0, col]
            packed[out+1] = raw_mk[row1, col]
            out += 2

So phys u16 slot `p` decomposes (p//2 = pair_idx, p%2 = which row in pair):
  col_in_tile = pair_idx % 32         ; rest = pair_idx // 32
  row_pair    = rest % 2              ; rest //= 2
  m32_group   = rest % (M//32)        ; rest //= (M//32)
  kt          = rest % (K//32)        ; rest //= (K//32)
  row4_phase  = rest                  (0..7)
  col = kt*32 + col_in_tile
  row0 = m32_group*32 + row4_phase*4 + row_pair*2
  row = row0 + (1 if which==1 else 0)
"""
import numpy as np


def inv_crouton16_row4(surf_u16_flat, M, K):
    """surf_u16_flat: 1-D uint16 array length M*K (the crouton16 row4 surface).
    Returns (M, K) uint16 row-major."""
    assert M % 32 == 0 and K % 32 == 0
    n_m32 = M // 32
    n_kt = K // 32
    out = np.empty((M, K), dtype=np.uint16)
    n_pairs = (M * K) // 2
    # build via vectorized decomposition of pair index
    pair_idx = np.arange(n_pairs, dtype=np.int64)        # 0 .. M*K/2-1
    col_in_tile = pair_idx % 32
    rest = pair_idx // 32
    row_pair = rest % 2
    rest = rest // 2
    m32_group = rest % n_m32
    rest = rest // n_m32
    kt = rest % n_kt
    rest = rest // n_kt
    row4_phase = rest
    col = kt * 32 + col_in_tile
    row0 = m32_group * 32 + row4_phase * 4 + row_pair * 2
    # even phys slot (which=0) -> row0, odd (which=1) -> row0+1
    # surf_u16_flat[2*pair] = raw[row0, col]; surf[2*pair+1] = raw[row0+1, col]
    out[row0, col] = surf_u16_flat[0::2]
    out[row0 + 1, col] = surf_u16_flat[1::2]
    return out


def inv_crouton16_bytes(surf_bytes, M, K):
    u16 = np.frombuffer(surf_bytes, dtype=np.uint16)
    assert u16.size == M * K, (u16.size, M * K)
    return inv_crouton16_row4(u16, M, K)


def pack_a16_crouton16_row4_surface(raw_mk):
    """Forward pack, copied from prepare_owned_inputs.py for round-trip self-test."""
    m, k = raw_mk.shape
    packed = np.empty(raw_mk.size, dtype=raw_mk.dtype)
    out = 0
    for row4_phase in range(8):
        for kt in range(k // 32):
            k_base = kt * 32
            for m32_group in range(m // 32):
                for row_pair in range(2):
                    row0 = m32_group * 32 + row4_phase * 4 + row_pair * 2
                    row1 = row0 + 1
                    for col in range(k_base, k_base + 32):
                        packed.reshape(-1)[out] = raw_mk[row0, col]
                        packed.reshape(-1)[out + 1] = raw_mk[row1, col]
                        out += 2
    return packed


if __name__ == "__main__":
    import sys
    # round-trip self-consistency on oracle activation + 3 random
    ORACLE = "/disk1/V81Dev/test/qcom_htp/example/qnn_matmul_profile/output_native_w4a16_conv_256"
    act = np.load(f"{ORACLE}/actRaw_u16.npy")   # (256,256) = A natural (M,K)
    M = K = 256
    surf = pack_a16_crouton16_row4_surface(act)
    back = inv_crouton16_row4(surf, M, K)
    print(f"[oracle act] round-trip bit-exact: {np.array_equal(back, act)}")
    ok = True
    # deterministic pseudo-random (no Math.random needed: use arange seeds)
    for seed_off in range(3):
        rng = np.random.default_rng(1000 + seed_off)
        r = rng.integers(0, 65536, size=(M, K), dtype=np.uint16)
        s2 = pack_a16_crouton16_row4_surface(r)
        b2 = inv_crouton16_row4(s2, M, K)
        eq = np.array_equal(b2, r)
        ok = ok and eq
        print(f"[random#{seed_off}] round-trip bit-exact: {eq}")
    print("ALL ROUND-TRIP PASS" if ok else "ROUND-TRIP FAIL")
    sys.exit(0 if ok else 1)
