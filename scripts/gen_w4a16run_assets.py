#!/usr/bin/env python3
"""gen_w4a16run_assets.py — ④ 门板测资产生成 (105 本地, 与例 44 配对)

case 生成 (→ /disk1/GEHTP/kernels/assets/w4a16run/):
  s256          闭包 256³ 资产: actRaw_u16 → f16; scale=1/7; yexp=Y_gold real
  r256_256_256  随机 256³, act ~±3 (测 A_s 归一化)
  r256_1024_2048  随机 (transformer q/k/v 形状)
  r32_1024_3584   随机 M=32 (pad 256 路径)

期望输出 = host 数学 (act_f32 @ (wq·S)) → f16 RNE — 设备输出有 act 量化
噪声, 板侧判据 cos ≥ 0.999 (容差, 与闭包 float_ref ±36 LSB 同量级)。
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, "/4090disk2/htpw4a16_v81/qcom_htp_link/example/handwritten_hmx_matmul")
from prepare_owned_inputs import pack_w4_kblock32_nmajor_k4_lohi, pack_native_a16_bias

OUT = "/disk1/GEHTP/kernels/assets/w4a16run"
S256 = "/disk1/GEHTP/kernels/assets/s256"
ORACLE = "/4090disk2/htpw4a16_v81/oracle"


def f16(x):
    return np.asarray(x, dtype=np.float16)


def write_case(name, m, k, n, act_f32, wq_int8, scale_f32, yexp_f32, f_override):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.txt"), "wb") as f:
        f.write(struct.pack("<IIIf", m, k, n, f_override))
    f16(act_f32).tofile(os.path.join(d, "act.f16.raw"))
    pack_w4_kblock32_nmajor_k4_lohi(wq_int8).tofile(os.path.join(d, "packed_weight.raw"))
    pack_native_a16_bias(4, wq_int8.astype(np.int32))[0].tofile(
        os.path.join(d, "folded_bias.raw"))
    kt, nt = k // 32, n // 32
    atbl = np.zeros(8 * kt, np.uint32)
    for mt in range(8):
        for i in range(kt):
            atbl[mt * kt + i] = (mt * kt + i) * 0x800
    atbl.tofile(os.path.join(d, "act_table.raw"))
    otbl = np.zeros(8 * nt, np.uint32)
    for mt in range(8):
        for i in range(nt):
            otbl[mt * nt + i] = (mt * nt + i) * 0x800
    otbl.tofile(os.path.join(d, "out_table.raw"))
    wq_rms = float(np.sqrt((wq_int8.astype(np.float64)**2).sum(axis=0).mean() / wq_int8.shape[0]))
    sc = np.concatenate([f16(scale_f32), f16([wq_rms])])
    sc.tofile(os.path.join(d, "scale.f16.raw"))
    f16(yexp_f32).tofile(os.path.join(d, "yexp.f16.raw"))
    print(f"  {name}: m={m} k={k} n={n} act range [{act_f32.min():.3f}, {act_f32.max():.3f}]")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []

    # ---- case 1: 闭包 256³ ----
    act_u16 = np.load(f"{ORACLE}/actRaw_u16.npy")            # (256,256) u16 a16 域
    w_kn = np.load(f"{ORACLE}/wRaw_KN.npy").astype(np.int8)  # (256,256) int8
    ygold = np.fromfile(f"{S256}/Y_gold_2563.raw", dtype="<u2")  # (N,M) 线性
    assert act_u16.shape == (256, 256) and w_kn.shape == (256, 256)
    act_f32 = (act_u16.astype(np.float32) - 32768.0) / 32767.0   # real ∈ [-1,1]
    yreal = (ygold.astype(np.float32) - 32768.0) / 32767.0
    yexp = yreal.reshape(256, 256).T.copy()                     # (M,N) 取向
    scale = np.full(256, 1.0 / 7.0, np.float32)
    write_case("s256", 256, 256, 256, act_f32, w_kn, scale, yexp, 1.0)
    manifest.append("s256")

    # ---- 随机 cases (真实统计: 高斯 act/权重 — C_int 量级贴近 transformer) ----
    rng = np.random.default_rng(7)
    for name, m, k, n in [
        ("r256_256_256", 256, 256, 256),
        ("r256_1024_2048", 256, 1024, 2048),
        ("r32_1024_3584", 32, 1024, 3584),
    ]:
        # 每列高斯权重 (σ ~ U(0.8,1.2)), 按列归一量化到 ±7 (真实 per-channel 语义)
        W = rng.normal(0, rng.uniform(0.8, 1.2, (1, n)), (k, n)).astype(np.float64)
        mx = np.abs(W).max(axis=0)
        scale = np.where(mx > 0, mx / 7.0, 1.0).astype(np.float32)
        wq = np.clip(np.rint(W / scale[None, :]), -7, 7).astype(np.int8)
        # 高斯 act σ=1.0 截断 ±11 (真实 post-norm: RMS=1, A_s≈10, 归一化
        # RMS≈0.09 — 与 transformer 实测首 GEMM rms_norm=0.084 同量级)
        act = np.clip(rng.normal(0, 1.0, (m, k)), -11.0, 11.0).astype(np.float32)
        yexp = (act.astype(np.float64) @ (wq.astype(np.float64) * scale)).astype(np.float32)
        write_case(name, m, k, n, act, wq, scale, yexp, 0.0)  # 0=运行时自适应 f
        manifest.append(name)

    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    sys.exit(main())
