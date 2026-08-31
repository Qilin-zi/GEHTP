# Examples — A Guided Tour of All 33 Test Cases (English)

This document is the English-language companion to [TEST_CASES.md](TEST_CASES.md)
(the Chinese per-case checklist) and [EXAMPLES_cn.md](EXAMPLES_cn.md) (the Chinese
version of this tour). Where [README.md](README.md) answers *how to run* the
examples and TEST_CASES.md gives the terse per-case gate list, this document is
the prose introduction: **what each example is, why it exists, what it actually
exercises on the device, and what a green PASS line proves.**

It is written so a reader who has never seen the project can pick it up cold and
understand every one of the 33 examples without reading the source first.

> **Status on the target device** (`52f67807`, V81, unsigned CDSP PD): all 33
> examples build, sign, deploy, and run via `./build_examples.sh all`. The last
> full run (2026-08-16) reported **181 gated checks PASS, 0 FAIL** across 32
> examples, plus example 15's own column-format checklist ending in
> `--- ALL PASS ---` (its per-item lines are not part of the 181).
> Performance figures live in [../PERF_REPORT.md](../PERF_REPORT.md); per-unit
> API manuals in [../docs/api_v22_*.md](../docs/) and
> [../docs/api_v23_*.md](../docs/).

---

## Table of Contents

1. [The Three Generations](#1-the-three-generations)
2. [How the Examples Are Built](#2-how-the-examples-are-built)
3. [The Shared Harness (`common/`)](#3-the-shared-harness-common)
4. [What "PASS" Means Here](#4-what-pass-means-here)
5. [The 33 Examples, One by One](#5-the-33-examples-one-by-one)
   - V2.1 operator regression (01–15)
   - V2.2 engineering units (16–22)
   - V2.3 engineering units (23–33)
6. [Tolerance Matrix Across Examples](#6-tolerance-matrix-across-examples)
7. [Running Everything at Once](#7-running-everything-at-once)
8. [Writing Your Own Example](#8-writing-your-own-example)

---

## 1. The Three Generations

The 33 examples are not 33 unrelated demos — they are three layers of one story:

| Generation | Examples | Layer | Question they answer |
|------------|----------|-------|----------------------|
| V2.1 | 01–15 | **Operator layer** | Is every single HVX/HMX math operator numerically equal to its textbook definition? |
| V2.2 | 16–22 | **Engineering units (U1–U8)** | Do the system parts lifted from closed modules (VTCM weight cache, W4A16 engine, GDN state machine, two-domain execution, op-list engine, threads) behave correctly on real hardware? |
| V2.3 | 23–33 | **Engineering units (U9–U19)** | Do the second wave of units (cache fences, arena, golden harness, worker pool, precision bridge, tree GDN, KV cache, graph step, GEMM dispatch, rollback, buffer audit) hold up? |

A PASS in generation one means "this operator equals the math." A PASS in
generations two and three means "this *mechanism* (cache protocol, concurrency
protocol, audit contract) behaves exactly as specified on silicon" — usually
expressed as byte-exactness against an independent oracle, not just self-consistency.

Unit-to-example map (manuals in `../docs/`):

| Unit | Name | Example | Unit | Name | Example |
|------|------|---------|------|------|---------|
| U1 | wtcache (VTCM weight cache) | 16 | U11 | harness (golden case runner) | 25 |
| U2 | dcmem (arena/file/DMA) | 17, 20, 22 | U12 | wpool (resident worker pool) | 26 |
| U3 | dcthread (threads) | 22 | U13 | pxbridge (f32↔f16↔i16) | 27 |
| U4 | w4a16 (W4A16 HMX engine) | 17, 21, 31 | U14 | gdntree (tree GDN) | 28 |
| U5 | gdnsm (GDN state machine) | 19 | U15 | kvcache (KV slot cache) | 29 |
| U6 | oplist (op-list engine) | 21, 30 | U16 | graphstep (whole-step exec) | 30 |
| U7 | dualdom (two domains) | 20 | U17 | gemmdispatch (MatMul routing) | 31 |
| U8 | smallm (pad-256 GEMV) | 18, 31 | U18 | rbr (rollback / partial accept) | 32 |
| U9 | fence (cache handoff) | 23 | U19 | bledger (buffer audit) | 33 |
| U10 | arena (dual-pool allocator) | 24, 28 | | | |

---

## 2. How the Examples Are Built

Each example lives in its own subdirectory (`NN_name/main.c`) and is compiled
into an **independent** shared object `test_NN_name.so` that links against
`libhvxhmx_v23.so`. The pipeline, orchestrated by
[`build_examples.sh`](build_examples.sh), is identical for every example:

1. **Compile** `main.c` + `common/example_util.c` with the canonical V81 flags
   (`-mv81 -O2 -mhvx -mhvx-length=128B -mhmx -shared -fPIC`), linking
   `-lhvxhmx_v23` and only `-lc -ldl -lgcc` (libqurt.a is not PIC; qurt/HAP
   symbols resolve at runtime from the DSP process).
2. **Sign** the `.so` with SWIV (`swiv_build_utility.py`). This is mandatory:
   the unsigned-PD loader rejects any unsigned `.so`, and silently rejects any
   `.so` containing an unresolvable UNDEF symbol.
3. **Deploy** to `52f67807:/data/local/tmp/hvxhmx23/` (plus the shared assets
   in `assets/`, the oplist blobs, and the golden files).
4. **Run** on the cDSP via `run_main_on_hexagon 3 test_NN_name.so`
   (PD 3 = cDSP; example 20 also uses PD 4 for its second domain).
5. **Read back** the result file `NN_name.txt` that the example itself wrote;
   the script pulls it into `../results/`.

Run one example or all of them:

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh            # all 33
./build_examples.sh 02         # just 02_convf16_gemm
./build_examples.sh 32_rbr     # same, by full name
```

The result file is the single source of truth. A typical good run ends with a
line of the form `--- summary: N pass, 0 fail ---`, and the script's final
banner reads `TOTAL 181 pass 0 FAIL / ALL GREEN`.

---

## 3. The Shared Harness (`common/`)

All 33 `main.c` files share the same skeleton, provided by
[`common/example_util.h`](common/example_util.h) and
[`common/example_util.c`](common/example_util.c). Understanding the harness
once makes every example readable at a glance.

The skeleton is:

1. `ex_open_result("NN_name")` — opens the result file on the device as the
   output channel (absolute path under `/data/local/tmp/hvxhmx23/`; relative
   `fopen` paths fail silently on the DSP).
2. Runtime bring-up — V2.1 examples call `hmx_runtime_setup(2 MB)`; V2.2/V2.3
   unit examples usually open their unit context (`wtcache_open`, `fence_*`,
   arena init, …), which internally powers up HVX/HMX and takes the VTCM.
3. `ex_fill_*` — fill a buffer using a **fixed-seed LCG** (deterministic, no
   `rand`). Variants exist for u8/i8/u16/i16/i32/f16, each taking a seed and a
   scale/range so values land in a controlled interval (fp16 inputs stay inside
   ±0.5, away from the denormal cliff).
4. **Call the unit under test.**
5. **Compute a golden** — either a plain scalar C implementation of the math
   (operator examples), an independent host-reproducible oracle (unit examples),
   or an archived artifact from the closed modules (`Y_gold_2563.raw`).
6. **Compare element-by-element**, take the max error → `ex_check(label, err, tol)`.
   The check records PASS when `err <= tol`. Boolean conditions must be passed
   as `cond ? 0 : 1` — polarity inversion here is a real, historical bug.
7. `return ex_summary()` — returns 1 if any check failed, 0 otherwise.

The crucial property is step 5: **the golden is independent of the code under
test.** A PASS genuinely proves "the hardware path equals the math/spec," not
merely "the library agrees with itself."

The V2.3 harness unit (U11, example 25) formalizes this skeleton into a reusable
case-runner with sha-pinned outputs; new golden cases can be written against it
instead of hand-rolling another main.c.

---

## 4. What "PASS" Means Here

Every gated check is `ex_check(label, err, tol)`:

- `err` is the measured error — max absolute element-wise error for numeric
  gates, or a 0/1 status for contract gates.
- `tol` is the acceptance threshold. Integer families use `tol = 0`
  (bit-exact); fp16 families use `tol = 1` (one ULP); engineering units use
  `tol = 0` for byte-exact mechanisms.
- Gate accounting: the script counts `[PASS]` bracket lines only. 01–14
  contribute 41, 16–22 contribute 42 (example 20 includes one per-segment
  `steps_dumped` gate for ser/a/b), 23–33 contribute 98 — **181 total**.
  Example 15 prints a column-format checklist (`maxrel=... tol=... PASS`)
  which the summary intentionally does not count.

A PASS line therefore means: *on this specific, reproducible input, on real
silicon, the measured behavior met the stated bound.* It is not a coverage
metric and not a smoke test.

---

## 5. The 33 Examples, One by One

### Part I — V2.1 operator regression (01–15)

These answer one question: **does every operator in the library equal its
mathematical definition?** They all follow the template of Section 3: LCG
inputs → operator → scalar golden → element-wise compare.

---

#### 01_runtime_init — runtime lifecycle smoke test

**What it is.** The only example that calls *no operator*. It exists to prove
the runtime layer itself works before anything else is layered on top.

**What it checks (4 gates):**

| Check | Pass condition | What it proves |
|-------|----------------|----------------|
| `hmx_runtime_setup(2 MB)` | returns 0 | cDSP/fastrpc is up; HMX/HVX powered on; VTCM reserved and zeroed |
| `VTCM base non-NULL` | `get_vtcm_base() != NULL` | the VTCM allocation actually succeeded |
| `VTCM size >= request` | `get_vtcm_size() >= 2 MB` | the reservation was honored (this device typically grants the full 16 MB block) |
| `hmx_perf_now_us monotonic` | two successive reads non-decreasing | the HAP qtimer works — every benchmark below depends on it |

**Why it is a separate example.** Every HMX operator requires `setup()` first,
otherwise the first HMX instruction faults (`CX_FAULT`). This example is the
environment health probe: if it fails, the other 32 cannot run, and the fix is
almost always host-side (reconnect the board, restart adb).

**Data setup.** None — no buffers.

---

#### 02_convf16_gemm — fp16 GEMM, single tile

**What it is.** The canonical fp16 correctness proof: one 32×32×32 fp16 matrix
multiply on the **real HMX engine**.

**Operator.** `hmx_convf16(act, wgt, bias, out, 32, 32, 32)`.

**Math.** `out[m,n] = bias[n] + Σ_k act[m,k] · wgt[k,n]`, accumulated in the
HMX fp32-class accumulator and truncated to fp16 by `cvt.hf`.

**Data setup.** M = K = N = 32. `act`/`wgt` filled with seed 7/9 at scale 0.01
(values land around ±0.5, deliberately away from the ±1.0 boundary). `bias`
seed 11 at scale 0.01.

**Golden.** Scalar fp32 accumulation per `(m, n)`, then cast to `__fp16`.

**Pass criterion.** `maxerr ≤ 1` (one ULP).

**What a PASS proves.** The real HMX fp16 systolic path is numerically correct
— not an HVX emulation, not a scalar fallback. This is the foundational
correctness check for the fp16 family.

---

#### 03_convbbb_int8 — u8 × u8 → u8 GEMM

**What it is.** The foundational int8-family correctness check.

**Operator.** `hmx_convbbb(act_u8, wgt_u8, bias_i32, out_u8, 32, 32, 32)`.

**Math.** `out[m,n] = sat_u8( bias[n] + Σ_k act[m,k] · wgt[k,n] )` — int32
accumulation followed by unsigned saturation to 0..255.

**Data setup.** `act` u8 seed 7 range 8, `wgt` u8 seed 9 range 8, `bias` i32
seed 11 centered at 200 half-width 100.

**Golden.** Scalar int32 accumulation with manual saturation.

**Pass criterion.** `maxerr = 0` (**bit-exact**).

**What a PASS proves.** The int8 path is correct. On this part HMX int8 is a
silent NOP, so the library routes int8 GEMM through HVX
(`vmpyacc` + `vasr_sat`). Note the bias type is `int32_t` for integer families,
not `__fp16`.

---

#### 04_convhbh_u16 — u8 × i8 → u16 GEMM (wide dynamic range)

**What it is.** The u16-output family correctness check, exercising two
functions in one example.

**Operators.** `hmx_convhbh` and `hmx_convhhh`. Both are u8 × i8 → u16; they
differ only in the HMX writeback format tag (`:2x1` vs `:2x2`).

**Math.** `out[m,n] = sat_u16( bias[n] + Σ_k act_u8[m,k] · wgt_i8[k,n] )`.

**Data setup.** `act` u8 seed 7 range 4; `wgt` **int8** seed 9 range 6; `bias`
i32 seed 11 (200, 100).

**Golden.** Scalar int32 accumulation with u16 saturation.

**Pass criterion.** `maxerr = 0` for **both** functions.

**What a PASS proves.** The u16-output family (which preserves the wide
accumulator range instead of clamping to u8) is correct, and the two format
variants are mathematically identical.

---

#### 05_i16_weight_convs — the i16-weight family bundle

**What it is.** Four i16-weight families validated in one example, all
`u8 act × i16 wgt`.

| Function | Output | Format |
|----------|--------|--------|
| `hmx_convbcb` | u8 (saturated) | — |
| `hmx_convbnb` | u8 (saturated) | base form |
| `hmx_convhch` | u16 | `:2x2` writeback |
| `hmx_convhnh` | u16 | `:2x1` writeback |

**Pass criterion.** `maxerr = 0` for all four (4 gates).

**What a PASS proves.** `bcb`/`bnb` are mathematically equivalent, `hch`/`hnh`
are mathematically equivalent, and all are bit-exact. Practical note: i16
weights cost roughly 2× the bandwidth of i8 — prefer i8 when the model allows.

---

#### 06_dwconv — depthwise convolution (fp16 + u8)

**What it is.** 3×3 depthwise-separable convolution: each channel is convolved
independently with its own spatial kernel.

**Layout.** `act[H][W][C]` row-major, **no padding**; `wgt[C][9]` channel-major;
`bias[C]`; out-of-bounds samples are skipped (clamped, not zero-padded) — the
golden's boundary `if` must match the kernel exactly.

**Data setup.** H = 5, W = 5, C = 9. fp16 seeds 7/9/11 scale 0.01; u8 seeds
13/15/17.

**Pass criterion.** fp16 ≤ 1 ULP; u8 exact (2 gates).

**What a PASS proves.** The depthwise path (spatial convolution, not GEMM) is
correct in both precisions.

---

#### 07_add — elementwise fp16 add (residual)

**What it is.** The residual-add of ResNet-style networks.

**Operator.** `hmx_add(a, b, bias, out, 32, 32)` — `out = max(0, a + b + bias)`.

**Pass criterion.** Q10-scaled error `round(|d|·1024) ≤ 1`, i.e. `|d| < 1/1024` —
tighter than a raw ULP because the path has no deep accumulation.

---

#### 08_divide — HVX integer divide, five variants

**What it is.** All five HVX divide functions in one example, including
divide-by-zero saturation.

| Function | Precision | Divide-by-zero | Rounding |
|----------|-----------|----------------|----------|
| `hvhx_divide_u8` | u8 | → 0xFF | truncation |
| `hvhx_floor_divide_u8` | u8 | → 0xFF | floor |
| `hvhx_divide_u16` | u16 | → 0xFFFF | truncation |
| `hvhx_floor_divide_u16` | u16 | → 0xFFFF | floor |
| `hvhx_divide_flat_i32` | i32 | → ±INT32_MAX | **round-to-nearest** |

**Data setup.** N = 1024 each; ranges chosen to avoid a flood of divides-by-zero
but **eight zero divisors deliberately injected** per array.

**Pass criterion.** u8/u16 exact; i32 ≤ 1 (5 gates).

> **Watch out.** The i32 path is round-to-nearest, **not truncation**. A naive
> truncating golden disagrees by one. This was a real bug in the project's
> history; do not "simplify" the golden.

---

#### 09_activation — HardSwish + PReLU

**What it is.** Two HVX activation functions.

- `hvhx_hardswish_flat_u16` — MobileNetV3 HardSwish `x·clamp(x+3,0,6)/6`; the
  u16 input is the bit pattern of an **int16 two's-complement Q12** value; the
  vector path approximates 1/6 ≈ 2731/16384, tolerance 2 LSB.
- `hvhx_prelu_u8` — PReLU in offset-binary u8 (zero point 0x80), slope Q7,
  tolerance 1.

> **Watch out.** The HardSwish input is `int16`, not unsigned — golden
  comparisons must read it as `(int16_t)in`.

---

#### 10_reduction — depth-axis reductions, five variants

**What it is.** Reductions along the depth axis: argmin/argmax, find-max,
top-1, reduce-sum. Inputs are flat `[hw][d]`; outputs are one result per row.

| Function | Output |
|----------|--------|
| `hvhx_argminmax_depth_crouton_b` | per-row min/max + indices (u8) |
| `hvhx_argminmax_depth_flat_h` | same (u16) |
| `hvhx_find_max_and_index_in_depth_b` | per-row max + index |
| `hvhx_top1_qu8_dLE32_cr2flt` | per-row top-1 (value + index) |
| `hvhx_reducesum_depth_u8` | per-row sum |

**Pass criterion.** Exact for all five.

> **Watch out.** Names containing `crouton` describe the *internal* processing;
  the **input is still flat row-major**. Do not pack it.

---

#### 11_lookup_unpack — table lookup + weight unpack

**What it is.** Two HVX data-movement primitives.

- `hvhx_table_lookup_flat_u8` — `out[i] = table[idx[i]]`, 256-entry LUT.
- `hvhx_unpack_weights` — 4-bit → 8-bit unpack: `out[2i] = (in[i]>>4)&0xF`,
  `out[2i+1] = in[i]&0xF`.

**Pass criterion.** Byte-exact for both. This is the sentinel that the
library's vgather-free (unsigned-PD-safe) lookup strategy has not regressed.

---

#### 12_multitile_gemm — large fp16 GEMM

**What it is.** `hmx_convf16` at M/N/K > 32, exercising the internal multi-tile
loop across four dimension combinations:

| Run | Dimensions | Stresses |
|-----|------------|----------|
| 1 | 64 × 32 × 32 | M > 32 |
| 2 | 32 × 32 × 64 | N > 32 |
| 3 | 32 × 64 × 32 | K > 32 (accumulator spans tiles) |
| 4 | 64 × 64 × 64 | all axes > 32 |

**Pass criterion.** ≤ 1 ULP for all four. This proves **multi-tile correctness
of the public wrapper** — the K-loop accumulation with a single `clracc` is the
historically off-by-one-prone part.

---

#### 13_compat_dlsym — old-project compatibility via dlsym

**What it is.** A runtime symbol-resolution test: `dlopen` the V2.1-era
`libhvxhmx_v2.so` (copied to the work dir by the build script), then `dlsym`
six symbols (four v73-named, two v81 new-geometry), call each, compare to the
scalar golden from example 03.

**Pass criterion.** `dlopen` succeeds + all six symbols exact (7 gates).

**What a PASS proves.** (a) the shipped library loads cleanly — no unresolvable
UNDEF, which the unsigned PD would silently reject; (b) the v73/v75/v79
compat wrappers work; (c) legacy projects that integrate via dlopen+dlsym keep
working. Run it with `ADSP_LIBRARY_PATH`/`CDSP_LIBRARY_PATH` set to the work
dir, or the dlopen target is not found.

---

#### 14_hmx_peak_gemm — raw HMX K-loop at peak throughput (advanced)

**What it is.** The marquee operator-layer performance example. The same
M = 32, N = 32, K = 256 fp16 GEMM is computed two ways:

1. **Raw K-loop** — NK = 8 act/wgt 32×32 slices packed into VTCM croutons
   **once**, then a hand-written loop issues `clracc → bias → 8 back-to-back
   activation.hf / weight.hf pairs (one shared accumulator) → cvt`. This is the
   documented way to approach the HMX hardware peak.
2. **Public `hmx_convf16` wrapper** — same dimensions, for contrast.

**Correctness gate.** Raw K-loop vs scalar golden ≤ 1 ULP — the low-level path
is *numerically right*, not merely fast.

**Throughput (measured on 52f67807):** raw K-loop ≈ **12.34 TFLOPS**
(0.04 µs/call) vs wrapper ≈ 1.9 GFLOPS — a ~6300× gap, because the wrapper
re-gathers and re-packs every tile, starving the array. The lesson (pre-pack +
raw K-loop for large GEMM) is exactly what the W4A16 engine of example 17
implements. Raise the `NK` macro to amortize `clracc`/`cvt` and approach the
~20.4 TFLOPS theoretical peak.

---

#### 15_v2_llm_ops — the V2 LLM operator layer, full sweep

**What it is.** The V2.1 **operator-family regression**: the `hvhx_v2_*` layer
(60 symbols) that LLM inference actually calls — rms_norm family, l2_norm,
sqrt, sqr, sigmoid, tanh, exp, log, scale, inverse, mul, softmax, and a mini
end-to-end pipeline (manual pack + dequant + GEMM + manual extract, transfer
round-trip, residual output).

**Output format.** Unlike every other example it prints a **column-format
checklist** — `name  maxrel=... tol=... PASS` — one line per item, ending in
`--- ALL PASS ---`. These lines are *not* counted in the 181-gate total; the
example is accounted separately.

**Tolerances.** f32 relative-error bounds of 0.01–0.02 per family; the GEMM
stage of the mini pipeline carries a wider 0.06 bound (dequant path).

**Timing lines.** Nine `PERF` lines: rms_norm_mul n=1024 ≈ 0.30 µs
(~44 GB/s effective), mul ≈ 0.12 µs (~102 GB/s), sigmoid ≈ 0.95 µs, softmax ≈
1.52 µs. Small sizes are call-overhead-dominated; the mul figure approaches
HVX streaming bandwidth.

**What a PASS proves.** The convenience layer LLM code actually links against
is numerically sound across its whole surface, not just its flagship kernels.

---

### Part II — V2.2 engineering units (16–22)

V2.2 kept every V2.1 operator (01–15 run unchanged against the new library) and
added units ported from closed modules. Their PASS means "the *mechanism* is
correct on silicon," usually as byte-exactness against an independent oracle.
Manuals: `../docs/api_v22_*.md`.

---

#### 16_wtcache_pin — U1: the VTCM weight cache (pin + ring)

**What it is.** The LLM-decode memory strategy: weights are **pinned** into
VTCM once (DMA in, resident, reused every step), while per-step activations
flow through a depth-4 prefetch **ring** (DDR→VTCM in, VTCM→DDR out).

**What it checks (9 gates):**

1. Three weights of different sizes (128K/64K/96K) pinned, each verified
   **bit-exact** against its DDR source (3 gates);
2. The three pin slots are distinct addresses; the slots land inside the
   declared pin region;
3. A 1 MB pin verified bit-exact, timed → DDR→VTCM bandwidth (~16 GB/s: 1 MB
   in ~64 µs);
4. An 8-tile 4+4 ring walk: every prefetched tile's content is byte-exact, and
   after drain every moved-back tile equals its source (2 gates);
5. After all ring traffic, the pinned weights are **re-verified intact** —
   ring activity must never corrupt the pin region (the T6 lesson).

**The contract worth remembering.** `wtcache_ring_next`'s third argument is
**this round's** output-slot DDR destination — the ring holds it pending until
the next call or drain, guaranteeing the caller has finished writing before the
DMA reads. Passing the *previous* round's destination shifts everything by one
beat — a real bug this example permanently guards against.

---

#### 17_w4a16_gemm — U4: the W4A16 HMX engine at 256³

**What it is.** The permanent regression anchor for the W4A16 engine
(4-bit weight × 16-bit activation, the workhorse GEMM of the library).

**What it checks (3 gates):**

1. One engine invoke on the canonical 256×256×256 shape; the output surface is
   crouton16_row4, so it is first **decoded to a linear (M,N) matrix**
   (`minv_crouton`, the device mirror of host `inv_crouton16.py`); the decoded
   result must be **byte-exact against `Y_gold_2563.raw`** — the archived
   oracle from the 256³ bit-exact closure: **65536/65536 elements**.
2. A second invoke must reproduce the first byte-for-byte (determinism).
3. 100 timed invocations → median latency (~26 µs ≈ 1.29 TFLOPS at this small
   shape) for the report.

**The gold-pairing trap.** `Y_gold_2563.raw` corresponds to the input
`act_surface.raw` (the t10 baseline surface). The sibling assets
`act_variants/v0.raw` are random variants (~5.7% element-agreement — feeding
them looks like an engine failure but is an input mistake), and `Y_ref_v0.raw`
is an int8-full-precision scalar gold that does **not** track the w4 engine at
all. The t10 closure never gated on it; neither does this example.

---

#### 18_smallm_gemv — U8: small-M GEMV via pad-256

**What it is.** The decode-shape answer: LLM decode steps have M = 1..16
tokens, but the W4A16 tile structure (`m_t = 8`) hard-requires **M to be a
multiple of 256** (M = 32/128 measurably fail). The engineering solution is
**pad-256**: pad the activation surface to 256 rows with the neutral value
32768 (the symmetric-quantization zero), run the full engine, take the real
rows.

**What it checks (5 gates, all on decoded linear outputs):**

1. M=1(padded) row 0 == full M=256 run's row 0 — padding does not pollute real
   rows;
2. M=16(padded) rows 0..15 == full run's rows 0..15;
3. Pad-row outputs are invariant across two different inputs (they really are
   neutral);
4. Row 0 tracks the input (not a constant);
5. Cost gate: median invoke at M=1 equals M=256 (measured Δ = 0%) — the
   tile-walk-bound property that makes padding the *economically* right answer
   for GEMV.

---

#### 19_gdn_sm — U5: the GDN recurrent state machine

**What it is.** The four GDN (Gated DeltaNet) kernel families against their
scalar oracles, with LCG inputs (host-reproducible, no asset transfer). For a
recurrent state machine, "correct" means **after 100 steps the state is still
correct** — accumulated drift is the enemy.

**What it checks (8 gates):**

1. f16→f32→f16 round-trip idempotence over 20k values including subnormals
   (a historical incident point);
2. Conv step vs oracle (cos ≥ 0.9999); block-conv state byte-exact vs the
   stepped state;
3. A guarded bit-exact rerun: a 64KB guard band around the state region must
   be untouched after the run (out-of-bounds-write detection);
4. A 100-token delta-rule chunk loop vs per-token oracle, for both outputs and
   final state (cos gates);
5. solve-tri vs back-substitution oracle;
6. Chunk-size freedom: a length-16 block run as 8+8 equals the whole block.

---

#### 20_dualdomain — U7: two CDSP protection domains, one binary

**What it is.** A **shard executor**: `run_main_on_hexagon <dom>
test_20_dualdomain.so dd <tag> <start> <len>` (note the fourth argument is a
**length**, not an end index — `dd b 8 16` fails range-check; it must be
`dd b 8 8`).

**Orchestration (done by the build script):** run `ser` on domain 3 (16 steps,
the serial baseline); then run `a` (domain 3, steps 0–8) and `b` (domain 4,
steps 8–16) **concurrently**. Each step computes, writes its 128KB output,
hashes it with the on-device sha256, and deletes the dump. The host-side
`analyze_dd.py` then proves the **split-equivalence contract**: every a/b step
hash equals the matching ser step hash (gate), plus the per-segment
`steps_dumped` gates (3) and a steps-complete gate.

**Measured:** ser ≈ 545 µs vs a/b ≈ 272/274 µs each → **~2.0× speedup**
(report-only; scheduling jitter makes it 1.99–2.00 across runs).

**What a PASS proves.** Splitting a stateful pipeline across two PDs is
*bitwise* equivalent to running it serially — the foundation for the
"scale by adding domains, not threads" conclusion (contrast example 22).

---

#### 21_oplist_exec — U6: blob parse + execute, with independent gold

**What it is.** The op-list engine: a whole op sequence (PIN weights → MATMUL →
RMSNORM → …) compiled on the host into a `.wtop` blob (weights included),
executed on-device by one `wt_exec_run()` call, with per-op timing.

**What it checks (6 gates):**

1. **Negative parsing**: five deliberately corrupted blob copies (bad magic,
   version, endian flag, slot count, op count) must each be rejected with the
   exact expected error code — the error path is part of the contract;
2. The device emits W3 report lines for the blob; the build script diffs them
   **line-by-line against the host `wt_inspect` tool** (same parser source on
   both sides — any diff is a transport bug);
3. `blob_w4` (single MATMUL) executes; then `blob_w5` across an engine
   re-initialization must be byte-exact (engine reuse is sound);
4. The decoded MATMUL output within **40 LSB of the independent K2560 scalar
   gold** (nominal closed-module value: 37 LSB);
5. The RMSNORM stage bit-exact (**0 ULP**) against a same-algorithm scalar
   mirror embedded in the example;
6. Suite completion.

**Timing read-out:** w4 MATMUL op ≈ 24.7 ms, w5 ≈ 46.9 ms full-path (weight
restage + DMA + compute) — the gap between these and the ~26 µs pure invoke of
example 17 is exactly why U1 wtcache exists.

---

#### 22_dualcore_threads — U3: threads are for correctness, not throughput

**What it is.** The same-domain concurrency verdict, made permanent. Two
`dc_spawn` workers (explicit 64KB stacks), a start barrier, two engines carved
from one arena, one shared DMA mutex, and a VTCM flag handshake (0→1→2).

**What it checks (6 gates):**

1. Both concurrent engine outputs byte-exact vs the serial reference (2 gates);
2. Three pure-HVX operators (`dc_hvx_load`/`dc_norm_i16`/`dc_dot_u64`) equal
   their main-thread values — cross-thread determinism (3 gates);
3. The flag handshake completes.

**The soul of the example — the hmx_lock handoff.** The HMX lock is
**held-by-thread**: whichever thread invokes HMX must hold it. The template:

```
main:    wtcache_hmx_unlock()      ← release (main held it since open)
         spawn workers
worker:  wtcache_hmx_lock()  → invoke → wtcache_hmx_unlock()
main:    join; wtcache_hmx_lock()  ← re-acquire
```

Skipping the handoff crashes the PD outright (return ≈ −2147482611), it does
not produce a FAIL line.

**Measured:** concurrent/serial ratio ≈ 0.96 — slightly *slower*, exactly as
the closed-module conclusion predicts (single DMA engine + global HMX lock).
Throughput scaling comes from example 20's domains; threads buy concurrency
*correctness*.

---

### Part III — V2.3 engineering units (23–33)

The second wave: infrastructure units (fence, arena, harness, wpool), numerical
bridges (pxbridge, gdntree), inference-state managers (kvcache, graphstep,
gemmdispatch), and integrity machinery (rbr, bledger). Manuals:
`../docs/api_v23_*.md`. Dependency note: fence/arena/harness/pxbridge/rbr/
bledger are standalone; wpool and kvcache build on fence; gdntree on
fence+arena+pxbridge; graphstep on wpool; gemmdispatch on the example-17/18
engines.

---

#### 23_fence — U9: the direction-paired cache-fence decision table

**What it is.** The V2.2 "four cache rules" (CPU→DMA flush, CPU-written VTCM
flush, DMA-written invalidate, close on exit) distilled into **one API**
(`fence_handoff`) driven by a decision table over (writer, reader, memory
domain). All V2.3 code routes its cache maintenance through this unit instead
of hand-written `qurt_mem_cache_clean` calls.

**What it checks (7 gates):**

1. **Decision table**: all 32 (writer, reader, domain) combinations resolved by
   `fence_op_for` equal the built-in expectation table; the nonsensical
   HMX-writes-DDR cell is always rejected (`FO_INVALID`);
2. **Invalid combinations rejected** by the parameter-domain checks;
3. **CPU→DMA over DDR** (rule ①): 200 rounds of variable-length LCG patterns,
   fenced, DMA'd DDR→VTCM with bypass, compared byte-for-byte;
4. **CPU→HMX→CPU over VTCM**: the W4A16 engine runs 20 rounds — same input is
   bit-exact every round *and* a different input must produce a different
   output (input sensitivity catches a cached/stale read that a pure
   round-trip would miss);
5. **CPU→HVX over VTCM**: memset patterns are visible to an HVX checksum
   (pattern-sensitive, so a stale cache would be caught) and the round trip
   reproduces;
6. **CPU→DMA over VTCM, reverse direction** (move-back): VTCM→DDR byte-exact —
   the source-FLUSH half of the contract.

**What a PASS proves.** The fence table encodes the cache protocol correctly
for every legal direction pair, and each fence actually does physical work on
silicon (bypass DMA reads prove it, not just "no crash").

---

#### 24_arena — U10: the dual-pool aligned arena

**What it is.** The allocator discipline unit: a DDR pool plus VTCM scratch
with alignment guarantees, free, and coalescing — because hot paths must not
`memalign`/`free` per call (a real +38 ms jitter bug from the closed modules).

**What it checks (4 gates):**

1. **1000 rounds** of random size/align alloc/free: every returned pointer
   honors its requested alignment (128B or 2KB), and live-pointer bookkeeping
   stays exact;
2. **No leak, full coalescing**: after releasing everything, `used == 0` and
   `largest_free == capacity` — no fragmentation residue;
3. **Mixed-tenant coexistence**: f32 GDN states and 2KB-aligned W4A16 faces
   allocated/freed interleaved over 100 rounds — the GDN state content is
   never corrupted by face allocations (the class of bug where one tenant's
   free-block merge hands out a block still logically owned by another);
4. **Fragmentation bound**: after random churn and releasing 90%,
   `largest_free ≥ 80%` of capacity — merges really happen.

---

#### 25_harness — U11: the golden-case runner

**What it is.** The meta-example: the framework (`harness.h`) that turns "a
case with inputs, a runner, and an oracle" into a sha-pinned record. It proves
itself by **re-running two previously hand-closed cases through the framework**
and requiring identical results.

**What it checks (13 gates):**

1. Case `gdn_sm` (the core of example 19): oracle cos gate + the framework's
   direct run byte-identical to the hand-written run (2+2 gates across first
   run and rerun);
2. Case `w4a16_gold` (the core of example 17): `act_surface` → decode vs
   `Y_gold_2563` **65536/65536 byte-exact** (again across rerun);
3. Case `solve_tri`: a pure-scalar case proving the framework does not require
   an engine;
4. **sha determinism**: two full runs of the same case produce an identical
   `harn_last_sha` — outputs are pinned bit-for-bit, so any future regression
   flips the hash.

**What a PASS proves.** New golden cases can be written against the harness
and will reproduce the hand-written closures exactly — with a stable
content hash as the regression signal.

---

#### 26_wpool — U12: the resident worker pool

**What it is.** The concurrency-work engineering answer: spawn-per-op costs
90×+ in thread creation (the op82 conclusion), so jobs go to **persistent
workers**. The pool owns the hmx-lock handoff so engine jobs are safe.

**What it checks (5 gates):**

1. **Random arrival**: 24 mixed jobs (norm/dot/hvxload) submitted in random
   order — every pool result equals the main-thread serial reference
   value-for-value (concurrency changes timing, never numbers);
2. **HMX gate**: two engine jobs using the unlock→job-lock→relock handoff
   produce output byte-exact vs serial invocation — the example-22 template,
   productized;
3. **spawn16**: 16 norm jobs via the pool all value-exact;
4. **The economics gate**: pool path must beat spawn-per-op — measured
   **109 µs vs 484 µs (4.4×)** for the same 16 jobs;
5. **Stress**: 5 rounds × 12 jobs; `done`/`executed` counters equal total
   submissions, every value correct.

---

#### 27_pxbridge — U13: the precision bridge, unipolar contract

**What it is.** The conversion layer between f32, f16, and symmetric INT16
(zp = −32768) — the boundary every quantized pipeline crosses. The example
runs the full contract **at four scale factors** (hence 20 gates: 4 scales ×
5 checks, plus linearity, extremes, and batch).

**What it checks:**

1. **f32→f16→f32 round trip**: 100k random values, error within the 0.5-ULP
   envelope (subnormal neighborhood included);
2. **INT16 symmetric decode**: error ≤ scale/2 (the half-step envelope), and
   zero ⇔ code 0x8000 exactly — zero is represented by exactly one code;
3. **Negative clamps to zero code** (the unipolar contract: this bridge never
   produces negative decoded values);
4. **f16↔i16 combination bridge**: composed conversion within scale/2 + a
   half-ULP of the direct path;
5. **Code space linear**: adjacent codes decode to values differing by exactly
   `scale`, across **all 65535 steps**;
6. **Endpoint clamping**: ±overflow inputs land on 0xFFFF/0x0000 and decode
   one-sided-bounded;
7. **Batch == scalar**: vector API byte-identical to the scalar API.

---

#### 28_gdn_tree — U14: tree-shaped GDN, closed form vs kernel

**What it is.** Tree-structured GDN evaluation (the draft-model shape): nodes
form a forest, each child decays its parent's committed state. The unit ships
**three mutually independent implementations**: a serial recursion oracle, a
closed-form solution, and the device f16 kernel.

**What it checks (8 gates;** T = 8/16/32 × 3 random topologies, D = 64,
LCG-reproducible**):**

1. **Closed form vs serial oracle** (all f32): outputs *and* per-node commit
   states at cos = 1.0000000 — the host math closure (2 gates);
2. **Device f16 kernel vs closed form**: output cos ≥ 0.999, per-node state
   min cos ≥ 0.999 (2 gates);
3. **Kernel rerun bit-exact**;
4. **Topology validation**: `parent[0] ≠ −1` and `parent[i] ≥ i` are both
   rejected — out-of-order DAGs are not silently mis-evaluated;
5. **INT16 decay curve**: the quantized stepwise decay product vs the f32
   `exp(path-sum)` reference stays within `depth·6e-4 + 1e-3`, and is never
   worse than the f16 path at any depth — the quantization budget for deep
   trees;
6. State buffers come from the U10 arena and are **fully freed** at exit.

---

#### 29_kvcache — U15: the KV slot cache

**What it is.** Token-KV cache management in slot form — the inference-state
unit every decode loop leans on. A host **shadow model** mirrors every
operation, and every gate is a byte-level or counter-level match against it.

**Configuration:** 64 slots × 256B (K 128B + V 128B).

**What it checks (8 gates):**

1. Slot sizes not a multiple of 128B are **rejected at init**;
2. **append of 64 positions**: per-slot K/V bytes and the position map match
   the shadow; every `lookup` hits;
3. **Wrap-around dynamics** (posIdsIdx): after 3N appends into N slots, old
   positions (< 2N) miss, new positions hit, and the **evict counter is
   exact**;
4. **scatter rewrite isolation**: a scatter touches only its own slot's bytes
   and posmap entry — neighbor-slot canaries stay intact; invalid slot ids
   rejected;
5. **Slot 128B alignment** holds for both K and V faces at all times;
6. **DMA bypass read-back**: after append (with its built-in fence), a real
   `dc_dma_once` reads a slot's K face — byte-identical to the shadow. This is
   the cache rule physically verified, not assumed;
7. **verify-rewrite semantics**: scattering the same slot again (target
   correction) makes the read return the corrected value while `lookup(pos)`
   still hits.

> The **pos ≡ slot (mod N)** congruence trap: positions must be congruent to
> their slot index or lookups miss forever — test positions use the
> `64k + slot` form.

---

#### 30_graph_step — U16: whole-step execution vs per-op dispatch

**What it is.** The graph-step engine: an op list executed either **fused**
(one `wt_exec_run` call) or **split** (per-op `run_range`), which must be
indistinguishable in results. The example synthesizes its blob **on-device**
from the s256 assets — no host packer dependency — with the op sequence
`NOP / PIN(wt) / MATMUL / PIN(wt) / RMSNORM / SILU` (exercising the new
`OP_SILU_F16 = 4`).

**What it checks (8 gates):**

1. `wt_parse` accepts the synthesized blob (v1, with SILU);
2. The fused run completes;
3. **Stats exact**: ops = 6, nop = 1, matmul = 1, rmsnorm = 1, silu = 1,
   pin = 2, skipped = 1 — the counters tell the true story of what ran;
4. **PIN-skip semantics**: before the engine exists, a PIN op only book-keeps
   (`skipped`); once the engine is built, PIN truly stages. First (fused) run
   → skipped = 1; second (split) run with the engine ready → skipped = 0;
5. The split run completes;
6. **Fused vs split: all three temp surfaces byte-identical** — the central
   equivalence contract;
7. PIN-skip counter reaches zero in the second run;
8. **SILU vs scalar oracle** within a half-ULP f16 envelope.

Fused-vs-split total timing (and a per-op table) is reported, not gated.

---

#### 31_gemm_dispatch — U17: the MatMul three-route decision

**What it is.** The front door for every matrix multiply: a router that picks
one of three execution routes by shape, plus the executors behind each route.
This is where the "kernel single-invoke ABI is fixed at M=256" reality is
handled.

**Routes:**

| Route | Shape | Executor |
|-------|-------|----------|
| `GR_SMALLM` | M = 1 (decode GEMV) | pad-256 + full engine, take row 0 |
| `GR_DENSE_F16` | small M, small K/N | dense f16 GEMM |
| `GR_W4A16` | M = 256 / 512 (multiple of 256) | W4A16 engine, 256-row blocks |

**What it checks (8 gates):**

1. **Decision table**: M swept 1..600 × K/N ∈ {64, 256, 2560} — every point
   matches the routing rules (1800+ decisions, 0 mismatch); boundaries pinned:
   M=1 SMALLM, M=32/128 DENSE, M=256/512 W4A16;
2. **GR_W4A16 at 256³**: full engine vs `Y_gold_2563` byte-exact
   (65536/65536) — the example-17 anchor, reachable through the router;
3. **M=512 split**: the dispatch layer splits into 2×256 blocks (the kernel's
  descriptor hard-codes M=256; a monolithic 512 face produces garbage in the
   upper rows — the V2.3 lesson) — upper half == gold A, lower half == gold B,
   both byte-exact; plus a crouton512 encode/decode round-trip identity;
4. **GR_SMALLM**: M=1 row 0 == full-engine row 0 == gold row 0 (two gates);
   cost gate — M=1 median (1510 µs) within 50% of M=256 (1028 µs), measured
   ratio 1.47;
5. **GR_DENSE_F16**: M=32/128, K=N=64 random f16 vs an f32-accumulating scalar
   oracle: cos ≥ 0.9999 and max|Δ| within the half-ULP envelope.

---

#### 32_rbr — U18: recurrent-state rollback with partial acceptance

**What it is.** The speculative-decoding safety net for **stateful** engines:
when a tree round accepts only some of its drafted steps, the engine state must
roll back to the accepted prefix and replay from there — and after 16 mixed
rounds the engine must be **bitwise** where an honest serial run would be.
Implements the verification pyramid of
`docs/P0-recurrent-state-rollback.md` with a deterministic f32 mock engine
(conv state + recurrent state per group) so bitwise judgment is meaningful.

**The three levels (8 gates):**

- **L1 module contract**: snapshot round-trip restores a deliberately polluted
  state byte-exact; SKIP is **one-shot** and consumed/invalidated by every
  in-state rewrite branch (the CLEAR branch once leaked a pending SKIP into
  the next frame — a real bug this gate pins); NOSNAP / STALE / FROZEN /
  out-of-range rewind are all rejected;
- **L2 round-level assertions**: the **defect fingerprint** (equation 1:
  `replay_in == phantom_out`) — a deliberately defective engine (unconditional
  out→in state copy at setup) is *detected* by the fingerprint and deviates
  from baseline; restore-effectiveness (equation 2: `replay_in ==
  snapshot_src` byte-exact); replayed KV-row overwrites are idempotent;
- **L3 end-to-end**: 16 mixed-accept rounds (including a forced reject-all at
  j=0 and a full-accept flow) — final engine state + KV rows + **both n_past
  ledgers** are bitwise-equal to the non-speculative baseline; counters exact
  (16 snapshots, restore = rewind = 6, skip = 5 in the zero-touch flow).

---

#### 33_bledger — U19: the Buffer Ledger, dataflow provenance audit

**What it is.** The "consumption of never-produced data errors immediately"
unit: every buffer row carries provenance (which writer produced it, with
which quantization tag), and the ledger flags the classic uninitialized-read
bug families at the moment of consumption instead of silently propagating
garbage. Built on the test template of
`docs/P1-uninitialized-read-dataflow.md`, simulated as a logits pipeline
(`[rows][VOC]` f32, writers W_PREFILL / W_TREE / W_DECODE / W_LATE, hash-noise
distributions with a deterministic dominant peak so rank/argmax judgments are
stable).

**What it checks (9 gates):**

1. **Contract table**: the full verdict matrix (OK/NEVER/CANARY/RELEASED/
   WRITER/QTAG) across writer/reader/state combinations;
2. **T1 cold start**: consuming an unproduced row → immediate `NEVER`; writing
   then converging the chain repairs it;
3. **T2 row misalignment**: verify = NEVER **and the E2 rank reverse-lookup
   localizes which row** is misaligned;
4. **T3 canary patrol**: post-canary reads report `CANARY` with the 0xAA fill
   pattern, plus a full-row deep rank that distinguishes H2 from H1a;
5. **T4 released buffers**: reads after release → `RELEASED`, with the
   pre-release canary bit pattern as evidence;
6. **T7 double-write detection**: an unread row overwritten by a *different*
   writer increments the overwrite counter; overwrite-after-read is legal;
7. **T5+T6**: new-path consumption without prior write → NEVER; qtag
   write/read mismatch → QTAG; a single configuration source resolves it;
8. **T8 end-to-end stethoscope**: the P1 §5 incident replayed — a garbage
   draft engine (d1) paired with a correct verifier (d2) collapses the
   acceptance histogram to all-1s with a break every round; the repaired
   engine gives breaks = 0, the regression anchor (argmax of the first tree
   proposal == argmax of the prefill last row) holds, and the histogram goes
   all-3s;
9. **L1 timeline**: the text-mode timeline (P1 §4.4) is non-empty and carries
   statistics.

---

## 6. Tolerance Matrix Across Examples

| Family | Tolerance | Examples |
|--------|-----------|----------|
| Integer GEMM (bbb / hbh / bcb / bnb / hch / hnh) | **0 (bit-exact)** | 03, 04, 05, 13 |
| u8 depthwise, reductions, lookup, unpack, divides (u8/u16) | **0 (bit-exact)** | 06, 08, 10, 11 |
| HVX divide i32 (round-to-nearest) | ≤ 1 | 08 |
| HardSwish (Q12 approximation) / PReLU | ≤ 2 LSB / ≤ 1 | 09 |
| fp16 GEMM / depthwise / add / K-loop | ≤ 1 ULP (add: < 1/1024) | 02, 06, 07, 12, 14 |
| V2 LLM ops (f32) | rel ≤ 0.01–0.02 (GEMM stage 0.06) | 15 |
| W4A16 vs archived oracle / pad-256 / engine reuse | **0 (byte-exact)** | 17, 18, 20, 22, 26, 31 |
| W4A16 vs K2560 independent scalar gold | ≤ 40 LSB (nominal 37) | 21 |
| RMSNORM vs same-algorithm mirror | **0 ULP** | 21 |
| SILU vs scalar oracle | ≤ half-ULP f16 | 30 |
| GDN conv/delta vs oracle | cos ≥ 0.9999 | 19, 25 |
| GDN solve-tri vs oracle | cos ≥ 0.99995 | 19, 25 |
| gdntree closed-vs-serial / kernel-vs-closed | cos = 1.0 / ≥ 0.999 | 28 |
| pxbridge f16 round trip | ≤ 0.5 ULP envelope | 27 |
| pxbridge INT16 symmetric | ≤ scale/2 (half-step) | 27 |
| gemm M=512 split vs separate golds | **0 (byte-exact)** | 31 |
| dense f16 vs scalar oracle | cos ≥ 0.9999 + ≤ 0.5 ULP | 31 |
| kvcache vs host shadow / wpool HMX / graph_step temps | **0 (byte-exact)** | 26, 29, 30 |
| rbr all levels (L1/L2/L3) | **0 (bitwise vs baseline)** | 32 |
| bledger state machine / counters | **0 (exact match)** | 33 |

---

## 7. Running Everything at Once

```bash
cd /disk1/V81Dev/hvxhmx_libsV2.3/examples
./build_examples.sh all          # compile / sign / deploy / run all 33
./build_examples.sh 20           # one example (number or full name)
```

Each example writes its result to the device and the script pulls everything
into `../results/` (example 20 additionally keeps its per-segment ser/a/b logs
and the host cross-check file; example 21 appends the host-vs-device W3 diff
gate). Expected end banner:

```
=== summary (results/) ===
  TOTAL                            181 pass   0 FAIL
=== ALL GREEN ===
```

To inspect one result:

```bash
grep -E 'PASS|FAIL' ../results/32_rbr.txt
```

If anything fails, [TEST_CASES.md](TEST_CASES.md) lists every gate per case,
and [../PERF_REPORT.md](../PERF_REPORT.md) §V2.3 collects the hard-won device
lessons (UNDEF rejections, skel version, hmx_lock handoff, crouton decoding,
one-shot flags, …).

---

## 8. Writing Your Own Example

Copy any existing `main.c` as a template
([`01_runtime_init/main.c`](01_runtime_init/main.c) is the smallest; a V2.3
unit example like [`24_arena/main.c`](24_arena/main.c) shows the unit-context
style). The recipe:

1. `#include "hvxhmx_v23.h"` and `#include "example_util.h"`.
2. Declare buffers as `static ... __attribute__((aligned(128)))` globals —
   never VLAs (the DSP stack is tiny; 128B alignment is an HVX load
   requirement).
3. Bring up the runtime: `hmx_runtime_setup()` for bare operators, or the
   unit context (`wtcache_open` / arena / fence) for unit code. Take the HMX
   lock on whichever thread invokes.
4. Start with `ex_open_result("your_name")` and end with
   `return ex_summary()` — and close/teardown the context on **both** the PASS
   and FAIL paths.
5. Write the golden independently (scalar loops, a host shadow model, or an
   archived artifact) and compare element-by-element; pick the tolerance from
   the matrix above. Boolean conditions go in as `cond ? 0 : 1`.
6. If your output comes from the W4A16 engine, decode the crouton surface
   first — never compare raw surface bytes or use "row offsets."
7. Add your example to the `EXAMPLES` array in
   [`build_examples.sh`](build_examples.sh), then run
   `./build_examples.sh your_name`.

The compile/link/sign/deploy/run sequence is identical to the library's own —
see [`build_examples.sh`](build_examples.sh) for the exact `hexagon-clang`
invocation.
