# HTP Step Expansion Rules — Empirical Analysis from 8 Comparison Samples

> Derived from parsing opData sections of 8 real SDK context .bin files
> Date: 2026-08-07

## 1. Sample Overview

| Name | Steps | Cmp | Mem | DMA | Sync | Ops | Input Shape |
|---|---|---|---|---|---|---|---|
| simple_linear | 19 | 11 | 3 | 4 | 1 | Trans+Reshape+FC+Reshape+Trans | [1,4,3] |
| linear_8x4 | 19 | 11 | 3 | 4 | 1 | same | [1,8,4] |
| linear_4x8 | 19 | 11 | 3 | 4 | 1 | same | [1,4,8] |
| linear_4x3_w4 | 18 | 10 | 3 | 4 | 1 | same (W[4,4]) | [1,4,3] |
| fc_only | 14 | 8 | 3 | 2 | 1 | FC only | [1,3,4] |
| trans_only | 13 | 8 | 3 | 1 | 1 | Transpose only | [1,4,3] |
| reshape_only | 9 | 7 | 2 | 0 | 0 | Reshape only | [1,4,3] |
| two_fc | 17 | 10 | 3 | 3 | 1 | Two FCs | [1,3,4] |

## 2. Framework Steps (Always Present)

### 2.1 InputNodeSetup (steps 0-1, always 2 steps)

**Step 0** — InputNodeSetup null (constant):
```
cnt=0, compute, f2=0x82D28377, blk=0x1003
extras=[1, 0, 0, 0, 0]
```

**Step 1** — InputNodeSetup shape (hash-dependent):
```
cnt=0, compute, f2=htp_shape_hash([1, batch, dim0, dim1]), blk=0x1003
extras=[1, 1, dim0, dim1, ???]
```

**Key finding**: HTP internally represents tensors as 4D (NCHW). A 3D input [batch, dim0, dim1] is padded to [1, batch, dim0, dim1] (adding channel=1). The hash is computed from `extras[1:]` = [1, batch, dim0, dim1].

Verified hash values:
- [1,1,4,3] → 0x9DCAC54A  (simple_linear, trans_only, reshape_only input)
- [1,1,8,4] → 0xA1CB89D7  (linear_8x4 input)
- [1,1,4,8] → 0x9DCAC54F  (linear_4x8 input)
- [1,1,3,4] → 0x9CCA9428  (fc_only, two_fc input)

### 2.2 InputSlice (step 2, always 1 step)

```
cnt=1, compute, f2=0x02, blk=0x1000|block
extras=[3, 3, 4, 3, 4, 4, 2, 1, flag]
```

**Key finding**: InputSlice extras are **constant** across ALL samples (except last element):
- `[3, 3, 4, 3, 4, 4, 2, 1]` — fixed tiling params for HVX vector width 4
- `flag` = 0x10020 if input-side Transpose present, 0x10000 if not

### 2.3 perm_load (step 3, always 1 step)

```
cnt=2, memory, f2=0x02, blk=0x1000|block
extras=[0, 1]  — always fixed
```

### 2.4 Output Tail (last 3-5 steps, always present)

**OutputNodeSetup** (present if has DMA/sync):
```
cnt=0, compute, f2=htp_shape_hash([1, batch, out_dim0, out_dim1]), blk=0x1003
extras=[1, 1, out_dim0, out_dim1, ???]
```

**SyncOp** (present if has DMA):
```
cnt=sync_tid, sync, f2=0x03, blk=0x1000|block
extras=[6, 0x0E, ..., 3, 3]
```

**perm_load (output)**:
```
cnt=2, memory, f2=0x03, blk=0x1000|block
extras=[0, 1]  — always fixed
```

**Transpose_impl (output)**:
```
cnt=3, compute, f2=0x03, blk=0x1000|block
extras=[1]  — always fixed
```

**OutputSlice (output)**:
```
cnt=output_tid, compute, f2=0x00, blk=0x1000|block
extras=[0]  — always fixed
```

## 3. Op-Specific Step Patterns

### 3.1 Reshape Only (adds 2 steps)

Between InputSlice and output tail:
```
Step A: cnt=3, compute, f2=0x02, blk=0x1000|block, extras=[1]
Step B: cnt=4, compute, f2=0x03, blk=0x1000|block, extras=[tid, D0, D1, D1]
```

No DMA, no sync, no OutputNodeSetup.

### 3.2 Transpose Only (adds 5 steps)

```
Step A: cnt=3, memory, f2=0x02, extras=[2, 1]    — layout_convert
Step B: cnt=4, compute, f2=0x11, extras=[...]     — flat_from_vtcm
Step C: cnt=4, compute, f2=0x12, extras=[...]     — fc_prep
Step D: cnt=5, DMA, f2=0x11, extras=[...]         — DMA SET (output)
```
Plus OutputNodeSetup + SyncOp in the output tail.

### 3.3 FC Only (adds 6 steps)

```
Step A: cnt=3, memory, f2=0x02, extras=[2, 1]    — layout_convert
Step B: cnt=4, DMA, f2=0x02, extras=[...]         — DMA SET (W load)
Step C: cnt=5, compute, f2=0x10, extras=[...]     — MatMul (x·W)
Step D: cnt=5, compute, f2=0x11, extras=[...]     — MatMul (+b)
Step E: cnt=6, DMA, f2=0x12, extras=[...]         — DMA SET (output)
```
Plus OutputNodeSetup + SyncOp in the output tail.

### 3.4 simple_linear: Transpose+Reshape+FC+Reshape+Transpose (adds 11 steps)

```
Step A: cnt=3, memory, f2=0x02, extras=[2, 1]    — layout_convert
Step B: cnt=4, compute, f2=0x11, extras=[...]     — flat_from_vtcm (reshape prep)
Step C: cnt=4, compute, f2=0x12, extras=[...]     — fc_prep
Step D: cnt=5, DMA, f2=0x11, extras=[...]         — DMA SET (W)
Step E: cnt=6, DMA, f2=0x11, extras=[...]         — DMA SET (b)
Step F: cnt=4, compute, f2=0x14, extras=[...]     — MatMul (x·W)
Step G: cnt=4, compute, f2=0x15, extras=[...]     — MatMul (+b)
Step H: cnt=7, DMA, f2=0x16, extras=[...]         — DMA SET (output_fc)
Step I: cnt=4, compute, f2=0x1A, extras=[...]     — flat_from_vtcm (reshape out)
Step J: cnt=8, DMA, f2=0x1A, extras=[...]         — DMA SET (output_ncf)
```
Plus OutputNodeSetup + SyncOp in the output tail.

### 3.5 Two FCs (adds 9 steps)

```
Step A: cnt=3, memory, f2=0x02, extras=[2, 1]    — layout_convert
Step B: cnt=4, DMA, f2=0x02, extras=[...]         — DMA SET (W1)
Step C: cnt=5, compute, f2=0x10, extras=[...]     — MatMul1 (x·W1)
Step D: cnt=5, compute, f2=0x11, extras=[...]     — MatMul1 (+b1)
Step E: cnt=6, DMA, f2=0x14, extras=[...]         — DMA SET (intermediate)
Step F: cnt=5, compute, f2=0x12, extras=[...]     — MatMul2 (·W2)
Step G: cnt=5, compute, f2=0x13, extras=[...]     — MatMul2 (+b2)
Step H: cnt=6, DMA, f2=0x16, extras=[...]         — DMA SET (output)
```
Plus OutputNodeSetup + SyncOp in the output tail.

## 4. f2 (Phase Index) Allocation

f2 values represent dependency phases. Observations:

| Pattern | f2 values | Description |
|---|---|---|
| Input side | 0x02 | InputSlice, perm_load, layout_convert, input DMA |
| Output side | 0x03 | output perm_load, Transpose_impl, OutputSlice, SyncOp |
| OutputSlice | 0x00 | Always 0x00 |
| FC compute | 0x10, 0x11, 0x12, 0x13, ... | Sequential per FC block |
| Transpose prep | 0x11, 0x12 | flat_from_vtcm + fc_prep |
| DMA tags | 0x02, 0x11, 0x12, 0x14, 0x16, 0x1A | Allocated based on dependency |

**Key insight**: DMA tags match the f2 of dependent compute ops. E.g.:
- DMA W tag=0x11 ↔ flat_from_vtcm f2=0x11 (depends on W)
- DMA out_ncf tag=0x1A ↔ flat_from_vtcm f2=0x1A (depends on ncf)

**Allocation pattern for simple_linear**:
- 0x02: input phase (InputSlice, perm, layout, input DMA)
- 0x11: weight phase (flat_from_vtcm, DMA W+b)
- 0x12: FC prep phase
- 0x14: MatMul phase 1
- 0x15: MatMul phase 2
- 0x16: output_fc DMA
- 0x1A: output_ncf phase (flat_from_vtcm, DMA)
- 0x03: output tail
- 0x00: OutputSlice

## 5. VTCM Block Allocation

Block refs are allocated by the FancyAllocator based on tensor lifetimes and sizes. The allocation is NOT sequential — blocks are reused after tensors are consumed.

Block ranges observed:
- reshape_only: 0x15-0x1C (8 blocks)
- trans_only: 0x15-0x24 (10 blocks)
- fc_only: 0x19-0x24 (8 blocks)
- simple_linear: 0x19-0x31 (14 blocks)
- two_fc: 0x1D-0x2A (9 blocks)

**TODO**: Reverse-engineer FancyAllocator for correct block assignment.

## 6. Shape-Dependent vs Fixed Values

### Fixed (shape-independent):
- InputSlice extras: [3, 3, 4, 3, 4, 4, 2, 1, flag]
- perm_load extras: [0, 1]
- layout_convert extras: [2, 1]
- Transpose_impl extras: [1]
- OutputSlice extras: [0]
- InputNodeSetup null: f2=0x82D28377, extras=[1,0,0,0,0]
- blk=0x1003 for node setup steps

### Shape-dependent:
- InputNodeSetup shape: f2 = htp_shape_hash([1, batch, dim0, dim1])
- OutputNodeSetup: f2 = htp_shape_hash([1, batch, out_dim0, out_dim1])
- InputSlice flag: 0x10020 (with Transpose) or 0x10000 (without)
- DMA extras: encode tensor IDs and dimension values
- MatMul extras: encode dimensions (e.g., dim0, dim1 in extras)

### linear_4x3_w4 difference (18 vs 19 steps):
When W is square (W[4,4] with input[1,4,3]):
- Step 10 (MatMul +b) has fewer extras: [0, 0x00000003, 0x00030004] instead of
  [0, 0x80000007, 0xCCCC0001, 0x40001000, 2, 0x00030004]
- The OutputNodeSetup step is missing (merged or skipped)
- This saves 1 step

## 7. DMA Extras Patterns

### Weight DMA (small format, 6 extras):
```
[counter, phase, W_tid, 0, 1, 0x00020004]
```
Used for: fc_only W, two_fc W1, simple_linear b

### Weight DMA (large format, 11 extras):
```
[counter, phase, W_tid, b_tid, out_tid, 2, 0x80000000|W_tid, 0x40001100, dim0, dim1, 0x10000]
```
Used for: simple_linear W (when W and b are in same channel)

### Output DMA (11 extras):
```
[counter, phase, ..., 0, 0x80000000|out_tid, 0x40001100, dim0, dim1, 0x00020008]
```

### Output DMA (small format, 8 extras):
```
[counter, phase, ..., 0, dim_val, 0x00020004]
```

## 8. Step Generation Algorithm (Pseudocode)

```
function generate_steps(graph):
    steps = []
    
    # Framework: input setup
    steps += input_node_setup_null()      # step 0
    steps += input_node_setup_shape(input) # step 1
    steps += input_slice(input, has_transpose)  # step 2
    steps += perm_load(input_perm)         # step 3
    
    # Op-specific
    has_dma = false
    for op in graph.ops:
        if op is Transpose:
            steps += layout_convert()
            steps += flat_from_vtcm()
            steps += fc_prep()
            steps += dma_output()
            has_dma = true
        elif op is Reshape:
            steps += reshape_prep()
            steps += reshape_compute()
        elif op is FC:
            steps += layout_convert()
            steps += dma_weight(W)
            steps += matmul_xW()
            steps += matmul_bias()
            steps += dma_output()
            has_dma = true
    
    # Framework: output tail
    if has_dma:
        steps += output_node_setup(output)
        steps += sync_op()
    steps += perm_load(output_perm)
    steps += transpose_impl()
    steps += output_slice()
    
    return steps
```
