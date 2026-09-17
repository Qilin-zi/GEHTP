# Phase 4 Integration Test Design

## Overview

Four test graphs (diamond, MatMul+Add, Transformer block, linear chain) feed
into a single end-to-end pipeline test. Each graph exercises all 4 Phase 4
sub-modules. Inputs and expected outputs are specified explicitly.

---

## Test Graphs

### Graph 1: Diamond (A -> B, A -> C, B -> D, C -> D)

Tests: lifetime overlap reuse between non-adjacent ops.

```
         -- B (Relu) --
A (Input)                D (Add) -> E (Output)
         -- C (Relu) --
```

| Tensor | Dims         | Bytes  | Lifetime [begin, end] |
|--------|--------------|--------|----------------------|
| A      | [1, 4, 1024] | 16384  | [0, 2]               |
| B      | [1, 4, 1024] | 16384  | [1, 3]               |
| C      | [1, 4, 1024] | 16384  | [2, 3]               |
| D      | [1, 4, 1024] | 16384  | [3, 4]               |
| E      | [1, 4, 1024] | 16384  | [4, 4]               |

### Graph 2: MatMul + Add (FC block)

Tests: weight/bias const handling, cost-aware spill priority.

```
X (Input) --            -- M (MatMul) -- A (Add) -> Y (Output)
W (Const) --/  \-- B (Const) --/
```

| Tensor | Dims       | Bytes | Op       | Cost (table) | Lifetime   |
|--------|------------|-------|----------|-------------|------------|
| X      | [1, 64]    | 256   | Input    | 1.0         | [0, 1]     |
| W      | [64, 128]  | 32768 | Const    | (skipped)   | [1, 1]     |
| B      | [128]      | 512   | Const    | (skipped)   | [1, 1]     |
| M      | [1, 128]   | 512   | MatMul   | 500.0       | [1, 2]     |
| A      | [1, 128]   | 512   | Add      | 20.0        | [2, 3]     |
| Y      | [1, 128]   | 512   | Output   | 1.0         | [3, 3]     |

### Graph 3: Transformer block (2-layer)

Tests: multi-op lifetime chains, multicast (single-NSP no-op), large graph.

```
Input
  -> LayerNorm -> Q/K/V split -> 3x Linear -> Attention (MatMul+Softmax+MatMul)
  -> Residual Add -> LayerNorm -> FFN (MatMul -> Gelu -> MatMul)
  -> Residual Add -> Output
```

Simplified to 8 ops for testability:

```
I (Input) -> L (LayerNorm) -> M (MatMul, QKV) -> S (Softmax) -> O (MatMul, attn)
          -> R (Add, residual) -> F (MatMul, FFN) -> G (Gelu) -> P (Output)
```

| Tensor | Dims          | Bytes  | Op        | Cost (table) | Lifetime   |
|--------|---------------|--------|-----------|-------------|------------|
| I      | [1, 64, 64]  | 16384  | Input     | 1.0         | [0, 5]     |
| L      | [1, 64, 64]  | 16384  | LayerNorm | 200.0       | [1, 1]     |
| M      | [1, 64, 192] | 49152  | MatMul    | 500.0       | [1, 2]     |
| S      | [1, 4, 64]   | 1024   | Softmax   | 300.0       | [2, 3]     |
| O      | [1, 64, 64]  | 16384  | MatMul    | 500.0       | [3, 4]     |
| R      | [1, 64, 64]  | 16384  | Add      | 20.0         | [4, 6]     |
| F      | [1, 64, 256] | 65536  | MatMul    | 500.0       | [5, 6]     |
| G      | [1, 64, 256] | 65536  | Gelu     | 50.0         | [6, 7]     |
| P      | [1, 64, 64]  | 16384  | Output   | 1.0          | [7, 7]     |

### Graph 4: Linear chain (A -> B -> C -> D -> E)

Tests: maximum sequential reuse (each tensor only overlaps with neighbors).

```
A (Input) -> B (Relu) -> C (Relu) -> D (Relu) -> E (Output)
```

| Tensor | Bytes  | Lifetime |
|--------|--------|----------|
| A      | 16384  | [0, 1]   |
| B      | 16384  | [1, 2]   |
| C      | 16384  | [2, 3]   |
| D      | 16384  | [3, 4]   |
| E      | 16384  | [4, 4]   |

---

## Sub-module Inputs and Expected Outputs

### 4.1: VTCM Lifetime Reuse

#### Input (per graph)

```
requests = [{op_id, size, life_begin, life_end} for each non-const tensor]
budget   = 3 MB (get_vtcm_tile_size)
alignment = 128
```

#### Expected output

**Graph 1 (Diamond):**

| FFD order | Tensor | Offset  | Reused from | Total used |
|-----------|--------|---------|-------------|------------|
| 1st       | A      | 0       | (new)       | 16384      |
| 2nd       | B      | 16384   | (new)       | 32768      |
| 3rd       | C      | 32768   | (new)       | 49152      |
| 4th       | D      | 0       | **A**       | 49152      |
| 5th       | E      | 16384   | **B**       | 49152      |

```
total_vtcm_used      = 49152    (3 blocks, not 5)
vtcm_saved_by_reuse  = 32768    (2 blocks saved)
A.offset == D.offset == 0
B.offset == E.offset == 16384
C.offset == 32768
spilled = 0
```

**Graph 2 (MatMul+Add):**

| Tensor | Offset  | Reused from | Notes                         |
|--------|---------|-------------|-------------------------------|
| X      | 0       | (new)       |                               |
| M      | 256     | (new)       | overlaps X [0,1] at 1        |
| A      | 768     | (new)       | overlaps M [1,2] at 2         |
| Y      | 1280    | (new)       | overlaps A [2,3] at 3         |

```
total_vtcm_used = 1792  (no reuse — all sequential overlap)
(W and B are const → skipped)
```

**Graph 3 (Transformer):**

| Tensor | Offset  | Reused from | Reason                          |
|--------|---------|-------------|----------------------------------|
| F      | 0       | (new)       | largest (65536), placed first   |
| G      | 65536   | (new)       | overlaps F [5,6] at 6           |
| I      | 131072  | (new)       | overlaps F (I [0,5] ∩ F [5,6]) |
| M      | 147456  | (new)       | overlaps I at 1                |
| O      | 196608  | (new)       | overlaps R [4,6] at 4          |
| R      | 213000  | (new)       | overlaps O                      |
| S      | 0       | **F**       | S [2,3] ∩ F [5,6] = empty      |
| P      | 65536   | **G**       | P [7,7] ∩ G [6,7] = empty      |

```
total_vtcm_used < 9 * sum(sizes)  (reuse saves at least 2 blocks)
spilled = 0  (budget 3MB >> demand)
```

**Graph 4 (Linear chain):**

```
A.offset == C.offset == 0      (A [0,1] and C [2,3] don't overlap)
B.offset == D.offset == 16384  (B [1,2] and D [3,4] don't overlap)
E.offset == 0                  (E [4,4] and A [0,1] don't overlap)
total_vtcm_used = 3 * 16384 = 49152  (5 tensors in 3 blocks)
vtcm_saved_by_reuse = 2 * 16384 = 32768
```

#### Verification

```
for each graph:
  assert len(vtcm_allocations) == num_non_const_tensors
  assert all(not spilled for all entries)  # budget sufficient
  assert total_vtcm_used < sum(all tensor sizes)  # reuse happened
  assert A.offset == D.offset  (graph 1)  # specific reuse check
  assert total_vtcm_used == 3 * 16384  (graph 1)
```

---

### 4.2: DMA Sync Token

#### Input

```
DMA operations for each graph's weight/bias loading:
  Graph 2: SET(W) at pos 5, SET(b) at pos 6 (reuse 0x11), WAIT at pos 10
  Graph 3: SET(W_qkv) at pos 3, WAIT at pos 5
           SET(W_ffn) at pos 8, WAIT at pos 10
```

#### Expected output (SynctokenManager)

**Graph 2:**

```
tokens = [{
  token_id: 0x11,
  signal_pos: 5,
  signal_name: "DmaCheckpointSet(W)",
  wait_positions: [10]
}]
next_token_id = 0x13
validate() = true
```

**Graph 3:**

```
tokens = [
  {token_id: 0x11, signal_pos: 3, wait_positions: [5]},
  {token_id: 0x13, signal_pos: 8, wait_positions: [10]}
]
next_token_id = 0x15
validate() = true
```

#### Verification

```
assert token_count() == expected_count
assert validate() == true
assert all wait_pos > signal_pos  (ordering)
assert next_token_id starts at 0x11, increments by 2
```

---

### 4.3: Multicast Optimization

#### Input

```
Single-NSP mode: mcsends = []  (no cross-NSP consumers)
max_mcast_buffer_size = vtcm_size_ (3 MB)
```

#### Expected output

```
result = []  (no-op in single-NSP)
supercast_count = 0
result_mcsend_count = 0
```

#### Multi-NSP simulated verification

```
# Two sends, same MCID, disjoint receivers → merge
sends = [
  {tag=1, sender=0, payload=1024, mcids=[10], receivers=[1,2]},
  {tag=2, sender=0, payload=512,  mcids=[10], receivers=[3,4]},
]
result = optimize(sends, 2)
assert result.size() == 1
assert result[0].receivers == [1,2,3,4]
assert result[0].num_mcids == 1  (Phase 4.3 fix)
assert result[0].payload_size == 1536

# Capacity reject: merged payload > buffer
opt.set_max_mcast_buffer_size(1024)
result = optimize(sends, 2)
assert result.size() == 2  (not merged, exceeds capacity)

# Receiver conflict: partial overlap
sends = [
  {tag=1, sender=0, payload=1024, mcids=[10], receivers=[1,2]},
  {tag=2, sender=0, payload=512,  mcids=[10], receivers=[2,3]},
]
result = optimize(sends, 2)
assert result.size() == 2  (not merged, partial receiver overlap)
```

---

### 4.4: Cost Model

#### Input

```
OpDesc for each op (op_name, output_dims, nsp_count, vtcm_budget)
InferenceMode = Performance (default)
```

#### Expected output (table lookup, desc=null)

| Graph | Op       | Table cost | Analytical (desc provided)      |
|-------|----------|-----------|----------------------------------|
| 1     | Relu     | 10.0      | 4096 * 0.1 = 409.6              |
| 1     | Add      | 20.0      | 4096 * 0.2 = 819.2              |
| 2     | MatMul   | 500.0     | 128 * 4.0 = 512.0               |
| 2     | Add      | 20.0      | 128 * 0.2 = 25.6                |
| 3     | LayerNorm| 200.0     | 4096 * 2.0 = 8192.0             |
| 3     | MatMul   | 500.0     | 12288 * 4.0 = 49152.0           |
| 3     | Softmax  | 300.0     | 256 * 3.0 = 768.0               |
| 3     | Gelu     | 50.0      | 16384 * 0.1 = 1638.4 (default)  |
| 4     | Relu     | 10.0      | 4096 * 0.1 = 409.6              |

#### tcm_migration cost-aware priority

```
priority = size / (access_freq * op_cost)

Graph 2 (MatMul+Add):
  X: 256  / (1 * 1.0)   = 256     (Input, 1 consumer)
  M: 512  / (1 * 500.0) = 1       (MatMul, highest cost → lowest priority → keep VTCM)
  A: 512  / (1 * 20.0)  = 25      (Add, medium cost)
  Y: 512  / (1 * 1.0)   = 512     (Output, lowest cost → highest priority → spill first)

  Spill order: Y > X > A > M
  (MatMul stays in VTCM; Output spills first)
```

#### Verification

```
assert has_mlp_model() == true
assert get_prediction("MatMul", nullptr, nullptr, {}) == 500.0  (table)
assert get_prediction("Relu",  nullptr, &desc,   {}) == 4096 * 0.1  (analytical)
assert get_prediction("UnknownOp", nullptr, &desc, {}) > 0.0  (MLP fallback)

# MLP architecture
mlp = get_mlp_model()
assert mlp.weights.size() == 2        (2 layers)
assert mlp.weights[0].size() == 32    (4×8)
assert mlp.weights[1].size() == 4     (1×4)
assert mlp.use_linear[0] == false     (hidden = ReLU)
assert mlp.use_linear[1] == true      (output = linear)
```

---

## End-to-End Pipeline Test

### Input

```cpp
for each of the 4 graphs:
  GraphPrepare gp;
  build graph (append_node calls)
  HexagonNNEnv env;
  gp.prepare(env);
```

### Final output and verification

```
For each graph, after prepare():

  ┌─────────────────────────────────────────────────────────────────┐
  │ 4.1 VTCM Allocation                                            │
  │                                                                 │
  │   Output: gp.get_vtcm_allocations()                             │
  │   = {op_id → (offset, block_id, spilled)}                      │
  │                                                                 │
  │   Verify:                                                       │
  │     1. all non-const tensors have an allocation entry           │
  │     2. no tensor spilled (budget 3MB >> demand)                │
  │     3. total_vtcm_used < sum(tensor sizes)  (reuse happened)    │
  │     4. specific offset sharing (graph-dependent, see above)     │
  │     5. all offsets are 128-byte aligned                         │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ 4.2 DMA Sync Token (via OpEmitter)                             │
  │                                                                 │
  │   Output: emitter.synctoken_manager().get_tokens()              │
  │   = [{token_id, signal_pos, signal_name, wait_positions}]      │
  │                                                                 │
  │   Verify:                                                       │
  │     1. token IDs start at 0x11, increment by 2                 │
  │     2. validate() == true (all waits have preceding signals)    │
  │     3. token_count matches number of DMA groups                 │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ 4.3 Multicast Optimization                                     │
  │                                                                 │
  │   Output: optimizer statistics                                  │
  │   = {supercast_count, original_mcsend_count, result_count}     │
  │                                                                 │
  │   Verify:                                                       │
  │     1. single-NSP: supercast_count == 0 (no-op)                │
  │     2. multi-NSP simulation: merges with disjoint receivers     │
  │     3. capacity constraint rejects oversized merges             │
  │     4. receiver conflict (partial overlap) rejected            │
  │     5. num_mcids == mcids.size() (Phase 4.3 fix)                │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ 4.4 Cost Model                                                  │
  │                                                                 │
  │   Output: CostSource predictions + tcm_migration priorities   │
  │                                                                 │
  │   Verify:                                                       │
  │     1. has_mlp_model() == true after init_for_soc              │
  │     2. MLP architecture: 8→4→1, ReLU hidden, linear output      │
  │     3. table lookup: MatMul=500.0, Relu=10.0, Add=20.0          │
  │     4. analytical: Relu(4096) = 4096 * 0.1 = 409.6              │
  │     5. NSP scaling: 4 NSP → cost / 4                            │
  │     6. mode scaling: Power=0.7×, Bandwidth=0.5×                  │
  │     7. MLP fallback: unknown op → positive cost                 │
  │     8. tcm_migration: high-cost ops (MatMul) stay in VTCM,     │
  │        low-cost ops (Output, Reshape) spill first               │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ Cross-module integration                                        │
  │                                                                 │
  │   Verify:                                                       │
  │     1. VTCM allocation respects tcm_migration spill flags       │
  │        (tensors with flags2 & 0x40 are skipped)                 │
  │     2. Cost-aware priority affects which tensors are spilled    │
  │     3. DMA sync tokens cover all weight/bias DMA ops            │
  │     4. Multicast optimization runs after VTCM allocation        │
  │     5. No regression: all existing tests still pass             │
  └─────────────────────────────────────────────────────────────────┘
```

### Success criteria summary

| Criterion | Graph 1 | Graph 2 | Graph 3 | Graph 4 |
|-----------|---------|---------|---------|---------|
| VTCM blocks used < tensors | 3 < 5 | 4 < 4 (no reuse) | 7 < 9 | 3 < 5 |
| VTCM saved > 0 | 32768 | 0 | > 0 | 32768 |
| Specific offset reuse | A=D, B=E | (none) | S=F, P=G | A=C=E, B=D |
| DMA tokens valid | (n/a) | 1 token | 2 tokens | (n/a) |
| Mcast no-op (single NSP) | yes | yes | yes | yes |
| Cost: MatMul stays VTCM | (n/a) | yes | yes | (n/a) |
| Cost: Output spills first | (n/a) | yes | (n/a) | (n/a) |
