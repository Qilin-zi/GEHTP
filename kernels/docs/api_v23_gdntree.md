# api_v23_gdntree — U14 树形 GDN 串行 oracle + kernel

源: `src/hvx/gdn_tree.c` (kernel) + `gdn_ref.c` 扩展 (oracle) · 头: `include/gdn_tree.h`
· 例: `28_gdn_tree` (8 门 PASS)

## 与 gdn_sm 的关系

例 19 gdn_sm 是**单 token 链式**递推; U14 扩成任意 **树拓扑** (每 token 一个 parent,
`parent[0]=-1`, `parent[i]<i`), 对应 DFlash draft 树 (T=8/16/32)。数学同源
(g_i/β_i/k_i/v_i/q_i per-token), 状态传递沿树边。

## 闭式解 (host oracle, 与串行递归 cos=1.0000000 精确)

```
Pg[i] = g_i + Pg[parent(i)]                      (累计 log 衰减, 沿祖先链)
L[j][i] = exp(Pg_j − Pg_i)·(k_j·k_i)·β_j         i ∈ Anc(j)
B[j][n] = β_j·(v_j[n] − exp(Pg_j)·(S0·k_j)[n])
前代换解 (I+L)U = B   (L 非零即祖先 → 下三角)
y_j = s·[ exp(Pg_j)·(S0·q_j) + Σ_{i∈Anc(j)∪{j}} exp(Pg_j−Pg_i)·(q_j·k_i)·u_i ]
S_a = exp(Pg_a)·S0 + Σ_{j: parent(j)=a} exp(Pg_a−Pg_j)·u_j·k_jᵀ
```

## API

```c
/* oracle (host/device 同码): parent[] 树, 每 token g/β/k/v/q */
void ref_delta_tree(const float* g, const float* beta, const float* k,
                    const float* v, const float* q, const int* parent,
                    uint32_t ntok, uint32_t d, uint32_t dstate,
                    const float* S0, float* Y, float* S_out);
void ref_tree_closed(...);   /* 闭式解 (对拍串行用) */

/* 设备 kernel: f16 输入面, 串行拓扑递推 */
int gdn_tree_serial_f16(const uint16_t* g_f16, ..., const int* parent,
                        uint32_t ntok, ..., uint16_t* y_f16, uint16_t* S_out_f16);
```

## 设备门 (28_gdn_tree, 2026-08-16)

- 拓扑: 35% 链混合随机树, T=8/16/32, d=64, dstate=128
- `closed_vs_serial_y_cos` / `closed_vs_serial_state_cos`: host 数学闭合 (cos=1.0000000)
- `kernel_vs_closed_y_cos` / `kernel_vs_closed_state_cos`: **设备 f16 kernel cos=1.000000**
- `kernel_rerun_bitexact` / `topology_invalid_rejected` (parent 违序拒绝)
- INT16 衰减连乘精度曲线 (sc16=1/65535): 深度 0..7 max rel err 7e-6..2.5e-5

## 坑

- 全部缓冲用 U10 arena 承载 (池 = 2×面 + 512KB 才够, 紧池 alloc FAIL)。
- `parent[i] < i` 是 kernel 前置 (拓扑序), 违序必须拒 — 非退化门在入口。
