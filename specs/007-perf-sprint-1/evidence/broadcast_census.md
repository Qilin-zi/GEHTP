# 007 broadcast 形态普查（T004）

**Date**: 2026-10-09
**Source**: 与 T003 同源（v4.wtop + v4.op_ts.bin）；解析走 `scripts/gehtp_prof.py` 的 `parse_wtop`/`parse_op_ts`（正典解析器 import 复用，无手搓 blob）。
**判定函数**: 本文件 Python 参照实现 = 设备/编译器两侧 C++ 判定的语义基准（见下"判定规则"）。

## 结论（总表）

| 形态 | n | total_us | %族 | 说明 |
|---|---|---|---|---|
| form2(fill) 行循环+标量铺行 | 1256 | 5,729,909 | 73.1% | 末轴 in=1→out=W 逐行铺；主力 |
| form2(rowcopy) 行循环+行拷贝 | 150 | 1,172,799 | 15.0% | 末轴 in=out=W 连续行拷 |
| form1(fill) 整块标量铺 | 179 | 906,782 | 11.6% | b_elems=1（in 全 1），标量 1→n |
| form1(repeat) 整块重复 | 37 | 31,818 | 0.4% | 连续块重复 M 次 |
| **form3(wrap/gather)** | **0** | **0** | 0% | **本图不存在** |

**核心结论**：form3 为零 → 三形态执行体可覆盖全族 100%，类 1/2 全部 HVX 化后：
- 918 笔 dev=1 回落全消（P1 门可达）；
- 704 笔现 planned=scalar 也可改计划 HVX（形态判据替代 n%32 代理判据），全族 7.84s 都在射程内（保守按带宽算 ≤1s 门）。

## 形态 × 偏差切片（与 baseline.md 918/704 双向核销）

| 形态 | dev=1（planned HVX→scalar） | planned scalar | 合计 |
|---|---|---|---|
| form2(fill) | 698 / 3.99s | 558 / 1.74s | 1256 |
| form2(rowcopy) | 78 / 0.53s | 72 / 0.64s | 150 |
| form1(fill) | 105 / 0.41s | 74 / 0.50s | 179 |
| form1(repeat) | 37 / 0.03s | 0 | 37 |
| **合计** | **918 / 4.97s** | **704 / 2.87s** | 1622 / 7.84s |

## 代表形状（in_d[4] → out_d[4]，右对齐 numpy 广播）

- form1(fill): (1,1,1,1)→(1,32,16,1) n=512 ×72；(1,1,1,1)→(16,1,64,64) ×54；(1,1,1,1)→(1,16,64,128) ×18
- form1(repeat): (1,1,1,16)→(1,1,32,16) ×36
- form2(fill): (1,16,64,1)→(1,16,64,128) ×54；(1,32,16,1)→(1,32,16,128) ×36；(16,1,64,1)→(16,1,64,64) ×18（69 种形状）
- form2(rowcopy): (1,1,64,64)→(16,1,64,64) ×72；(16,1,1,64)→(16,1,64,64) ×18；(1,1,64,64)→(1,16,64,64) ×18

## 判定规则（两侧实现必须同语义；注释互指本节）

```
r  = out 有效 rank（去尾 1；r==0 时置 1）
ir = min(4, r)；lead = r - ir（in 右对齐，前 lead 轴隐式广播）
末轴配对：in_d[ir-1] vs out_d[r-1]
  相等           → inner = copy
  in==1          → inner = fill
  其他（wrap）   → form3
外轴 d ∈ [0, ir-1)：in_d[d]==out_d[d+lead]（copy）或 in==1（broadcast）皆可；
  否则（1<in<out 的 wrap 轴）→ form3
外轴全 broadcast → form1（inner=fill 即整块铺 / copy 即整块重复）
否则             → form2（inner=fill 铺行 / copy 拷行）
```

工程要点：
- W = out_d[r-1] 元素数；行数 = n/W；form2 外层用里程计（odometer）递增坐标免逐行 div/mod，行内坐标增量预计算。
- form1(fill) 即 b_elems==1 特例，可用 HVX 半字广播寄存器整段 vstore。
- 形态判定纯形状函数（in_d/out_d → 1/2/3），emit 与 exec 同式；本图 form3=0 不代表一般图没有，类 3 保留标量兜底（合法 planned scalar）。
