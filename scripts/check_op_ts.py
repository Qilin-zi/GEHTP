#!/usr/bin/env python3
"""check_op_ts.py — op_ts.bin 与 blob/manifest 交叉校验 (验收闭环)

复用 gehtp_prof 的解析 (WTS1 头 24B / ver 1-4), 不手搓二进制判读。
校验项 (全部硬门, 任一违例 exit 1):
  1. 长度一致: op_ts n_ops == blob n_ops (陈旧/未重跑即 fail)
  2. 字段域:   engine/planned_engine ∈ {0,1,2}, deviation ∈ {0,1,2}
  3. 时间合理: dur_us ≥ 0; 单段串行时 start 非降 (--no-monotonic 放宽)
  4. ver≥3:   T020 不变量 deviation==0 ⟺ planned==actual
              deviation==1 ⟹ planned≠actual (允许的回落)
              deviation==2 ⟹ planned≠actual (禁止回落→硬报错中止)
     ver<3 由解析器补 planned=engine/deviation=0, 不变量自然满足。
     T020 抽样 K>0: 写入方恒写全量槽、零槽=未采样, 零槽跳过校验且长度门不变。
用法:
  check_op_ts.py --wtop m.wtop --op-ts prof/m.op_ts.bin [--no-monotonic]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gehtp_prof  # noqa: E402

ENGINE_NAMES = {0: "scalar", 1: "HVX", 2: "HMX"}


def main():
    ap = argparse.ArgumentParser(description="op_ts.bin 交叉校验门")
    ap.add_argument("--wtop", required=True)
    ap.add_argument("--op-ts", required=True)
    ap.add_argument("--no-monotonic", action="store_true",
                    help="start 非降检查降为 warn (多段/并行执行时用)")
    args = ap.parse_args()

    blob_ver, _slots, ops = gehtp_prof.parse_wtop(args.wtop)
    wall_us, ts, sample = gehtp_prof.parse_op_ts(args.op_ts)

    errors, warns = [], []

    if len(ts) != len(ops):
        errors.append(f"长度不一致: op_ts n={len(ts)} != blob n_ops={len(ops)} "
                      f"(op_ts.bin 陈旧或未重跑; gehtp run 跑前已清设备侧)")
    if wall_us is not None and wall_us == 0:
        errors.append("wall_us=0 (记录未写入?)")

    inv0 = inv1 = inv2 = 0
    bad_start = None
    prev_start = None
    for i, (s, d, dus, db, eng, pl, dv, fb) in enumerate(ts):
        tag = f"op[{i}]"
        if sample and not (s or d or dus or db or eng or pl or dv or fb):
            continue  # T020 抽样 K>0: 零槽=未采样, 跳过校验
        if eng not in ENGINE_NAMES:
            errors.append(f"{tag} engine={eng} 越域")
        if pl not in ENGINE_NAMES:
            errors.append(f"{tag} planned_engine={pl} 越域")
        if dv not in (0, 1, 2):
            errors.append(f"{tag} deviation={dv} 越域")
        if d is not None and d < 0:
            errors.append(f"{tag} dur_us={d} < 0")
        if s is not None:
            if prev_start is not None and s < prev_start:
                if bad_start is None:
                    bad_start = i
                if not args.no_monotonic:
                    errors.append(f"{tag} start={s} < 前一 op start={prev_start} "
                                  f"(执行序乱/多段; 确属并行则加 --no-monotonic)")
            prev_start = s
        # T020 不变量
        if dv == 0 and pl != eng:
            inv0 += 1
        elif dv != 0 and pl == eng:
            inv1 += 1
        if dv == 2 and pl == eng:
            inv2 += 1

    if inv0:
        errors.append(f"不变量违例×{inv0}: deviation==0 但 planned≠actual "
                      f"(T020 契约: 静默替换)")
    if inv1:
        errors.append(f"不变量违例×{inv1}: planned==actual 但 deviation≠0")
    if inv2:
        errors.append(f"不变量违例×{inv2}: deviation==2 但 planned==actual "
                      f"(禁止回落 op 未按契约中止)")

    # 统计口径 (判读用, 不做门)
    n_eng = {v: 0 for v in ENGINE_NAMES}
    n_dev = {0: 0, 1: 0, 2: 0}
    for rec in ts:
        if sample and not any(rec):
            continue
        eng, dv = rec[4], rec[6]
        if eng in n_eng:
            n_eng[eng] += 1
        if dv in n_dev:
            n_dev[dv] += 1
    print(f"blob_ver={blob_ver} ops={len(ops)} ts={len(ts)} wall={wall_us or '?'}us "
          f"sample={sample or 'full'}")
    print("engine 分布: " + ", ".join(f"{ENGINE_NAMES[k]}={v}" for k, v in n_eng.items()))
    print(f"deviation 分布: 0={n_dev[0]} 1(回落)={n_dev[1]} 2(硬报错)={n_dev[2]}")
    if bad_start is not None:
        warns.append(f"首个 start 回退出现在 op[{bad_start}] (多段执行属正常)")

    for w in warns:
        print(f"warn: {w}", file=sys.stderr)
    if errors:
        for e in errors[:20]:
            print(f"FAIL: {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"... 共 {len(errors)} 条违例", file=sys.stderr)
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
