#!/usr/bin/env python3
"""analyze_dd.py — 例 20 双域编排的 host 侧对拍
用法: python3 analyze_dd.py <results_dir>
输入: <dir>/20_dualdomain_{ser,a,b}.txt (设备拉回)
判定 (写成与设备 result 相同的 [PASS]/[FAIL] 行 → <dir>/20_dualdomain_host.txt):
  dd_split_equiv  : a[0:8)+b[8:16) 每步 sha256 == ser 同步 (切分等价)
  dd_steps_all    : 三段合计 16 步 sha 全部落盘
报告 (不设门): 加速比 = ser.wall / max(a.wall, b.wall)
"""
import re
import sys
from pathlib import Path


def parse(p: Path):
    shas, wall = {}, None
    for line in p.read_text().splitlines():
        m = re.match(r"sha step(\d+) ([0-9a-f]{64})", line)
        if m:
            shas[int(m.group(1))] = m.group(2)
        m = re.search(r"wall_us=(\d+)", line)
        if m:
            wall = int(m.group(1))
    return shas, wall


def main():
    d = Path(sys.argv[1])
    ser, wser = parse(d / "20_dualdomain_ser.txt")
    a, wa = parse(d / "20_dualdomain_a.txt")
    b, wb = parse(d / "20_dualdomain_b.txt")

    lines = ["=== 20_dualdomain_host (host 对拍) ==="]
    bad = [k for k in range(0, 8) if a.get(k) != ser.get(k)]
    bad += [k for k in range(8, 16) if b.get(k) != ser.get(k)]
    lines.append(f"[{'PASS' if not bad else 'FAIL'}] %-36s err=%d tol=0"
                 % ("dd_split_equiv", len(bad)))
    if bad:
        lines.append(f"  mismatch steps: {bad}")
    n = len(ser) + len(a) + len(b)
    lines.append(f"[{'PASS' if n == 32 else 'FAIL'}] %-36s err=%d tol=0"
                 % ("dd_steps_all", abs(32 - n)))
    if wser and wa and wb and wb > 0:
        sp = wser / max(wa, wb)
        lines.append(f"dd speedup ser={wser}us a={wa}us b={wb}us "
                     f"→ {sp:.3f}x (报告, 不设门)")
    (d / "20_dualdomain_host.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
