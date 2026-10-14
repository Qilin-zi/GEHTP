import struct
import sys

import pytest

import check_op_ts

REC_FMT = {1: "<II", 2: "<IIIII", 3: "<IIIIIII", 4: "<IIIIIIII"}


def make_op_ts(path, ver, recs, sample=0, wall=12345):
    with open(path, "wb") as f:
        f.write(b"WTS1")
        f.write(struct.pack("<III", ver, len(recs), sample))
        f.write(struct.pack("<Q", wall))
        for r in recs:
            f.write(struct.pack(REC_FMT[ver], *r))
    return path


def run_gate(monkeypatch, tmp_path, recs, ver=4, sample=0, n_ops=None):
    op_ts = make_op_ts(tmp_path / "m.op_ts.bin", ver, recs, sample)
    n_ops = len(recs) if n_ops is None else n_ops
    monkeypatch.setattr(check_op_ts.gehtp_prof, "parse_wtop",
                        lambda _p: (ver, [], [None] * n_ops))
    monkeypatch.setattr(sys, "argv",
                        ["check_op_ts.py", "--wtop", "m.wtop", "--op-ts", str(op_ts)])
    return check_op_ts.main()


def test_v4_valid_passes(monkeypatch, tmp_path, capsys):
    recs = [(0, 10, 0, 0, 2, 2, 0, 0), (10, 20, 0, 0, 1, 1, 0, 0), (30, 5, 0, 0, 0, 0, 0, 0)]
    assert run_gate(monkeypatch, tmp_path, recs) == 0
    assert "GATE PASS" in capsys.readouterr().out


def test_v4_sampled_zero_slots_skipped(monkeypatch, tmp_path):
    recs = [(0, 10, 0, 0, 2, 2, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (10, 20, 0, 0, 1, 1, 0, 0)]
    assert run_gate(monkeypatch, tmp_path, recs, sample=2) == 0


def test_v2_planned_follows_engine(monkeypatch, tmp_path):
    recs = [(0, 10, 0, 0, 1), (10, 20, 0, 0, 2), (30, 5, 0, 0, 2)]
    assert run_gate(monkeypatch, tmp_path, recs, ver=2) == 0


def test_monotonic_break_fails(monkeypatch, tmp_path):
    recs = [(50, 10, 0, 0, 2, 2, 0, 0), (10, 20, 0, 0, 1, 1, 0, 0)]
    assert run_gate(monkeypatch, tmp_path, recs) == 1


def test_length_mismatch_fails(monkeypatch, tmp_path):
    recs = [(0, 10, 0, 0, 2, 2, 0, 0)]
    assert run_gate(monkeypatch, tmp_path, recs, n_ops=3) == 1


def test_parse_op_ts_roundtrip_all_vers(tmp_path):
    import gehtp_prof
    cases = {
        1: ((1, 2), (1, 2, 0, 0, 0, 0, 0, 0)),
        2: ((1, 2, 3, 4, 1), (1, 2, 3, 4, 1, 1, 0, 0)),  # ver<3: planned=engine 口径
        3: ((1, 2, 3, 4, 1, 1, 0), (1, 2, 3, 4, 1, 1, 0, 0)),
        4: ((1, 2, 3, 4, 1, 1, 0, 5), (1, 2, 3, 4, 1, 1, 0, 5)),
    }
    for ver, (raw, expect) in cases.items():
        p = make_op_ts(tmp_path / f"v{ver}.bin", ver, [raw], sample=0, wall=777)
        wall, ts, sample = gehtp_prof.parse_op_ts(str(p))
        assert wall == 777 and sample == 0 and len(ts) == 1
        assert ts[0] == expect
