import sys

import numpy as np

import check_device_golden


def write(path, arr, dtype):
    np.asarray(arr, dtype=dtype).tofile(path)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["check_device_golden.py", *argv])
    return check_device_golden.main()


ROW = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
ROW_FLIP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.0]  # argmax 7→6, 数值近似不变


def test_single_pair_top1_gate(monkeypatch, tmp_path):
    write(tmp_path / "dev.f16.raw", [ROW, ROW], np.float16)
    write(tmp_path / "gold.f32.raw", [ROW, ROW_FLIP], np.float32)
    base = ["--device", str(tmp_path / "dev.f16.raw"),
            "--gold", str(tmp_path / "gold.f32.raw"),
            "--gold-dtype", "f32", "--rows", "2", "--cos-min", "0.99"]
    assert run_main(monkeypatch, [*base, "--top1-min", "1/2"]) == 0
    assert run_main(monkeypatch, [*base, "--top1-min", "2/2"]) == 1


def test_batch_rate_gate_not_per_pair_perfection(monkeypatch, tmp_path, capsys):
    write(tmp_path / "dev.f16.raw", [ROW, ROW], np.float16)
    write(tmp_path / "gold_0.f32.raw", [ROW, ROW], np.float32)        # 2/2
    write(tmp_path / "gold_1.f32.raw", [ROW, ROW_FLIP], np.float32)   # 1/2
    base = ["--device", str(tmp_path / "dev.f16.raw"),
            "--gold-pattern", str(tmp_path / "gold_{i}.f32.raw"),
            "--prompts", "2", "--gold-dtype", "f32",
            "--rows", "2", "--cos-min", "0.99"]
    # 合计 3/4: 按命中率 3/4 应过门 (原实现逐对全对 → 此例假 FAIL)
    assert run_main(monkeypatch, [*base, "--top1-min", "3/4"]) == 0
    assert "批量 top1 合计 3/4" in capsys.readouterr().out
    assert run_main(monkeypatch, [*base, "--top1-min", "4/4"]) == 1


def test_cos_gate_fail(monkeypatch, tmp_path):
    write(tmp_path / "dev.f16.raw", [ROW, ROW_FLIP], np.float16)
    shifted = (np.array([ROW, ROW_FLIP]) + 3.0).tolist()
    write(tmp_path / "gold.f32.raw", shifted, np.float32)
    rc = run_main(monkeypatch, ["--device", str(tmp_path / "dev.f16.raw"),
                                "--gold", str(tmp_path / "gold.f32.raw"),
                                "--gold-dtype", "f32", "--rows", "2",
                                "--cos-min", "0.99"])
    assert rc == 1
