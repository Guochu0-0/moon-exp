"""finetune 里不依赖 GPU / LoFTR 的部分。"""
import numpy as np

from finetune.compare import compare


def test_compare_identical_and_diff():
    rng = np.random.default_rng(0)
    a = {"R__p0": rng.random((5, 5), dtype=np.float32), "R__p1": np.zeros((0, 5), np.float32)}
    assert compare(a, {k: v.copy() for k, v in a.items()})["identical"]

    b = {k: v.copy() for k, v in a.items()}
    b["R__p0"][0, 2] += 0.5
    r = compare(a, b)
    assert not r["identical"] and abs(r["max_abs_xy"] - 0.5) < 1e-6

    r = compare(a, {"R__p0": a["R__p0"][:3], "R__p2": a["R__p1"]})
    assert r["n_diff_count"] == 1 and r["only_a"] == ["R__p1"] and r["only_b"] == ["R__p2"]
