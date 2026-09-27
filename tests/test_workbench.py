"""工作台的记录格式、数据集读取与评价口径。用合成的小数据集，不依赖真实数据。"""
import math

import numpy as np
import pytest

from workbench import protocol
from workbench.dataset import Dataset, read_tif
from workbench.evaluate import evaluate_identity, evaluate_run
from workbench.records import PredWriter, check_runs, load_runs

A_TRUE = [[1.0, 0.0, 3.0], [0.0, 1.0, -4.0]]


def write_tif(path, arr):
    import tifffile

    tifffile.imwrite(path, arr, compression="zlib", rowsperstrip=4, photometric="minisblack",
                     planarconfig="contig" if arr.ndim == 3 else None)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "Processed_Data"
    rng = np.random.default_rng(0)
    for split in ("Val", "Test"):
        for roi in ("ROI_002", "ROI_010"):
            d = root / split / roi
            for k in ("Optical", "SAR", "Label"):
                (d / k).mkdir(parents=True)
            for n in (0, 2, 10):   # 数值序：0, 2, 10（字符串序会把 10 排在 2 前）
                write_tif(d / "Optical" / f"patch_{n}.tif", rng.integers(0, 255, (16, 16), dtype=np.uint8))
                write_tif(d / "SAR" / f"patch_{n}.tif", rng.random((16, 16, 4), dtype=np.float32))
                if n == 2:
                    continue      # 无标注 pair
                opt = rng.uniform(0, 16, (8, 2))
                sar = opt + [3.0, -4.0]
                lab = np.c_[opt[:, 0], -opt[:, 1], sar[:, 0], -sar[:, 1]]   # y 存为负值
                np.savetxt(d / "Label" / f"patch_{n}.txt", lab, delimiter="\t", fmt="%.8f")
    return Dataset(tmp_path)


def test_pairs_numeric_order_and_labels(dataset):
    assert dataset.pairs("test") == ["ROI_002/patch_0", "ROI_002/patch_2", "ROI_002/patch_10",
                                     "ROI_010/patch_0", "ROI_010/patch_2", "ROI_010/patch_10"]
    assert "ROI_002/patch_2" not in dataset.labelled("test")
    opt, sar = dataset.checkpoints("test", "ROI_002/patch_0")
    assert (opt >= 0).all() and np.allclose(sar - opt, [3, -4])


def test_read_tif_matches_tifffile(tmp_path):

    a = np.arange(8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
    p = tmp_path / "x.tif"
    write_tif(p, a)
    assert np.array_equal(read_tif(p), a)


def test_pair_error_and_failure():
    opt = np.array([[0.0, 0.0], [10.0, 0.0]])
    sar = opt + [3.0, 4.0]
    assert protocol.pair_error([[1, 0, 3], [0, 1, 4]], opt, sar) == pytest.approx(0.0)
    assert protocol.pair_error([[1, 0, 0], [0, 1, 0]], opt, sar) == pytest.approx(5.0)
    assert protocol.pair_error(None, opt, sar) == math.inf


def test_summary_counts_failures_in_denominator():
    err = np.array([1.0, 4.0, 30.0, math.inf])
    s = protocol.summarize(err)
    assert s["fail_rate"] == 0.25
    assert s["sr@5"] == 0.5
    assert s["auc@10"] == pytest.approx(((1 - 0.1) + (1 - 0.4)) / 4)
    assert s["median"] == pytest.approx(17.0)
    assert protocol.summarize(np.array([1.0, math.inf, math.inf]))["median"] == math.inf
    lo, hi = s["auc@10_ci"]
    assert lo <= s["auc@10"] <= hi


def test_writer_roundtrip_and_eval(dataset, tmp_path):
    runs = tmp_path / "runs"
    (runs / "B0").mkdir(parents=True)
    (runs / "B0" / "exp.toml").write_text('id = "B0"\ntitle = "t"\nstatus = "baseline"\n', encoding="utf-8")
    (runs / "E1").mkdir()
    (runs / "E1" / "exp.toml").write_text('id = "E1"\ntitle = "t"\nstatus = "kept"\nparent = "B0"\ninit = "B0/good"\n',
                                          encoding="utf-8")
    for split in ("val", "test"):
        pairs = dataset.pairs(split)
        with PredWriter(runs / "B0", "good", split) as w:
            for p in pairs:
                w.write(p, A_TRUE, matches=np.zeros((5, 4)), n_inliers=np.int64(5))
        with PredWriter(runs / "B0", "bad", split) as w:
            w.write(pairs[0], None, fail="few_inliers")      # 其余有标注 pair 缺预测 → 失败
        with PredWriter(runs / "E1", "main", split) as w:
            for p in pairs:
                w.write(p, [[1, 0, float("nan")], [0, 1, 0]])  # 非有限 → 失败

    loaded = load_runs(runs)
    assert check_runs(loaded) == []
    assert loaded["B0"].methods == ["bad", "good"]
    pr = loaded["B0"].preds("good", "test")["ROI_010/patch_10"]
    assert pr.A == A_TRUE and pr.extra == {"n_inliers": 5}
    assert loaded["B0"].matches("good", "test", "ROI_010/patch_10").shape == (5, 4)
    assert loaded["E1"].preds("main", "val")["ROI_002/patch_0"].fail == "non_finite"

    m = evaluate_run(dataset, loaded["B0"])["methods"]
    good, bad = m["good"]["test"], m["bad"]["test"]
    assert good["summary"]["sr@1"] == 1.0 and good["coverage"]["missing"] == 0
    assert bad["summary"]["fail_rate"] == 1.0 and bad["coverage"]["missing"] == 3
    assert len(good["errors"]) == len(dataset.labelled("test")) == 4
    ident = evaluate_identity(dataset)["test"]["summary"]   # 恒等仿射下误差恒为 5 px
    assert ident["sr@3"] == 0.0 and ident["sr@10"] == 1.0


def test_check_catches_bad_records(tmp_path):
    (tmp_path / "X").mkdir()
    (tmp_path / "X" / "exp.toml").write_text('id = "X"\ntitle = "t"\nstatus = "done"\nparent = "Y"\n', encoding="utf-8")
    problems = check_runs(load_runs(tmp_path))
    assert any("status" in p for p in problems) and any("父实验" in p for p in problems)
