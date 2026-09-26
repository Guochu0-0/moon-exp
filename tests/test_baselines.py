"""baseline runner：TIFF 读取、输入映射、坐标回映、match → fit → eval 链路。用合成数据，不依赖真实数据与 GPU。"""
import json

import numpy as np
import pytest

from baselines import inputs
from baselines.adapters.base import long_side_size, to_original
from baselines.fit import fit_affine
from baselines.tif import read_tif

tifffile = pytest.importorskip("tifffile")
cv2 = pytest.importorskip("cv2")


@pytest.mark.parametrize("kw", [
    dict(compression="zlib", rowsperstrip=4),
    dict(compression=None, rowsperstrip=5),
    dict(compression="zlib", tile=(16, 16)),
    dict(compression="zlib", rowsperstrip=8, planarconfig="separate"),
])
@pytest.mark.parametrize("dtype,shape", [(np.uint8, (20, 24)), (np.float32, (20, 24, 4)), (np.int16, (20, 24))])
def test_read_tif(tmp_path, kw, dtype, shape):
    if kw.get("planarconfig") == "separate" and len(shape) == 2:
        pytest.skip("单波段无 planar 之分")
    rng = np.random.default_rng(0)
    a = (rng.random(shape) * 100).astype(dtype)
    kw = dict(kw)
    if len(shape) == 3:
        kw.setdefault("planarconfig", "contig")
    p = tmp_path / "x.tif"
    # separate 时 tifffile 要 (S, H, W)；读回来统一是 H×W×S
    w = a.transpose(2, 0, 1) if kw.get("planarconfig") == "separate" else a
    tifffile.imwrite(p, w, photometric="minisblack", **kw)
    b = read_tif(p)
    assert b.dtype == a.dtype and b.shape == a.shape
    np.testing.assert_array_equal(a, b)


def test_read_tif_bigendian(tmp_path):
    a = np.arange(48, dtype=np.float32).reshape(6, 8)
    p = tmp_path / "be.tif"
    tifffile.imwrite(p, a, byteorder=">", compression="zlib")
    np.testing.assert_array_equal(read_tif(p), a)


def test_sar_mapping():
    rng = np.random.default_rng(0)
    sar = rng.random((64, 64, 4), dtype=np.float32) * 0.1
    sar[0, 0, :2] = 0.0          # −∞ dB → 下限
    sar[1, 1, 0] = np.nan
    x = inputs.sar_p2p98(sar)
    assert x.dtype == np.float32 and x.shape == (64, 64)
    assert x.min() == 0.0 and x.max() == 1.0 and np.isfinite(x).all()
    assert inputs.sar_db(sar)[0, 0] == inputs.DB_FLOOR
    # 逐 patch 拉伸：对 S1 的整体缩放不变（dB 平移）
    np.testing.assert_allclose(inputs.sar_p2p98(sar * 7.0), x, atol=1e-5)
    for name in inputs.SAR:
        y = inputs.get("sar", name)(sar)
        assert y.min() >= 0 and y.max() <= 1
    assert inputs.get("optical", "div255")(np.array([[0, 255]], np.uint8)).tolist() == [[0.0, 1.0]]


def test_to_original_matches_cv2_resize():
    """在 resize 后的图上定位一个点，映射回原图后应落在原位置。"""
    h = w = 64
    img = np.zeros((h, w), np.float32)
    img[20, 37] = 1.0
    img = cv2.GaussianBlur(img, (0, 0), 3)
    hn, wn = long_side_size(h, w, 105, df=8)
    assert (hn, wn) == (104, 104)
    r = cv2.resize(img, (wn, hn))
    ys, xs = np.mgrid[:hn, :wn]
    c = np.array([(xs * r).sum(), (ys * r).sum()]) / r.sum()
    np.testing.assert_allclose(to_original(c, h, w, hn, wn)[0], [37, 20], atol=0.05)
    np.testing.assert_allclose(to_original([[3.0, 4.0]], h, w, h, w), [[3.0, 4.0]])


def test_fit_affine_recovers_and_fails():
    rng = np.random.default_rng(0)
    A = np.array([[1.02, 0.01, 5.0], [-0.01, 0.98, -3.0]])
    src = rng.uniform(0, 512, (200, 2))
    dst = (src + 0.5) @ A[:, :2].T + A[:, 2] - 0.5
    dst[:40] = rng.uniform(0, 512, (40, 2))                       # 20% 外点
    M = np.c_[src, dst, np.ones(200)].astype(np.float32)
    Ahat, inl, fail = fit_affine(M, 3.0)
    assert fail is None and inl[40:].all() and not inl[:40].any()
    np.testing.assert_allclose(Ahat, A, atol=1e-2)
    assert fit_affine(M[:2], 3.0)[2] == "few_matches"


def test_identity_pipeline_equals_unregistered(tmp_path, monkeypatch):
    """identity 适配器走完 match → fit → eval，结果应与工作台的「未配准」逐 pair 相同。"""
    from baselines import fit, match
    from workbench.dataset import Dataset
    from workbench.evaluate import evaluate_identity, evaluate_run
    from workbench.records import load_runs

    root = tmp_path / "Processed_Data"
    rng = np.random.default_rng(1)
    for split in ("Val", "Test"):
        d = root / split / "ROI_001"
        for k in ("Optical", "SAR", "Label"):
            (d / k).mkdir(parents=True)
        for n in (0, 1, 3):
            tifffile.imwrite(d / "Optical" / f"patch_{n}.tif", rng.integers(0, 255, (64, 64), dtype=np.uint8),
                             compression="zlib")
            tifffile.imwrite(d / "SAR" / f"patch_{n}.tif", rng.random((64, 64, 4), dtype=np.float32),
                             compression="zlib", planarconfig="contig", photometric="minisblack")
            if n == 1:
                continue
            opt = rng.uniform(0, 64, (8, 2))
            sar = opt + rng.normal(0, 2, (8, 2))
            np.savetxt(d / "Label" / f"patch_{n}.txt", np.c_[opt[:, 0], -opt[:, 1], sar[:, 0], -sar[:, 1]],
                       delimiter="\t")
    run = tmp_path / "runs" / "B0"
    run.mkdir(parents=True)
    (run / "exp.toml").write_text('id = "B0"\ntitle = "t"\nstatus = "baseline"\n', encoding="utf-8")
    cfg = tmp_path / "identity.json"
    cfg.write_text(json.dumps({"method": "identity", "adapter": "identity", "params": {"step": 16},
                               "input": {"optical": "div255", "sar": "p2p98"}}), encoding="utf-8")
    out = tmp_path / "raw"
    for split in ("val", "test"):
        match.main([str(cfg), "--split", split, "--data", str(tmp_path), "--out", str(out), "--device", "cpu"])
        fit.main([str(out / "identity"), "--split", split, "--run", str(run), "--data", str(tmp_path)])
    meta = json.loads((out / "identity" / "val.meta.json").read_text(encoding="utf-8"))
    assert meta["n_pairs"] == 2 and meta["n_errors_this_session"] == 0

    ds = Dataset(tmp_path)
    got = evaluate_run(ds, load_runs(tmp_path / "runs")["B0"])["methods"]["identity"]
    ref = evaluate_identity(ds)
    for s in ("val", "test"):
        assert got[s]["coverage"]["missing"] == 0
        np.testing.assert_allclose(got[s]["errors"], ref[s]["errors"], atol=1e-4)
