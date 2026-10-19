"""难版评测的已知扰动（#138）：T 的抽样、numpy warp 的坐标约定、按扰动后真值评测、工作台 API。"""
import json

import numpy as np
import pytest

from test_workbench import A_TRUE, dataset, write_exp  # noqa: F401 — dataset 是 fixture
from test_workbench_api import api, light, runs  # noqa: F401 — fixture
from moonlib import perturb as P
from workbench import perturb as WP
from workbench.evaluate import evaluate_run
from workbench.records import check_runs, load_runs

SHIFT = [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]


def compose(T, A):
    H = lambda M: np.r_[np.asarray(M, float), [[0, 0, 1]]]
    return (H(T) @ H(A))[:2].tolist()


def test_sample_levels_and_shared_direction():
    a, b = P.sample("shift12", "val", "ROI_1/patch_3"), P.sample("shift128", "val", "ROI_1/patch_3")
    assert np.allclose(a[:, :2], np.eye(2)) and np.isclose(np.linalg.norm(a[:, 2]), 12)
    assert np.isclose(np.linalg.norm(b[:, 2]), 128) and np.allclose(a[:, 2] / 12, b[:, 2] / 128)
    r = P.sample("rot10", "test", "ROI_1/patch_3")
    assert np.isclose(abs(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), 10)
    assert np.allclose(r[:, :2] @ [256, 256] + r[:, 2], [256, 256])          # 绕 patch 中心
    s = P.sample("scale20", "test", "ROI_1/patch_3")
    assert np.isclose(abs(s[0, 0] - 1), 0.2) and s[0, 1] == 0
    assert np.array_equal(P.sample("rot3", "val", "x/p"), P.sample("rot3", "val", "x/p"))
    signs = {np.sign(P.sample("rot3", "val", f"R/patch_{i}")[1, 0]) for i in range(20)}
    assert signs == {-1, 1}


def test_write_and_load_round_trip(tmp_path):
    P.write(tmp_path, {"val": ["R/patch_0"]}, variants=("shift64",))
    T = P.load(tmp_path, "shift64", "val")["R/patch_0"]
    assert np.allclose(T, P.sample("shift64", "val", "R/patch_0"))


def test_numpy_warp_moves_content_to_T_of_position():
    img = np.zeros((16, 16), np.float32)
    img[5, 4] = 1.0                                    # 中心约定 (4, 5)
    T = np.array([[1.0, 0, 3], [0, 1, 2]])
    out = WP.warp(img, T)
    assert out[7, 7] == pytest.approx(1.0) and out.sum() == pytest.approx(1.0)
    R = P.sample("rot20", "val", "x/p", hw=16)
    out = WP.warp(img, R)
    q = R[:, :2] @ [4.5, 5.5] + R[:, 2] - 0.5          # 角点约定换算
    yy, xx = np.mgrid[0:16, 0:16]
    c = np.array([(out * xx).sum(), (out * yy).sum()]) / out.sum()
    assert np.allclose(c, q, atol=0.05)


def setup_perturbed(runs, dataset, T=SHIFT, perturb_flag=True):
    d = write_exp(runs, "E7", "perturb = true\n" if perturb_flag else "")
    (d / "perturb" / "s").mkdir(parents=True)
    Ts = {p: T for p in dataset.labelled("val")}
    Ts["ROI_010/patch_10"] = [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]]   # 全部检查点出画面
    (d / "perturb" / "s" / "val.json").write_text(json.dumps({"T": Ts}), encoding="utf-8")
    light(runs, dataset, "E7", "m")
    light(runs, dataset, "E7", "m__s", A=compose(T, A_TRUE))
    return d


def test_eval_uses_perturbed_truth_and_keeps_out_of_frame(runs, dataset):
    """画面外的检查点照算（#159）：全部出画面的那一对也在，误差仍为 0；只另记画面外的点数。"""
    setup_perturbed(runs, dataset)
    met = evaluate_run(dataset, load_runs(runs)["E7"])["methods"]
    assert met["m"]["val"]["summary"]["median"] == pytest.approx(0)
    v = met["m__s"]["val"]
    assert v["summary"]["n"] == len(dataset.labelled("val")) and v["summary"]["median"] == pytest.approx(0, abs=1e-9)
    assert v["perturb"]["variant"] == "s" and v["perturb"]["outside"]["pairs"] == 1
    assert v["perturb"]["outside"]["points"] >= 8                   # 那一对的 8 个，另有平移出界的
    view = WP.dataset_for(dataset, load_runs(runs)["E7"], "m__s")
    opt, sar = view.checkpoints("val", "ROI_010/patch_10")
    assert len(opt) == len(dataset.checkpoints("val", "ROI_010/patch_10")[0]) and (sar[:, 0] >= 16).all()


def test_scale_error_measured_in_original_pixels(runs, dataset):
    S = [[1.2, 0.0, -1.6], [0.0, 1.2, -1.6]]                       # 绕 16 px patch 中心放大 20%
    off = [[1.0, 0.0, 4.0], [0.0, 1.0, -4.0]]                      # 比真值偏 1 px
    d = setup_perturbed(runs, dataset, T=S)
    light(runs, dataset, "E7", "m__s", A=compose(S, off))
    met = evaluate_run(dataset, load_runs(runs)["E7"])["methods"]["m__s"]["val"]
    assert met["summary"]["median"] == pytest.approx(1.0)
    assert d.exists()


def test_without_flag_suffix_is_ordinary_variant(runs, dataset):
    setup_perturbed(runs, dataset, perturb_flag=False)
    met = evaluate_run(dataset, load_runs(runs)["E7"])["methods"]
    assert "perturb" not in met["m__s"]["val"] and met["m__s"]["val"]["summary"]["median"] > 1


def test_check_flags_missing_T(runs, dataset):
    d = setup_perturbed(runs, dataset)
    light(runs, dataset, "E7", "m__s", split="test", A=A_TRUE)
    problems, _ = check_runs(load_runs(runs))
    assert any("perturb/s/" in p for p in problems)
    (d / "perturb" / "s" / "test.json").write_text(json.dumps({"T": {}}), encoding="utf-8")
    assert not check_runs(load_runs(runs))[0]


def test_api_compare_aligns_and_pair_view_uses_method_frame(api, runs, dataset):
    setup_perturbed(runs, dataset)
    code, c = api.get("/api/compare?method=E7/m__s&ref=E7/m&split=val")
    assert code == 200 and c["subset"] is None and len(c["pairs"]) == len(dataset.labelled("val"))
    assert len(c["method"]["errors"]) == len(c["ref"]["errors"]) == len(c["pairs"])
    assert c["diff"]["median"] == pytest.approx(0, abs=1e-9)

    code, d = api.get("/api/pair?method=E7/m__s&ref=E7/m&split=val&pair=ROI_002/patch_0&points=0")
    assert code == 200 and d["perturb"] == "E7/s"
    assert d["method"]["error"] == pytest.approx(0, abs=1e-9) and d["ref"]["error"] == pytest.approx(0, abs=1e-9)
    assert np.allclose(d["ref"]["A"], compose(SHIFT, A_TRUE))      # 参考方法的仿射换到画面坐标系
    opt, sar = (np.array(d["checkpoints"][k]) for k in ("opt", "sar"))
    assert np.allclose(sar - opt, [5, -3])

    code, d = api.get("/api/pair?method=E7/m&ref=E7/m__s&split=val&pair=ROI_002/patch_0&points=0")
    assert d["perturb"] is None and np.allclose(d["ref"]["A"], A_TRUE)

    code, d = api.get("/api/pair?method=E7/m__s&split=val&pair=ROI_010/patch_10&points=0")
    assert code == 200 and d["checkpoints"] is not None
    assert d["method"]["error"] == pytest.approx(np.hypot(98, 1))     # 预测按 SHIFT，T 却是 +100 px：画面外照样量出偏差


def test_api_image_with_perturb(api, runs, dataset):
    import urllib.request
    import urllib.error

    setup_perturbed(runs, dataset)
    url = api.base + "/img?split=val&pair=ROI_002/patch_0&kind=sar"
    with urllib.request.urlopen(url) as r0, urllib.request.urlopen(url + "&perturb=E7/s") as r1:
        assert r0.status == r1.status == 200 and r0.read() != r1.read()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "&perturb=E7/nope")
