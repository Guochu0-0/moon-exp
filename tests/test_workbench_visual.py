"""节点页「可视化结果」用的 API：单个 pair 的仿射、检查点与点对，以及比较结果里的 pair 编号。
临时 runs/ + 合成数据集（真值是平移 (3, -4)），经真实 HTTP。"""
import numpy as np
import pytest

from test_workbench import A_TRUE, dataset, write_exp  # noqa: F401 — dataset 是 fixture
from test_workbench_api import api, light, runs  # noqa: F401 — api、runs 是 fixture
from workbench.records import PredWriter

PAIR = "ROI_002/patch_10"
# matcher 坐标约定（整数 = 像素中心）的 N×4；写入端 +0.5。第 1 个点符合真值，第 2 个偏 (5, 4)，残差 √41
M4 = np.array([[1, 1, 4, -3], [2, 2, 10, 2]], dtype=np.float32)


def pair(api, method, ref=None, p=PAIR, split="val", points=True):
    q = f"/api/pair?split={split}&pair={p}&method={method}" + (f"&ref={ref}" if ref is not None else "")
    return api.get(q + ("" if points else "&points=0"))


def write(runs, rid, method, rows, split="val"):
    """rows: {pair: (A, fail, matches, conf)}，没列出的 pair 不写。"""
    write_exp(runs, rid)
    with PredWriter(runs / rid, method, split, repo=runs) as w:
        for p, (A, fail, m, conf) in rows.items():
            w.write(p, A, fail=fail, matches=m, conf=conf)


def test_pair_with_matches_gives_points_conf_and_residual_to_estimated_affine(api, runs, dataset):
    write(runs, "E1", "main", {PAIR: (A_TRUE, None, M4, [0.9, 0.2])})
    code, d = pair(api, "E1/main")
    assert code == 200, d
    assert (d["split"], d["pair"], d["no"], d["inlier_px"]) == ("val", PAIR, 2, 3.0)   # 编号含无标注 pair
    assert d["size"] == [16, 16] and d["sar_size"] == [16, 16]
    assert np.allclose(np.array(d["checkpoints"]["sar"]) - d["checkpoints"]["opt"], [3, -4])
    assert d["ref"] is None
    me = d["method"]
    assert (me["key"], me["label"], me["exp"], me["A"], me["fail"]) == ("E1/main", "E1", "E1", A_TRUE, None)
    assert me["error"] == pytest.approx(0.0)
    mt = me["matches"]
    assert (mt["state"], mt["n"]) == ("ok", 2)
    assert mt["opt"] == [[1.5, 1.5], [2.5, 2.5]] and mt["sar"] == [[4.5, -2.5], [10.5, 2.5]]   # +0.5 约定
    assert mt["conf"] == pytest.approx([0.9, 0.2])
    assert mt["resid"] == pytest.approx([0.0, 41 ** .5], abs=1e-3)


def test_pair_without_local_matches_suggests_sync_command(api, runs, dataset):
    write(runs, "E1", "main", {PAIR: (A_TRUE, None, None, None)})     # 点对留在服务器上：本地没有 npz
    code, d = pair(api, "E1/main")
    assert code == 200
    assert d["method"]["matches"] == {"state": "not_synced", "sync": "python -m workbench sync E1"}


def test_empty_matches_and_matches_without_conf(api, runs, dataset):
    write(runs, "E1", "main", {PAIR: (A_TRUE, None, np.zeros((0, 5)), None),
                               "ROI_010/patch_0": (A_TRUE, None, M4, None)})          # N×4 不给 conf
    mt = pair(api, "E1/main")[1]["method"]["matches"]
    assert mt == {"state": "ok", "n": 0, "opt": [], "sar": [], "conf": None, "resid": []}   # 0×5 自然退化
    mt = pair(api, "E1/main", p="ROI_010/patch_0")[1]["method"]["matches"]
    assert mt["n"] == 2 and mt["conf"] is None                                       # 没有 conf：整列 NaN → null


def test_failed_pair_gives_reason_and_points_without_residual(api, runs, dataset):
    write(runs, "E1", "main", {PAIR: (None, "few_inliers", M4, [0.9, 0.2]),
                               "ROI_010/patch_0": (None, "error: CUDA OOM", M4, [0.9, 0.2])})
    d = pair(api, "E1/main")[1]["method"]
    assert (d["A"], d["fail"], d["error"]) == (None, "few_inliers", None)            # 失败：误差记 ∞ → null
    assert d["matches"]["n"] == 2 and d["matches"]["resid"] is None                  # 没有仿射就判不了内外点
    d = pair(api, "E1/main", p="ROI_010/patch_0")[1]["method"]
    assert d["fail"] == "error: CUDA OOM" and d["matches"] == {"state": "absent"}    # 出错的 pair 不写点对
    d = pair(api, "E1/main", p="ROI_010/patch_10")[1]["method"]                      # 没有预测行：按失败计
    assert (d["A"], d["fail"], d["error"]) == (None, "no_pred", None)


def test_pair_reference_side_and_options(api, runs, dataset):
    write(runs, "E1", "main", {PAIR: (A_TRUE, None, M4, [0.9, 0.2])})
    write_exp(runs, "B0", 'baseline = true\n[methods.roma]\nname = "RoMa"\ncaveat = "点数是采样出来的"\n')
    light(runs, dataset, "B0", "roma", A=[[1, 0, 0], [0, 1, 0]])
    code, d = pair(api, "E1/main", "B0/roma")
    assert code == 200
    r = d["ref"]
    assert (r["key"], r["label"], r["exp"], r["caveat"]) == ("B0/roma", "B0", "B0", "点数是采样出来的")   # 单方法实验只写编号
    assert r["error"] == pytest.approx(5.0)
    assert r["matches"]["state"] == "not_synced" and r["matches"]["sync"] == "python -m workbench sync B0"
    d = pair(api, "E1/main", "identity")[1]["ref"]
    assert (d["label"], d["A"], d["matches"]) == ("未配准", [[1, 0, 0], [0, 1, 0]], None)
    assert d["error"] == pytest.approx(5.0)
    d = pair(api, "E1/main", "B0/roma", points=False)[1]                             # 缩略图只要仿射与检查点
    assert d["method"]["matches"] is None and d["ref"]["matches"] is None
    unlabelled = pair(api, "E1/main", p="ROI_002/patch_2")[1]
    assert unlabelled["checkpoints"] is None and unlabelled["method"]["error"] is None
    assert pair(api, "E1/main", p="ROI_002/patch_99")[0] == 400
    assert pair(api, "E1/nope")[0] == 400
    assert pair(api, "identity")[0] == 400


def test_compare_lists_labelled_pairs_with_numbers(api, runs, dataset):
    write_exp(runs, "E1")
    light(runs, dataset, "E1", "main")
    code, c = api.get("/api/compare?method=E1/main&split=val")
    assert code == 200
    assert c["pairs"] == ["ROI_002/patch_0", "ROI_002/patch_10", "ROI_010/patch_0", "ROI_010/patch_10"]
    assert c["no"] == [0, 2, 3, 5]                                                   # 与 errors 一一对应
    assert len(c["method"]["errors"]) == 4
