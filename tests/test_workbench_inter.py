"""中间结果：InterWriter 写 → 经 HTTP API 读回。临时 runs/ + 合成数据集（16×16，真值是平移 (3, -4)）。"""
import io
import json

import numpy as np
import pytest

from test_workbench import A_TRUE, dataset, write_exp  # noqa: F401 — dataset 是 fixture
from test_workbench_api import api, runs  # noqa: F401 — api、runs 是 fixture
from test_workbench_visual import PAIR, pair, write
from workbench.records import InterWriter

OTHER = "ROI_010/patch_0"     # 有标注、但中间结果不覆盖的 pair


def png(api, path):
    import urllib.request

    from PIL import Image

    with urllib.request.urlopen(api.base + path) as r:
        assert r.headers["Content-Type"] == "image/png"
        return np.asarray(Image.open(io.BytesIO(r.read())))


def inter(api, name, method="E1/main", p=PAIR, split="val"):
    return api.get(f"/api/inter?split={split}&pair={p}&method={method}&name={name}")


def lit(runs, rid="E1"):
    write(runs, rid, "main", {PAIR: (A_TRUE, None, None, None), OTHER: (A_TRUE, None, None, None)})


def test_four_kinds_roundtrip_and_partial_coverage(api, runs, dataset):
    lit(runs)
    conf = np.arange(32, dtype=np.float32).reshape(4, 8) / 31   # 低分辨率：显示时按 frame 拉伸到 16×16
    pts = np.array([[1, 2, 0.5], [7, 3, 2.0]])
    flow = np.zeros((8, 8, 2), np.float32)
    flow[..., 0], flow[..., 1] = 3, -4                         # 光学 → SAR 的位移：恰好是真值平移
    rgb = np.zeros((5, 6, 3), np.uint8)
    rgb[..., 1] = 200
    with InterWriter(runs / "E1", "main", "certainty", "val", kind="scalar", frame="opt", desc="RoMa certainty") as w:
        w.write(PAIR, conf)
    with InterWriter(runs / "E1", "main", "kpts", "val", kind="points", frame="sar", desc="关键点", unit="分") as w:
        w.write(PAIR, pts)
    with InterWriter(runs / "E1", "main", "warp", "val", kind="flow", frame="opt") as w:
        w.write(PAIR, flow)
    with InterWriter(runs / "E1", "main", "resp", "val", kind="image", frame="sar") as w:
        w.write(PAIR, rgb)

    meta = json.loads((runs / "E1/inter/main/certainty/meta.json").read_text(encoding="utf-8"))
    assert meta == {"kind": "scalar", "frame": "opt", "desc": "RoMa certainty", "unit": ""}

    # pair 接口给出清单：名称、种类、坐标系、说明、单位，以及这个 pair 的状态
    side = pair(api, "E1/main")[1]["method"]
    assert [(x["name"], x["kind"], x["frame"], x["state"]) for x in side["inter"]] == [
        ("certainty", "scalar", "opt", "ok"), ("kpts", "points", "sar", "ok"),
        ("resp", "image", "sar", "ok"), ("warp", "flow", "opt", "ok")]
    assert side["inter"][1]["desc"] == "关键点" and side["inter"][1]["unit"] == "分"
    # 只覆盖部分 pair：其余 pair 标 absent
    assert {x["state"] for x in pair(api, "E1/main", p=OTHER)[1]["method"]["inter"]} == {"absent"}

    code, d = inter(api, "certainty")
    assert code == 200, d
    assert (d["kind"], d["state"], d["shape"]) == ("scalar", "ok", [4, 8])
    assert (d["vmin"], d["vmax"]) == pytest.approx((0.0, 1.0))
    heat = png(api, d["png"])
    assert heat.shape == (4, 8, 4)                              # 原分辨率 RGBA，页面按 frame 拉伸到 patch 大小
    assert not np.array_equal(heat[0, 0], heat[-1, -1])          # 低值和高值颜色不同

    d = inter(api, "kpts")[1]
    assert d["kind"] == "points" and d["png"] is None
    assert d["points"] == [[1.5, 2.5, 0.5], [7.5, 3.5, 2.0]]    # 写入端 +0.5，与点对同一约定
    assert (d["vmin"], d["vmax"]) == (0.5, 2.0)

    d = inter(api, "warp")[1]
    assert (d["kind"], d["shape"]) == ("flow", [8, 8, 2])
    assert (d["vmin"], d["vmax"]) == pytest.approx((5.0, 5.0))  # 位移大小
    warped = png(api, d["png"])                                 # SAR 按位移场摆回光学坐标
    assert warped.shape == (16, 16, 4)
    from workbench.dataset import to_display
    sar = to_display(dataset.sar("val", PAIR))
    assert np.array_equal(warped[5, 5, 0], sar[1, 8])            # (x, y) = (5, 5) → SAR (8, 1)
    assert warped[0, 0, 3] == 0 and warped[10, 2, 3] == 255     # 落到 SAR 外的像素透明

    d = inter(api, "resp")[1]
    assert d["kind"] == "image"
    assert np.array_equal(png(api, d["png"])[..., :3], rgb)      # 图片原样返回，拉伸交给页面

    d = inter(api, "certainty", p=OTHER)[1]
    assert d["state"] == "absent" and d["png"] is None


def test_shape_checked_by_kind(runs):
    write_exp(runs, "E1")
    cases = [("scalar", np.zeros((4, 4, 2))), ("points", np.zeros((3, 2))), ("flow", np.zeros((4, 4))),
             ("flow", np.zeros((4, 4, 3))), ("image", np.zeros((4, 4, 2), np.uint8)), ("scalar", np.zeros((0, 4)))]
    for kind, bad in cases:
        with InterWriter(runs / "E1", "main", kind, "val", kind=kind, frame="opt") as w:
            with pytest.raises(ValueError, match=kind):
                w.write(PAIR, bad)
    with pytest.raises(ValueError, match="kind"):
        InterWriter(runs / "E1", "main", "x", "val", kind="heatmap", frame="opt")
    with pytest.raises(ValueError, match="frame"):
        InterWriter(runs / "E1", "main", "x", "val", kind="scalar", frame="both")


def test_not_synced_lists_inter_with_sync_command(api, runs, dataset):
    lit(runs)
    with InterWriter(runs / "E1", "main", "certainty", "val", kind="scalar", frame="opt") as w:
        w.write(PAIR, np.ones((4, 4)))
    (runs / "E1/inter/main/certainty/val.npz").unlink()           # 数据留在服务器上，只有 meta.json 在 git 里
    side = pair(api, "E1/main")[1]["method"]
    assert side["inter"] == [{"name": "certainty", "kind": "scalar", "frame": "opt", "desc": "", "unit": "",
                              "state": "not_synced", "sync": "python -m workbench sync E1 --extra certainty"}]
    d = inter(api, "certainty")[1]
    assert d["state"] == "not_synced" and d["sync"] == "python -m workbench sync E1 --extra certainty"


def test_image_kind_not_synced_and_unknown_name(api, runs, dataset):
    lit(runs)
    with InterWriter(runs / "E1", "main", "resp", "val", kind="image", frame="sar") as w:
        w.write(PAIR, np.zeros((4, 4), np.uint8))
    assert (runs / "E1/inter/main/resp/val/ROI_002__patch_10.png").is_file()
    import shutil
    shutil.rmtree(runs / "E1/inter/main/resp/val")
    assert inter(api, "resp")[1]["state"] == "not_synced"
    code, d = inter(api, "nope")
    assert code == 404 and "nope" in d["error"]


def test_splits_kept_apart_and_rewrite_replaces(api, runs, dataset):
    lit(runs)
    write(runs, "E1", "main", {PAIR: (A_TRUE, None, None, None)}, split="test")
    with InterWriter(runs / "E1", "main", "c", "val", kind="scalar", frame="opt") as w:
        w.write(PAIR, np.full((2, 2), 3.0))
    with InterWriter(runs / "E1", "main", "c", "test", kind="scalar", frame="opt") as w:
        w.write(PAIR, np.full((2, 2), 7.0))
    code, d = inter(api, "c")
    assert code == 200, d
    assert d["vmax"] == 3.0
    assert inter(api, "c", split="test")[1]["vmax"] == 7.0
    with InterWriter(runs / "E1", "main", "c", "val", kind="scalar", frame="opt") as w:   # 重跑：整份替换
        w.write(OTHER, np.zeros((2, 2)))
    assert inter(api, "c")[1]["state"] == "absent"
