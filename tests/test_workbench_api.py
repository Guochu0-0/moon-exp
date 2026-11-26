"""工作台 v2 的 HTTP JSON API：画布摘要与实验编辑。临时 runs/ + 合成数据集，经真实 HTTP 读写，再看磁盘。"""
import json
import shutil
import threading
import tomllib
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from test_workbench import A_TRUE, dataset, write_exp  # noqa: F401 — dataset 是 fixture
from workbench import server
from workbench.records import PredWriter


class Client:
    def __init__(self, base):
        self.base = base

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body or {})

    def put(self, path, body):
        return self.call("PUT", path, body)

    def delete(self, path):
        return self.call("DELETE", path)

    def exps(self):
        code, d = self.get("/api/data")
        assert code == 200
        return {e["id"]: e for e in d["experiments"]}


@pytest.fixture
def runs(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def api(runs, dataset, tmp_path):
    runs.mkdir(exist_ok=True)
    wb = server.Workbench(runs, dataset, tmp_path, ds_v2=server.Dataset(tmp_path / "moon-v2"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(wb))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield Client(f"http://127.0.0.1:{httpd.server_address[1]}")
    httpd.shutdown()
    httpd.server_close()


def light(runs, dataset, rid, method, split="val", A=A_TRUE):
    with PredWriter(runs / rid, method, split, repo=runs) as w:
        for p in dataset.pairs(split):
            w.write(p, A)


def toml(runs, rid):
    return tomllib.loads((runs / rid / "exp.toml").read_text(encoding="utf-8"))


def canvas(runs):
    return json.loads((runs / "canvas.json").read_text(encoding="utf-8"))


def test_summary_lists_experiments_with_lit_state_and_metrics(api, runs, dataset):
    write_exp(runs, "B0", 'baseline = true\n[methods.roma]\nname = "RoMa outdoor"\n')
    write_exp(runs, "E1", 'parent = "B0"\ninit = "B0/roma"\n')
    light(runs, dataset, "B0", "roma")
    light(runs, dataset, "B0", "loftr", A=[[1, 0, 0], [0, 1, 0]])   # 恒等仿射：误差恒 5 px
    light(runs, dataset, "B0", "spsg", split="test")                # 只有 Test：没有 Val 指标

    e = api.exps()
    b0, e1 = e["B0"], e["E1"]
    assert b0["lit"] and not e1["lit"]
    assert b0["baseline"] and not e1["baseline"]
    assert (e1["parent"], e1["init"], e1["title"]) == ("B0", "B0/roma", "t")
    ms = b0["methods"]
    assert [m["id"] for m in ms] == ["roma", "loftr", "spsg"]            # 按 AUC@10（Val）降序，没有的排最后
    assert ms[0]["auc"] == pytest.approx(1.0) and ms[0]["name"] == "RoMa outdoor"
    assert ms[1]["auc"] == pytest.approx(0.5) and ms[2]["auc"] is None
    assert e1["methods"] == []
    for x in (b0, e1):
        assert isinstance(x["x"], (int, float)) and isinstance(x["y"], (int, float))


def test_create_writes_toml_and_shows_unlit_with_default_position(api, runs):
    write_exp(runs, "B0", "baseline = true\n")
    write_exp(runs, "E1")
    code, r = api.post("/api/exp", {"title": "想法"})
    assert code == 200 and r["id"] == "E2"                               # 自动建议下一个 E<n>
    t = toml(runs, "E2")
    assert t["id"] == "E2" and t["title"] == "想法" and len(t["date"]) == 10 and "parent" not in t
    assert t["data_version"] == "v2"
    e2 = api.exps()["E2"]
    assert not e2["lit"] and isinstance(e2["x"], (int, float))
    assert canvas(runs)["experiments"]["E2"] == {"x": e2["x"], "y": e2["y"]}

    code, r = api.post("/api/exp", {"id": "E9", "title": "", "x": 12.4, "y": -30})
    assert code == 200 and canvas(runs)["experiments"]["E9"] == {"x": 12, "y": -30}
    assert api.post("/api/exp", {"id": "E9"})[0] == 400                  # 已存在
    assert api.post("/api/exp", {"id": "../x"})[0] == 400                # 非法编号
    assert api.post("/api/exp", {"id": "E10", "data_version": "v1"})[0] == 200
    assert toml(runs, "E10")["data_version"] == "v1"


def test_two_data_versions_share_canvas_and_use_their_own_evaluation(api, runs, dataset, tmp_path):
    from workbench.__main__ import main

    root = tmp_path / "moon-v2"
    shutil.copytree(dataset.root / "Test", root / "Test")
    roi = dataset.pairs("val")[0].split("/")[0]
    shutil.copytree(dataset.root / "Val" / roi, root / "Val" / "ROI_001",
                    ignore=shutil.ignore_patterns("Label"))
    shutil.copytree(dataset.root / "Val" / roi, root / "Test" / "ROI_035")
    ds2 = server.Dataset(root)
    write_exp(runs, "E1")
    write_exp(runs, "E2", 'parent = "E1"\ndata_version = "v2"\n')
    for s in ("val", "test"):
        light(runs, dataset, "E1", "main", s)
    provenance = {"data_version": "v2", "step": 20, "weights_sha256": "test-checkpoint"}
    with PredWriter(runs / "E2", "main", "test", repo=runs, provenance=provenance) as w:
        for pair in ds2.pairs("test"):
            w.write(pair, A_TRUE)
    with PredWriter(runs / "E2", "main", "val", repo=runs, provenance=provenance) as w:
        for pair in ds2.pairs("val"):
            w.write(pair, None, fail="few_matches", n_matches=0)
    peak = runs / "E2/sweep/main/peak.json"
    peak.parent.mkdir(parents=True)
    peak.write_text(json.dumps({**provenance, "metric": "val_inlier_mean",
                                "candidates": [{"step": 20, "summary": {"inlier_mean": 0}}]}))
    test_preds = runs / "E2/preds/main/test.jsonl"
    test_preds.write_bytes(test_preds.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    cards = api.exps()
    assert cards["E2"]["parent"] == "E1"
    assert cards["E1"]["metric"]["split"] == "val" and cards["E2"]["metric"]["split"] == "test"
    assert cards["E1"]["methods"][0]["auc"] == cards["E2"]["methods"][0]["auc"] == 1

    status, detail = api.get("/api/exp/E2")
    assert status == 200 and detail["default_split"] == "test"
    method = detail["methods"][0]
    assert method["selection"]["step"] == 20
    assert method["delta"] == {} and method["delta_reason"]["test"]
    assert method["refs"]["test"] != "E1/main"
    status, comparison = api.get("/api/compare?method=E2/main&ref=E1/main")
    assert status == 200 and comparison["split"] == "test"
    assert comparison["diff"] is None and comparison["comparison_reason"]
    assert comparison["method"]["summary"]["n"] > comparison["ref"]["summary"]["n"]
    assert method["unmatched_reference"]["test"]["key"] == "E1/main"

    status, val = api.get("/api/compare?method=E2/main&split=val")
    assert status == 200 and val["method"]["inlier_rates"] == [0] * len(ds2.pairs("val"))
    pair = ds2.pairs("val")[0]
    assert api.get(f"/api/pair?method=E2/main&split=val&pair={pair}")[1]["checkpoints"] is None
    with urllib.request.urlopen(f"{api.base}/img?exp=E2&split=val&kind=opt&pair={pair}") as response:
        assert response.read().startswith(b"\x89PNG")
    pair = ds2.pairs("test")[-1]
    with urllib.request.urlopen(f"{api.base}/img?exp=E2&split=test&kind=sar&pair={pair}") as response:
        assert response.read().startswith(b"\x89PNG")

    main(["--runs", str(runs), "--data", str(dataset.root), "--data-v2", str(root), "eval", "E1", "E2"])
    old, new = [json.loads((runs / rid / "metrics.json").read_text())["methods"]["main"] for rid in ("E1", "E2")]
    assert old["val"]["summary"]["auc@10"] == 1
    assert new["val"]["summary"]["inlier_mean"] == 0
    assert new["test"]["summary"]["n"] == comparison["method"]["summary"]["n"]

    (runs / "E1/preds/main/test.jsonl").unlink()
    _, cross_split = api.get("/api/compare?method=E2/main&ref=E1/main")
    assert cross_split["ref"]["split"] == "val"
    assert cross_split["diff"] is None and cross_split["comparison_reason"]
    _, detail = api.get("/api/exp/E2")
    assert detail["methods"][0]["unmatched_reference"]["test"]["split"] == "val"

    test_preds.write_bytes(test_preds.read_bytes().replace(b'"A":', b'"changed":'))
    assert api.get(f"/api/pair?method=E2/main&split=test&pair={pair}")[0] == 400

    shutil.move(str(root / "Test"), str(root / "Test-unavailable"))
    cards = api.exps()
    assert cards["E1"]["methods"][0]["auc"] == 1
    assert cards["E2"]["methods"][0]["auc"] is None
    warning = cards["E2"]["warnings"][0]
    assert "Moon-v2 Test" in warning and "MOON_DATA_V2" in warning
    assert api.get("/api/exp/E2")[1]["warnings"] == [warning]


def test_invalid_data_version_leaves_other_nodes_usable(api, runs, dataset):
    write_exp(runs, "E1")
    write_exp(runs, "E2", 'data_version = "unknown"\n')
    light(runs, dataset, "E1", "main")
    cards = api.exps()
    assert cards["E1"]["methods"][0]["auc"] == 1
    assert "data_version" in cards["E2"]["warnings"][0]
    assert api.get("/api/exp/E2")[0] == 200


def test_derive_places_right_of_parent_and_siblings_downward(api, runs, dataset):
    write_exp(runs, "B0", "baseline = true\n")
    light(runs, dataset, "B0", "roma")
    api.put("/api/canvas", {"experiments": {"B0": {"x": 100, "y": 50}}})
    _, a = api.post("/api/exp", {"parent": "B0", "init": "B0/roma"})
    _, b = api.post("/api/exp", {"parent": "B0"})
    e = api.exps()
    ea, eb = e[a["id"]], e[b["id"]]
    assert toml(runs, a["id"])["init"] == "B0/roma" and toml(runs, b["id"])["parent"] == "B0"
    assert ea["x"] > 100 + 300 and ea["y"] == 50                        # 父实验右侧
    assert eb["x"] == ea["x"] and eb["y"] > ea["y"] + 100               # 兄弟往下排，不重叠


def test_unplaced_experiments_get_default_positions_without_overlap(api, runs):
    write_exp(runs, "B0")
    for rid in ("E1", "E2", "E3"):       # 终端 / agent 写的：canvas.json 里没有
        write_exp(runs, rid, 'parent = "B0"\n')
    e = api.exps()
    assert all(e[k]["x"] > e["B0"]["x"] for k in ("E1", "E2", "E3"))
    ys = sorted(e[k]["y"] for k in ("E1", "E2", "E3"))
    assert ys[1] - ys[0] >= 100 and ys[2] - ys[1] >= 100
    assert not (runs / "canvas.json").exists()                           # 读接口不写文件


def test_creating_does_not_move_unplaced_experiments(api, runs):
    write_exp(runs, "B0")
    write_exp(runs, "E1", 'parent = "B0"\n')
    shown = {k: (e["x"], e["y"]) for k, e in api.exps().items()}
    api.post("/api/exp", {"x": shown["B0"][0], "y": shown["B0"][1] - 50})   # 落在 B0 附近
    after = api.exps()
    assert {k: (after[k]["x"], after[k]["y"]) for k in shown} == shown


def test_new_preds_light_up_on_refetch(api, runs, dataset):
    write_exp(runs, "E1")
    assert not api.exps()["E1"]["lit"]
    light(runs, dataset, "E1", "main")
    e1 = api.exps()["E1"]
    assert e1["lit"] and e1["methods"][0]["auc"] == pytest.approx(1.0)
    light(runs, dataset, "E1", "main", A=[[1, 0, 0], [0, 1, 0]])        # 重写 preds → 指标跟着变
    assert api.exps()["E1"]["methods"][0]["auc"] == pytest.approx(0.5)


def test_refetch_only_reevaluates_changed_experiments(api, runs, dataset, monkeypatch):
    for rid in ("B0", "E1"):
        write_exp(runs, rid)
        light(runs, dataset, rid, "main")
    api.exps()
    calls = []
    real = server.pair_errors
    monkeypatch.setattr(server, "pair_errors", lambda *a: calls.append(a) or real(*a))
    api.exps()
    assert calls == []
    light(runs, dataset, "E1", "main", A=[[1, 0, 0], [0, 1, 0]])
    api.exps()
    assert len(calls) == 1


def test_reparent_and_cycle_rejected(api, runs, dataset):
    write_exp(runs, "B0")
    write_exp(runs, "E1", 'parent = "B0"\n')
    write_exp(runs, "E2", 'parent = "E1"\n')
    light(runs, dataset, "E1", "main")
    code, _ = api.post("/api/exp/E2/parent", {"parent": "B0"})
    assert code == 200 and toml(runs, "E2")["parent"] == "B0"
    code, _ = api.post("/api/exp/E1/parent", {"parent": "E2", "init": "E2/main"})   # 已点亮也能改父
    assert code == 200

    before = (runs / "B0" / "exp.toml").read_bytes()
    code, r = api.post("/api/exp/B0/parent", {"parent": "E1"})                     # B0 → E1 → E2 → B0
    assert code == 400 and "环" in r["error"]
    assert (runs / "B0" / "exp.toml").read_bytes() == before
    assert api.post("/api/exp/B0/parent", {"parent": "B0"})[0] == 400
    assert api.post("/api/exp/E2/parent", {"parent": "Y"})[0] == 400
    assert api.post("/api/exp/E2/parent", {"parent": "B0", "init": "E1/main"})[0] == 400  # init 不在新父实验上


def test_disconnect_parent_keeps_other_fields(api, runs):
    write_exp(runs, "B0")
    write_exp(runs, "E1", 'parent = "B0"\ninit = "B0/roma"\nbaseline = false\n# 注释保留\n'
                          '[methods.main]\nname = "M"\n')
    code, _ = api.post("/api/exp/E1/parent", {"parent": None})
    assert code == 200
    text = (runs / "E1" / "exp.toml").read_text(encoding="utf-8")
    t = tomllib.loads(text)
    assert "parent" not in t and "init" not in t
    assert t["methods"] == {"main": {"name": "M"}} and t["baseline"] is False and "# 注释保留" in text
    assert api.exps()["E1"]["parent"] is None


def test_rename_cascades_and_refuses_lit(api, runs, dataset):
    write_exp(runs, "B0")
    write_exp(runs, "E1", 'parent = "B0"\n')
    write_exp(runs, "E2", 'parent = "E1"\ninit = "E1/main"\n')
    write_exp(runs, "E3", 'parent = "E1"\n')
    api.put("/api/canvas", {"experiments": {"E1": {"x": 1, "y": 2}, "E2": {"x": 3, "y": 4}}})
    code, _ = api.post("/api/exp/E1/rename", {"id": "E7"})
    assert code == 200
    assert not (runs / "E1").exists() and toml(runs, "E7")["id"] == "E7" and toml(runs, "E7")["parent"] == "B0"
    assert toml(runs, "E2")["parent"] == "E7" and toml(runs, "E2")["init"] == "E7/main"
    assert toml(runs, "E3")["parent"] == "E7"
    c = canvas(runs)["experiments"]
    assert "E1" not in c and c["E7"] == {"x": 1, "y": 2} and c["E2"] == {"x": 3, "y": 4}

    assert api.post("/api/exp/E7/rename", {"id": "E2"})[0] == 400          # 目标已存在
    light(runs, dataset, "B0", "main")
    code, r = api.post("/api/exp/B0/rename", {"id": "B1"})
    assert code == 400 and "点亮" in r["error"] and (runs / "B0").exists()


def test_set_title(api, runs):
    write_exp(runs, "E1")
    assert api.post("/api/exp/E1/title", {"title": '新"标题"'})[0] == 200
    assert toml(runs, "E1")["title"] == '新"标题"'


def test_delete_unlit_only(api, runs, dataset):
    write_exp(runs, "B0")
    write_exp(runs, "E1", 'parent = "B0"\n')
    write_exp(runs, "E2", 'parent = "E1"\ninit = "E1/main"\n')
    light(runs, dataset, "B0", "main")
    api.put("/api/canvas", {"experiments": {"E1": {"x": 0, "y": 0}}})
    code, r = api.delete("/api/exp/B0")
    assert code == 400 and "点亮" in r["error"] and (runs / "B0" / "exp.toml").exists()
    assert api.delete("/api/exp/E1")[0] == 200
    assert not (runs / "E1").exists()
    assert "parent" not in toml(runs, "E2") and "init" not in toml(runs, "E2")   # 子实验变为根
    assert "E1" not in canvas(runs)["experiments"]
    assert api.delete("/api/exp/E1")[0] == 404


def test_save_canvas_sorts_and_drops_missing(api, runs):
    write_exp(runs, "B0")
    write_exp(runs, "E1")
    note = [{"id": "s1", "x": 0, "y": 300, "w": 220, "h": 120, "text": "注"}]
    code, _ = api.put("/api/canvas", {"experiments": {"E1": {"x": 5, "y": 6}, "Z9": {"x": 0, "y": 0},
                                                      "B0": {"x": 1, "y": 2}}, "stickies": note})
    assert code == 200
    text = (runs / "canvas.json").read_text(encoding="utf-8")
    c = json.loads(text)
    assert list(c["experiments"]) == ["B0", "E1"] and c["stickies"] == note and c["groups"] == []
    assert text.startswith('{\n  "experiments": {\n    "B0"') and text.endswith("}\n")
    api.put("/api/canvas", {"experiments": {"B0": {"x": 9, "y": 9}}})          # 只给 experiments：其余保留
    c = canvas(runs)
    assert c["stickies"] == note and c["experiments"] == {"B0": {"x": 9, "y": 9}, "E1": {"x": 5, "y": 6}}
    _, d = api.get("/api/data")
    assert d["canvas"]["stickies"] == note


def test_groups_and_stickies_round_trip_in_stable_format(api, runs):
    write_exp(runs, "B0")
    groups = [{"id": "g2", "x": 10.6, "y": -20, "w": 460, "h": 300, "title": "待跑：等算力", "color": "c3"},
              {"id": "g1", "x": 0, "y": 0, "w": 800, "h": 260, "title": "零样本基线", "color": "c1"}]
    stickies = [{"id": "s1", "x": 0, "y": 300.4, "w": 230, "h": 80, "text": "Test 只在定稿时跑\n平时只看 Val"}]
    assert api.put("/api/canvas", {"groups": groups, "stickies": stickies})[0] == 200
    text = (runs / "canvas.json").read_text(encoding="utf-8")
    c = json.loads(text)
    assert c["groups"] == [{"color": "c3", "h": 300, "id": "g2", "title": "待跑：等算力", "w": 460, "x": 11, "y": -20},
                           {"color": "c1", "h": 260, "id": "g1", "title": "零样本基线", "w": 800, "x": 0, "y": 0}]
    assert c["stickies"] == [{"h": 80, "id": "s1", "text": "Test 只在定稿时跑\n平时只看 Val", "w": 230, "x": 0, "y": 300}]
    assert text == json.dumps(c, ensure_ascii=False, indent=2, sort_keys=True) + "\n"   # key 排序、缩进 2 格
    assert '    {\n      "color": "c3",\n      "h": 300,' in text
    _, d = api.get("/api/data")
    assert d["canvas"] == {"groups": c["groups"], "stickies": c["stickies"]}


def test_group_drag_saves_members_and_frame_together(api, runs):
    write_exp(runs, "B0")
    write_exp(runs, "E1")
    g = {"id": "g1", "x": 0, "y": 0, "w": 800, "h": 300, "title": "组", "color": "c2"}
    s = {"id": "s1", "x": 20, "y": 60, "w": 230, "h": 80, "text": "注"}
    api.put("/api/canvas", {"experiments": {"B0": {"x": 20, "y": 150}, "E1": {"x": 900, "y": 0}},
                            "groups": [g], "stickies": [s]})
    moved = {"experiments": {"B0": {"x": 120, "y": 200}}, "groups": [{**g, "x": 100, "y": 50}],
             "stickies": [{**s, "x": 120, "y": 110}]}                      # 前端拖标题栏后一次 PUT
    assert api.put("/api/canvas", moved)[0] == 200
    c = canvas(runs)
    assert c["experiments"] == {"B0": {"x": 120, "y": 200}, "E1": {"x": 900, "y": 0}}
    assert (c["groups"][0]["x"], c["groups"][0]["y"]) == (100, 50)
    assert (c["stickies"][0]["x"], c["stickies"][0]["y"]) == (120, 110)


@pytest.mark.parametrize("body", [
    {"groups": [{"id": "g1", "x": 0, "y": 0, "w": 9, "h": 9, "title": "t", "color": "red"}]},     # 不在 4 种颜色里
    {"groups": [{"id": "g1", "x": 0, "y": 0, "w": 9, "h": 9, "title": "t"}]},                     # 缺 color
    {"groups": [{"id": "g1", "x": "0", "y": 0, "w": 9, "h": 9, "title": "t", "color": "c1"}]},    # 坐标不是数值
    {"groups": ["g1"]},
    {"stickies": [{"id": "s1", "x": 0, "y": 0, "w": 9, "h": 9, "text": 3}]},                      # text 不是字符串
    {"stickies": [{"id": "", "x": 0, "y": 0, "w": 9, "h": 9, "text": ""}]},                       # 空 id
    {"stickies": [{"id": "s1", "x": 0, "y": 0, "w": 9, "h": 9, "text": ""}] * 2},                 # id 重复
])
def test_bad_groups_or_stickies_rejected_without_writing(api, runs, body):
    write_exp(runs, "B0")
    api.put("/api/canvas", {"experiments": {"B0": {"x": 1, "y": 2}}})
    before = (runs / "canvas.json").read_bytes()
    code, r = api.put("/api/canvas", body)
    assert code == 400 and r["error"]
    assert (runs / "canvas.json").read_bytes() == before


def test_hand_edited_canvas_with_bad_items_does_not_block_writes(api, runs):
    write_exp(runs, "B0")
    (runs / "canvas.json").write_text(json.dumps({
        "experiments": {"B0": {"x": 1, "y": 2}},
        "groups": [{"id": "g1", "x": 0, "y": 0, "w": 800, "h": 260, "title": "手写的"},   # 漏了 color
                   {"title": "没有 id"}],
        "stickies": [{"id": "s1", "x": 5, "y": 6, "text": "没写 w、h"}]}), encoding="utf-8")
    _, d = api.get("/api/data")
    assert d["canvas"]["groups"] == [{"id": "g1", "x": 0, "y": 0, "w": 800, "h": 260, "title": "手写的", "color": "c1"}]
    assert d["canvas"]["stickies"][0]["text"] == "没写 w、h"
    assert api.put("/api/canvas", {"experiments": {"B0": {"x": 9, "y": 9}}})[0] == 200
    c = canvas(runs)
    assert c["experiments"]["B0"] == {"x": 9, "y": 9} and c["groups"][0]["color"] == "c1"
    assert c["stickies"][0]["id"] == "s1"


def test_v1_endpoints_gone(api):
    assert api.get("/api/reload")[0] == 404        # /api/pair 已按 v2 重新定义，见 test_workbench_visual.py


def test_delete_clears_only_links_to_deleted(api, runs):
    write_exp(runs, "B0")
    write_exp(runs, "E1")
    write_exp(runs, "E2", 'parent = "B0"\ninit = "E1/main"\n')     # 手写的不一致 init：只清 init，parent 保留
    assert api.delete("/api/exp/E1")[0] == 200
    assert toml(runs, "E2")["parent"] == "B0" and "init" not in toml(runs, "E2")


def test_bad_payloads_rejected_without_side_effects(api, runs):
    assert api.post("/api/exp", {"id": "E1", "x": "a", "y": 0})[0] == 400
    assert not (runs / "E1").exists()
    write_exp(runs, "B0")
    assert api.put("/api/canvas", {"stickies": "abc"})[0] == 400
