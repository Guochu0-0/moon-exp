"""节点页用的 API：实验详情、参考方法、方法与参考方法的比较、Notes 读写。临时 runs/ + 合成数据集，经真实 HTTP。"""
import pytest

from test_workbench import dataset, write_exp  # noqa: F401 — dataset 是 fixture
from test_workbench_api import api, light, runs  # noqa: F401 — api、runs 是 fixture

IDENTITY = [[1, 0, 0], [0, 1, 0]]   # 合成数据集的真值是平移 (3, -4)：恒等仿射误差恒 5 px


def detail(api, rid):
    code, d = api.get(f"/api/exp/{rid}")
    assert code == 200, d
    return d


def defaults(d):
    return {m["id"]: m["ref"] for m in d["methods"]}


def option_keys(d):
    return [[o["key"] for o in g["options"]] for g in d["reference"]["groups"]]


@pytest.fixture
def tree(runs, dataset):
    """B0（基线）：roma 误差 0、loftr 误差 5；E1 从 B0/loftr 起步；E2 是 B0m 式变体；R 是无父的非基线。"""
    write_exp(runs, "B0", 'baseline = true\n[methods.loftr]\nname = "LoFTR outdoor"\n')
    light(runs, dataset, "B0", "roma")
    light(runs, dataset, "B0", "loftr", A=IDENTITY)
    write_exp(runs, "E1", 'parent = "B0"\ninit = "B0/loftr"\n')
    light(runs, dataset, "E1", "main")
    write_exp(runs, "E2", 'parent = "B0"\n')
    light(runs, dataset, "E2", "loftr__v")
    light(runs, dataset, "E2", "spsg__v", A=IDENTITY)
    write_exp(runs, "R")
    light(runs, dataset, "R", "main", A=IDENTITY)
    return runs


def test_reference_defaults(api, tree):
    assert defaults(detail(api, "E1")) == {"main": "B0/loftr"}                 # 有 init 用 init
    assert defaults(detail(api, "E2")) == {"loftr__v": "B0/loftr",             # 父实验有同名基础方法
                                           "spsg__v": "B0/roma"}               # 否则父实验 Val 主指标最好的
    assert defaults(detail(api, "B0")) == {"roma": None, "loftr": None}        # 基线默认无参考
    assert defaults(detail(api, "R")) == {"main": "identity"}                  # 无父的非基线：未配准


def test_reference_candidates(api, tree):
    d = detail(api, "E2")
    labels = [g["label"] for g in d["reference"]["groups"]]
    assert labels == ["父实验 B0", "本实验 E2", "实验 E1", "实验 R", "其他"]
    assert option_keys(d) == [["B0/roma", "B0/loftr"],                          # 按 Val 主指标降序
                              ["E2/loftr__v", "E2/spsg__v"],
                              ["E1/main"], ["R/main"],
                              ["identity", None]]
    names = {o["key"]: o["label"] for g in d["reference"]["groups"] for o in g["options"]}
    assert names["B0/loftr"] == "B0 / LoFTR outdoor" and names["E1/main"] == "E1"
    assert names["identity"] == "未配准" and names[None] == "无"
    splits = {o["key"]: o["splits"] for g in d["reference"]["groups"] for o in g["options"]}
    assert splits["B0/roma"] == ["val"] and splits["identity"] == ["val", "test"]   # 供页面置灰 Test
    assert option_keys(detail(api, "B0"))[0] == ["B0/roma", "B0/loftr"]       # 无父：第一组是本实验


def test_detail_info_names_and_delta_vs_parent_base(api, tree, runs):
    write_exp(runs, "E3", 'parent = "E1"\n')
    d = detail(api, "B0")
    assert (d["lit"], d["baseline"], d["parent"], d["children"]) == (True, True, None, ["E1", "E2"])
    for f in (runs / "B0" / "preds").glob("*/val.meta.json"):
        f.write_text('{"commit": "abc", "dirty": false}')
    assert d["commit"]["state"] == "unknown"                                  # 合成 runs/ 不是 git 仓库
    c = detail(api, "B0")["commit"]
    assert (c["state"], c["commit"]) == ("single", "abc")
    e2 = detail(api, "E2")
    ms = {m["id"]: m for m in e2["methods"]}
    assert [m["id"] for m in e2["methods"]] == ["loftr__v", "spsg__v"]        # 按 Val 主指标降序
    assert ms["loftr__v"]["name"] == "LoFTR outdoor · v"                      # 沿父链回退，加「· 变体」
    assert ms["loftr__v"]["base"] == "B0/loftr" and ms["spsg__v"]["base"] is None
    assert ms["loftr__v"]["delta"] == {"val": pytest.approx(0.5)}             # AUC@10：1.0 − 0.5
    assert ms["spsg__v"]["delta"] == {}                                       # 父实验没有同名基础方法
    assert ms["loftr__v"]["results"]["val"]["summary"]["auc@10"] == pytest.approx(1.0)
    assert len(ms["loftr__v"]["results"]["val"]["errors"]) == 4               # 每个有标注 pair 一个
    assert "test" not in ms["loftr__v"]["results"]

    e3 = detail(api, "E3")
    assert (e3["lit"], e3["methods"], e3["children"], e3["init"]) == (False, [], [], None)
    assert e3["commit"]["state"] == "unknown"
    assert api.get("/api/exp/Z9")[0] == 404


def compare(api, method, ref, split="val"):
    q = f"/api/compare?method={method}&split={split}" + (f"&ref={ref}" if ref is not None else "")
    return api.get(q)


def test_compare_method_with_reference_and_paired_difference(api, tree):
    code, c = compare(api, "E1/main", "B0/loftr")
    assert code == 200
    assert c["method"]["label"] == "E1" and c["ref"]["label"] == "B0 / LoFTR outdoor"
    assert c["method"]["summary"]["auc@10"] == pytest.approx(1.0) and c["ref"]["summary"]["auc@10"] == pytest.approx(0.5)
    assert len(c["ref"]["errors"]) == 4
    d = c["diff"]
    assert d["auc@10"] == pytest.approx(0.5) and d["auc@10_ci"] == pytest.approx([0.5, 0.5])   # 逐对差恒定
    assert d["sr@5"] == pytest.approx(0.0) and d["median"] == pytest.approx(-5.0) and d["fail_rate"] == 0
    assert "median_ci" not in d

    code, c = compare(api, "E2/spsg__v", "identity")                          # 未配准：同样是误差 5 px
    assert code == 200 and c["ref"]["label"] == "未配准" and c["diff"]["auc@10"] == pytest.approx(0.0)
    code, c = compare(api, "B0/roma", None)
    assert code == 200 and c["ref"] is None and c["diff"] is None
    assert compare(api, "B0/roma", "B0/nope")[0] == 400
    assert compare(api, "B0/roma", "B0/loftr", "test")[0] == 400               # 没有 Test 结果


def test_notes_empty_then_created_on_first_save_with_lf(api, runs):
    write_exp(runs, "E1")
    assert detail(api, "E1")["notes"] == ""
    assert not (runs / "E1" / "notes.md").exists()                            # 读不创建
    code, _ = api.put("/api/exp/E1/notes", {"text": "## 假设\r\n\r\n中文 ![](extra/a.png)\r\n"})
    assert code == 200
    assert (runs / "E1" / "notes.md").read_bytes() == "## 假设\n\n中文 ![](extra/a.png)\n".encode("utf-8")
    assert detail(api, "E1")["notes"] == "## 假设\n\n中文 ![](extra/a.png)\n"
    assert api.put("/api/exp/E1/notes", {"text": 3})[0] == 400
    assert api.put("/api/exp/Z9/notes", {"text": ""})[0] == 404
    assert not (runs / "Z9").exists()


def test_detail_launch(api, tree):
    """启动记录：只列有 launch/<方法>.json 的方法，原样返回（含事后补记的字段）；没有时为空。"""
    import json
    rec = [{"backfilled": "2026-10-03", "reliability": "推断", "commit": None, "entry": "python -m finetune.train",
            "args": {"lr": 1e-5}, "args_source": "args.json"}]
    (tree / "B0" / "launch").mkdir()
    (tree / "B0" / "launch" / "loftr.json").write_text(json.dumps(rec), encoding="utf-8")
    (tree / "B0" / "launch" / "gone.json").write_text(json.dumps(rec), encoding="utf-8")   # 没有这个方法：不列
    assert detail(api, "B0")["launch"] == {"loftr": rec}
    assert detail(api, "E1")["launch"] == {}
