"""工作台的记录格式、数据集读取与评价口径。用合成的小数据集，不依赖真实数据。"""
import json
import math
import subprocess
import tomllib

import numpy as np
import pytest

from workbench import __main__ as cli
from workbench import protocol
from workbench.dataset import Dataset, read_tif
from workbench.evaluate import evaluate_identity, evaluate_run
from workbench.records import MATCHES_CAP, PredWriter, check_runs, load_runs

A_TRUE = [[1.0, 0.0, 3.0], [0.0, 1.0, -4.0]]


def write_exp(runs, rid, body=""):
    d = runs / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp.toml").write_text(f'id = "{rid}"\ntitle = "t"\n{body}', encoding="utf-8")
    return d


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
    write_exp(runs, "B0", "baseline = true\n")
    write_exp(runs, "E1", 'parent = "B0"\ninit = "B0/good"\n')
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
    assert check_runs(loaded) == ([], [])
    assert loaded["B0"].methods == ["bad", "good"]
    pr = loaded["B0"].preds("good", "test")["ROI_010/patch_10"]
    assert pr.A == A_TRUE and pr.extra == {"n_inliers": 5}
    assert loaded["B0"].matches("good", "test", "ROI_010/patch_10").shape == (5, 5)
    assert loaded["E1"].preds("main", "val")["ROI_002/patch_0"].fail == "non_finite"

    m = evaluate_run(dataset, loaded["B0"])["methods"]
    good, bad = m["good"]["test"], m["bad"]["test"]
    assert good["summary"]["sr@1"] == 1.0 and good["coverage"]["missing"] == 0
    assert bad["summary"]["fail_rate"] == 1.0 and bad["coverage"]["missing"] == 3
    assert len(good["errors"]) == len(dataset.labelled("test")) == 4
    ident = evaluate_identity(dataset)["test"]["summary"]   # 恒等仿射下误差恒为 5 px
    assert ident["sr@3"] == 0.0 and ident["sr@10"] == 1.0


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """临时 git 仓库，PredWriter 从这里取 commit 和 dirty。"""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "c1")
    return r


def test_pred_writer_matches_roundtrip(dataset, tmp_path, repo):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", "baseline = true\n")
    p0, p1, p2, p3, p4 = dataset.pairs("val")[:5]
    rng = np.random.default_rng(0)
    n = MATCHES_CAP + 500
    big = np.c_[rng.uniform(0, 500, (n, 4)), rng.permutation(n)].astype(np.float32)
    small = np.array([[0, 1, 2, 3], [10, 11, 12, 13]], np.float32)
    with PredWriter(runs / "B0", "m", "val", repo=repo) as w:
        w.write(p0, A_TRUE, matches=big)                                  # N×5，超过上限 → 按 conf 取前 MATCHES_CAP
        w.write(p1, A_TRUE, matches=small)                                # N×4，没有 conf → NaN
        w.write(p2, A_TRUE, matches=small, conf=[0.2, 0.9])               # N×4 + 单独的 conf
        w.write(p3, None, fail="few_matches", matches=np.zeros((0, 5)))   # 0 个点 → 0×5
        w.write(p4, None, fail="error: boom", matches=small)              # 出错 → 不写 key
    r = load_runs(runs)["B0"]

    got = r.matches("m", "val", p0)
    keep = big[np.argsort(-big[:, 4])[:MATCHES_CAP]]
    assert got.shape == (MATCHES_CAP, 5) and got.dtype == np.float32
    np.testing.assert_array_equal(got[:, 4], keep[:, 4])                 # conf 降序
    np.testing.assert_array_equal(got[:, :4], keep[:, :4] + 0.5)         # 坐标 +0.5

    got = r.matches("m", "val", p1)
    np.testing.assert_array_equal(got[:, :4], small + 0.5)
    assert np.isnan(got[:, 4]).all()
    np.testing.assert_allclose(r.matches("m", "val", p2)[:, 4], [0.2, 0.9])
    assert r.matches("m", "val", p3).shape == (0, 5)
    assert r.matches("m", "val", p4) is None
    assert r.preds("m", "val")[p4].fail == "error: boom"

    meta = json.loads((runs / "B0" / "preds" / "m" / "val.meta.json").read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("commit", "dirty")} == {"commit": head(repo), "dirty": False} and meta["host"]
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    with PredWriter(runs / "B0", "m", "test", repo=repo) as w:
        w.write(p0, A_TRUE)
    meta = json.loads((runs / "B0" / "preds" / "m" / "test.meta.json").read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("commit", "dirty")} == {"commit": head(repo), "dirty": True} and meta["host"]
    assert not (runs / "B0" / "preds" / "m" / "test_matches.npz").exists()   # 没给过点对就不写 npz


def test_pred_writer_ignores_untracked_and_runs_for_dirty(tmp_path, repo):
    (repo / "runs" / "B0").mkdir(parents=True)
    (repo / "runs" / "B0" / "old.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "c2")
    (repo / "runs" / "B0" / "old.txt").write_text("b\n", encoding="utf-8")   # 改了已跟踪的实验产物
    (repo / "scratch.py").write_text("", encoding="utf-8")                    # 未跟踪文件
    with PredWriter(repo / "runs" / "B0", "m", "val", repo=repo):
        pass
    meta = json.loads((repo / "runs" / "B0" / "preds" / "m" / "val.meta.json").read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("commit", "dirty")} == {"commit": head(repo), "dirty": False} and meta["host"]


def test_lit_and_field_warnings(tmp_path):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", 'baseline = true\ndate = "2026-09-26"\nstatus = "baseline"\nfoo = 1\n'
                          '[methods.a]\nname = "A"\nbar = 2\n')
    write_exp(runs, "E1", 'parent = "B0"\ninit = "B0/a"\n')
    with PredWriter(runs / "B0", "a", "val", repo=tmp_path):
        pass
    loaded = load_runs(runs)
    b0, e1 = loaded["B0"], loaded["E1"]
    assert b0.lit and not e1.lit
    assert b0.baseline and not e1.baseline
    assert (b0.title, b0.date, e1.parent, e1.init) == ("t", "2026-09-26", "B0", "B0/a")
    assert b0.parent is None and b0.init is None
    w = "\n".join(b0.warnings)
    assert "status" in w and "foo" in w and "bar" in w
    assert e1.warnings == []


def test_method_display_name_falls_back_along_parent_chain(dataset, tmp_path):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", 'baseline = true\n[methods.loftr]\nname = "LoFTR outdoor"\ncaveat = "点数是采样的"\n')
    write_exp(runs, "B0m", 'parent = "B0"\nbaseline = true\n[methods.roma__minmax]\nname = "RoMa 自定"\n')
    write_exp(runs, "E1", 'parent = "B0m"\n')
    for rid, m in (("B0", "loftr"), ("B0m", "loftr__minmax"), ("B0m", "roma__minmax"), ("B0m", "spsg__minmax")):
        with PredWriter(runs / rid, m, "val", repo=tmp_path) as w:
            for p in dataset.pairs("val"):
                w.write(p, A_TRUE)
    loaded = load_runs(runs)
    assert loaded["B0m"].methods == ["loftr__minmax", "roma__minmax", "spsg__minmax"]
    b0m, e1 = loaded["B0m"], loaded["E1"]
    assert loaded["B0"].method_info("loftr") == {"name": "LoFTR outdoor", "caveat": "点数是采样的"}
    assert b0m.method_info("loftr__minmax") == {"name": "LoFTR outdoor · minmax", "caveat": "点数是采样的"}
    assert e1.method_info("loftr__zscore_2p5") == {"name": "LoFTR outdoor · zscore_2p5", "caveat": "点数是采样的"}
    assert b0m.method_info("roma__minmax") == {"name": "RoMa 自定", "caveat": None}     # 自己写了就不回退
    assert b0m.method_info("spsg__minmax") == {"name": "spsg__minmax", "caveat": None}  # 父链上都没有
    assert e1.method_info("main") == {"name": "main", "caveat": None}


def test_commit_summary(dataset, tmp_path, repo):
    runs = tmp_path / "runs"
    for rid in ("S", "M", "U", "N"):
        write_exp(runs, rid)
    pair = dataset.pairs("val")[0]
    c1 = head(repo)
    for rid, m in (("S", "a"), ("S", "b"), ("M", "a")):
        with PredWriter(runs / rid, m, "val", repo=repo) as w:
            w.write(pair, A_TRUE)
    (repo / "code.py").write_text("x = 3\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "c2")
    c2 = head(repo)
    with PredWriter(runs / "M", "b", "val", repo=repo) as w:
        w.write(pair, A_TRUE)
    with PredWriter(runs / "U", "a", "val", repo=repo) as w:
        w.write(pair, A_TRUE)
    (runs / "U" / "preds" / "a" / "val.meta.json").unlink()   # v1 时代写的：没有 meta

    loaded = load_runs(runs)
    s, m, u, n = (loaded[k].commit() for k in ("S", "M", "U", "N"))
    assert s["state"] == "single" and s["commit"] == c1 and s["dirty"] is False
    assert m["state"] == "multiple" and m["commit"] is None
    assert m["methods"]["a"]["val"]["commit"] == c1 and m["methods"]["b"]["val"]["commit"] == c2
    assert u["state"] == "unknown" and u["methods"] == {"a": {"val": None}}
    assert n["state"] == "unknown" and n["methods"] == {}     # 未点亮


def test_notes(tmp_path):
    runs = tmp_path / "runs"
    d = write_exp(runs, "B0")
    write_exp(runs, "E1")
    (d / "notes.md").write_bytes("## 假设\n\n一句话\n".encode("utf-8"))
    loaded = load_runs(runs)
    assert loaded["B0"].notes == "## 假设\n\n一句话\n"
    assert loaded["E1"].notes == ""


def test_check_v2_rules(tmp_path):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", 'baseline = true\nstatus = "baseline"\nhypothesis = "x"\n')   # 旧字段 → 警告
    write_exp(runs, "E1", 'parent = "E2"\n')                                           # E1 ↔ E2 成环
    write_exp(runs, "E2", 'parent = "E1"\n')
    write_exp(runs, "E3", 'parent = "Y"\n')                                            # 父实验缺失
    write_exp(runs, "E4", 'parent = "B0"\ninit = "Z/main"\n')                          # init 缺失
    d = runs / "E5"
    d.mkdir()
    (d / "exp.toml").write_text('id = "E6"\ntitle = "t"\n', encoding="utf-8")         # id 与目录名不一致
    problems, warnings = check_runs(load_runs(runs))
    assert any(p.startswith("E1") and "环" in p for p in problems)
    assert any(p.startswith("E3") and "父实验" in p for p in problems)
    assert any(p.startswith("E4") and "init" in p for p in problems)
    assert any(p.startswith("E5") and "E6" in p for p in problems)
    assert not any(p.startswith("B0") for p in problems)
    assert any(w.startswith("B0") and "status" in w and "hypothesis" in w for w in warnings)


def test_check_accepts_unlit_v2_records(tmp_path):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", "baseline = true\n")
    write_exp(runs, "E1", 'parent = "B0"\n')
    assert check_runs(load_runs(runs)) == ([], [])


def test_cli_new_writes_v2_template(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", "baseline = true\n")
    cli.main(["--runs", str(runs), "new", "E1", "--parent", "B0", "--title", '含"引号"的标题'])
    cli.main(["--runs", str(runs), "new", "E2"])
    e1 = tomllib.loads((runs / "E1" / "exp.toml").read_text(encoding="utf-8"))
    e2 = tomllib.loads((runs / "E2" / "exp.toml").read_text(encoding="utf-8"))
    assert set(e1) == {"id", "title", "parent", "date"} and set(e2) == {"id", "title", "date"}
    assert e1["id"] == "E1" and e1["parent"] == "B0" and e1["title"] == '含"引号"的标题'
    assert len(e1["date"]) == 10
    with pytest.raises(SystemExit) as e:
        cli.main(["--runs", str(runs), "check"])
    assert e.value.code == 0 and "记录无问题" in capsys.readouterr().out


def test_cli_check_exit_codes(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", 'status = "baseline"\n')
    with pytest.raises(SystemExit) as e:
        cli.main(["--runs", str(runs), "check"])
    assert e.value.code == 0 and "status" in capsys.readouterr().out     # 只有警告不算失败
    write_exp(runs, "E1", 'parent = "Y"\n')
    with pytest.raises(SystemExit) as e:
        cli.main(["--runs", str(runs), "check"])
    assert e.value.code == 1


def test_cli_new_with_init_and_rejects_bad_parent(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_exp(runs, "B0", "baseline = true\n")
    cli.main(["--runs", str(runs), "new", "E1", "--parent", "B0", "--init", "B0/roma"])
    text = (runs / "E1" / "exp.toml").read_text(encoding="utf-8")
    assert tomllib.loads(text)["init"] == "B0/roma" and "# init" not in text
    with pytest.raises(SystemExit) as e:
        cli.main(["--runs", str(runs), "new", "E2", "--parent", "Y"])
    assert "Y" in str(e.value.code) and not (runs / "E2").exists()
