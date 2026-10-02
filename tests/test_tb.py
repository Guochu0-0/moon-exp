"""训练图表：tfevents 解析（续训重叠按 tag 截断、version_N 分开、add_scalars 子目录）、scalars API、TensorBoard 子进程。
日志在本地按 TFRecord 帧格式现写，文件名带指定的时间戳，和 SummaryWriter / Lightning 写出的一样。"""
import json
import urllib.request

import pytest

pytest.importorskip("tensorboard")

from tensorboard.compat.proto import event_pb2, summary_pb2  # noqa: E402
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: E402

from test_workbench import dataset, write_exp  # noqa: E402,F401 — dataset 是 fixture
from test_workbench_api import api, runs  # noqa: E402,F401 — api、runs 是 fixture
from workbench import tb  # noqa: E402


def events(d, ts, points, tail=b""):
    """在目录 d 写一个 events 文件：points 是 [(step, {tag: value})]；另写一条图片，解析时应忽略。"""
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"events.out.tfevents.{ts}.host.123.0"
    with open(path, "wb") as f:
        w = RecordWriter(f)
        w.write(event_pb2.Event(wall_time=ts, file_version="brain.Event:2").SerializeToString())
        for step, vals in points:
            s = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=k, simple_value=v) for k, v in vals.items()])
            w.write(event_pb2.Event(wall_time=ts, step=step, summary=s).SerializeToString())
        img = summary_pb2.Summary.Image(height=1, width=1, colorspace=1, encoded_image_string=b"x")
        w.write(event_pb2.Event(step=0, summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag="match", image=img)])).SerializeToString())
        f.write(tail)
    return path


def steps(series):
    return series["step"]


def test_resume_overlap_truncated_per_tag(tmp_path):
    d = tmp_path / "version_1"
    # 第一次：train/loss 的 step 是 global_step 9…129，val/auc 的 step 是 epoch 0…2；在 step 130 崩溃
    events(d, 1000, [(s, {"train/loss": s / 100}) for s in range(9, 130, 10)]
           + [(e, {"val/auc": e / 10}) for e in range(3)] + [(0, {"only_old": 1.0})])
    # 续训从 step 100 的 ckpt 起：train/loss 109…239，val/auc 从 epoch 2 起
    events(d, 2000, [(s, {"train/loss": -s / 100}) for s in range(109, 240, 10)]
           + [(e, {"val/auc": -e / 10}) for e in (2, 3)])
    tags = tb.read_dir(d)
    loss = tags["train/loss"]
    assert steps(loss) == list(range(9, 100, 10)) + list(range(109, 240, 10))   # 旧文件里 ≥109 的点丢掉
    assert loss["value"][9] == pytest.approx(0.99) and loss["value"][10] == pytest.approx(-1.09)
    # 按 tag 截断：val/auc 以它自己在新文件里的起点 2 为界；若按整个文件的最小 step（2）截，train/loss 会被清空
    assert steps(tags["val/auc"]) == [0, 1, 2, 3] and tags["val/auc"]["value"][2] == pytest.approx(-0.2)
    assert steps(tags["only_old"]) == [0]                                      # 新文件里没有的 tag 原样保留
    assert "match" not in tags                                                 # 只取 simple_value


def test_files_sorted_by_timestamp_not_name(tmp_path):
    d = tmp_path / "scalars"
    events(d, 1000, [(5, {"loss": 2.0})])        # 名字按字符串排在前面，时间戳却更晚
    events(d, 999, [(0, {"loss": 1.0}), (5, {"loss": 1.5}), (9, {"loss": 1.9})])
    s = tb.read_dir(d)["loss"]
    assert steps(s) == [0, 5] and s["value"] == [1.0, 2.0]


def test_truncated_tail_and_epoch_repeats_kept(tmp_path):
    d = tmp_path / "run"
    events(d, 1000, [(9, {"epoch": 0.0}), (9, {"epoch": 0.0}), (19, {"epoch": 1.0})], tail=b"\x10\x00\x00")   # 写到一半
    s = tb.read_dir(d)["epoch"]
    assert steps(s) == [9, 9, 19]               # 同一文件里重复的 step 不去重（Lightning 的 epoch tag 就是这样）


def test_logdir_versions_apart_and_add_scalars_subdirs(tmp_path):
    root = tmp_path / "tb" / "main"
    events(root / "lightning_logs" / "version_0", 1000, [(0, {"train/loss": 1.0}), (1, {"train/loss": 0.5})])
    events(root / "lightning_logs" / "version_1", 2000, [(0, {"train/loss": 3.0})])
    (root / "lightning_logs" / "version_1" / "checkpoints").mkdir()
    events(root / "scalars", 1000, [])                                         # add_scalars 的父目录：只有空文件
    events(root / "scalars" / "loss_train", 1000, [(0, {"loss": 1.0})])
    events(root / "scalars" / "loss_val", 1000, [(0, {"loss": 2.0})])
    events(root / "media", 1000, [(0, {"should_not_read": 1.0})])               # 约定：media/ 只放图片
    runs = tb.read_logdir(root)
    assert list(runs) == ["lightning_logs/version_0", "lightning_logs/version_1", "scalars/loss_train", "scalars/loss_val"]
    assert steps(runs["lightning_logs/version_0"]["train/loss"]) == [0, 1]
    assert runs["lightning_logs/version_1"]["train/loss"]["value"] == [3.0]    # 不同 version 不拼接
    assert runs["scalars/loss_val"]["loss"]["value"] == [2.0]


def test_long_series_thinned_keeps_extremes():
    s = {"step": list(range(10000)), "value": [float(i % 97) for i in range(10000)]}
    s["value"][4321] = -50.0
    t = tb.thin(s, 500)
    assert len(t["step"]) <= 500 and t["n"] == 10000
    assert t["step"][0] == 0 and t["step"][-1] == 9999
    assert min(t["value"]) == -50.0 and max(t["value"]) == 96.0


def test_scalars_api_and_section_only_when_logs(api, runs, dataset):
    write_exp(runs, "E1")
    write_exp(runs, "E2")
    events(runs / "E1" / "tb" / "main" / "scalars", 1000, [(0, {"loss": 1.0}), (1, {"loss": float("nan")})])
    code, d = api.get("/api/exp/E1")
    assert code == 200 and d["tb"] == ["main"]
    assert api.get("/api/exp/E2")[1]["tb"] == []                               # 没有日志：页面不显示这一节
    write_exp(runs, "E3")
    events(runs / "E3" / "tb" / "main" / "media", 1000, [(0, {"x": 1.0})])     # 只有 media/
    events(runs / "E3" / "tb" / "main" / "scalars", 1000, [])                  # 只有空文件
    assert api.get("/api/exp/E3")[1]["tb"] == []
    code, d = api.get("/api/exp/E1/scalars")
    assert code == 200, d
    assert d["logdir"] == "runs/E1/tb" and d["max_points"] == tb.MAX_POINTS
    assert d["methods"] == [{"method": "main", "runs": [
        {"run": "scalars", "tags": {"loss": {"step": [0, 1], "value": [1.0, None], "n": 2}}}]}]
    code, d = api.post("/api/exp/E2/tensorboard")
    assert code == 400 and "没有 TensorBoard 日志" in d["error"]


def test_tensorboard_subprocess_reused_and_closed(tmp_path):
    logdir = tmp_path / "tb"
    events(logdir / "main" / "scalars", 1000, [(0, {"loss": 1.0})])
    boards = tb.TensorBoards()
    try:
        url = boards.open(logdir)
        assert url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(url + "data/runs", timeout=10) as r:
            assert [x.replace("\\", "/") for x in json.loads(r.read())] == ["main/scalars"]
        assert boards.open(logdir) == url                                       # 同一 logdir 复用一个实例
        proc = boards.procs[str(logdir.resolve())][0]
    finally:
        boards.close()
    assert proc.poll() is not None                                              # 统一 terminate
    assert boards.procs == {}
