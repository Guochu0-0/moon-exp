"""纯标准库的 TB 标量写入器：帧与官方 RecordWriter 逐字节一致，Event 能被官方 proto 解析，workbench.tb 读得回来。"""
import io

import pytest

pytest.importorskip("tensorboard")

from tensorboard.compat.proto import event_pb2  # noqa: E402
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: E402

from workbench import tb, tbwriter  # noqa: E402


def test_crc32c_known_vector():
    assert tbwriter.crc32c(b"123456789") == 0xE3069283


def test_record_framing_matches_official():
    data = tbwriter._event(1.5, 7, tag="loss", value=0.25)
    buf = io.BytesIO()
    RecordWriter(buf).write(data)
    assert tbwriter._record(data) == buf.getvalue()


def test_event_parses_with_official_proto():
    e = event_pb2.Event.FromString(tbwriter._event(2.0, 300, tag="cexp", value=-1.25))
    assert (e.wall_time, e.step) == (2.0, 300)
    assert [(v.tag, v.simple_value) for v in e.summary.value] == [("cexp", -1.25)]
    assert event_pb2.Event.FromString(tbwriter._event(1.0, 0, file_version="brain.Event:2")).file_version == "brain.Event:2"


def test_roundtrip_through_workbench_reader(tmp_path):
    d = tmp_path / "tb" / "main" / "scalars"
    with tbwriter.ScalarWriter(d) as w:
        for step in (1, 2, 3):
            w.add_dict({"step": step, "loss": 1.0 / step, "pairs": ["a"], "ok": True, "lr": 1e-5}, step)
    series = tb.read_file(next(d.iterdir()))
    assert sorted(series) == ["loss", "lr"]
    assert [s for s, _ in series["loss"]] == [1, 2, 3]
    assert series["loss"][1][1] == pytest.approx(0.5)
