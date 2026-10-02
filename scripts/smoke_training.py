"""训练产物冒烟：合成一个实验（scalar 中间结果 + TB scalars），用无头 Chrome（CDP）走一遍真实页面，不进 pytest 默认集合。

    MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset python scripts/smoke_training.py [--python <带 tensorboard 的 python>] [--shots DIR]

在临时目录里建实验 E1：Val 上全部 pair 的仿射（恒等）、中间结果 certainty（scalar，光学坐标系，32×32，
留一个 pair 不写），以及 tb/main/scalars/ 下两个 events 文件（续训：第二个从 step 100 接着写）。
工作台用 --python 指定的解释器作子进程起（要装 tensorboard），然后覆盖：
- 可视化结果的详情里有「中间结果 certainty」视图，切过去是热力图叠加（图层 PNG 可取），切回卷帘后热力图消失；
  翻到没写的 pair 时显示「此 pair 没有该中间结果」；
- 训练图表一节按 tag 出两张图，续训重叠已截断；
- 「在 TensorBoard 中打开」返回的地址能打开，TensorBoard 列出 run；
- 强行结束工作台进程后，TensorBoard 随之结束（端口不再响应）。
依赖：Chrome 或 Edge、websocket-client（跑本脚本的解释器）；tensorboard（--python 指定的解释器）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

LOSS_FILES = [(1000, range(0, 130, 10), 1.0), (2000, range(100, 210, 10), 0.6)]   # (时间戳, steps, 起始 loss)


def write_tb(logdir: Path):
    """由 --python 的解释器执行（它装了 tensorboard）：两个 events 文件，模拟崩溃后从 step 100 续训。"""
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    logdir.mkdir(parents=True, exist_ok=True)
    for ts, steps, base in LOSS_FILES:
        with open(logdir / f"events.out.tfevents.{ts}.smoke.1.0", "wb") as f:
            w = RecordWriter(f)
            w.write(event_pb2.Event(wall_time=ts, file_version="brain.Event:2").SerializeToString())
            for s in steps:
                vals = {"train/loss": base * 0.98 ** (s / 10)}
                if s % 50 == 0:
                    vals["val/auc@5"] = 0.2 + s / 1000
                summ = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=k, simple_value=v) for k, v in vals.items()])
                w.write(event_pb2.Event(wall_time=ts + s, step=s, summary=summ).SerializeToString())


def build(runs: Path, data: str, python: str) -> tuple[str, str]:
    """建合成实验 E1，返回（写了中间结果的一个有标注 pair，没有写的那个 pair）。"""
    import numpy as np

    from workbench.dataset import Dataset
    from workbench.records import InterWriter, PredWriter

    ds = Dataset(data)
    d = runs / "E1"
    d.mkdir(parents=True)
    (d / "exp.toml").write_text('id = "E1"\ntitle = "合成：中间结果与训练图表"\n', encoding="utf-8")
    pairs = ds.pairs("val")
    with PredWriter(d, "main", "val", repo=REPO) as w:
        for p in pairs:
            w.write(p, [[1, 0, 0], [0, 1, 0]])
    skip = ds.labelled("val")[0]
    yy, xx = np.mgrid[0:32, 0:32] / 31
    with InterWriter(d, "main", "certainty", "val", kind="scalar", frame="opt", desc="合成置信图") as w:
        for i, p in enumerate(pairs):
            if p != skip:
                w.write(p, np.hypot(xx - 0.5, yy - (i % 7) / 6))
    subprocess.run([python, __file__, "--write-tb", str(d / "tb" / "main" / "scalars")], check=True)
    return ds.labelled("val")[1], skip


def alive(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2).close()
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--python", default=sys.executable, help="起工作台用的解释器，要装 tensorboard")
    ap.add_argument("--shots", help="把各步截图存到这个目录")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--write-tb", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.write_tb:
        return write_tb(Path(args.write_tb))
    from smoke_canvas import BROWSERS, Page, free_port   # 要 websocket-client；--write-tb 的子进程不需要

    if not args.data:
        sys.exit("需要数据集根目录：--data 或环境变量 MOON_DATA")
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        sys.exit("找不到 Chrome / Edge")

    tmp = Path(tempfile.mkdtemp(prefix="wb-smoke-train-"))
    runs = tmp / "runs"
    have, skip = build(runs, args.data, args.python)
    wport = free_port()
    wb = subprocess.Popen([args.python, "-m", "workbench", "--runs", str(runs), "--data", args.data, "serve", "--port", str(wport)],
                          cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{wport}/"
    bport = free_port()
    proc = subprocess.Popen([browser, f"--remote-debugging-port={bport}", f"--user-data-dir={tmp / 'profile'}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,950",
                             *([] if args.headed else ["--headless=new"]), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    ok, p = False, None
    try:
        t0 = time.time()
        while not alive(url + "api/data"):
            assert wb.poll() is None, f"工作台没起来：{wb.stderr.read().decode('utf-8', 'replace')}"
            assert time.time() - t0 < 30, "工作台 30 秒内没有就绪"
            time.sleep(.2)
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{bport}/json").read())
                break
            except OSError:
                time.sleep(.1)
        p = Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
        p.cmd("Page.enable")
        p.cmd("Page.addScriptToEvaluateOnNewDocument", source="window.__errors = [];"
              "addEventListener('error', e => __errors.push(String(e.message)));"
              "addEventListener('unhandledrejection', e => __errors.push(String(e.reason)));")
        p.cmd("Emulation.setDeviceMetricsOverride", width=1500, height=950, deviceScaleFactor=1, mobile=False)
        p.cmd("Page.navigate", url=url + "#/exp/E1")
        step = lambda name: (print(f"通过  {name}"), shots and p.shot(shots / f"{name}.png"))
        scroll = lambda sel: p.js(f"document.querySelector('{sel}').scrollIntoView({{block: 'start', behavior: 'instant'}}), true")

        # 章节：可视化结果与训练图表都在
        p.wait("document.querySelector('#sec-visual .detail .views')", 30, "可视化结果的详情出现")
        secs = p.js("[...document.querySelectorAll('.nav a[data-sec]')].map(a => a.textContent)")
        assert secs == ["实验信息", "结果", "可视化结果", "训练图表", "Notes"], secs

        # 中间结果视图：热力图叠加，可切换。先翻到一个写了中间结果的 pair
        find = lambda pair: p.js(f"document.querySelector('#vfind').value = {json.dumps(pair)}; document.querySelector('#vgo').click(); true")
        find(have)
        p.wait(f"document.querySelector('#sec-visual .detail .dhead').textContent.includes({json.dumps(have)})", what="翻到 " + have)
        btn = "#sec-visual .views button[data-v=\"inter:certainty\"]"
        p.wait(f"document.querySelector('{btn}')", what="有 certainty 视图")
        assert p.js(f"document.querySelector('{btn}').title") == "合成置信图"
        p.js(f"document.querySelector('{btn}').click()")
        p.wait("document.querySelector('#sec-visual image.heat')", 10, "热力图图层出现")
        href = p.js("document.querySelector('#sec-visual image.heat').getAttribute('href')")
        assert p.js(f"fetch({json.dumps(href)}).then(r => r.status + ' ' + r.headers.get('content-type'))") == "200 image/png"
        assert p.js("document.querySelector('#valpha') !== null")
        assert "32×32" in p.js("document.querySelector('#sec-visual .detail .figcap').textContent")
        scroll("#sec-visual .detail")
        step("01-热力图图层")
        p.js("document.querySelector('#sec-visual .views button[data-v=\"swipe\"]').click()")
        p.wait("!document.querySelector('#sec-visual image.heat')", what="切回卷帘后热力图消失")
        p.js(f"document.querySelector('{btn}').click()")
        p.wait("document.querySelector('#sec-visual image.heat')", 10, "再切回热力图")

        # 没有写中间结果的 pair
        find(skip)
        p.wait("[...document.querySelectorAll('#sec-visual .pane .msg')].some(m => m.textContent === '此 pair 没有该中间结果')",
               10, "此 pair 没有该中间结果")
        assert p.js(f"document.querySelector('#sec-visual .detail .dhead').textContent.includes({json.dumps(skip)})")
        step("02-此pair没有")

        # 训练图表：按 tag 出图，续训重叠已截断
        p.wait("document.querySelectorAll('#sec-train .tbchart').length === 2", 15, "训练图表两张")
        tags = p.js("[...document.querySelectorAll('#sec-train .tbchart figcaption')].map(x => x.textContent)")
        assert tags == ["train/loss", "val/auc@5"], tags
        ticks = p.js("[...document.querySelectorAll('#sec-train .tbchart')][0].querySelectorAll('.ax text').length")
        assert ticks >= 6, ticks
        scal = json.loads(urllib.request.urlopen(url + "api/exp/E1/scalars").read())
        steps = scal["methods"][0]["runs"][0]["tags"]["train/loss"]["step"]
        assert steps == list(range(0, 100, 10)) + list(range(100, 210, 10)), steps
        scroll("#sec-train")
        step("03-训练图表")

        # 在 TensorBoard 中打开
        p.js("document.querySelector('#sec-train .tbopen').click()")
        tb_url = p.wait("document.querySelector('#sec-train .tbstatus a')?.href", 120, "TensorBoard 地址")
        runs_list = json.loads(urllib.request.urlopen(tb_url + "data/runs", timeout=10).read())
        assert [r.replace("\\", "/") for r in runs_list] == ["main/scalars"], runs_list
        assert alive(tb_url)
        step("04-TensorBoard已打开")
        p.js("document.querySelector('#sec-train .tbopen').click()")
        assert p.wait("document.querySelector('#sec-train .tbstatus a')?.href", 30, "再次打开") == tb_url   # 复用

        errors = p.js("window.__errors")
        assert not errors, errors

        # 工作台退出（这里是强行结束）后 TensorBoard 随之结束
        wb.terminate()
        wb.wait(timeout=10)
        t0 = time.time()
        while alive(tb_url):
            assert time.time() - t0 < 15, "工作台结束 15 秒后 TensorBoard 仍在响应"
            time.sleep(.3)
        print("通过  05-工作台退出后 TensorBoard 结束")
        ok = True
    finally:
        if p:
            p.ws.close()
        if wb.poll() is None:
            wb.terminate()
        proc.terminate()
        proc.wait(timeout=10)
        time.sleep(.5)
        shutil.rmtree(tmp, ignore_errors=True)
    print("全部通过" if ok else "失败")


if __name__ == "__main__":
    main()
