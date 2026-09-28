"""画布冒烟：用无头 Chrome（CDP）对真实页面做一遍交互，不进 pytest 默认集合。

    MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset python scripts/smoke_canvas.py [--shots DIR]

把仓库 runs/ 里的 B0、B0m（exp.toml 与 preds/*.jsonl）复制到临时目录，起一个工作台服务，然后覆盖：
按 E 新建、菜单派生、拖到空白派生、拖到实验改父、成环被拒、已点亮实验的删除项置灰、
拖动后刷新位置保持、终端写 exp.toml 后几秒内出现、双击进入节点页并返回。
依赖：Chrome 或 Edge、websocket-client。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import websocket

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from workbench import server  # noqa: E402
from workbench.dataset import Dataset  # noqa: E402

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Page:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30, suppress_origin=True)
        self.n = 0

    def cmd(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr):
        r = self.cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS 出错：{expr}\n{r['exceptionDetails']}")
        return r["result"].get("value")

    def wait(self, expr, timeout=8.0, what=None):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = self.js(f"(() => {{ const v = ({expr}); return v instanceof Node ? true : v; }})()")
            if v:
                return v
            time.sleep(.1)
        raise AssertionError(f"等待超时：{what or expr}")

    def center(self, selector):
        r = self.js(f"(() => {{ const el = document.querySelector({json.dumps(selector)}); if (!el) return null;"
                    f" const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
        assert r, f"找不到元素 {selector}"
        return r

    def mouse(self, kind, x, y, button="left", clicks=1):
        buttons = {"left": 1, "right": 2}[button] if kind != "mouseReleased" else 0
        self.cmd("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, buttons=buttons, clickCount=clicks)

    def move(self, x, y):
        self.cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="none", buttons=0)

    def click(self, x, y, button="left", clicks=1):
        self.move(x, y)
        for c in range(1, clicks + 1):
            self.mouse("mousePressed", x, y, button, c)
            self.mouse("mouseReleased", x, y, button, c)

    def drag(self, a, b, steps=12):
        self.move(*a)
        self.mouse("mousePressed", *a)
        for i in range(1, steps + 1):
            x, y = a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps
            self.cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="left", buttons=1)
        self.mouse("mouseReleased", *b)

    def key(self, key, code=None, text=None):
        vk = {"Enter": 13, "Escape": 27, "Delete": 46}.get(key, ord(key.upper()) if len(key) == 1 else 0)
        base = dict(key=key, code=code or key, windowsVirtualKeyCode=vk)
        self.cmd("Input.dispatchKeyEvent", type="keyDown", **base, **({"text": text} if text else {}))
        self.cmd("Input.dispatchKeyEvent", type="keyUp", **base)

    def type(self, text):
        self.cmd("Input.insertText", text=text)

    def shot(self, path):
        Path(path).write_bytes(base64.b64decode(self.cmd("Page.captureScreenshot", format="png")["data"]))


def until(cond, what, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if cond():
                return
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(.1)
    raise AssertionError(f"等待超时：{what}")


def menu_click(p, label):
    btn = p.wait(f"""(() => {{ const b = [...document.querySelectorAll('.menu:not([hidden]) button')]
        .find(b => b.textContent.includes({json.dumps(label)})); if (!b) return null;
        const r = b.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2, b.disabled]; }})()""",
                 what=f"菜单项「{label}」")
    assert not btn[2], f"菜单项「{label}」不可用"
    p.click(btn[0], btn[1])


def node(rid):
    return f'.node[data-id="{rid}"]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--shots", help="把各步截图存到这个目录")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    if not args.data:
        sys.exit("需要数据集根目录：--data 或环境变量 MOON_DATA")
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        sys.exit("找不到 Chrome / Edge")

    tmp = Path(tempfile.mkdtemp(prefix="wb-smoke-"))
    runs = tmp / "runs"
    for rid in ("B0", "B0m"):
        src = REPO / "runs" / rid
        (runs / rid).mkdir(parents=True)
        shutil.copy(src / "exp.toml", runs / rid / "exp.toml")
        for f in src.glob("preds/*/*.jsonl"):
            (runs / rid / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, runs / rid / f.relative_to(src))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(server.Workbench(runs, Dataset(args.data), REPO)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"

    port = free_port()
    proc = subprocess.Popen([browser, f"--remote-debugging-port={port}", f"--user-data-dir={tmp / 'profile'}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,950",
                             *([] if args.headed else ["--headless=new"]), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    ok, p = False, None
    try:
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                break
            except OSError:
                time.sleep(.1)
        p = Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
        p.cmd("Page.enable")
        p.cmd("Page.addScriptToEvaluateOnNewDocument", source="window.__errors = [];"
              "addEventListener('error', e => __errors.push(String(e.message)));"
              "addEventListener('unhandledrejection', e => __errors.push(String(e.reason)));")
        p.cmd("Emulation.setDeviceMetricsOverride", width=1500, height=950, deviceScaleFactor=1, mobile=False)
        p.cmd("Page.navigate", url=url)
        step = lambda name: (print(f"通过  {name}"), shots and p.shot(shots / f"{name}.png"))
        fit = lambda: (p.key(".", "Period", "."), time.sleep(.2))
        toml = lambda rid: tomllib.loads((runs / rid / "exp.toml").read_text(encoding="utf-8"))

        # 浏览：B0 / B0m、方法行、init 连线
        p.wait(f"document.querySelector('{node('B0m')}')", 20, "画布出现 B0m")
        p.js("document.fonts.ready.then(() => true)")
        assert p.js(f"document.querySelectorAll('{node('B0')} .row').length") == 4
        assert p.js(f"document.querySelector('{node('B0')} .more').textContent").startswith("▸ 另外")
        assert p.js(f"document.querySelector('{node('B0')} .chip').textContent") == "基线"
        assert p.js("document.querySelectorAll('.edge').length") == 1
        assert "实验画布" in p.js("document.querySelector('.title').textContent")
        step("01-浏览")

        # 按 E 新建：建议 E1，立即进入标题编辑
        p.click(700, 800)                 # 先点空白，让画布拿到焦点
        p.move(700, 800)
        p.key("e", "KeyE", "e")
        p.wait(f"document.querySelector('{node('E1')} .ttl[contenteditable=true]')", what="E1 进入标题编辑")
        p.type("按 E 新建")
        p.key("Enter")
        until(lambda: toml("E1")["title"] == "按 E 新建", "E1 标题写回 exp.toml")
        assert "parent" not in toml("E1")
        assert p.js(f"document.querySelector('{node('E1')}').classList.contains('off')")
        step("02-按E新建")

        # 菜单派生：B0 的方法行右键 →「从 <方法> 派生新实验」，自动带 init
        m = p.js(f"document.querySelector('{node('B0')} .row').dataset.m")
        p.click(*p.center(f"{node('B0')} .row .mn"), button="right")
        menu_click(p, f"从 {m} 派生新实验")
        p.wait(f"document.querySelector('{node('E2')} .ttl[contenteditable=true]')", what="E2 进入标题编辑")
        p.key("Escape")
        assert toml("E2")["parent"] == "B0" and toml("E2")["init"] == f"B0/{m}"
        e2 = p.js(f"(() => {{ const a = document.querySelector('{node('B0')}').getBoundingClientRect(),"
                  f" b = document.querySelector('{node('E2')}').getBoundingClientRect(); return b.left > a.right; }})()")
        assert e2, "派生的实验应在父实验右侧"
        step("03-菜单派生")

        # 拖到空白派生：从 B0m 的通用端口拖到空白处
        a = p.center(f'{node("B0m")} .io .port.out')
        p.drag(a, (a[0] + 60, 880))
        p.wait(f"document.querySelector('{node('E3')} .ttl[contenteditable=true]')", what="E3 进入标题编辑")
        p.key("Enter")
        assert toml("E3")["parent"] == "B0m" and "init" not in toml("E3")
        step("04-拖到空白派生")

        # 拖到实验改父：E1 的通用端口拖到 E3 上
        fit()
        p.drag(p.center(f'{node("E1")} .io .port.out'), p.center(f"{node('E3')} .hdr"))
        until(lambda: toml("E3")["parent"] == "E1", "E3 的父实验改为 E1")
        step("05-拖到实验改父")

        # 成环被拒：E3 的端口拖到 E1 上（E1 → E3 已存在）
        before = (runs / "E1" / "exp.toml").read_bytes()
        fit()
        p.drag(p.center(f'{node("E3")} .io .port.out'), p.center(f"{node('E1')} .hdr"))
        p.wait("document.querySelector('.toast.on') && document.querySelector('.toast').textContent.includes('环')",
               what="成环提示")
        assert (runs / "E1" / "exp.toml").read_bytes() == before
        step("06-成环被拒")

        # 已点亮实验的删除项置灰；Del 跳过并提示
        p.click(*p.center(f"{node('B0')} .hdr .nid"), button="right")
        dis = p.wait("""[...document.querySelectorAll('.menu:not([hidden]) button')].find(b => b.textContent.includes('删除实验'))
                        ?.disabled === true""", what="删除项置灰")
        assert dis
        step("07-删除项置灰")
        p.key("Escape")
        p.click(*p.center(f"{node('B0')} .hdr .nid"))
        p.key("Delete")
        p.wait("document.querySelector('.toast').textContent.includes('不能删除')", what="Del 跳过提示")
        assert (runs / "B0").exists()

        # 删除未点亮实验（E2 没有子实验）
        p.click(*p.center(f"{node('E2')} .hdr .nid"))
        p.key("Delete")
        p.wait(f"!document.querySelector('{node('E2')}')", what="E2 消失")
        assert not (runs / "E2").exists()

        # 拖动后刷新，位置保持
        before_b0 = json.loads((runs / "canvas.json").read_text(encoding="utf-8"))["experiments"].get("B0")
        b = p.center(f"{node('B0')} .hdr .nid")
        p.drag(b, (b[0] + 40, b[1] + 30))
        canvas = lambda: json.loads((runs / "canvas.json").read_text(encoding="utf-8"))["experiments"]
        until(lambda: canvas().get("B0") != before_b0, "B0 的坐标写回 canvas.json")
        pos = canvas()["B0"]
        p.cmd("Page.reload")
        p.wait(f"document.querySelector('{node('B0')}')", 10)
        assert p.js(f"[parseFloat(document.querySelector('{node('B0')}').style.left), "
                    f"parseFloat(document.querySelector('{node('B0')}').style.top)]") == [pos["x"], pos["y"]]
        step("08-拖动后刷新")

        # 终端直接写 exp.toml，几秒内出现在画布上
        (runs / "E9").mkdir()
        (runs / "E9" / "exp.toml").write_text('id = "E9"\ntitle = "终端写的"\nparent = "B0"\n', encoding="utf-8")
        p.wait(f"document.querySelector('{node('E9')}')", 8, "E9 出现在画布上")
        step("09-终端新建")

        # 双击进入节点页，「← 画布」返回，实验居中且选中
        p.click(*p.center(f"{node('E1')} .hdr .nid"), clicks=2)
        p.wait("location.hash === '#/exp/E1' && document.querySelector('#page .back')", what="进入节点页")
        step("10-节点页")
        p.click(*p.center("#page .back"))
        p.wait(f"""(() => {{ const el = document.querySelector('{node('E1')}.sel'); if (!el || location.hash !== '#/') return false;
                   const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
                   return Math.abs(cx - innerWidth / 2) < 5 && Math.abs(cy - innerHeight / 2) < 5; }})()""",
               what="返回画布，E1 居中并选中")
        # Enter 也能进入
        p.key("Enter")
        p.wait("location.hash === '#/exp/E1'", what="Enter 进入节点页")
        step("11-返回画布")
        ok = True
        print("画布冒烟全部通过")
    except Exception:
        if p is not None:
            try:
                print("页面文字：", p.js("document.body.innerText.slice(0, 800)"))
                print("页面错误：", p.js("window.__errors"))
                if shots:
                    p.shot(shots / "fail.png")
            except Exception as e:  # noqa: BLE001
                print("取页面状态失败：", e)
        raise
    finally:
        proc.terminate()
        httpd.shutdown()
        time.sleep(.5)
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
