"""画布冒烟：用无头 Chrome（CDP）对真实页面做一遍交互，不进 pytest 默认集合。

    MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset python scripts/smoke_canvas.py [--shots DIR]

把仓库 runs/ 里的 B0、B0m（exp.toml 与 preds/*.jsonl）复制到临时目录，起一个工作台服务，然后覆盖：
按 E 新建、菜单派生、拖到空白派生、拖到实验改父、成环被拒、已点亮实验的删除项置灰、
拖动后刷新位置保持、终端写 exp.toml 后几秒内出现、便签新建与编辑、Ctrl+G 打组并重命名、
拖分组框标题带走框内实验和便签、分组框调大小与换色、`?` 面板、框选加多选拖动、Ctrl+A、
各类右键菜单的内容、Del 删除选中对象时跳过已点亮实验、双击进入节点页并返回。
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

    # mods：CDP 修饰键位掩码，Alt=1、Ctrl=2、Meta=4、Shift=8
    def mouse(self, kind, x, y, button="left", clicks=1, mods=0):
        buttons = {"left": 1, "right": 2}[button] if kind != "mouseReleased" else 0
        self.cmd("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, buttons=buttons, clickCount=clicks,
                 modifiers=mods)

    def move(self, x, y):
        self.cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="none", buttons=0)

    def click(self, x, y, button="left", clicks=1, mods=0):
        self.move(x, y)
        for c in range(1, clicks + 1):
            self.mouse("mousePressed", x, y, button, c, mods)
            self.mouse("mouseReleased", x, y, button, c, mods)

    def drag(self, a, b, steps=12, mods=0):
        self.move(*a)
        self.mouse("mousePressed", *a, mods=mods)
        for i in range(1, steps + 1):
            x, y = a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps
            self.cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="left", buttons=1, modifiers=mods)
        self.mouse("mouseReleased", *b, mods=mods)

    def key(self, key, code=None, text=None, mods=0):
        vk = {"Enter": 13, "Escape": 27, "Delete": 46, "?": 191}.get(key, ord(key.upper()) if len(key) == 1 else 0)
        base = dict(key=key, code=code or key, windowsVirtualKeyCode=vk, modifiers=mods)
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


CTRL, SHIFT = 2, 8


def menu_items(p):
    return p.js("[...document.querySelectorAll('.menu:not([hidden]) button')].map(b => b.querySelector('span').textContent)")


def blank_near(p, x, y, w=260, h=140):
    """离 (x, y) 最近、w×h 范围内没有卡片 / 便签 / 分组框 / 外框控件的点，返回该矩形的中心。"""
    r = p.js(f"""(() => {{
        const free = (cx, cy) => {{
            for (let dx = -{w} / 2; dx <= {w} / 2; dx += 20) for (let dy = -{h} / 2; dy <= {h} / 2; dy += 20) {{
                const el = document.elementFromPoint(cx + dx, cy + dy);
                if (!el || !el.classList.contains('viewport')) return false;
            }}
            return true;
        }};
        let best = null;
        for (let cx = 160; cx < innerWidth - 160; cx += 20) for (let cy = 110; cy < innerHeight - 110; cy += 20) {{
            const d = Math.hypot(cx - {x}, cy - {y});
            if ((!best || d < best[2]) && free(cx, cy)) best = [cx, cy, d];
        }}
        return best && best.slice(0, 2);
    }})()""")
    assert r, "画布上找不到空白处"
    return r


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
        canvas_all = lambda: json.loads((runs / "canvas.json").read_text(encoding="utf-8"))

        # 空白处右键：新建实验 / 便签 / 分组框、打组（没选中时置灰）、适应窗口
        fit()
        p.key("Escape")
        bx, by = blank_near(p, 750, 475)
        p.click(bx, by, button="right")
        p.wait("document.querySelector('.menu:not([hidden])')", what="空白处菜单")
        items = menu_items(p)
        assert items == ["新建实验", "新建便签", "新建分组框", "把选中的对象打成一组", "适应窗口"], items
        assert p.js("[...document.querySelectorAll('.menu button')].find(b => b.textContent.includes('打成一组')).disabled")
        step("12-空白处菜单")

        # 菜单新建便签：随即编辑，Shift+Enter 换行，Enter 确认，写进 canvas.json
        menu_click(p, "新建便签")
        p.wait("document.querySelector('.sticky .body[contenteditable=true]')", what="便签进入编辑")
        p.type("先看 Val")
        p.key("Enter", text="\r", mods=SHIFT)   # 带 text 才会真的插入换行
        p.type("再定稿")
        p.key("Enter")
        until(lambda: [x["text"] for x in canvas_all()["stickies"]] == ["先看 Val\n再定稿"], "便签文字写回 canvas.json")
        sid = canvas_all()["stickies"][0]["id"]
        sticky = f'.sticky[data-s="{sid}"]'
        assert p.js(f"document.querySelector('{sticky} .hdr').textContent") == "便签"
        # 便签右键：编辑、删除
        p.click(*p.center(f"{sticky} .body"), button="right")
        assert menu_items(p) == ["编辑", "删除便签"], menu_items(p)
        p.key("Escape")
        # 双击编辑
        p.click(*p.center(f"{sticky} .body"), clicks=2)
        p.wait(f"document.querySelector('{sticky} .body[contenteditable=true]')", what="双击便签进入编辑")
        p.key("a", "KeyA", mods=CTRL)
        p.type("Test 只在定稿时跑")
        p.key("Enter")
        until(lambda: canvas_all()["stickies"][0]["text"] == "Test 只在定稿时跑", "便签改写写回 canvas.json")
        step("13-便签")

        # 把便签挪到 B0 正下方，再单击 B0、Ctrl+单击便签，Ctrl+G 打组并重命名
        b0 = p.js(f"(() => {{ const r = document.querySelector('{node('B0')}').getBoundingClientRect(); return [r.left, r.bottom]; }})()")
        s0 = p.js(f"(() => {{ const r = document.querySelector('{sticky}').getBoundingClientRect(); return [r.left, r.top]; }})()")
        hc = p.center(f"{sticky} .hdr")
        p.drag(hc, (hc[0] + b0[0] + 10 - s0[0], hc[1] + b0[1] + 30 - s0[1]))
        p.click(*p.center(f"{node('B0')} .hdr .nid"))
        p.click(*p.center(f"{sticky} .hdr"), mods=CTRL)
        assert p.js("document.querySelectorAll('.node.sel, .sticky.sel').length") == 2
        # 实验上有多选时，菜单里有「打成一组」
        p.click(*p.center(f"{node('B0')} .hdr .nid"), button="right")
        assert "把选中的 2 个打成一组" in menu_items(p), menu_items(p)
        p.key("Escape")
        p.wait("document.querySelector('.menu[hidden]')", what="Esc 关闭菜单")
        assert p.js("document.querySelectorAll('.node.sel, .sticky.sel').length") == 2
        p.key("g", "KeyG", mods=CTRL)
        p.wait("document.querySelector('.grp .gt[contenteditable=true]')", what="Ctrl+G 后进入分组重命名")
        p.type("零样本基线")
        p.key("Enter")
        until(lambda: [g["title"] for g in canvas_all()["groups"]] == ["零样本基线"], "分组标题写回 canvas.json")
        g = canvas_all()["groups"][0]
        assert g["color"] in ("c1", "c2", "c3", "c4")
        grp = f'.grp[data-g="{g["id"]}"]'
        step("14-Ctrl+G打组")

        # 拖分组框标题：框内的 B0 和便签一起移动，松手后坐标都已保存
        c0 = canvas_all()
        g0, e0, n0 = c0["groups"][0], c0["experiments"]["B0"], c0["stickies"][0]
        t = p.center(f"{grp} .gt")
        p.drag((t[0] - 60, t[1]), (t[0] - 60 + 90, t[1] + 60))
        until(lambda: canvas_all()["groups"][0]["x"] != g0["x"], "分组框坐标写回 canvas.json")
        c1 = canvas_all()
        dx, dy = c1["groups"][0]["x"] - g0["x"], c1["groups"][0]["y"] - g0["y"]
        assert dx > 0 and dy > 0
        assert (c1["experiments"]["B0"]["x"] - e0["x"], c1["experiments"]["B0"]["y"] - e0["y"]) == (dx, dy)
        assert (c1["stickies"][0]["x"] - n0["x"], c1["stickies"][0]["y"] - n0["y"]) == (dx, dy)
        step("15-拖分组框")

        # 按 G 在鼠标处新建分组框（找一块放得下的空白），再拖右下角手柄调大小
        p.key("-", "Minus", "-")
        p.key("-", "Minus", "-")
        k = p.js("parseFloat(document.querySelector('.pc').textContent) / 100")
        cx, cy = blank_near(p, 750, 475, 460 * k + 120, 300 * k + 80)
        p.move(cx - 230 * k, cy - 150 * k)
        p.key("g", "KeyG", "g")
        until(lambda: len(canvas_all()["groups"]) == 2, "按 G 新建的分组框写进 canvas.json")
        g2 = canvas_all()["groups"][1]
        assert (g2["w"], g2["h"], g2["title"]) == (460, 300, "新分组")
        rz = p.center(f'.grp[data-g="{g2["id"]}"] .rz')
        p.drag(rz, (rz[0] + 70, rz[1] + 40))
        until(lambda: canvas_all()["groups"][1]["w"] > 460 and canvas_all()["groups"][1]["h"] > 300, "分组框尺寸写回 canvas.json")
        step("16-分组框调大小")

        # 分组框右键：在此新建实验、重命名、换颜色、贴合框内内容、删除（内容保留）
        p.click(*p.center(f"{grp} .gt"), button="right")
        items = menu_items(p)
        assert items == ["在此新建实验", "重命名", "换颜色", "贴合框内内容", "删除分组框（内容保留）"], items
        color0 = canvas_all()["groups"][0]["color"]
        menu_click(p, "换颜色")
        until(lambda: canvas_all()["groups"][0]["color"] != color0, "换颜色写回 canvas.json")
        assert p.js(f"document.querySelector('{grp}').classList.contains({json.dumps(canvas_all()['groups'][0]['color'])})")
        # 贴合框内内容：空框只提示；把便签拖进新框再贴合，框收紧到便签外加留白
        grp2 = f'.grp[data-g="{g2["id"]}"]'
        p.click(*p.center(f"{grp2} .gt"), button="right")
        menu_click(p, "贴合框内内容")
        p.wait("document.querySelector('.toast.on') && document.querySelector('.toast').textContent.includes('框里没有内容')",
               what="空框贴合提示")
        p.key("Escape")                       # 先取消选择（B0 和便签还选着），只拖便签
        assert p.js("document.querySelectorAll('.sel').length") == 0
        hc, gc = p.center(f"{sticky} .hdr"), p.center(grp2)
        p.drag(hc, gc)
        p.click(*p.center(f"{grp2} .gt"), button="right")
        menu_click(p, "贴合框内内容")
        until(lambda: canvas_all()["groups"][1]["w"] == canvas_all()["stickies"][0]["w"] + 60, "贴合后框收紧")
        step("17-分组框菜单")

        # ? 打开快捷键面板，Esc 关闭；左下角按钮也能开关
        p.key("?", "Slash", "?", mods=SHIFT)
        p.wait("getComputedStyle(document.querySelector('.keys')).display !== 'none'", what="? 打开快捷键面板")
        assert "Ctrl" in p.js("document.querySelector('.keys').textContent")
        step("18-快捷键面板")
        p.key("Escape")
        p.wait("getComputedStyle(document.querySelector('.keys')).display === 'none'", what="Esc 关闭快捷键面板")
        p.click(*p.center(".keysbtn"))
        p.wait("document.querySelector('.keys.on')", what="快捷键按钮打开面板")
        p.click(*p.center(".keysbtn"))
        p.wait("!document.querySelector('.keys.on')", what="快捷键按钮关闭面板")

        # 框选：Ctrl+从空白处拖，框住 E1 和 E3；再拖其中一个，两个一起移动
        fit()
        p.key("Escape")
        u = p.js(f"""(() => {{ const a = document.querySelector('{node('E1')}').getBoundingClientRect(),
                    b = document.querySelector('{node('E3')}').getBoundingClientRect();
                    return [Math.min(a.left, b.left), Math.min(a.top, b.top), Math.max(a.right, b.right), Math.max(a.bottom, b.bottom)]; }})()""")
        start, end = (u[0] - 12, u[1] - 12), (u[2] + 12, u[3] + 12)
        assert p.js(f"!document.elementFromPoint({start[0]}, {start[1]}).closest('.node, .sticky, .gt, .rz')"), \
            "框选起点应在空白处或分组框体上"
        p.drag(start, end, mods=CTRL)
        selected = p.js("[...document.querySelectorAll('.node.sel')].map(e => e.dataset.id)")
        assert {"E1", "E3"} <= set(selected), selected
        p.click(*p.center(f"{node('E1')} .hdr .nid"), mods=CTRL)      # Ctrl 单击：减选，再加回来
        assert not p.js(f"document.querySelector('{node('E1')}').classList.contains('sel')")
        p.click(*p.center(f"{node('E1')} .hdr .nid"), mods=CTRL)
        before = canvas_all()["experiments"]
        a = p.center(f"{node('E3')} .hdr .nid")
        p.drag(a, (a[0] + 50, a[1] + 40))
        until(lambda: canvas_all()["experiments"]["E3"] != before["E3"], "多选拖动写回 canvas.json")
        after = canvas_all()["experiments"]
        d1 = (after["E1"]["x"] - before["E1"]["x"], after["E1"]["y"] - before["E1"]["y"])
        d3 = (after["E3"]["x"] - before["E3"]["x"], after["E3"]["y"] - before["E3"]["y"])
        assert d1 == d3 and d1 != (0, 0), (d1, d3)
        # 在多选中单击（不拖）：只留这一个
        p.click(*p.center(f"{node('E3')} .hdr .nid"))
        assert p.js("[...document.querySelectorAll('.node.sel, .sticky.sel')].map(e => e.dataset.id)") == ["E3"]
        step("19-框选与多选拖动")

        # Ctrl+A 全选实验和便签；单击空白取消选择
        p.key("a", "KeyA", mods=CTRL)
        n_all = p.js("document.querySelectorAll('.node').length + document.querySelectorAll('.sticky').length")
        assert p.js("document.querySelectorAll('.node.sel, .sticky.sel').length") == n_all
        bx, by = blank_near(p, 750, 475, 40, 40)
        p.click(bx, by)
        assert p.js("document.querySelectorAll('.sel').length") == 0

        # Del 删除选中对象：已点亮的 B0 跳过并提示，便签删掉
        p.click(*p.center(f"{node('B0')} .hdr .nid"))
        p.click(*p.center(f"{sticky} .hdr"), mods=CTRL)
        p.key("Delete")
        p.wait("document.querySelector('.toast').textContent.includes('已跳过')", what="Del 跳过已点亮实验")
        until(lambda: canvas_all()["stickies"] == [], "便签从 canvas.json 删除")
        assert (runs / "B0").exists() and p.js(f"!!document.querySelector('{node('B0')}')")
        step("20-Del删除选中")

        # 双击进入节点页，「← 画布」返回，实验居中且选中
        p.click(*p.center(f"{node('E1')} .hdr .nid"), clicks=2)
        p.wait("location.hash === '#/exp/E1' && document.querySelector('#page .back')", what="进入节点页")
        step("21-节点页")
        p.click(*p.center("#page .back"))
        p.wait(f"""(() => {{ const el = document.querySelector('{node('E1')}.sel'); if (!el || location.hash !== '#/') return false;
                   const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
                   return Math.abs(cx - innerWidth / 2) < 5 && Math.abs(cy - innerHeight / 2) < 5; }})()""",
               what="返回画布，E1 居中并选中")
        # Enter 也能进入
        p.key("Enter")
        p.wait("location.hash === '#/exp/E1'", what="Enter 进入节点页")
        step("22-返回画布")
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
