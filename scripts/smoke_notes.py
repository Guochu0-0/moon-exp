"""Notes 编辑器（Vditor IR）回归：现有 Notes 写回、显示约定、输入法、快捷键。

    python scripts/smoke_notes.py [--runs <runs 目录>] [--gh-render] [--out <目录>]

把 --runs（默认仓库的 runs/）下每个实验的 exp.toml、notes.md、extra/、launch/ 复制到临时目录（不复制 preds/、tb/，
实验都按未点亮显示，节点页打开得快），起工作台和无头 Chrome / Edge：

1. 写回：每份 notes.md 在编辑器里打开，切到「源码」取出此时保存会写入的内容（不改动直接保存写进文件的就是它），
   要求切回后不显示未保存；和原文去掉空白后逐行比，列出不同的行。--gh-render 时再把原文和写回的内容都交给
   GitHub 的 Markdown 接口（gh api markdown，gfm，以本仓库为上下文）渲染，比较两份 HTML。
2. 显示约定（另建一个测试实验）：相对路径的附件图片能显示；`#` 标题显示为 16px；链接单击不打开、Ctrl+单击才打开
   （外链原样，相对路径按实验目录）；段内换行照源码分行显示。
3. 输入法：用 CDP 模拟中文输入法组字（先出拼音、再上屏汉字），上屏后只留汉字，组字期间的按键不触发快捷键。
4. 快捷键：Ctrl+B、Ctrl+2 / Ctrl+0、Ctrl+Shift+]、列表项中间按 Tab、Ctrl+U 被吞掉、Ctrl+/、Ctrl+S、「撤销改动」。

--out 给出时把写回的内容存成 <out>/<id>.md，方便对照。依赖：Chrome 或 Edge、websocket-client；--gh-render 要 gh 已登录。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_canvas import BROWSERS, REPO, Page, free_port, until  # noqa: E402
from workbench import server  # noqa: E402
from workbench.dataset import Dataset  # noqa: E402

CTRL, ALT, SHIFT = 2, 1, 8
TEST = "ZZnotes"
TEST_MD = """# 标题一

第一行
第二行

[附件](extra/a.png) 和 [外链](https://example.com/x)

![图](extra/a.png)

- 甲乙丙
- 丁戊己

尾段
"""


def github_html(text: str) -> str:
    r = subprocess.run(["gh", "api", "markdown", "--input", "-"], capture_output=True, check=True,
                       input=json.dumps({"text": text, "mode": "gfm", "context": "Guochu0-0/moon-exp"}).encode("utf-8"))
    return re.sub(r">\s+<", "><", r.stdout.decode("utf-8")).strip()


def strip_lines(s: str) -> list[str]:
    """去掉空白；表格分隔行的连字符按列宽补齐过，统一成 ---。"""
    lines = (re.sub(r"\s+", "", l) for l in s.split("\n"))
    return [re.sub(r"-{3,}", "---", l) if re.fullmatch(r"\|?(:?-+:?\|)*:?-+:?\|?", l) else l for l in lines if l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--gh-render", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        sys.exit("找不到 Chrome / Edge")

    tmp = Path(tempfile.mkdtemp(prefix="wb-notes-"))
    runs = tmp / "runs"
    src_runs = Path(args.runs)
    ids = []
    for src in sorted(p for p in src_runs.iterdir() if (p / "exp.toml").exists()):
        (runs / src.name).mkdir(parents=True)
        shutil.copy(src / "exp.toml", runs / src.name / "exp.toml")
        for d in ("extra", "launch"):
            if (src / d).is_dir():
                shutil.copytree(src / d, runs / src.name / d)
        if (src / "notes.md").exists():
            shutil.copy(src / "notes.md", runs / src.name / "notes.md")
            ids.append(src.name)
    t = runs / TEST
    (t / "extra").mkdir(parents=True)
    (t / "exp.toml").write_text(f'id = "{TEST}"\ntitle = "Notes 编辑器测试"\ndate = "2026-10-06"\n', encoding="utf-8")
    (t / "notes.md").write_bytes(TEST_MD.encode("utf-8"))
    png = next(src_runs.glob("*/extra/**/*.png"))
    shutil.copy(png, t / "extra" / "a.png")
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(server.Workbench(runs, Dataset(tmp), REPO)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    port = free_port()
    proc = subprocess.Popen([browser, f"--remote-debugging-port={port}", f"--user-data-dir={tmp / 'profile'}",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1500,950",
                             *([] if args.headed else ["--headless=new"]), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok, p, failed = False, None, []
    try:
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                break
            except OSError:
                time.sleep(.1)
        p = Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
        p.cmd("Page.enable")
        p.cmd("Page.addScriptToEvaluateOnNewDocument", source="window.__errors = []; window.__opened = [];"
              "addEventListener('error', e => __errors.push(String(e.message)));"
              "addEventListener('unhandledrejection', e => __errors.push(String(e.reason)));"
              "window.open = u => { __opened.push(String(u)); return null; }; window.confirm = () => true;")
        p.cmd("Emulation.setDeviceMetricsOverride", width=1500, height=950, deviceScaleFactor=1, mobile=False)
        p.cmd("Page.navigate", url=url)
        p.wait("document.querySelector('.title')", 20, "画布")

        NB = "#page #sec-notes"
        foot = lambda b: f"document.querySelector('{NB} .notes-foot .{b}')"

        def open_exp(rid):
            p.js(f"location.hash = '#/exp/{rid}'")
            p.wait(f"{foot('status')}?.textContent === 'runs/{rid}/notes.md' && document.querySelector('{NB} .vditor-ir')",
                   30, f"{rid} 的 Notes 编辑器载入")

        def value():   # 切到源码读出保存会写入的内容，再切回
            p.js(f"{foot('mode')}.click()")
            v = p.js(f"document.querySelector('{NB} textarea.notes-src').value")
            p.js(f"{foot('mode')}.click()")
            return v

        def key(code, key_, mods=0, vk=0):
            base = dict(key=key_, code=code, windowsVirtualKeyCode=vk or (ord(key_.upper()) if len(key_) == 1 else 0),
                        modifiers=mods)
            p.cmd("Input.dispatchKeyEvent", type="rawKeyDown", **base)
            p.cmd("Input.dispatchKeyEvent", type="keyUp", **base)
            time.sleep(.15)

        def caret(text, at_end=True):   # 把光标放在含 text 的文字节点里 text 的末尾（或开头）
            assert p.js(f"""(() => {{
                const root = document.querySelector('{NB} .vditor-ir .vditor-reset'); root.focus();
                const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
                for (let n; (n = w.nextNode());) {{ const k = n.nodeValue.indexOf({json.dumps(text)}); if (k < 0) continue;
                  const r = document.createRange(); r.setStart(n, k + {len(text) if at_end else 0}); r.collapse(true);
                  const s = getSelection(); s.removeAllRanges(); s.addRange(r); return true; }}
                return false; }})()"""), f"找不到文字 {text}"

        # ---- 1. 写回 ----
        print(f"写回：{len(ids)} 份 Notes（{src_runs}）")
        for rid in ids:
            open_exp(rid)
            orig = (runs / rid / "notes.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
            v = value()
            dirty = not p.js(f"{foot('save')}.disabled")
            if out:
                (out / f"{rid}.md").write_bytes(v.encode("utf-8"))
            a, b = strip_lines(orig), strip_lines(v)
            diff = [x for x in b if x not in a] + [x for x in a if x not in b]
            gh = ""
            if args.gh_render:
                gh = "GitHub 渲染相同" if github_html(orig) == github_html(v) else "GitHub 渲染不同"
            line_changes = sum(1 for x, y in zip(orig.split("\n"), v.split("\n")) if x != y) + abs(orig.count("\n") - v.count("\n"))
            print(f"  {rid:5s} 原文 {orig.count(chr(10)):4d} 行，改动约 {line_changes:3d} 行；去空白后不同 {len(diff)} 行"
                  f"{'；' + gh if gh else ''}{'；切回后显示未保存' if dirty else ''}")
            for x in diff[:6]:
                print(f"        {x[:100]}")
            if dirty or diff or gh.endswith("不同"):
                failed.append(rid)

        # ---- 2. 显示约定 ----
        open_exp(TEST)
        p.wait(f"""(() => {{ const im = document.querySelector('{NB} .vditor-ir img');
                   return im && im.getAttribute('src').endsWith('/runs/{TEST}/extra/a.png') && im.complete && im.naturalWidth > 0; }})()""",
               10, "相对路径的附件图片显示")
        assert p.js(f"getComputedStyle(document.querySelector('{NB} .vditor-ir h1')).fontSize") == "16px"
        assert p.js(f"""(() => {{ const e = [...document.querySelectorAll('{NB} .vditor-ir p')].find(e => e.textContent.includes('第一行'));
                   return e.innerText.includes('第一行\\n第二行') && e.getBoundingClientRect().height > 1.5 * parseFloat(getComputedStyle(e).lineHeight); }})()"""), \
            "段内换行没有分行显示"
        print("通过  附件图片、标题降两级、段内换行分行")
        link = lambda t: p.js(f"""(() => {{ const a = [...document.querySelectorAll('{NB} .vditor-ir [data-type="a"]')].find(a => a.textContent.includes('{t}'));
                   const r = (a.querySelector('.vditor-ir__link') || a).getBoundingClientRect(); return [r.left + 4, r.top + r.height / 2]; }})()""")
        p.click(*link("附件"))
        time.sleep(.2)
        assert p.js("__opened.length") == 0, "单击就打开了链接"
        p.click(*link("附件"), mods=CTRL)
        p.click(*link("外链"), mods=CTRL)
        opened = p.js("__opened")
        assert opened == [f"/runs/{TEST}/extra/a.png", "https://example.com/x"], opened
        print("通过  单击不打开链接，Ctrl+单击打开（相对路径按实验目录）")

        # ---- 3. 输入法 ----
        caret("尾段")
        p.cmd("Input.imeSetComposition", text="z", selectionStart=1, selectionEnd=1)
        p.cmd("Input.imeSetComposition", text="zhong", selectionStart=5, selectionEnd=5)
        p.cmd("Input.dispatchKeyEvent", type="rawKeyDown", key="Process", code="KeyB", windowsVirtualKeyCode=229, modifiers=CTRL)
        p.cmd("Input.dispatchKeyEvent", type="keyUp", key="Process", code="KeyB", windowsVirtualKeyCode=229, modifiers=CTRL)
        p.cmd("Input.imeSetComposition", text="中", selectionStart=1, selectionEnd=1)
        p.cmd("Input.insertText", text="中文")
        time.sleep(.3)
        v = value()
        assert "尾段中文\n" in v and "zhong" not in v and "**" not in v, v[-80:]
        assert not p.js(f"{foot('save')}.disabled"), "输入后保存按钮没有亮"
        print("通过  输入法组字上屏，组字期间的按键不触发快捷键")

        # ---- 4. 快捷键 ----
        p.js(f"""(() => {{ const e = [...document.querySelectorAll('{NB} .vditor-ir p')].find(e => e.textContent.includes('尾段'));
                  const r = document.createRange(); const n = [...e.childNodes].find(n => n.nodeType === 3 && n.nodeValue.includes('尾段'));
                  const k = n.nodeValue.indexOf('尾段'); r.setStart(n, k); r.setEnd(n, k + 2);
                  const s = getSelection(); s.removeAllRanges(); s.addRange(r); }})()""")
        key("KeyB", "b", CTRL)
        assert "**尾段**中文" in value(), "Ctrl+B"
        caret("第二行")
        key("Digit2", "2", CTRL)
        assert "## 第一行\n" in value(), "Ctrl+2：\n" + value()   # 段内换行时整段算一块，标题落在第一行
        caret("第一行")
        key("Digit0", "0", CTRL)
        assert "## " not in value(), "Ctrl+0"
        caret("中文")
        key("BracketRight", "]", CTRL | SHIFT, 221)
        assert re.search(r"^- \*\*尾段\*\*中文", value(), re.M), "Ctrl+Shift+]"
        caret("丁", at_end=False)
        before = value()
        key("KeyU", "u", CTRL)
        assert value() == before, "Ctrl+U 没被吞掉"
        caret("丁戊")   # 第二项中间（第一项前面没有可挂靠的项，本来就缩进不了）
        key("Tab", "Tab", 0, 9)
        assert re.search(r"^ +- 丁戊己", value(), re.M), "Tab 在列表项中间缩进：\n" + value()
        print("通过  Ctrl+B、Ctrl+2 / Ctrl+0、Ctrl+Shift+]、列表项中间 Tab、Ctrl+U 被吞掉")
        key("Slash", "/", CTRL, 191)
        assert p.js(f"!document.querySelector('{NB} textarea.notes-src').hidden && {foot('mode')}.getAttribute('aria-pressed') === 'true'")
        want = p.js(f"document.querySelector('{NB} textarea.notes-src').value")
        key("Slash", "/", CTRL, 191)
        key("KeyS", "s", CTRL)
        until(lambda: (runs / TEST / "notes.md").read_bytes() == want.encode("utf-8"), "Ctrl+S 写入 notes.md")
        p.wait(f"{foot('save')}.disabled && {foot('status')}.textContent.startsWith('已保存')", what="保存后不再显示未保存")
        caret("丁")
        p.type("戊")
        p.wait(f"!{foot('save')}.disabled", what="改动后可以保存")
        p.js(f"{foot('revert')}.click()")
        assert value() == want and p.js(f"{foot('save')}.disabled"), "撤销改动"
        print("通过  Ctrl+/ 切源码、Ctrl+S 保存、撤销改动")
        errs = p.js("__errors")
        assert not errs, errs
        ok = not failed
        print("Notes 回归全部通过" if ok else f"写回有差异：{', '.join(failed)}")
    except Exception:
        if p is not None:
            try:
                print("页面错误：", p.js("window.__errors"))
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
