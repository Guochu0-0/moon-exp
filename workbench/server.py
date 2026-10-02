"""本地服务：读写 runs/，给画布和节点页提供 HTTP JSON API 与静态页。

读接口
    GET  /api/data                   全部实验摘要（画布用）+ canvas.json 里的便签、分组框
    GET  /api/exp/<id>               单个实验详情（节点页用）：信息、commit、各方法结果、参考方法默认值与候选、notes
    GET  /api/compare?method=&ref=&split=
                                     方法 vs 参考方法（<实验>/<方法>、identity = 未配准、省略 = 无）：两边结果与差值
    GET  /img?split=&pair=&kind=     原图 PNG（opt / sar）
    GET  /runs/<id>/extra/...        附件
写接口（成功返回 {"ok": true, ...}；被拒绝时 4xx + {"error": "..."}）
    POST   /api/exp                  新建实验 {id?, title?, parent?, init?, x?, y?}
    POST   /api/exp/<id>/parent      改父 / init {parent, init?}；parent 为 null 即断开
    POST   /api/exp/<id>/title       改标题 {title}
    POST   /api/exp/<id>/rename      改编号 {id}（仅未点亮）
    DELETE /api/exp/<id>             删除（仅未点亮）
    PUT    /api/canvas               保存画布 {experiments?, groups?, stickies?}；experiments 按 key 合并，
                                     groups / stickies 整体替换（字段见 canvas.py）
    PUT    /api/exp/<id>/notes       保存 notes.md {text}（UTF-8、LF；第一次保存时才创建）

前端定期重拉 /api/data，所以它必须便宜：exp.toml 每次现读（很小），AUC 按 preds 文件的 mtime 和大小缓存，
只有变了的方法才重算。
"""
from __future__ import annotations

import io
import json
import mimetypes
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import canvas, protocol, reference
from .dataset import SPLIT_DIRS, Dataset, to_display
from .evaluate import IDENTITY, evaluate_preds, jsonable, pair_errors
from .records import (SPLITS, EditError, NotFound, Pred, Run, create_experiment, delete_experiment, load_runs,
                      next_id, rename_experiment, save_notes, set_parent, set_title)

STATIC = Path(__file__).parent / "static"
CARD_AUC = 10.0           # 画布卡片上的数值条：AUC@10（Val），不算 bootstrap CI
CARD_SPLIT = "val"
DELTA = "auc@10"          # 节点页全表末列：相对父实验同名基础方法的 ΔAUC@10
CTYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".html": "text/html; charset=utf-8", ".woff2": "font/woff2", ".svg": "image/svg+xml"}


class Workbench:
    def __init__(self, runs_dir: Path, ds: Dataset, repo: Path):
        self.runs_dir, self.ds, self.repo = Path(runs_dir), ds, repo
        self.lock = threading.RLock()
        self._auc: dict[Path, tuple[tuple, float | None]] = {}
        self._res: dict[Path, tuple[tuple, dict]] = {}
        self._identity: dict[str, dict] = {}

    # ---------------- 读 ----------------

    def _card_auc(self, r: Run, method: str) -> float | None:
        p = r.dir / "preds" / method / f"{CARD_SPLIT}.jsonl"
        if not p.exists():
            return None
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._auc.get(p)
        if hit is None or hit[0] != sig:
            v = protocol.auc(pair_errors(self.ds, CARD_SPLIT, r.preds(method, CARD_SPLIT)), CARD_AUC)
            hit = self._auc[p] = (sig, v)
        return hit[1]

    def _result(self, r: Run, method: str, split: str) -> dict | None:
        """一个方法在一个 split 上的 summary 与逐 pair 误差（同 metrics.json），按 preds 文件的 mtime 和大小缓存。
        "_err" 是未取整的误差数组（失败为 ∞），只在服务端用，不下发。"""
        p = r.dir / "preds" / method / f"{split}.jsonl"
        if not p.exists():
            return None
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._res.get(p)
        if hit is None or hit[0] != sig:
            hit = self._res[p] = (sig, self._evaluate(split, r.preds(method, split)))
        return hit[1]

    def _identity_result(self, split: str) -> dict:
        if split not in self._identity:
            self._identity[split] = self._evaluate(split, {p: Pred(p, IDENTITY) for p in self.ds.pairs(split)})
        return self._identity[split]

    def _evaluate(self, split: str, preds: dict) -> dict:
        e = evaluate_preds(self.ds, split, preds)
        return {"summary": e["summary"], "errors": e["errors"], "_err": pair_errors(self.ds, split, preds)}

    def _score(self, runs: dict[str, Run]):
        main = protocol.protocol_info()["main"]

        def score(rid: str, m: str) -> float | None:
            res = self._result(runs[rid], m, "val")
            return res["summary"][main] if res else None
        return score

    def detail(self, rid: str) -> dict:
        """节点页：一个实验的全部事实。"""
        with self.lock:
            runs = load_runs(self.runs_dir)
            if rid not in runs:
                raise NotFound(f"实验 {rid} 不存在")
            r, score = runs[rid], self._score(runs)
            p = runs.get(r.parent) if r.parent else None
            methods = []
            for m in reference.ranked(r, score):
                res = {s: x for s in SPLITS if (x := self._result(r, m, s))}
                base = reference.base_method(m)
                base = f"{p.id}/{base}" if p is not None and base in p.methods else None
                delta = {}
                for s, x in res.items():
                    b = self._result(p, base.partition("/")[2], s) if base else None
                    if b and x["summary"][DELTA] is not None and b["summary"][DELTA] is not None:
                        delta[s] = x["summary"][DELTA] - b["summary"][DELTA]
                methods.append({"id": m, **r.method_info(m), "ref": reference.default_ref(r, m, runs, score),
                                "base": base, "delta": delta, "results": {k: _public(x) for k, x in res.items()}})
            return jsonable({"id": rid, "title": r.title, "parent": r.parent, "init": r.init,
                             "baseline": r.baseline, "date": r.date, "lit": r.lit, "methods": methods,
                             "children": [k for k, x in runs.items() if x.parent == rid],
                             "commit": r.commit(), "delta_key": DELTA, "warnings": r.warnings,
                             "notes": r.notes,
                             "reference": {"groups": reference.groups(r, runs, score)},
                             "protocol": protocol.protocol_info()})

    def compare(self, q: dict) -> dict:
        """方法 vs 参考方法：两边的 summary 与逐 pair 误差，以及差值（方法 − 参考方法）。"""
        with self.lock:
            runs = load_runs(self.runs_dir)
            split = q.get("split", "val")
            if split not in SPLITS:
                raise EditError(f"split 应为 {' / '.join(SPLITS)}")
            side = {}
            for name in ("method", "ref"):
                key = q.get(name) or None
                try:
                    target = reference.parse(key, runs)
                except KeyError:
                    raise EditError(f"找不到方法 {key}") from None
                if name == "method" and not isinstance(target, tuple):
                    raise EditError("method 应写成 <实验>/<方法>")
                if target is None:
                    side[name] = None
                    continue
                res = self._identity_result(split) if target == reference.UNREGISTERED else self._result(*target, split)
                if res is None:
                    raise EditError(f"{key} 没有 {split} 结果")
                side[name] = {"key": key, "label": reference.label(key, runs), **res}
            a, b = side["method"], side["ref"]
            diff = reference.paired_diff(a["_err"], b["_err"], a["summary"], b["summary"]) if b else None
            return jsonable({"split": split, "method": _public(a), "ref": b and _public(b), "diff": diff})

    def data(self) -> dict:
        with self.lock:
            runs = load_runs(self.runs_dir)
            c = canvas.load(self.runs_dir)
            exps = {}
            for rid, r in runs.items():
                ms = [{"id": m, **r.method_info(m), "auc": self._card_auc(r, m)} for m in r.methods]
                ms.sort(key=lambda x: (x["auc"] is None, -(x["auc"] or 0)))
                exps[rid] = {"id": rid, "title": r.title, "parent": r.parent, "init": r.init, "baseline": r.baseline,
                             "date": r.date, "lit": r.lit, "methods": ms, "warnings": r.warnings}
            pos = canvas.place({k: {"parent": e["parent"], "h": self._height(e, exps)} for k, e in exps.items()},
                               c["experiments"])
            for k, e in exps.items():
                e.update(pos[k], placed=k in c["experiments"])
            return jsonable({"metric": {"key": f"auc@{CARD_AUC:g}", "split": CARD_SPLIT},
                             "experiments": list(exps.values()),
                             "canvas": {"groups": c["groups"], "stickies": c["stickies"]}})

    @staticmethod
    def _height(e: dict, exps: dict) -> int:
        """卡片高度估计：默认显示前几个方法，被子实验用作 init 的方法总是显示。"""
        shown = {m["id"] for m in e["methods"][:canvas.SHOWN_METHODS]}
        shown |= {x["init"].partition("/")[2] for x in exps.values()
                  if x["parent"] == e["id"] and (x["init"] or "").startswith(e["id"] + "/")}
        return canvas.node_height(len(e["methods"]), len(shown & {m["id"] for m in e["methods"]}), e["lit"])

    # ---------------- 写 ----------------

    def _pin(self):
        """把还没存过坐标的实验按当前显示的位置写进 canvas.json。

        默认位置是现算的，别的实验一动它就可能跟着挪；所以在画布上做任何改动之前先把它们钉住，
        页面上看到的布局就不会因为这次改动而跳动。只有写接口调用它，读接口从不写文件。
        """
        c = canvas.load(self.runs_dir)
        loose = {e["id"]: {"x": e["x"], "y": e["y"]} for e in self.data()["experiments"] if not e["placed"]}
        if loose:
            c["experiments"].update(loose)
            canvas.save(self.runs_dir, c, load_runs(self.runs_dir))

    def create(self, body: dict) -> dict:
        xy = {a: body.get(a) for a in ("x", "y")}
        if any(v is not None and not _num(v) for v in xy.values()):
            raise EditError("x、y 应为数值")
        with self.lock:
            self._pin()
            rid = body.get("id") or next_id(self.runs_dir)
            create_experiment(self.runs_dir, rid, body.get("title") or "", body.get("parent") or None,
                              body.get("init") or None)
            if None in xy.values():   # 由 canvas.place 摆到父实验右侧、往下避开已有实验
                e = next(e for e in self.data()["experiments"] if e["id"] == rid)
                xy = {"x": e["x"], "y": e["y"]}
            c = canvas.load(self.runs_dir)
            c["experiments"][rid] = xy
            canvas.save(self.runs_dir, c, load_runs(self.runs_dir))
            return {"ok": True, "id": rid}

    def set_parent(self, rid: str, body: dict) -> dict:
        with self.lock:
            self._pin()
            set_parent(self.runs_dir, rid, body.get("parent"), body.get("init"))
            return {"ok": True}

    def save_notes(self, rid: str, body: dict) -> dict:
        with self.lock:
            save_notes(self.runs_dir, rid, body.get("text"))
            return {"ok": True}

    def set_title(self, rid: str, body: dict) -> dict:
        with self.lock:
            set_title(self.runs_dir, rid, str(body.get("title", "")))
            return {"ok": True}

    def rename(self, rid: str, body: dict) -> dict:
        with self.lock:
            new = str(body.get("id", ""))
            self._pin()
            rename_experiment(self.runs_dir, rid, new)
            c = canvas.load(self.runs_dir)
            if rid in c["experiments"]:
                c["experiments"][new] = c["experiments"].pop(rid)
            canvas.save(self.runs_dir, c, load_runs(self.runs_dir))
            return {"ok": True, "id": new}

    def delete(self, rid: str) -> dict:
        with self.lock:
            self._pin()
            delete_experiment(self.runs_dir, rid)
            canvas.save(self.runs_dir, canvas.load(self.runs_dir), load_runs(self.runs_dir))
            return {"ok": True}

    def save_canvas(self, body: dict) -> dict:
        with self.lock:
            self._pin()
            c = canvas.load(self.runs_dir)
            for k, v in (body.get("experiments") or {}).items():
                if not (isinstance(v, dict) and all(_num(v.get(a)) for a in ("x", "y"))):
                    raise EditError(f"{k} 的坐标应为 {{x, y}} 数值")
                c["experiments"][k] = v
            try:
                for key, clean in (("groups", canvas.clean_groups), ("stickies", canvas.clean_stickies)):
                    if key in body:
                        c[key] = clean(body[key])
            except ValueError as e:
                raise EditError(str(e)) from e
            canvas.save(self.runs_dir, c, load_runs(self.runs_dir))
            return {"ok": True}

    # ---------------- 影像 ----------------

    @lru_cache(maxsize=512)
    def image_png(self, split: str, pair: str, kind: str) -> bytes:
        from PIL import Image

        img = self.ds.optical(split, pair) if kind == "opt" else self.ds.sar(split, pair)
        buf = io.BytesIO()
        Image.fromarray(to_display(img)).save(buf, format="PNG", optimize=False, compress_level=3)
        return buf.getvalue()


def _public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


EXP_ACTION = re.compile(r"/api/exp/([^/]+)/(parent|title|rename)")
EXP_ITEM = re.compile(r"/api/exp/([^/]+)")
EXP_NOTES = re.compile(r"/api/exp/([^/]+)/notes")


def make_handler(wb: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=200):
            self.send(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                      "application/json; charset=utf-8", code)

        def send_file(self, f: Path):
            ctype = CTYPES.get(f.suffix) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") and "charset" not in ctype:
                ctype += "; charset=utf-8"
            self.send(f.read_bytes(), ctype)

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                d = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            except ValueError as e:
                raise EditError(f"请求体不是合法 JSON：{e}") from e
            if not isinstance(d, dict):
                raise EditError("请求体应为 JSON 对象")
            return d

        def route(self, method: str):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = unquote(u.path)
            if method == "GET":
                if path in ("/", "/index.html"):
                    return self.send_file(STATIC / "index.html")
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC.resolve() in f.parents and f.is_file():
                        return self.send_file(f)
                if path == "/api/data":
                    return self.send_json(wb.data())
                if m := EXP_ITEM.fullmatch(path):
                    return self.send_json(wb.detail(m.group(1)))
                if path == "/api/compare":
                    return self.send_json(wb.compare(q))
                if path == "/img":
                    if q.get("split") in SPLIT_DIRS and q.get("kind") in ("opt", "sar") \
                            and q.get("pair") in wb.ds.pairs(q["split"]):
                        return self.send(wb.image_png(q["split"], q["pair"], q["kind"]), "image/png")
                if path.startswith("/runs/"):
                    rid, _, rel = path[len("/runs/"):].partition("/")
                    d = wb.runs_dir / rid
                    if (d / "exp.toml").exists():
                        f = (d / rel).resolve()
                        if (d / "extra").resolve() in f.parents and f.is_file():
                            return self.send_file(f)
            elif method == "POST":
                if path == "/api/exp":
                    return self.send_json(wb.create(self.body()))
                if m := EXP_ACTION.fullmatch(path):
                    rid, act = m.groups()
                    return self.send_json({"parent": wb.set_parent, "title": wb.set_title,
                                           "rename": wb.rename}[act](rid, self.body()))
            elif method == "PUT":
                if path == "/api/canvas":
                    return self.send_json(wb.save_canvas(self.body()))
                if m := EXP_NOTES.fullmatch(path):
                    return self.send_json(wb.save_notes(m.group(1), self.body()))
            elif method == "DELETE":
                if m := EXP_ITEM.fullmatch(path):
                    return self.send_json(wb.delete(m.group(1)))
            self.send_json({"error": "not found"}, 404)

        def handle_method(self, method):
            try:
                self.route(method)
            except NotFound as e:
                self.send_json({"error": str(e)}, 404)
            except EditError as e:
                self.send_json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001 — 本地工具，把错误直接回给页面
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_GET(self):
            self.handle_method("GET")

        def do_POST(self):
            self.handle_method("POST")

        def do_PUT(self):
            self.handle_method("PUT")

        def do_DELETE(self):
            self.handle_method("DELETE")

    return Handler


def serve(runs_dir: Path, data_root: Path, repo: Path, host: str, port: int):
    wb = Workbench(runs_dir, Dataset(data_root), repo)
    httpd = ThreadingHTTPServer((host, port), make_handler(wb))
    print(f"工作台：http://{host}:{port}/  （{len(load_runs(runs_dir))} 个实验；runs/ 的改动几秒内自动出现在画布上）")
    httpd.serve_forever()
