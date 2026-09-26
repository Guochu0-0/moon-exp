"""本地服务：读 runs/ 与本地数据集，提供思路树页面、指标数据和按需渲染的影像。"""
from __future__ import annotations

import io
import json
import mimetypes
import subprocess
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import protocol
from .dataset import SPLIT_DIRS, Dataset, to_display
from .evaluate import evaluate_identity, evaluate_run, jsonable
from .records import SPLITS, load_runs

STATIC = Path(__file__).parent / "static"


def git_info(repo: Path, commit: str | None, parent_commit: str | None) -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None

    if not commit:
        return {}
    out = {"subject": git("show", "-s", "--format=%s", commit)}
    if parent_commit and parent_commit != commit:
        out["diffstat"] = git("diff", "--stat=100", parent_commit, commit)
    return out


class Workbench:
    def __init__(self, runs_dir: Path, ds: Dataset, repo: Path):
        self.runs_dir, self.ds, self.repo = runs_dir, ds, repo
        self.reload()

    def reload(self):
        self.runs = load_runs(self.runs_dir)
        self.metrics = {rid: evaluate_run(self.ds, r) for rid, r in self.runs.items()}
        self.identity = evaluate_identity(self.ds)
        self._data = None

    def data(self) -> dict:
        if self._data is None:
            runs = []
            for rid, r in self.runs.items():
                pc = self.runs[r.parent].meta.get("commit") if r.parent in self.runs else None
                runs.append({
                    "id": rid,
                    "meta": r.meta,
                    "methods": r.methods,
                    "metrics": self.metrics[rid]["methods"],
                    "git": git_info(self.repo, r.meta.get("commit"), pc),
                    "extras": [p.relative_to(r.dir).as_posix() for p in r.extras()],
                })
            self._data = jsonable({
                "protocol": protocol.protocol_info(),
                "pairs": {s: self.ds.pairs(s) for s in SPLITS},
                "labelled": {s: self.ds.labelled(s) for s in SPLITS},
                "identity": self.identity,
                "runs": runs,
            })
        return self._data

    def pair(self, split: str, pair: str) -> dict:
        cp = self.ds.checkpoints(split, pair)
        preds = {}
        for rid, r in self.runs.items():
            for m in r.methods:
                p = r.preds(m, split).get(pair)
                if p is not None:
                    mt = r.matches(m, split, pair)
                    preds[f"{rid}/{m}"] = {"A": p.A, "fail": p.fail, "extra": p.extra,
                                           "matches": None if mt is None else np.round(mt, 2).tolist()}
        return jsonable({
            "split": split, "pair": pair,
            "opt_pts": None if cp is None else cp[0].tolist(),
            "sar_pts": None if cp is None else cp[1].tolist(),
            "preds": preds,
        })

    @lru_cache(maxsize=512)
    def image_png(self, split: str, pair: str, kind: str) -> bytes:
        from PIL import Image

        img = self.ds.optical(split, pair) if kind == "opt" else self.ds.sar(split, pair)
        buf = io.BytesIO()
        Image.fromarray(to_display(img)).save(buf, format="PNG", optimize=False, compress_level=3)
        return buf.getvalue()


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

        def send_json(self, obj):
            self.send(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                      "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = unquote(u.path)
            try:
                if path in ("/", "/index.html"):
                    return self.send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC.resolve() in f.parents and f.is_file():
                        return self.send(f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                if path == "/api/data":
                    return self.send_json(wb.data())
                if path == "/api/reload":
                    wb.reload()
                    return self.send_json({"ok": True})
                if path == "/api/pair":
                    return self.send_json(wb.pair(q["split"], q["pair"]))
                if path == "/img":
                    if q["split"] in SPLIT_DIRS and q["kind"] in ("opt", "sar") and q["pair"] in wb.ds.pairs(q["split"]):
                        return self.send(wb.image_png(q["split"], q["pair"], q["kind"]), "image/png")
                if path.startswith("/runs/"):
                    rid, _, rel = path[len("/runs/"):].partition("/")
                    r = wb.runs.get(rid)
                    if r:
                        f = (r.dir / rel).resolve()
                        if (r.dir / "extra").resolve() in f.parents and f.is_file():
                            ctype = mimetypes.guess_type(f.name)[0] or "text/plain"
                            if ctype.startswith("text/") or f.suffix in (".md", ".json"):
                                ctype = (ctype if ctype.startswith("text/") else "text/plain") + "; charset=utf-8"
                            return self.send(f.read_bytes(), ctype)
                self.send(b"not found", "text/plain", 404)
            except Exception as e:  # noqa: BLE001 — 本地工具，把错误直接回给页面
                self.send(f"{type(e).__name__}: {e}".encode("utf-8"), "text/plain; charset=utf-8", 500)

    return Handler


def serve(runs_dir: Path, data_root: Path, repo: Path, host: str, port: int):
    wb = Workbench(runs_dir, Dataset(data_root), repo)
    httpd = ThreadingHTTPServer((host, port), make_handler(wb))
    print(f"工作台：http://{host}:{port}/  （{len(wb.runs)} 个实验；改了 runs/ 后点页面上的「重新加载」）")
    httpd.serve_forever()
