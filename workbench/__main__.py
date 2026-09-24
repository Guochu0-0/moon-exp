"""python -m workbench {serve,eval,check,new} …"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO / "runs"
DATA_ENV = "MOON_DATA"  # 数据集根目录，例如 G:/Lunar_Optical_SAR_Registration_Dataset

EXP_TEMPLATE = '''id = "{id}"
title = "{title}"
parent = {parent}
# init = "B0/roma"      # 可选：从父实验的哪个方法起步；父实验有多个方法时用来定比较基准
date = "{date}"
commit = "{commit}"
status = "running"      # baseline / running / kept / dropped

hypothesis = """
"""

change = """
"""

# 结论只依据 Val
verdict = """
"""

next = """
"""
'''


def _data_root(args) -> Path:
    root = args.data or os.environ.get(DATA_ENV)
    if not root:
        sys.exit(f"需要数据集根目录：--data 或环境变量 {DATA_ENV}")
    return Path(root)


def cmd_serve(args):
    from .server import serve

    serve(Path(args.runs), _data_root(args), REPO, args.host, args.port)


def cmd_eval(args):
    from .dataset import Dataset
    from .evaluate import evaluate_run, write_metrics
    from .records import load_runs

    ds = Dataset(_data_root(args))
    runs = load_runs(Path(args.runs))
    for rid in args.ids or list(runs):
        m = evaluate_run(ds, runs[rid])
        path = write_metrics(runs[rid], m)
        main = m["protocol"]["main"]
        for meth, splits in m["methods"].items():
            line = "  ".join(f"{s} {main}={v['summary'][main]:.3f} 缺{v['coverage']['missing']}"
                             for s, v in splits.items())
            print(f"{rid}/{meth}: {line}")
        print(f"  → {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")


def cmd_check(args):
    from .records import check_runs, load_runs

    problems = check_runs(load_runs(Path(args.runs)))
    for p in problems:
        print(p)
    print("记录无问题" if not problems else f"{len(problems)} 个问题")
    sys.exit(1 if problems else 0)


def cmd_new(args):
    d = Path(args.runs) / args.id
    if d.exists():
        sys.exit(f"{d} 已存在")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (d / "extra").mkdir(parents=True)
    (d / "exp.toml").write_text(EXP_TEMPLATE.format(
        id=args.id, title=args.title, parent=f'"{args.parent}"' if args.parent else '""',
        date=datetime.date.today().isoformat(), commit=commit), encoding="utf-8")
    print(f"已建 {d}/exp.toml（commit {commit}）")


def main():
    ap = argparse.ArgumentParser(prog="python -m workbench")
    ap.add_argument("--runs", default=str(DEFAULT_RUNS), help="实验记录目录（默认 runs/）")
    ap.add_argument("--data", help=f"数据集根目录（默认读环境变量 {DATA_ENV}）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="启动工作台页面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("eval", help="按评价协议算指标，写 runs/<id>/metrics.json")
    p.add_argument("ids", nargs="*")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("check", help="检查实验记录的一致性")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("new", help="新建一个实验目录")
    p.add_argument("id")
    p.add_argument("--parent")
    p.add_argument("--title", default="")
    p.set_defaults(fn=cmd_new)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
