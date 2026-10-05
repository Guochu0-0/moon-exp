"""python -m workbench {serve,eval,check,new,sync} …"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO / "runs"
DATA_ENV = "MOON_DATA"  # 数据集根目录，例如 G:/Lunar_Optical_SAR_Registration_Dataset


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
        if runs[rid].analysis:
            print(f"{rid}: 分析没有预测，跳过")
            continue
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

    problems, warnings = check_runs(load_runs(Path(args.runs)))
    for w in warnings:
        print(f"警告 {w}")
    for p in problems:
        print(p)
    print("记录无问题" if not problems else f"{len(problems)} 个问题")
    sys.exit(1 if problems else 0)


def cmd_new(args):
    from .records import EditError, create_experiment

    try:
        d = create_experiment(Path(args.runs), args.id, args.title, args.parent, args.init, args.analysis)
    except EditError as e:
        sys.exit(str(e))
    print(f"已建 {d}/exp.toml")


def cmd_sync(args):
    from . import sync

    r = sync.sync(args.ids, Path(args.runs), host=args.host, remote_root=args.remote_root,
                  extras=args.extra, tb_all=args.tb == "all")
    for rid in r.missing:
        print(f"远端没有 runs/{rid}/")
    print(f"拉取 {len(r.fetched)} 个文件（{r.fetched_bytes / 2**20:.1f} MB），跳过 {len(r.skipped)} 个已有且未变的文件")
    if r.missing:
        sys.exit(1)


def main(argv=None):
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
    p.add_argument("--init", help="<父实验>/<方法>")
    p.add_argument("--title", default="")
    p.add_argument("--analysis", action="store_true", help="建分析：不训练、没有预测，只有 Notes 和附件")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("sync", help="从服务器按实验拉回点对、中间结果、TB 日志（ckpt 永不拉）")
    p.add_argument("ids", nargs="+")
    p.add_argument("--extra", action="append", default=[], metavar="NAME", help="要拉的中间结果名，可重复")
    p.add_argument("--tb", choices=("all",), help="默认只拉 TB scalars；all 拉整个 tb/")
    p.add_argument("--host", help="ssh 主机别名（默认读环境变量 MOON_SYNC_HOST，再默认 xufang154外网）")
    p.add_argument("--remote-root", help="远端仓库根目录（默认读 MOON_SYNC_ROOT，再默认服务器上的 YGC/moon-exp）")
    p.set_defaults(fn=cmd_sync)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
