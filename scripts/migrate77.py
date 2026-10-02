"""一次性：把 results/finetune/ 下的旧训练产物迁进 runs/<id>/（#77「历史代码与结果整理」）。

    /opt/envs/wb/bin/python scripts/migrate77.py            # dry-run：只打印计划
    /opt/envs/wb/bin/python scripts/migrate77.py --apply    # 执行

在本票 worktree 里运行；产物挪到 worktree 的 runs/，结票时由 wt.sh close 挪回主 checkout。
每个方法的旧目录 results/finetune/<src>/ 按 scripts/finetune/run.py 的布局拆开：

  ckpt/<m>/          ckpt_*.pt、log.jsonl、args.json、train.log、sweep.log、test.log 等，及 <src>.driver.log
  sweep/<m>/S|match|neg、peak.json、collapse.json
  preds/<m>/         选中 step 的预测；git 里已有的不覆盖，只核对并回填 meta 的 commit

ckpt 只留峰值与最后一个，其余挪进 YGC/_archive/2026-10/ckpt/<id>/<m>/。全部是 rename（同一 gpfs），不删文件。
preds 的 meta 里 commit 按 #77 的 blob 比对回填，标注「事后重建」。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

Y = Path("/remote-home/xufang/YGC")
OLD = Y / "results/finetune"
A6K = Y / "_archive/2026-10/a6000-finetune"   # A6000 上的 Q8 Q9 Q10 Q12 Q12p，原样拷来
ARC = Y / "_archive/2026-10"
REPO = Path(__file__).resolve().parents[1]
H150 = "bcb626e99cea"   # 150 的容器 hostname
# 150 时钟慢 13h07m：这样换算后 M7–M10 的占位是 12:08:22，紧接加入它们的 1cd5d54（12:08:20），且 M8 用了 b7246eb 才有的 --cert-out
LAG150 = 13 * 3600 + 7 * 60

# 训练代码的提交（#77 blob 比对）
C_FT49 = "ea09457"   # moon-exp-ft49 副本：C 系
C_FT51 = "b21bc77"   # moon-exp-ft51 副本：P1–P10、Q0–Q7（任务清单就地改过，见 runs/P|Q/code/）
C_R2 = "840927b"     # moon-exp-r2（gpfs）：P11–P16、Q11、Q11p、Q13
C_R2A = "5a08a30"    # moon-exp-r2（A6000）：Q8 Q9 Q10 Q12 Q12p
ROMA = [("8457109", "2026-09-30 22:21:19"), ("015aeb4", "2026-09-30 23:04:08"), ("b7246eb", "2026-10-01 12:06:36")]

S_MAP = {"S1": ("S1", "main"), "S1s1": ("S1", "seed1"), "S1s2": ("S1", "seed2"), "S2": ("S2", "main")}


def methods():
    """(源目录, 实验 id, 方法名)"""
    out = []
    for d in sorted(OLD.iterdir()):
        if not d.is_dir():
            continue
        if re.fullmatch(r"[CPQM]\d+[a-z0-9]*", d.name):
            out.append((d, d.name[0], d.name))
        elif d.name in S_MAP:
            out.append((d, *S_MAP[d.name]))
    if A6K.exists():
        out += [(d, "Q", d.name) for d in sorted(A6K.iterdir()) if d.is_dir()]
    return out


def claim_time(m):
    """M 系：领取时间（北京时间）。占位文件只记了时分秒，日期取文件 mtime；150 按 LAG150 校正。"""
    h = OLD / "_claims" / m / "host"
    host = h.read_text().split()[0]
    t = h.stat().st_mtime + (LAG150 if host == H150 else 0)
    return host, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def commit_for(exp, m, src):
    n = int(re.match(r"[A-Z](\d+)", m).group(1)) if exp in "CPQM" else None
    if exp == "C":
        return C_FT49, "moon-exp-ft49 副本的代码与此提交逐文件一致"
    if exp == "P":
        return (C_FT51, "moon-exp-ft51 副本") if n <= 10 else (C_R2, "moon-exp-r2 副本（gpfs）")
    if exp == "Q":
        if src.parent == A6K:
            return C_R2A, "A6000 上 moon-exp-r2 副本的代码与此提交逐文件一致"
        return (C_FT51, "moon-exp-ft51 副本") if n <= 7 else (C_R2, "moon-exp-r2 副本（gpfs）")
    if exp == "M":
        host, t = claim_time(m)
        c = [c for c, ct in ROMA if ct <= t][-1]
        return c, f"moon-exp-roma 副本被就地覆盖过；按领取时间 {t}（{host}）取此前最后一个改过训练代码的提交"
    return None, None


def step_of(p):
    return int(re.search(r"(\d+)$", p.stem if p.suffix else p.name).group(1))


def peak_of(src, sw_S):
    if (src / "peak.json").exists():
        return json.loads((src / "peak.json").read_text())["step"], "peak.json"
    ms = json.loads((sw_S / "metrics.json").read_text(encoding="utf-8"))["methods"]
    pts = [(int(k[4:]), v["val"]["summary"]["auc@5"]) for k, v in ms.items()
           if k.startswith("step") and "val" in v and int(k[4:]) > 0]
    return max(pts, key=lambda p: p[1])[0], "metrics.json"


class Plan:
    def __init__(self, apply):
        self.apply, self.n = apply, 0

    def mv(self, a: Path, b: Path):
        if b.exists():
            raise SystemExit(f"目标已存在：{b}")
        self.n += 1
        print(f"  mv {a} -> {b}")
        if self.apply:
            b.parent.mkdir(parents=True, exist_ok=True)
            os.rename(a, b)

    def write(self, p: Path, text: str):
        print(f"  write {p}")
        if self.apply:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

    def cp(self, a: Path, b: Path):
        print(f"  cp {a} -> {b}")
        if self.apply:
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a, b)


def migrate(P: Plan, src: Path, exp: str, m: str, warn: list):
    R = REPO / "runs" / exp
    ck, sw = R / "ckpt" / m, R / "sweep" / m
    S_old = src / "sweep" / "S"
    st, how = peak_of(src, S_old)
    ckpts = sorted(src.glob("ckpt_*.pt"), key=step_of)
    last = step_of(ckpts[-1])
    keep = {st, last}
    print(f"[{exp}/{m}] <- {src}  peak={st}（{how}） last={last}  ckpt {len(ckpts)} 个，留 {sorted(keep)}")

    # 选中 step 的 Val / Test 是否齐
    sp = S_old / "preds" / f"step{st}"
    splits = [s for s in ("val", "test") if (sp / f"{s}.jsonl").exists()]
    if "val" not in splits:
        warn.append(f"{exp}/{m}: 峰值 step{st} 没有 val 预测")
    if "test" not in splits:
        tested = sorted(step_of(p.parent) for p in (S_old / "preds").glob("step*/test.jsonl"))
        warn.append(f"{exp}/{m}: 峰值 step{st} 没有 Test" + (f"（Test 评在 {tested}）" if tested else "（未评 Test）"))

    # 1. preds：先算出要写的内容（从旧位置读），再挪目录
    dst = R / "preds" / m
    c, basis = commit_for(exp, m, src)
    full = None
    if c:
        full = os.popen(f"git -C {REPO} rev-parse {c}").read().strip()
    meta = {"commit": full, "dirty": None, "reconstructed": f"事后重建（#77）：{basis}"} if c else None
    for s in splits:
        if (dst / f"{s}.jsonl").exists():
            same = (dst / f"{s}.jsonl").read_bytes() == (sp / f"{s}.jsonl").read_bytes()
            if not same:
                warn.append(f"{exp}/{m}: git 里的 preds/{s}.jsonl 与 step{st} 不一致，未覆盖")
        else:
            for f in sp.glob(f"{s}*"):
                if not f.name.endswith(".meta.json"):
                    P.cp(f, dst / f.name)
        old = json.loads((dst / f"{s}.meta.json").read_text()) if (dst / f"{s}.meta.json").exists() else {}
        if meta and not old.get("commit"):
            P.write(dst / f"{s}.meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    # 2. sweep（peak.json 要读 S/metrics.json，先算好再挪）
    pk = None
    if not (src / "peak.json").exists():
        ms = json.loads((S_old / "metrics.json").read_text(encoding="utf-8"))["methods"][f"step{st}"]
        pk = {"step": st, "metric": "val auc@5", "val": ms["val"]["summary"]}
        if "test" in ms:
            pk["test"] = ms["test"]["summary"]
    for name in ("S", "match", "neg"):
        if (src / "sweep" / name).exists():
            P.mv(src / "sweep" / name, sw / name)
        elif (src / name).exists():
            P.mv(src / name, sw / name)
    for name in ("peak.json", "collapse.json"):
        if (src / name).exists():
            P.mv(src / name, sw / name)
    if pk:
        P.write(sw / "peak.json", json.dumps(pk, indent=1) + "\n")
    rest = [p for p in (src / "sweep").iterdir() if p.name not in ("S", "match", "neg")] if (src / "sweep").exists() else []
    for p in rest:
        P.mv(p, sw / p.name)

    # 3. ckpt：留峰值与最后一个，其余进归档
    for p in ckpts:
        P.mv(p, (ck if step_of(p) in keep else ARC / "ckpt" / exp / m) / p.name)
    for p in sorted(src.iterdir()):
        if p.name in ("sweep", "match", "neg", "peak.json", "collapse.json") or p.name.startswith("ckpt_"):
            continue
        P.mv(p, ck / p.name)
    drv = src.parent / f"{src.name}.driver.log"
    if drv.exists():
        P.mv(drv, ck / "driver.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", help="只处理这些源目录名")
    a = ap.parse_args()
    assert os.stat(OLD).st_dev == os.stat(REPO).st_dev, "源与 worktree 不在同一文件系统"
    P, warn = Plan(a.apply), []
    for src, exp, m in methods():
        if a.only and src.name not in a.only:
            continue
        migrate(P, src, exp, m, warn)
    print(f"\n共 {P.n} 次 mv{'（已执行）' if a.apply else '（dry-run）'}")
    for w in warn:
        print("WARN", w)


if __name__ == "__main__":
    main()
