"""给旧实验补写启动记录 runs/<id>/launch/<方法>.json（「保全旧实验出处」#85）。一次性脚本，留作出处。

    python scripts/backfill_launch85.py <gpfs 拷贝目录> [--write]

<gpfs 拷贝目录> 是从 gpfs 只读拷回本地的 YGC/ 子集（相对路径不变）：
  moon-exp/runs/<id>/ckpt/<m>/args.json        训练时写下的参数（C、M、P、Q、S1、S2）
  _archive/2026-10/results/finetune/<run>/args.json   R 系训练时写下的参数
  results/baselines/<m>/<split>.meta.json      B0 匹配时写下的 meta（配置、命令行、commit、起止时间、环境）
  results/baselines_ablation/<m>/val.meta.json、_cfg/<m>.json   B0m 同上
  _archive/migrate77_apply.log                 #77 迁移日志（原位置）
不加 --write 只打印汇总与要打 tag 的 commit；加了写出启动记录和 scripts/tag_exp85.sh。

每个方法一个列表，与 workbench.launch 写的启动记录同构（一次启动一条），另加：
  backfilled      补记日期；有这个字段即为事后补记，不是启动时自动写下的
  reliability     commit 的可靠程度：跑时记录 / 跑时记录但 dirty / 事后补记 / 推断 / 未知
  tag             commit 不在 main 上时，保全它的 tag 名
  entry           入口（模块或脚本）
  args            完整参数（训练脚本的 argparse 结果，或基线的匹配参数）；查不到为 null
  args_source     参数从哪里来
  config          模型配置（configs/baselines/*.json 的内容）
  origin          产物原来在哪里
  env             运行环境（主机、GPU、各库版本），有记录时才有
  notes           补充说明（几处来源不一致等）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
Y = "/remote-home/xufang/YGC"
TODAY = "2026-10-03"

# runs/<id> 的方法 → 归档目录 _archive/2026-10/results/finetune/<run>（R 系；依据 exp.toml 显示名与旧 notes）
R_RUNS = {
    "R": {"pair": "R1", "pair_match": "R2", "match": "R3", "pair_x4": "R4", "sigg05": "R5", "cfog": "R6",
          "l2sp_pure": "R7", "K8": "R8", "l2sp": "R9", "from_b0": "R10", "small_sigma": "R11",
          "fine_only": "R19", "fine_accum": "R20", "pair_x4_accum": "R21"},
    "R26": {"main": "R26", "seed1": "R35", "pairs256": "R36", "placebo": "R34"},
    "R27": {"main": "R27", "seed1": "R27s1", "seed2": "R27s2", "placebo": "R27p", "lr1e-4": "R37",
            "steps8000": "R38", "K8": "R39", "sigg05": "R40", "cfog": "R41", "match": "R44", "from_b0": "R42",
            "on_S1s1": "R27b1", "on_S1s2": "R27b2"},
    "R28": {"main": "R28", "placebo": "R28p", "noaccum": "R29", "noaccum_sigg05": "R31", "noaccum_K8": "R32",
            "noaccum_cfog": "R33", "single_stage": "R43"},
}
COARSE = ("B0c", "S1c", "Cc", "Qc", "Pc")   # #70 的粗级单独评估：只匹配 + 拟合，不训练


def git(*a) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip() if r.returncode == 0 else None


def full(c: str | None) -> str | None:
    return git("rev-parse", "--verify", "-q", f"{c}^{{commit}}") if c else None


def on_main(c: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", c, "origin/main"],
                          capture_output=True).returncode == 0


def split_commit(raw) -> tuple[str | None, bool | None, str | None]:
    """meta / args 里的 commit 字段 → (完整 sha, dirty, 原文)。原文可能是 `<sha>-dirty`，
    也可能是导出副本里的标签（短 sha + 换行 + 日期）。"""
    if not raw:
        return None, None, None
    s = str(raw).strip()
    dirty = s.endswith("-dirty")
    head = s.removesuffix("-dirty").split()[0]
    return full(head), dirty, s


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def preds_meta(rid: str, m: str) -> dict:
    """val 优先；没有 val 时取 test。"""
    for s in ("val", "test"):
        d = load(RUNS / rid / "preds" / m / f"{s}.meta.json")
        if d is not None:
            return d
    return {}


def origins(G: Path) -> dict[str, str]:
    """#77 迁移日志里的 `[<id>/<方法>] <- <原位置>`。"""
    out = {}
    log = G / "_archive" / "migrate77_apply.log"
    for ln in log.read_text(encoding="utf-8").splitlines() if log.exists() else []:
        if m := re.match(r"\[(\S+?)/(\S+?)\] <- (\S+)", ln):
            out[f"{m[1]}/{m[2]}"] = m[3]
    return out


def base(commit, reliability, dirty=None) -> dict:
    return {"backfilled": TODAY, "reliability": reliability, "commit": commit, "dirty": dirty}


def from_meta_commit(meta: dict) -> tuple[str | None, str, list[str]]:
    """preds meta 的 commit 与可靠程度。#77 事后重建的带 reconstructed，按其说明定「事后补记」或「推断」。"""
    c, notes = full(meta.get("commit")), []
    if not c:
        return None, "未知", notes
    if rec := meta.get("reconstructed"):
        notes.append(f"预测文件的 commit 是事后重建的：{rec}")
        return c, ("推断" if "取此前最后一个" in rec else "事后补记"), notes
    if mf := meta.get("migrated_from"):
        if "run 级" in mf:
            notes.append(f"预测文件的 commit 来自旧格式里整个实验共用的一个 commit（{mf}）")
            return c, "推断", notes
    return c, ("跑时记录但 dirty" if meta.get("dirty") else "跑时记录"), notes


# ---------------- 训练类：C、M、P、Q、S1、S2、R 系 ----------------
def training(rid, m, args_json: Path | None, origin: str | None) -> list[dict]:
    meta = preds_meta(rid, m)
    a = load(args_json) if args_json else None
    mc, mrel, notes = from_meta_commit(meta)
    ac, adirty, araw = split_commit(a.get("commit")) if a else (None, None, None)
    exact = araw is not None and re.fullmatch(r"[0-9a-f]{40}(-dirty)?", araw) is not None
    if exact and ac:
        # 训练时从 git checkout 取到的完整 sha：比预测文件里的事后 commit 可靠
        commit, rel, dirty = ac, ("跑时记录但 dirty" if adirty else "跑时记录"), adirty
        notes = [] if mc == ac else [f"预测文件里记的是 {mc[:10] if mc else '未知'}（{mrel}），与训练时记下的不同；以训练时记下的为准"]
    else:
        commit, rel, dirty = mc, mrel, None
        if araw:
            notes.append(f"训练时记下的 commit 标签是「{araw.splitlines()[0]}」"
                         f"{'（另附时间 ' + araw.splitlines()[1] + '）' if len(araw.splitlines()) > 1 else ''}："
                         "当时从导出副本运行，标签取自副本的来源 commit，副本之后可能被改过；"
                         "以按文件内容比对得到的 commit 为准")
        elif a is not None:
            notes.append("训练时没有记下 commit")
    rec = base(commit, rel, dirty)
    if a is None:
        rec.update(entry=None, args=None, args_source="未找到训练时写下的参数文件", origin=origin, notes=notes)
        return [rec]
    cfg = a.get("config") or {}
    roma = "roma" in str(a["args"].get("config", ""))
    env = a.get("env") or {}
    rec.update(entry="python -m finetune.train_roma" if roma else "python -m finetune.train",
               args=a["args"], args_source="训练时写下的 args.json（argparse 展开默认值后的全部参数）",
               config=cfg, init_weights=a.get("weights"), n_train=a.get("n_train"),
               origin=origin, host=env.get("host"), env=env or None, notes=notes)
    return [rec]


# ---------------- 基线：B0、B0m ----------------
def baseline(rid, m, G: Path, raw_dir: Path, cfg_path: Path | None = None) -> list[dict]:
    out = []
    fit = {}
    for s in ("val", "test"):
        if (d := load(RUNS / rid / "preds" / m / f"{s}.meta.json")) is not None:
            fit[s] = d
    for s in ("val", "test"):
        raw = load(raw_dir / f"{s}.meta.json")
        if raw is None or s not in fit:
            continue
        c, dirty, txt = split_commit(raw.get("commit"))
        fc, fdirty = full(fit[s].get("commit")), fit[s].get("dirty")
        notes = []
        if c:
            rel = "跑时记录但 dirty" if dirty else "跑时记录"
        else:
            rel = "未知"
            notes.append("匹配时没有记下 commit（当时取 commit 超时，23d65aa 修复）")
        notes.append(f"由点对拟合仿射（baselines.fit）时的 commit："
                     f"{fc[:10] + ('（dirty）' if fdirty else '') if fc else '未知'}")
        if raw.get("repo_commit"):
            notes.append(f"匹配器上游代码（third_party 子模块）的 commit：{raw['repo_commit']}")
        if cfg_path is not None:
            notes.append(f"配置由 scripts/baselines/ablate.sh 从 {raw.get('config', {}).get('method', m).split('__')[0]} "
                         "的配置派生，只改 SAR 输入映射")
        argv = raw.get("argv") or []
        rec = base(c, rel, dirty)
        rec.update(split=s, entry="python -m baselines.match，再 python -m baselines.fit",
                   cmd=["python", "-m", "baselines.match", *argv[1:]] if argv else None,
                   args={"split": s, "weights": raw.get("weights"), "weights_sha256": raw.get("weights_sha256"),
                         "config_path": raw.get("config_path"), **(raw.get("config") or {})},
                   args_source="匹配时写下的 meta.json（完整配置）",
                   config=raw.get("config"), origin=f"{Y}/{raw_dir.relative_to(G).as_posix()}",
                   host=(raw.get("env") or {}).get("host"), start=raw.get("started"), end=raw.get("finished"),
                   status="ok" if raw.get("finished") else None, env=raw.get("env"), notes=notes)
        out.append(rec)
    return out


# ---------------- #70 粗级单独评估：B0c、S1c、Cc、Qc、Pc ----------------
COARSE_CKPT = {"anymatch_loftr": None, "main": "results/finetune/S1/ckpt_6000.pt", "C2": "results/finetune/C2/ckpt_500.pt",
               "Q4": "results/finetune/Q4/ckpt_1500.pt", "Q4s1": "results/finetune/Q4s1/ckpt_1500.pt",
               "P8": "results/finetune/P8/ckpt_8000.pt"}


def coarse(rid, m) -> list[dict]:
    meta = preds_meta(rid, m)
    c, rel, notes = from_meta_commit(meta)
    src, t = m.rsplit("_r", 1)
    ck = COARSE_CKPT[src]
    cfg_txt = git("show", f"{c}:configs/baselines/anymatch_loftr_coarse.json") if c else None
    rec = base(c, rel)
    rec.update(entry="python -m baselines.match（只用粗级匹配），再 python -m baselines.fit --ransac <阈值>",
               args={"config_path": "configs/baselines/anymatch_loftr_coarse.json",
                     "weights": f"{Y}/{ck}" if ck else "配置里的 zero-shot 权重",
                     "ransac": float(t), "name": f"{src}_r{t}"},
               args_source=f"任务脚本 runs/B0c/code/common.sh、fit.sh（与 {c[:10] if c else '?'} 一起提交）",
               config=json.loads(cfg_txt) if cfg_txt else None,
               origin=f"{Y}/results/coarse70/{src}-coarse（现归档在 {Y}/_archive/2026-10/results/coarse70/）",
               notes=notes + ["只评测、不训练；同一份粗级点对分别以 RANSAC 3 px 与 8 px 拟合"])
    return [rec]


def methods(rid):
    return sorted(p.name for p in (RUNS / rid / "preds").iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpfs", type=Path)
    ap.add_argument("--write", action="store_true")
    o = ap.parse_args()
    G, org = o.gpfs, origins(o.gpfs)
    recs: dict[tuple[str, str], list[dict]] = {}
    for rid in sorted(p.name for p in RUNS.iterdir() if (p / "exp.toml").exists()):
        for m in methods(rid):
            if rid == "B0":
                r = baseline(rid, m, G, G / "results" / "baselines" / m)
            elif rid == "B0m":
                r = baseline(rid, m, G, G / "results" / "baselines_ablation" / m, G / "results" / "baselines_ablation" / "_cfg" / f"{m}.json")
            elif rid in COARSE:
                r = coarse(rid, m)
            elif rid in R_RUNS:
                run = R_RUNS[rid][m]
                r = training(rid, m, G / "_archive" / "2026-10" / "results" / "finetune" / run / "args.json",
                             f"{Y}/results/finetune/{run}（现归档在 {Y}/_archive/2026-10/results/finetune/{run}/）")
            else:
                r = training(rid, m, G / "moon-exp" / "runs" / rid / "ckpt" / m / "args.json", org.get(f"{rid}/{m}"))
            if not r:
                print(f"!! {rid}/{m}: 没有可补记的内容", file=sys.stderr)
                continue
            recs[(rid, m)] = r
    # 不在 main 上的 commit → tag exp/<实验>-<短 commit>
    tags = {}
    for (rid, m), r in recs.items():
        for x in r:
            if x["commit"] and not on_main(x["commit"]):
                x["tag"] = f"exp/{rid}-{x['commit'][:7]}"
                tags[x["tag"]] = x["commit"]
    stat = {}
    for (rid, m), r in recs.items():
        for x in r:
            stat[(rid, x["reliability"], x["args"] is not None)] = stat.get((rid, x["reliability"], x["args"] is not None), 0) + 1
    for k, v in sorted(stat.items()):
        print(*k, v)
    print(f"{len(recs)} 个方法，{sum(map(len, recs.values()))} 条记录；{len(tags)} 个 tag：")
    for t, c in sorted(tags.items()):
        print(f"  {t} {c}")
    if o.write:
        for (rid, m), r in recs.items():
            p = RUNS / rid / "launch" / f"{m}.json"
            if p.exists():
                old = json.loads(p.read_text(encoding="utf-8"))
                if any(not x.get("backfilled") for x in old):
                    print(f"!! {p} 已有启动时写下的记录，跳过", file=sys.stderr)
                    continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        sh = ["#!/bin/bash", "# 给旧实验用过、不在 main 上的 commit 打 tag 并推到 GitHub（#85）。由 scripts/backfill_launch85.py 生成。",
              "# 在本机主 checkout 里运行：其中几个 commit 不在任何分支上，只在本机的 git 对象库里。", "set -e"]
        sh += [f"git tag -a {t} {c} -m '{t.split('/')[1].rsplit('-', 1)[0]} 的方法用过的 commit，不在 main 上（#85）'"
               for t, c in sorted(tags.items())]
        sh += ["git push origin " + " ".join(f"refs/tags/{t}" for t in sorted(tags))]
        (REPO / "scripts" / "tag_exp85.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
