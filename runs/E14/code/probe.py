"""对照（「【RoMa·RL】对照：粗级概率很平，是 RoMa 本身的特性还是不适应月球数据」#179）：只读模型，不训练。

    python runs/E14/code/probe.py [--sets moon moon_hard ...] [--n 200]

复用 runs/E12/code/probe.py 的第一部分（概率形状）：每对算粗级 gm_cls 的 softmax（40×40 格，每格 4096 个锚点），
逐格统计熵、mode 概率；温度 1 每格抽 group 份锚点，统计与 mode 不同的比例。格分「全部」与「有把握」两类：mode 那份
细化到尺度 1 的 certainty（sigmoid）在格内均值 > 0.5 的为有把握的格子（同 E12）。

各组（SETS）：模型 × 数据 × 扰动。
- moon / moon_val：zero-shot，月球光学对 SAR，Train / Val 均匀取 n 对，无扰动；
- moon_hard：zero-shot，同 moon 的 n 对，E12 同款扰动（难版几何 + 擦除，同一组随机数），复现 E12 作核对；
- optopt_hard：zero-shot，光学对它的扰动副本（与 moon_hard 同一组几何，不擦除）；
- mega / mega_gray：zero-shot，MegaDepth-1500 均匀取 n 对，RGB / 灰度复制三通道，直接缩放到 560×560（不保长宽比）；
- rns_r1 / rns_r2（_val）：E9 的 RoMa 伪标签 noisy student 第 1、2 轮在 Val 上选出的存档，月球 Train / Val，无扰动。

产物：runs/E14/extra/data/pairs.csv（逐对汇总）、summary.json（逐组的格统计与熵直方图）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402

E9 = "/remote-home/xufang/YGC/moon-exp/runs/E9"
CKPT = {"zs": "", "rns_r1": f"{E9}/ckpt/rns_r1/ckpt_7000.pt", "rns_r2": f"{E9}/ckpt/rns_r2/ckpt_6000.pt"}   # sweep/*/peak.json
MEGA = "/remote-home/xufang/YGC/dataset/megadepth_test_1500/phoenix/S6/zl548/MegaDepth_v1"
MEGA_INFO = REPO / "third_party/LoFTR/assets/megadepth_test_1500_scene_info"
HARD = {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False}
SETS = {   # 名 → (模型, 数据, split, 扰动)
    "moon": ("zs", "moon", "train", None),
    "moon_hard": ("zs", "moon", "train", {**HARD, "erase": True}),
    "optopt_hard": ("zs", "optopt", "train", {**HARD, "erase": False}),
    "mega": ("zs", "mega", None, None),
    "mega_gray": ("zs", "mega_gray", None, None),
    "moon_val": ("zs", "moon", "val", None),
    "rns_r1": ("rns_r1", "moon", "train", None),
    "rns_r1_val": ("rns_r1", "moon", "val", None),
    "rns_r2": ("rns_r2", "moon", "train", None),
    "rns_r2_val": ("rns_r2", "moon", "val", None),
}
BINS = np.linspace(0, np.log(4096), 84)


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, np.float64)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} | {"mean": round(float(x.mean()), 4)}


def load_model(name, weights_root):
    cfg = C.expand({"model": {"name": "roma", "init": CKPT[name]}})
    return roma.Model(cfg, weights_root)


def moon_set(model, data, split, pert, n):
    """月球对：PairSet 照 E12 取样（Train 每 len // n 取一对），扰动用同一组随机数（seed 0、主进程 torch 种子 0）。
    optopt 把 SAR 换成同一张光学图（同一映射），再照常施加几何扰动。"""
    kw = aug.data_kw(C.expand({"model": {"name": "roma"}, "aug": pert, "rloo": {}})["aug"]) if pert else {}
    ds = PairSet(data, split, model.resize, **model.input, seed=0, **kw)
    ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // n)][:n]
    return ds


class OptOpt(PairSet):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.data.sar, self.map_sar = self.data.optical, self.map_opt


def mega_pairs(n):
    """MegaDepth-1500：按 LoFTR 的 megadepth_test_1500.txt 顺序拼起 5 个场景文件的 1500 对，均匀取 n 对。"""
    out = []
    for stem in (MEGA_INFO / "megadepth_test_1500.txt").read_text().split():
        d = np.load(MEGA_INFO / f"{stem}.npz", allow_pickle=True)
        for (i0, i1), *_ in d["pair_infos"]:
            out.append((stem, d["image_paths"][i0], d["image_paths"][i1]))
    idx = np.linspace(0, len(out) - 1, n).round().astype(int)
    return [out[i] for i in idx]


def mega_image(rel, gray):
    scene, name = rel.split("/")[1], rel.split("/")[-1]
    hits = sorted(Path(MEGA, scene).glob(f"dense*/imgs/{name}"))
    if not hits:
        raise FileNotFoundError(f"{MEGA}/{scene}/dense*/imgs/{name}")
    im = Image.open(hits[0]).convert("L" if gray else "RGB").resize((roma.RES, roma.RES), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
    return (x[None] if gray else x.permute(2, 0, 1))[None]


def items(model, data_kind, split, pert, n, data_root):
    if data_kind in ("mega", "mega_gray"):
        for stem, a, b in mega_pairs(n):
            g = data_kind == "mega_gray"
            yield f"{stem}/{Path(a).stem}-{Path(b).stem}", mega_image(a, g), mega_image(b, g)
        return
    if data_kind == "optopt":
        kw = aug.data_kw(C.expand({"model": {"name": "roma"}, "aug": pert, "rloo": {}})["aug"])
        ds = OptOpt(data_root, split, model.resize, **model.input, seed=0, **kw)
        ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // n)][:n]
    else:
        ds = moon_set(model, data_root, split, pert, n)
    for i in range(len(ds)):
        it = ds[i]
        yield it["pair"], it["image0"][None], it["image1"][None]


def probe_pair(model, i0, i1, group):
    f1, f2 = rloo.features(model, i0, i1)
    cls, cert = rloo.coarse(model, f1, f2)
    cls = cls.float()
    C_, h, w = cls.shape[1:]
    P = cls.softmax(1)
    q, logq = P[0], cls[0].log_softmax(0)
    mode = q.argmax(0)
    _, m_ce = rloo.rollout(model, f1, f2, P, cert.float(), mode[None])
    conf = torch.nn.functional.adaptive_avg_pool2d(m_ce.sigmoid(), (h, w))[0, 0] > 0.5
    k = torch.multinomial(q.reshape(C_, -1).T, group, replacement=True).T.reshape(-1, h, w)
    H = -(q * logq).sum(0)
    return {"H": H, "pmode": q.max(0).values, "nonmode": (k != mode).float().mean(0), "conf": conf}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--group", type=int, default=16)
    a = ap.parse_args(argv)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows, summ, models = [], {}, {}
    for name in a.sets:
        mname, dkind, split, pert = SETS[name]
        if mname not in models:
            models.clear()
            torch.cuda.empty_cache()
            models[mname] = load_model(mname, a.weights_root)
        model = models[mname]
        torch.manual_seed(0)                          # 同 E12：PairSet 的扰动随机数取主进程 torch 种子
        cells = {k: [] for k in ("H", "pmode", "nonmode", "conf")}
        t0 = time.time()
        with torch.no_grad():
            for pair, i0, i1 in items(model, dkind, split, pert, a.n, a.data):
                c = probe_pair(model, i0.to("cuda"), i1.to("cuda"), a.group)
                cf = c["conf"]
                for k in cells:
                    cells[k].append(c[k].float().reshape(-1).cpu().numpy())
                rows.append({"set": name, "pair": pair, "H": float(c["H"].mean()), "H_med": float(c["H"].median()),
                             "pmode": float(c["pmode"].mean()), "nonmode": float(c["nonmode"].mean()),
                             "conf_frac": float(cf.float().mean()),
                             "H_conf": float(c["H"][cf].mean()) if cf.any() else np.nan})
        cat = {k: np.concatenate(v) for k, v in cells.items()}
        m = cat["conf"] > 0.5
        summ[name] = {"model": mname, "data": dkind, "split": split, "perturb": pert, "n_pairs": len(cells["H"]),
                      "cells_conf_frac": round(float(m.mean()), 4), "sec": round(time.time() - t0, 1),
                      "cell": {sub: {k: pct(cat[k][s]) for k in ("H", "pmode", "nonmode")} | {"n": int(s.sum())}
                               for sub, s in (("all", np.ones_like(m)), ("conf", m))},
                      "H_hist": {sub: np.histogram(cat["H"][s], BINS)[0].tolist() for sub, s in (("all", np.ones_like(m)),
                                                                                                ("conf", m))}}
        print(name, json.dumps(summ[name]["cell"]["all"]["H"]), summ[name]["cells_conf_frac"], flush=True)
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summ = {"group": a.group, "H_bins": BINS.round(4).tolist(), "uniform_entropy": round(float(np.log(4096)), 4),
            "ckpt": CKPT, "sets": summ}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
