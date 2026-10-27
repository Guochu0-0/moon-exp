"""E32 只用（#221）：定细化损失系数 w_f。只读，不训练、不更新参数。

AdamW 逐参数归一化，w_f 只改变粗匹配与细化两种信号在共用参数（DINOv2 的 LoRA、尺度 16 的 proj）里的比例。本脚本在
训练图对（带训练扰动，同 E32 配置的 [aug]）上照 Term 的算法算出：
- 粗匹配损失（RLOO 策略梯度项 + 粗级 certainty 的 BCE × cert_bce，即 E28 fm 的粗级损失；另记只有策略梯度项的 coarse_pol）；
- 细化各级（16、8、4）的逐级抽签损失（local，σ = k·s，fine_group 次，cauchy，画面外像素优势置零），各级分开；
- 尺度 16 细化 certainty 的 BCE（只记录，不进比值）。
每个损失单独对共用参数求梯度（autograd.grad，不碰 .grad），记梯度范数（LoRA 与 proj16 合起来，另分开记）。
尺度 8、4 的细化层只读 VGG 特征，梯度到不了共用参数，预期为 0。参照仿射拟合失败的对跳过（记 skip）。

    python runs/E32/code/wf_grad.py runs/E32/configs/base.toml --tag zs --n 100 [--ckpt <E19 gauss8 ckpt_1000.pt>]

输出 runs/E32/extra/data/wf_<tag>.jsonl（每对一行）与汇总 wf_<tag>.json（各量的中位数、比值的中位数）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R_ = Path(__file__).resolve().parents[1]
REPO = R_.parents[1]
sys.path.insert(0, str(REPO))
from finetune import config as C  # noqa: E402
from finetune.data import PairSet  # noqa: E402
from finetune.geom import HW  # noqa: E402
from finetune.models import roma as M  # noqa: E402
from finetune.parts import aug as AUG  # noqa: E402
from finetune.parts import rloo as R  # noqa: E402

SCALE = 1024.0      # 求导前乘上，免得半精度里梯度下溢（范数之比与它无关）


def norms(loss, groups):
    """loss 对各组参数的梯度范数 {组: 范数}，另给 all（全部组拼起来）。"""
    flat = [q for g in groups.values() for q in g]
    gr = torch.autograd.grad(loss * SCALE, flat, retain_graph=True, allow_unused=True)
    sq, out, i = 0.0, {}, 0
    for name, g in groups.items():
        s = sum(0.0 if x is None else float(x.float().square().sum()) for x in gr[i:i + len(g)])
        out[name] = (s ** 0.5) / SCALE
        sq += s
        i += len(g)
    out["all"] = (sq ** 0.5) / SCALE
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("TORCH_HOME", M.TORCH_HOME)
    torch.manual_seed(a.seed)
    cfg = C.load(a.config)
    p = cfg["rloo"]
    model = M.Model(cfg, os.environ["MOON_WEIGHTS"], device="cuda")
    if a.ckpt:
        M.load_weights(model, torch.load(a.ckpt, map_location="cpu"))
    dec = model.model.decoder
    groups = {"lora": [q for l in model.lora.values() for q in (l.A, l.B)],
              "proj16": [q for q in dec.proj["16"].parameters()]}
    ds = PairSet(os.environ["MOON_DATA"], "train", model.resize, **model.input, seed=a.seed, **AUG.data_kw(cfg["aug"]))
    idx = np.random.default_rng(a.seed).choice(len(ds), a.n, replace=False)
    out_dir = R_ / "extra" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_dir / f"wf_{a.tag}.jsonl", "w", encoding="utf-8") as fo:
        for j, i in enumerate(idx):
            b = ds[int(i)]
            i0, i1 = b["image0"][None].cuda(), b["image1"][None].cuda()
            f1, f2 = R.features(model, i0, i1)
            cls, cert = R.coarse(model, f1, f2)
            cls = cls.float()
            _, Cn, h, w = cls.shape
            logp = F.log_softmax(cls / p["temp"], 1)[0]
            with torch.no_grad():
                q = logp.exp()
                k = torch.multinomial(q.reshape(Cn, -1).T, p["group"], replacement=True).T.reshape(-1, h, w)
                P = cls.softmax(1)
                traj = {}
                flow, ce = R.refine(model, f1, f2, R.anchor_flow(P, P.argmax(1)), cert.float(), traj)
                A = R.fit_many(flow, ce, p["num"], p["ransac"])[0][0]
                row = {"pair": b["pair"], "i": int(i)}
                if A is None:
                    row["skip"] = True
                    fo.write(json.dumps(row) + "\n")
                    continue
                fc = R.anchor_flow(P.expand(p["group"], -1, -1, -1), k)
                r, d = R.ref_cells(fc, A, p["delta"], p["tau"], p["r_out"], p["shape"])
                m = R.in_frame(A, i1[0], h, w)
                adv = R.advantage(r, p["baseline"]) * m
            l_pol = -(adv * R._logp(logp, k)).mean()
            l_bce = F.binary_cross_entropy_with_logits(cert[0, 0].float(), m.float())
            losses = {"coarse": l_pol + p["cert_bce"] * l_bce, "coarse_pol": l_pol}
            H, W = f1[1].shape[-2:]
            for s in (16, 8, 4):
                fin, cin, _ = traj[s]
                mu, ce_s = R._mu_cert(dec, s, f1[s], f2[s], fin, cin, H, W)
                ms = R.in_frame(A, i1[0], *fin.shape[-2:])
                losses[f"f{s}"], _ = R.level_loss(mu, A, s, p["k"] * s / (HW / 2), p, "local", keep=ms.float())
                if s == 16:
                    losses["f16_bce"] = p["fine_cert_bce"] * F.binary_cross_entropy_with_logits(ce_s[0, 0].float(),
                                                                                                ms.float())
            row.update(in_frame=float(m.float().mean()), d_ref=float(d.mean()))
            for name, L in losses.items():
                row[name] = norms(L, groups)
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            rows.append(row)
            print(j, b["pair"], {n: round(row[n]["all"], 6) for n in losses}, flush=True)
            del f1, f2, cls, cert, losses, logp
    keys = ("coarse", "coarse_pol", "f16", "f8", "f4", "f16_bce")
    med = {n: {g: float(np.median([x[n][g] for x in rows])) for g in ("lora", "proj16", "all")} for n in keys}
    ratio = [x["coarse"]["all"] / x["f16"]["all"] for x in rows if x["f16"]["all"] > 0]
    summ = {"tag": a.tag, "ckpt": a.ckpt, "n": len(rows), "skipped": int(a.n - len(rows)), "median": med,
            "ratio_coarse_over_f16": {"median": float(np.median(ratio)), "q25": float(np.quantile(ratio, 0.25)),
                                      "q75": float(np.quantile(ratio, 0.75))},
            "ratio_of_medians": med["coarse"]["all"] / med["f16"]["all"]}
    (out_dir / f"wf_{a.tag}.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
