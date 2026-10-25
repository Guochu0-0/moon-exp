"""相邻存档之间各部分权重的相对改变量（#206，方法 2 的补充）：‖θ_后 − θ_前‖ / ‖θ_前‖，按部分（粗级 decoder、细化部分、
VGG，划分同 finetune.models.roma._part）；LoRA 按等效权重增量 ‖Δ(scale·BA)‖ 计，只给绝对值（原权重不在存档里）。
七个坍缩运行与三个健康对照的全部存档。只用 CPU。
用法（仓库根目录、loftr 环境）：PYTHONPATH=. python runs/E26/code/weights.py  → runs/E26/extra/data/weights.csv
"""
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
E = HERE.parent
sys.path.insert(0, str(HERE))
from finetune.models.roma import _part              # noqa: E402
from methods import COLLAPSED, HEALTHY              # noqa: E402

MAIN = Path("/remote-home/xufang/YGC/moon-exp/runs")


def flat(sd):
    out = {}
    for k, v in sd["model"].items():
        p = _part(k)
        if p and v.is_floating_point():
            out.setdefault(p, []).append(v.float().reshape(-1))
    out = {p: torch.cat(v) for p, v in out.items()}
    lo = sd.get("dinov2_lora", {})
    return out, {k: l["scale"] * (l["B"].float() @ l["A"].float()) for k, l in lo.items()}


def main():
    rows = []
    for e, r in COLLAPSED + HEALTHY:
        d = MAIN / e / "ckpt" / r
        steps = sorted(int(p.stem[5:]) for p in d.glob("ckpt_*.pt"))
        prev = None
        for s in steps:
            cur = flat(torch.load(d / f"ckpt_{s}.pt", map_location="cpu"))
            if prev is not None:
                (a, la), (b, lb) = prev[1], cur
                row = {"run": f"{e} {r}", "from": prev[0], "to": s}
                for p in sorted(a):
                    row[p] = float((b[p] - a[p]).norm() / a[p].norm())
                row["lora_dW"] = float(sum((lb[k] - la[k]).square().sum() for k in la) ** 0.5)
                row["lora_W"] = float(sum(lb[k].square().sum() for k in lb) ** 0.5)
                rows.append(row)
                print(row, flush=True)
            prev = (s, cur)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(E / "extra" / "data" / "weights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
