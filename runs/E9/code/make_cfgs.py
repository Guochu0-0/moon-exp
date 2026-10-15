"""E9 的推理配置：从 B0 的两个 AnyMatch 配置派生，只改 input（#150）。生成 runs/E9/code/cfg/<方法>.json，提交后再跑。

    python runs/E9/code/make_cfgs.py

方法名 <模型>__<变体>：s1 为 B0 口径的 S1（p2–p98）在同一批卡上重跑，作配对比较的参照；s1mm 为 RoMa 的 min–max 口径
（B0m 里 RoMa 最好的映射）；其余变体见 moonlib/inputs.py。融合（fuse_s1_cpr）不推理，由 fuse.py 合并点对。
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "cfg"
BASE = {"anymatch_loftr": "anymatch_loftr.json", "anymatch_roma": "anymatch_roma.json"}
SAR = ["sc", "oc", "cpr", "m", "mchi_odd", "mchi_even", "mchi_vol", "cpr_ml5", "cpr_ml9", "m_ml5", "m_ml9"]
VARIANTS = {   # 变体: (optical, sar)
    "s1": ("div255", "p2p98"),
    **{k: ("div255", k) for k in SAR},
    "optx4": ("xblur4", "p2p98"),
    "optx4y2": ("blur4x2", "p2p98"),
}
EXTRA = {"anymatch_roma": {"s1mm": ("div255", "minmax"), "mchi_rgb": ("div255", "mchi_rgb")}}   # LoFTR 只吃单通道

OUT.mkdir(exist_ok=True)
for model, f in BASE.items():
    base = json.loads((REPO / "configs/baselines" / f).read_text(encoding="utf-8"))
    for v, (opt, sar) in {**VARIANTS, **EXTRA.get(model, {})}.items():
        c = dict(base, method=f"{model}__{v}", input={"optical": opt, "sar": sar})
        (OUT / f"{model}__{v}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(sorted(p.stem for p in OUT.glob("*.json")))
