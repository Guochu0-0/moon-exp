"""被擦除区域可信度约束（#256；[rloo] erase_cert = true）：SAR 上被随机擦除（[aug] erase，掩码 batch["occ"]）的地方，
光学图上对到那里的格子 / 像素，可信度目标改为 0、不参与奖励打分。训练里怎么用见 finetune/parts/rloo.py 模块说明
「被擦除区域可信度约束」。

光学的哪些格子对到被擦处：用参照仿射把格子中心映到 SAR 上，按最近邻查擦除掩码（同 pseudo.occluded() 未交换的情形，
与 rloo.in_frame 用同一套坐标）。擦除掩码在 SAR 原网格（512）上、在已知扰动之后生成，与训练输入的 SAR 同一坐标系；
参照仿射也是光学 → 扰动后 SAR，所以不需要再换算。映到画面外的格子不算被擦（它们已由画面内掩码处理）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .pseudo import affine_norm, occluded


def erased(A, occ, h, w) -> torch.Tensor:
    """A：参照仿射（光学 → SAR，原网格 px，2×3）；occ：SAR 原网格上的擦除掩码 (1, H, W)（被擦为 1）。
    返回 (h, w) bool：光学 h×w 网格上，格子中心经 A 落在 SAR 被擦处的格子。"""
    An = torch.from_numpy(affine_norm(A)).to(occ.device).float()[None]
    return occluded((occ[None].float(), torch.zeros(1, dtype=torch.bool, device=occ.device)), An, h, w)[0]


def target(m: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """可信度目标与奖励掩码：画面内（m）且没被擦（~e）为 True。"""
    return m & ~e


def split_stats(logits: torch.Tensor, m: torch.Tensor, e: torch.Tensor, stats: dict, suffix: str = ""):
    """只记录、不参与训练：可信度 logits (h, w) 在被擦处与未擦处（画面内、没被擦）各自的 BCE（目标为新目标：被擦处 0、
    未擦处 1）与平均可信度（sigmoid），以及被擦处占画面内格子的比例。某一侧没有格子时那一侧不记。
    键：era_frac / l_cert_e / l_cert_u / cert_e / cert_u，加 suffix（细化级为 "_<尺度>"）。"""
    with torch.no_grad():
        x = logits.float()
        em, um = e & m, m & ~e
        n_in = int(m.sum())
        if n_in:
            stats.setdefault(f"era_frac{suffix}", []).append(float(em.sum()) / n_in)
        bce = F.binary_cross_entropy_with_logits(x, um.float(), reduction="none")
        c = x.sigmoid()
        for tag, sel in (("e", em), ("u", um)):
            if bool(sel.any()):
                stats.setdefault(f"l_cert_{tag}{suffix}", []).append(float(bce[sel].mean()))
                stats.setdefault(f"cert_{tag}{suffix}", []).append(float(c[sel].mean()))
