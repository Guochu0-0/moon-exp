"""finetune 里不依赖 GPU / LoFTR 的部分。"""
import numpy as np

from finetune.compare import compare


def test_compare_identical_and_diff():
    rng = np.random.default_rng(0)
    a = {"R__p0": rng.random((5, 5), dtype=np.float32), "R__p1": np.zeros((0, 5), np.float32)}
    assert compare(a, {k: v.copy() for k, v in a.items()})["identical"]

    b = {k: v.copy() for k, v in a.items()}
    b["R__p0"][0, 2] += 0.5
    r = compare(a, b)
    assert not r["identical"] and abs(r["max_abs_xy"] - 0.5) < 1e-6

    r = compare(a, {"R__p0": a["R__p0"][:3], "R__p2": a["R__p1"]})
    assert r["n_diff_count"] == 1 and r["only_a"] == ["R__p1"] and r["only_b"] == ["R__p2"]


def _grid(w=10, h=10, step=8):
    ii = np.arange(w * h)
    return np.stack([ii % w, ii // w], 1).astype(np.float64) * step


def test_pseudo_targets_translation():
    from finetune.pseudo import coarse_targets, fine_targets

    s = 512 / 640
    A = np.array([[1.0, 0, 8 * s], [0, 1.0, 0]])          # 原网格右移 6.4 px = 输入网格 8 px = 一个粗格
    i, j = coarse_targets(A, (10, 10), 8, s)
    assert len(i) == 90 and np.all(j == i + 1)            # 最右一列落到图外，不监督
    k0 = _grid()[:5]
    A2 = np.array([[1.0, 0, 10 * s], [0, 1.0, 0]])        # 输入网格右移 10 px
    g = fine_targets(A2, k0, k0 + [8, 0], 4.0, s)          # 粗级落在 +8，窗口半宽 4 px
    assert np.allclose(g, [[0.5, 0]] * 5)


def test_pseudo_loss_offline_and_online():
    import torch
    from finetune.pseudo import pseudo_loss

    s, w = 512 / 640, 10
    torch.manual_seed(0)
    logits = torch.randn(1, w * w, w * w, requires_grad=True)
    conf = torch.softmax(logits, 1) * torch.softmax(logits, 2)
    k0 = _grid()
    shift = np.array([9.0, 0])                              # 真实对应在 +9：细级偏移 0.25 窗口
    expec = torch.zeros(len(k0), 3)
    expec[:, 2] = 0.5
    expec.requires_grad_(True)
    ids = torch.arange(len(k0))
    data = {"conf_matrix": conf, "expec_f": expec, "hw0_c": (w, w), "hw0_i": (80, 80), "hw0_f": (40, 40), "W": 5,
            "b_ids": torch.zeros(len(k0), dtype=torch.long), "i_ids": ids, "j_ids": ids,
            "mkpts0_c": torch.from_numpy(k0), "mkpts1_c": torch.from_numpy(k0 + [8, 0]),
            "mkpts1_f": torch.from_numpy(k0 + shift)}
    A = np.array([[1.0, 0, 9 * s], [0, 1.0, 0]])
    loss, st = pseudo_loss(data, s, affines=[A])
    loss.backward()
    assert st["pairs_used"] == 1 and st["n_fine"] == 100 and st["n_coarse_pos"] > 0
    assert logits.grad.abs().sum() > 0 and expec.grad.abs().sum() > 0
    assert expec.grad[:, 0].mean() < 0                      # 梯度下降把 x 偏移往 +0.25 推

    loss, st = pseudo_loss(data, s, min_inliers=10)         # 在线：从 mkpts1_f 估出同一平移
    assert st["pairs_used"] == 1 and st["n_inliers"] == [100]
    loss, st = pseudo_loss(data, s, affines=[None])
    assert st["pairs_used"] == 0 and float(loss) == 0
