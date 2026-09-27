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


def _texture(n=128, shift=0, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0)
    return np.roll(img, shift, axis=1)          # 内容右移 shift px：光学 x → SAR x + shift


def test_pair_scores_prefers_true_shift():
    import torch
    from finetune.rl import FEATS, pair_scores

    for kind, ff in FEATS.items():
        F0 = ff(torch.from_numpy(_texture())[None, None])[0]
        F1 = ff(torch.from_numpy(_texture(shift=3))[None, None])[0]
        A = lambda t: np.array([[1.0, 0, t], [0, 1.0, 0]])
        sc = pair_scores(F0, F1, [A(0), A(3), A(1.5), A(5)], 1.0, kind)
        assert sc.argmax() == 1 and sc[1] > sc[2] > sc[0], kind


def test_diagnose_reward_surface_peak():
    from baselines import diagnose_reward as dr

    Fo, Fs = dr.cfog(_texture(160)), dr.cfog(_texture(160, shift=3))
    A = lambda t: np.array([[1.0, 0, t], [0, 1.0, 0]])
    dx, dy, _ = dr.peak(dr.surface(dr.warp_feat(Fo, A(3), Fs.shape[1:]), Fs, "cfog"))
    assert abs(dx) < 0.3 and abs(dy) < 0.3                  # 基准已对准：峰在 0
    dx, dy, _ = dr.peak(dr.surface(dr.warp_feat(Fo, A(1), Fs.shape[1:]), Fs, "cfog"))
    assert abs(dx + 2) < 0.3 and abs(dy) < 0.3              # 基准差 2 px：峰在 d = −2，A' = A − d 对准


def test_rl_pair_gradient_points_to_true_shift():
    import torch
    from finetune.rl import FEATS, rl_loss

    n, w = 128, 16
    k0 = _grid(w, w, 8)
    k0 = k0[(k0 >= 24).all(1) & (k0 < n - 24).all(1)]
    for kind, ff in FEATS.items():
        torch.manual_seed(0)
        mu = torch.zeros(len(k0), 3)
        mu.requires_grad_(True)
        data = {"expec_f": mu, "b_ids": torch.zeros(len(k0), dtype=torch.long), "W": 5, "hw0_i": (n, n),
                "hw0_f": (n // 2, n // 2), "mkpts0_c": torch.from_numpy(k0), "mkpts1_c": torch.from_numpy(k0),
                "conf_matrix": torch.zeros(1, 1, 1)}
        feats = (ff(torch.from_numpy(_texture())[None, None]), ff(torch.from_numpy(_texture(shift=2))[None, None]))
        for _ in range(20):
            loss, st = rl_loss(data, 1.0, K=8, sig_g=0.25, sig_i=0.05, w_pair=1.0, feats=feats, kind=kind)
            loss.backward()
        assert st["rl_ransac_fail"] == 0
        assert mu.grad[:, 0].mean() < 0 and abs(mu.grad[:, 1].mean()) < abs(mu.grad[:, 0].mean()), kind


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
