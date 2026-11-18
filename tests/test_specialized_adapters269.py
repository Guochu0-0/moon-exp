"""Integration contracts without downloading checkpoints or importing upstream packages."""
import numpy as np
import pytest

from baselines.adapters.fhreg import FHRegAdapter
from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.adapters.vmgga import VMGGAAdapter


def test_fhreg_keeps_outlier_and_maps_resize_centres():
    torch = pytest.importorskip("torch")
    adapter = FHRegAdapter.__new__(FHRegAdapter)
    adapter.device, adapter.image_size = "cpu", 512

    def forward(batch):
        assert batch["image_opt"].shape == (1, 3, 512, 512)
        assert batch["image_sar"].max() == pytest.approx(0.25)
        batch.update(mkpts0_f=torch.tensor([[0., 0.], [511., 511.]]),
                     mkpts1_f=torch.tensor([[7., 5.], [0., 0.]]))
        return batch

    adapter.model = forward
    kp0, kp1, conf = adapter.match(np.ones((256, 256), np.float32), np.full((256, 256), .25, np.float32))
    np.testing.assert_allclose(kp0, [[-.25, -.25], [255.25, 255.25]])
    np.testing.assert_allclose(kp1, [[3.25, 2.25], [-.25, -.25]])
    assert conf is None


def test_vmgga_excludes_padding_but_does_not_fit_geometry():
    torch = pytest.importorskip("torch")
    adapter = VMGGAAdapter.__new__(VMGGAAdapter)
    adapter.device = "cpu"

    def forward(batch):
        assert batch["image0"].shape == (1, 1, 32, 32)
        assert batch["image0"][0, 0, 31, 31] == 0
        batch.update(mkpts0_f=torch.tensor([[0., 0.], [10., 10.], [30., 30.]]),
                     mkpts1_f=torch.tensor([[3., 4.], [0., 19.], [1., 1.]]),
                     mconf=torch.tensor([.9, .7, .3]))

    adapter.model = forward
    kp0, kp1, conf = adapter.match(np.ones((21, 21), np.float32), np.ones((21, 21), np.float32))
    np.testing.assert_array_equal(kp0, [[0, 0], [10, 10]])
    np.testing.assert_array_equal(kp1, [[3, 4], [0, 19]])
    np.testing.assert_allclose(conf, [.9, .7])


def test_redfeat_uses_correct_modality_branches_and_handles_too_few_descriptors():
    adapter = ReDFeatAdapter.__new__(ReDFeatAdapter)
    seen = []

    def features(image, branch):
        seen.append(branch)
        return np.array([[0., 0.]]), np.ones((1, 8), np.float32)

    adapter._features = features
    kp0, kp1, conf = adapter.match(np.zeros((20, 20)), np.zeros((20, 20)))
    assert seen == ["forward1", "forward2"]
    assert kp0.shape == kp1.shape == (0, 2)
    assert conf is None


def test_fhreg_confidence_corresponds_to_filtered_fine_matches():
    torch = pytest.importorskip("torch")
    adapter = FHRegAdapter.__new__(FHRegAdapter)
    adapter.device, adapter.image_size = "cpu", 512

    class Model:
        config = {"matching_type": "fine"}

        def __call__(self, batch):
            # Three coarse candidates produced two reordered fine matches.
            batch.update(mkpts0_f=torch.tensor([[50., 20.], [10., 30.]]),
                         mkpts1_f=torch.tensor([[55., 20.], [15., 30.]]),
                         mconf=torch.tensor([.8, .7, .6]),
                         mconf_f=torch.tensor([.6, .8]))
            return batch

    adapter.model = Model()
    kp0, kp1, conf = adapter.match(np.zeros((512, 512), np.float32), np.zeros((512, 512), np.float32))
    assert len(conf) == len(kp0) == len(kp1) == 2
    np.testing.assert_allclose(conf, [.6, .8])


def test_fhreg_explicit_usm_uses_coarse_confidence():
    torch = pytest.importorskip("torch")
    adapter = FHRegAdapter.__new__(FHRegAdapter)
    adapter.device, adapter.image_size = "cpu", 512

    class Model:
        config = {"matching_type": "usm"}

        def __call__(self, batch):
            batch.update(mkpts0_f=torch.tensor([[10., 20.]]),
                         mkpts1_f=torch.tensor([[15., 20.]]), mconf=torch.tensor([.7]))
            return batch

    adapter.model = Model()
    _, _, conf = adapter.match(np.zeros((512, 512), np.float32), np.zeros((512, 512), np.float32))
    np.testing.assert_allclose(conf, [.7])


def test_redfeat_real_multiscale_normalization_nms_ranking_and_coordinates():
    torch = pytest.importorskip("torch")
    adapter = ReDFeatAdapter.__new__(ReDFeatAdapter)
    adapter.device, adapter.num_features = "cpu", 2
    adapter.min_size, adapter.max_size = 32, 64
    adapter.scale_factor = 2**.25
    adapter.threshold, adapter.border = .4, 5
    seen = []

    class Network:
        def forward1(self, tensor):
            seen.append(tensor.detach().clone())
            h, w = tensor.shape[-2:]
            desc = torch.zeros((1, 2, h, w))
            desc[0, 0] = h
            rep = torch.zeros((1, 1, h, w))
            rep[0, 0, 12, 12] = .5 + (64 - h) / 128
            rep[0, 0, 12, 13] = .4  # adjacent lower response is suppressed
            rep[0, 0, 18, 18] = .3  # below repeatability threshold
            rep[0, 0, 5, 5] = .99  # official detector's 10px border
            return desc, rep

    adapter.model = Network()
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 4096
    kp, desc = adapter._features(image, "forward1")
    assert [t.shape[-1] for t in seen] == [64, 54, 45, 38, 32]
    assert seen[0].mean().item() == pytest.approx(0., abs=1e-6)
    assert seen[0][0, 0].std().item() == pytest.approx(1., abs=1e-6)
    torch.testing.assert_close(seen[0][0, 0], seen[0][0, 1])
    np.testing.assert_allclose(desc[:, 0], [32, 38])
    np.testing.assert_allclose(kp, [[24.5, 24.5], [(12.5 * 64 / 38) - .5] * 2])
    detected, scores = adapter.detect(image, "optical")
    np.testing.assert_array_equal(detected, kp)
    np.testing.assert_allclose(scores, [.75, .5 + (64 - 38) / 128])

    def forward2(tensor):
        desc, rep = adapter.model.forward1(tensor)
        return desc, rep * .9

    adapter.model.forward2 = forward2
    sar_points, sar_scores = adapter.detect(image, "sar")
    np.testing.assert_array_equal(sar_points, kp)
    np.testing.assert_allclose(sar_scores, scores * .9)

    adapter.threshold = 1
    empty_points, empty_scores = adapter.detect(image, "sar")
    assert empty_points.shape == (0, 2)
    assert empty_scores.shape == (0,)


def test_redfeat_real_ratio_matching_preserves_valid_raw_points():
    adapter = ReDFeatAdapter.__new__(ReDFeatAdapter)
    adapter.ratio = .9
    features = {
        "forward1": (np.array([[2., 3.], [40., 50.]]), np.array([[0., 0.], [9.9, 9.9]], np.float32)),
        "forward2": (np.array([[200., 300.], [5., 6.], [60., 70.]]),
                     np.array([[0.01, 0.01], [10., 10.], [20., 20.]], np.float32)),
    }
    adapter._features = lambda image, branch: features[branch]
    kp0, kp1, conf = adapter.match(np.zeros((512, 512)), np.zeros((512, 512)))
    np.testing.assert_array_equal(kp0, [[2, 3], [40, 50]])
    np.testing.assert_array_equal(kp1, [[200, 300], [5, 6]])
    assert conf is None
