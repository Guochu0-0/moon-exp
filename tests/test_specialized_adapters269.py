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
