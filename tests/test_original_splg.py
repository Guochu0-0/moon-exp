"""SP+LG keeps official extraction coordinates, weights, and float inputs."""
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_original_splg_uses_defaults_and_preserves_coordinates(monkeypatch):
    from baselines.adapters.splg_original import OriginalSPLGAdapter

    calls = {"extract": []}

    class SuperPoint:
        def __init__(self, **kwargs):
            calls["sp"] = kwargs

        def eval(self):
            return self

        def to(self, device):
            return self

        def extract(self, image):
            calls["extract"].append(image.clone())
            # Official extract() already returns keypoints in the input grid.
            offset = len(calls["extract"]) - 1
            return {"keypoints": torch.tensor([[[3.25, 5.75], [9.5, 11.0]]]) + offset}

    class LightGlue(SuperPoint):
        def __init__(self, **kwargs):
            calls["lg"] = kwargs

        def __call__(self, features):
            return {"matches": torch.tensor([[[1, 0], [0, 1]]]),
                    "matching_scores0": torch.tensor([[0.4, 0.8]])}

    package = types.ModuleType("lightglue")
    package.LightGlue, package.SuperPoint = LightGlue, SuperPoint
    utils = types.ModuleType("lightglue.utils")
    utils.rbd = lambda values: {key: value[0] for key, value in values.items()}
    monkeypatch.setitem(sys.modules, "lightglue", package)
    monkeypatch.setitem(sys.modules, "lightglue.utils", utils)
    adapter = OriginalSPLGAdapter(repo="unused", device="cpu")
    image = np.full((17, 23), 0.1234567, dtype=np.float32)
    k0, k1, confidence = adapter.match(image, image)
    assert calls["sp"] == {} and calls["lg"] == {"features": "superpoint"}
    assert calls["extract"][0].shape == (1, 1, 17, 23)
    np.testing.assert_array_equal(calls["extract"][0][0, 0].numpy(), image)
    np.testing.assert_array_equal(k0, [[9.5, 11.0], [3.25, 5.75]])
    np.testing.assert_array_equal(k1, [[4.25, 6.75], [10.5, 12.0]])
    np.testing.assert_allclose(confidence, [0.8, 0.4])
    with pytest.raises(ValueError, match="official constructor"):
        OriginalSPLGAdapter(repo="unused", weights="minima.pth", device="cpu")


def test_original_roma_float_preprocessing_and_native_coordinates(monkeypatch):
    pytest.importorskip("torchvision")
    from baselines.adapters import roma_original

    calls = {}

    class Model:
        h_resized, w_resized = 28, 28
        upsample_res = (42, 42)

        def eval(self):
            return self

        def match(self, first, second, **kwargs):
            calls.update(first=first, second=second, **kwargs)
            return None, None

        def sample(self, warp, certainty):
            return torch.tensor([[-0.5, 0.0, 0.5, 0.0]]), torch.tensor([0.9])

        def to_pixel_coordinates(self, matches, ha, wa, hb, wb):
            return ((matches[:, :2] + 1) * torch.tensor([wa, ha]) / 2,
                    (matches[:, 2:] + 1) * torch.tensor([wb, hb]) / 2)

    def constructor(**kwargs):
        calls["constructor"] = kwargs
        return Model()

    package = types.ModuleType("romatch")
    package.roma_outdoor = constructor
    monkeypatch.setitem(sys.modules, "romatch", package)
    monkeypatch.setattr(roma_original, "load_ckpt", lambda path: {"official": True})
    adapter = roma_original.OriginalRoMaAdapter(repo="unused", weights="official.pth", device="cpu")
    image = np.full((18, 26), 0.1234567, dtype=np.float32)
    first, second, confidence = adapter.match(image, image)
    assert calls["first"].shape == (1, 3, 28, 28)
    assert calls["im_A_high_res"].shape == (1, 3, 42, 42)
    expected = (float(image[0, 0]) - 0.485) / 0.229
    torch.testing.assert_close(calls["first"][:, 0], torch.full((1, 28, 28), expected))
    torch.testing.assert_close(calls["im_A_high_res"][:, 0], torch.full((1, 42, 42), expected))
    assert calls["batched"] is True
    assert calls["constructor"]["weights"] == {"official": True}
    np.testing.assert_allclose(first, [[6.0, 8.5]])
    np.testing.assert_allclose(second, [[19.0, 8.5]])
    np.testing.assert_allclose(confidence, [0.9])
