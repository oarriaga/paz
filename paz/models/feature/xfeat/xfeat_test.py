import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax.numpy as jp
import pytest

from paz.models.feature.xfeat import backend
from paz.models.feature.xfeat.model import XFeatModel

torch = pytest.importorskip("torch")
import torch.nn.functional as F

WEIGHTS = os.environ.get("XFEAT_WEIGHTS")


def torch_sample(feature_chw, positions, height, width, mode):
    scale = torch.tensor([width - 1, height - 1], dtype=torch.float32)
    grid = 2.0 * torch.tensor(positions) / scale - 1.0
    grid = grid[None, :, None, :]
    sampled = F.grid_sample(torch.tensor(feature_chw)[None], grid,
                            mode=mode, align_corners=False)
    return sampled[0, :, :, 0].T.numpy()


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_sample_features_matches_torch(mode):
    rng = np.random.default_rng(0)
    feature = rng.standard_normal((16, 60, 80)).astype(np.float32)
    positions = rng.uniform(-20, 660, size=(200, 2)).astype(np.float32)
    reference = torch_sample(feature, positions, 480, 640, mode)
    feature_hwc = jp.asarray(np.transpose(feature, (1, 2, 0)))
    ours = backend.sample_features(feature_hwc, jp.asarray(positions),
                                   480, 640, mode)
    assert np.allclose(np.asarray(ours), reference, atol=1e-4)


def torch_mutual(descriptors1, descriptors2, min_cosine):
    similarity = torch.tensor(descriptors1) @ torch.tensor(descriptors2).T
    match12 = similarity.argmax(1)
    match21 = similarity.T.argmax(1)
    source = torch.arange(len(match12))
    mutual = match21[match12] == source
    if min_cosine > 0:
        mutual = mutual & (similarity.max(1).values > min_cosine)
    return source[mutual].numpy(), match12[mutual].numpy()


@pytest.mark.parametrize("min_cosine", [-1.0, 0.5])
def test_mutual_nearest_neighbors_matches_torch(min_cosine):
    rng = np.random.default_rng(1)
    descriptors1 = normalize(rng.standard_normal((300, 64)))
    descriptors2 = normalize(rng.standard_normal((280, 64)))
    reference = torch_mutual(descriptors1, descriptors2, min_cosine)
    ours = backend.mutual_nearest_neighbors(
        jp.asarray(descriptors1), jp.asarray(descriptors2), min_cosine)
    assert np.array_equal(ours[0], reference[0])
    assert np.array_equal(ours[1], reference[1])


def normalize(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_model_output_shapes():
    model = XFeatModel()
    image = np.zeros((1, 480, 640, 3), np.float32)
    features, keypoints, heatmap = model.predict(image, verbose=0)
    assert features.shape == (1, 60, 80, 64)
    assert keypoints.shape == (1, 60, 80, 65)
    assert heatmap.shape == (1, 60, 80, 1)


@pytest.mark.skipif(not WEIGHTS, reason="set XFEAT_WEIGHTS to xfeat.pt")
def test_forward_matches_reference_weights():
    from paz.models.feature.xfeat.port_weights import port_weights

    model = port_weights(WEIGHTS)
    state = torch.load(WEIGHTS, map_location="cpu")
    reference = reference_backbone(state)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (1, 480, 640, 3)).astype(np.float32)
    features = model.predict(image, verbose=0)[0]
    features = np.transpose(features, (0, 3, 1, 2))
    with torch.inference_mode():
        expected = reference(torch.tensor(np.transpose(image, (0, 3, 1, 2))))
    assert np.abs(features - expected.numpy()).max() < 1e-3


def reference_backbone(state):
    pytest.importorskip("modules.model", reason="XFeat reference repo needed")
    from modules.model import XFeatModel as TorchModel

    model = TorchModel().eval()
    model.load_state_dict(state)
    return lambda x: model(x)[0]
