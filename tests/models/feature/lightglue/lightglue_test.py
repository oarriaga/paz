import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax.numpy as jp
import pytest

from paz.models.feature.lightglue import model as lightglue

WEIGHTS = os.environ.get("LIGHTERGLUE_WEIGHTS")
REPO = os.environ.get("LIGHTGLUE_REPO")


def numpy_filter(scores, threshold):
    valid = scores[:-1, :-1]
    match_0, match_1 = valid.argmax(1), valid.argmax(0)
    mutual_0 = np.arange(len(match_0)) == match_1[match_0]
    mutual_1 = np.arange(len(match_1)) == match_0[match_1]
    strength_0 = np.where(mutual_0, np.exp(valid.max(1)), 0.0)
    valid_0 = mutual_0 & (strength_0 > threshold)
    valid_1 = mutual_1 & valid_0[match_1]
    return (np.where(valid_0, match_0, -1), np.where(valid_1, match_1, -1),
            strength_0, np.where(mutual_1, strength_0[match_1], 0.0))


def test_filter_matches_matches_reference():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((80, 65)).astype(np.float32)
    matches_0, matches_1, scores_0, scores_1 = numpy_filter(scores, 0.1)
    mask_0, mask_1 = jp.ones(scores.shape[0] - 1), jp.ones(scores.shape[1] - 1)
    ours = lightglue.filter_matches(jp.asarray(scores), mask_0, mask_1, 0.1)
    assert np.array_equal(np.asarray(ours[0]), matches_0)
    assert np.array_equal(np.asarray(ours[1]), matches_1)
    assert np.allclose(np.asarray(ours[2]), scores_0, atol=1e-4)
    assert np.allclose(np.asarray(ours[3]), scores_1, atol=1e-4)


def test_rotate_half_is_quarter_turn():
    rng = np.random.default_rng(1)
    x = jp.asarray(rng.standard_normal((10, 96)), jp.float32)
    rotated = lightglue.rotate_half(lightglue.rotate_half(x))
    assert np.allclose(np.asarray(rotated), -np.asarray(x), atol=1e-6)


@pytest.mark.skipif(not (WEIGHTS and REPO), reason="set LIGHTERGLUE_WEIGHTS "
                    "and LIGHTGLUE_REPO to the cvg/LightGlue checkout")
def test_matches_torch_reference(tmp_path):
    from paz.models.feature.lightglue.port_weights import port_weights

    weights_path = str(tmp_path / "lighterglue.weights.h5")
    port_weights(WEIGHTS).save_weights(weights_path)
    keypoints_0, descriptors_0 = random_features(0, 300)
    keypoints_1, descriptors_1 = correlated_features(descriptors_0, 1)
    size = jp.array([640.0, 480.0])
    match = lightglue.LighterGlue(weights=weights_path)
    ours = match(keypoints_0, descriptors_0, keypoints_1, descriptors_1,
                 size, size)
    expected = reference_matches(keypoints_0, descriptors_0, keypoints_1,
                                 descriptors_1)
    assert np.array_equal(ours.matches_0, expected)


def random_features(seed, count):
    rng = np.random.default_rng(seed)
    keypoints = rng.uniform([0, 0], [640, 480], (count, 2)).astype(np.float32)
    descriptors = normalize(rng.standard_normal((count, 64)))
    return jp.asarray(keypoints), jp.asarray(descriptors)


def correlated_features(descriptors_0, seed):
    rng = np.random.default_rng(seed)
    descriptors_0 = np.asarray(descriptors_0)
    keypoints = rng.uniform([0, 0], [640, 480], (280, 2)).astype(np.float32)
    noise = rng.normal(0, 0.05, (280, 64))
    descriptors = normalize(descriptors_0[:280] + noise)
    return jp.asarray(keypoints), jp.asarray(descriptors)


def normalize(descriptors):
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return (descriptors / norm).astype(np.float32)


def reference_matches(keypoints_0, descriptors_0, keypoints_1, descriptors_1):
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location(
        "reference_lightglue", os.path.join(REPO, "lightglue/lightglue.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = build_reference(module, torch)
    data = reference_data(torch, keypoints_0, descriptors_0, keypoints_1,
                          descriptors_1)
    with torch.inference_mode():
        return net(data)["matches0"][0].numpy()


def build_reference(module, torch):
    config = dict(input_dim=64, descriptor_dim=96, n_layers=6, num_heads=1,
                  flash=False, depth_confidence=-1, width_confidence=-1)
    net = module.LightGlue(features=None, **config).eval()
    state = torch.load(WEIGHTS, map_location="cpu")
    for arg in range(6):
        state = rename(state, arg)
    state = {key.replace("matcher.", ""): value
             for key, value in state.items()}
    net.load_state_dict(state, strict=False)
    return net


def rename(state, arg):
    for kind in ("self_attn", "cross_attn"):
        pattern = f"{kind}.{arg}", f"transformers.{arg}.{kind}"
        state = {key.replace(*pattern): value for key, value in state.items()}
    return state


def reference_data(torch, keypoints_0, descriptors_0, keypoints_1,
                   descriptors_1):
    def image(keypoints, descriptors):
        return {"keypoints": torch.tensor(np.asarray(keypoints))[None],
                "descriptors": torch.tensor(np.asarray(descriptors))[None],
                "image_size": torch.tensor([[640.0, 480.0]])}

    return {"image0": image(keypoints_0, descriptors_0),
            "image1": image(keypoints_1, descriptors_1)}
