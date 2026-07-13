import os

import numpy as np
import jax

from paz.models import DINOv2Small, DINOv2Base, DINOv2Large
from paz.models import DINOv2SmallFeatures

IMAGE_SHAPE = (70, 70, 3)
GRID = (5, 5)


def make_input(batch=2):
    return np.random.RandomState(0).randn(batch, *IMAGE_SHAPE).astype("float32")


def test_small_returns_class_and_patch_tokens():
    model = DINOv2Small(image_shape=IMAGE_SHAPE)
    class_token, patch_tokens = model(make_input())
    assert tuple(class_token.shape) == (2, 384)
    assert tuple(patch_tokens.shape) == (2, 25, 384)


def test_standard_output_is_two_plain_tensors():
    model = DINOv2Small(image_shape=IMAGE_SHAPE)
    output = model(make_input())
    assert not isinstance(output, dict)
    assert not hasattr(output, "_fields")
    assert len(output) == 2


def test_base_returns_class_and_patch_tokens():
    model = DINOv2Base(image_shape=IMAGE_SHAPE)
    class_token, patch_tokens = model(make_input())
    assert tuple(class_token.shape) == (2, 768)
    assert tuple(patch_tokens.shape) == (2, 25, 768)


def test_large_builds():
    model = DINOv2Large(image_shape=IMAGE_SHAPE)
    assert model.output_shape == [(None, 1024), (None, 25, 1024)]


def test_feature_model_returns_ordered_maps():
    model = DINOv2SmallFeatures(image_shape=IMAGE_SHAPE, out_layers=(5, 7, 9, 11))
    maps = model(make_input())
    assert not isinstance(maps, dict)
    assert not hasattr(maps, "_fields")
    assert len(maps) == 4
    for feature_map in maps:
        assert tuple(feature_map.shape) == (2, GRID[0], GRID[1], 384)


def test_small_save_and_reload_weights(tmp_path):
    model = DINOv2Small(image_shape=IMAGE_SHAPE)
    data = make_input()
    before = [np.array(tensor) for tensor in model(data)]
    path = os.path.join(tmp_path, "dinov2_small.weights.h5")
    model.save_weights(path)
    reloaded = DINOv2Small(image_shape=IMAGE_SHAPE)
    reloaded.load_weights(path)
    after = [np.array(tensor) for tensor in reloaded(data)]
    for expected, actual in zip(before, after):
        assert np.allclose(expected, actual, atol=1e-6)


def test_small_jit_matches_eager():
    model = DINOv2Small(image_shape=IMAGE_SHAPE)
    data = make_input()
    eager = np.array(model(data)[0])
    jitted = np.array(jax.jit(lambda x: model(x))(data)[0])
    assert np.allclose(eager, jitted, atol=1e-5)
