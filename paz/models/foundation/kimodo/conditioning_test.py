import jax.numpy as jp
import numpy as np

from paz.models.foundation.kimodo import conditioning, features, skeleton
from paz.models.foundation.kimodo.features_test import build_unit_stats


def test_root_path_sets_planar_columns_only():
    condition = conditioning.build_empty(10)
    frames = jp.array([2, 7])
    positions = jp.array([[0.5, -1.0], [1.5, 2.0]])
    condition = conditioning.add_root_path(condition, frames, positions)
    assert condition.mask.sum() == 4
    assert jp.allclose(condition.observed[2, :3], jp.array([0.5, 0.0, -1.0]))
    assert bool(condition.mask[7, 0]) and not bool(condition.mask[7, 1])


def test_heading_uses_cosine_and_sine():
    condition = conditioning.build_empty(4)
    condition = conditioning.add_heading(condition, jp.array([1]),
                                         jp.array([jp.pi / 2]))
    assert jp.allclose(condition.observed[1, 3:5], jp.array([0.0, 1.0]),
                       atol=1e-6)
    assert condition.mask[1, 3:5].all()


def test_keyframes_pin_root_heading_and_local_positions():
    root = jp.array([1.0, 0.8, -2.0])
    positions = (skeleton.NEUTRAL_JOINTS + root)[None]
    condition = conditioning.build_empty(5)
    condition = conditioning.add_keyframes(condition, jp.array([3]), positions)
    observed = condition.observed[3]
    assert jp.allclose(observed[:3], root)
    assert jp.allclose(observed[3:5], jp.array([1.0, 0.0]), atol=1e-6)
    local = observed[5:5 + 3 * skeleton.NUM_JOINTS].reshape(-1, 3)
    expected = skeleton.NEUTRAL_JOINTS + jp.array([0.0, 0.8, 0.0])
    assert jp.allclose(local, expected, atol=1e-6)
    assert int(condition.mask[3].sum()) == 5 + 3 * skeleton.NUM_JOINTS
    assert not condition.mask[2].any()


def test_normalize_returns_float_mask_and_feature_shapes():
    condition = conditioning.build_empty(6)
    condition = conditioning.add_root_path(condition, jp.array([0]),
                                           jp.array([[1.0, 1.0]]))
    normalized = conditioning.normalize(condition, build_unit_stats())
    assert normalized.observed.shape == (6, features.FEATURE_DIM)
    assert normalized.mask.dtype == jp.float32
    assert np.allclose(np.asarray(normalized.mask).sum(), 2.0)
