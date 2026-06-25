import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax.numpy as jp

from paz.models.transformers import logits


def test_soft_cap_none_is_identity():
    values = jp.array([[1.0, -2.0, 30.0]])
    out = np.asarray(logits.soft_cap(values, None))
    assert np.allclose(out, [[1.0, -2.0, 30.0]])


def test_soft_cap_matches_scaled_tanh():
    values = jp.array([[0.0, 5.0, -5.0]])
    out = np.asarray(logits.soft_cap(values, 10.0))
    expected = 10.0 * np.tanh(np.array([0.0, 5.0, -5.0]) / 10.0)
    assert np.allclose(out, expected)


def test_soft_cap_saturates_at_cap():
    values = jp.array([[1e6, -1e6]])
    out = np.asarray(logits.soft_cap(values, 30.0))
    assert np.allclose(out, [[30.0, -30.0]], atol=1e-3)


def test_apply_temperature_scales():
    values = jp.array([[2.0, 4.0]])
    out = np.asarray(logits.apply_temperature(values, 2.0))
    assert np.allclose(out, [[1.0, 2.0]])


def test_apply_top_k_keeps_k_largest():
    values = jp.array([[1.0, 3.0, 2.0, 0.0]])
    out = np.asarray(logits.apply_top_k(values, 2))
    assert out[0, 1] == 3.0 and out[0, 2] == 2.0
    assert np.isneginf(out[0, 0]) and np.isneginf(out[0, 3])


def test_apply_top_p_keeps_nucleus():
    values = jp.array([[0.0, 10.0, 9.0, -10.0]])
    out = np.asarray(logits.apply_top_p(values, 0.5))
    # 10.0 alone already exceeds 0.5 of the mass; only it survives.
    assert out[0, 1] == 10.0
    assert np.isneginf(out[0, 0]) and np.isneginf(out[0, 3])


def test_disabled_transforms_are_identity():
    values = jp.array([[1.0, 2.0, 3.0]])
    assert np.allclose(np.asarray(logits.apply_top_k(values, 0)), [[1, 2, 3]])
    assert np.allclose(np.asarray(logits.apply_top_p(values, 1.0)), [[1, 2, 3]])
