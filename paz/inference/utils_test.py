import jax.numpy as jp
import pytest

from paz.inference import utils


def test_validate_space_inv_returns_none():
    assert utils.validate_space("inv") is None


def test_validate_space_fwd_returns_none():
    assert utils.validate_space("fwd") is None


def test_validate_space_invalid_raises():
    with pytest.raises(ValueError):
        utils.validate_space("bad")


def test_slice_batch_selects_index():
    samples = {"x": jp.array([1.0, 2.0, 3.0])}
    sliced = utils.slice_batch(samples, 1)
    assert sliced["x"] == 2.0


def test_squeeze_pytree_squeezes_first_axis():
    samples = {"x": jp.ones((1, 2))}
    squeezed = utils.squeeze_pytree(samples)
    assert squeezed["x"].shape == (2,)


def test_squeeze_pytree_keeps_shape():
    samples = {"x": jp.ones((2,))}
    squeezed = utils.squeeze_pytree(samples)
    assert squeezed["x"].shape == (2,)
