import jax
import jax.numpy as jp
import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")
from tensorflow_probability.substrates import jax as tfp

from paz.backend import bijectors as B

tfb = tfp.bijectors

ROUNDS = 25
KEY = jax.random.PRNGKey(100)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def sample(key, shape, low, high):
    return jax.random.uniform(key, shape, jp.float64, low, high)


def assert_close(name, ours, theirs, atol=1e-9, rtol=1e-7):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert ours.shape == theirs.shape, f"{name}: shape"
    finite = jp.isfinite(ours) & jp.isfinite(theirs)
    assert jp.array_equal(jp.isfinite(ours), jp.isfinite(theirs)), name
    if int(finite.sum()):
        gap = jp.abs(ours[finite] - theirs[finite])
        assert bool(jp.all(gap <= atol + rtol * jp.abs(theirs[finite]))), name


def build_bijectors(key):
    scale = sample(key[1], (), 0.05, 5)
    low = sample(key[2], (), -4, -0.3)
    high = sample(key[3], (), 0.3, 4)
    power = sample(key[4], (), 0.3, 3.0)
    return [
        ("Shift", B.Shift(scale), tfb.Shift(scale), 0),
        ("Scale", B.Scale(scale), tfb.Scale(scale), 0),
        ("Sigmoid", B.Sigmoid(), tfb.Sigmoid(), 0),
        ("SigmoidBounded", B.Sigmoid(low, high), tfb.Sigmoid(low, high), 0),
        ("Exp", B.Exp(), tfb.Exp(), 0),
        ("Log", B.Log(), tfb.Log(), 0),
        ("Softplus", B.Softplus(), tfb.Softplus(), 0),
        ("Tanh", B.Tanh(), tfb.Tanh(), 0),
        ("NormalCDF", B.NormalCDF(), tfb.NormalCDF(), 0),
        ("Square", B.Square(), tfb.Square(), 0),
        ("Reciprocal", B.Reciprocal(), tfb.Reciprocal(), 0),
        ("Power", B.Power(power), tfb.Power(power), 0),
    ]


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_scalar_bijector_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
    domain = sample(key[0], (5,), 0.05, 4)
    for name, ours, theirs, event_ndims in build_bijectors(key):
        forward = jp.asarray(np.asarray(theirs(domain)).copy())
        assert_close(f"{name}.forward", ours(domain), theirs(domain))
        assert_close(f"{name}.inverse", ours.inverse(forward),
                     theirs.inverse(forward))
        assert_close(
            f"{name}.fldj",
            ours.forward_log_det_jacobian(domain, event_ndims),
            theirs.forward_log_det_jacobian(domain, event_ndims),
        )
        assert_close(
            f"{name}.ildj",
            ours.inverse_log_det_jacobian(forward, event_ndims),
            theirs.inverse_log_det_jacobian(forward, event_ndims),
        )


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_vector_bijector_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, 500 + round_index), 3)
    domain = sample(key[0], (4,), -3, 3)
    for name, ours, theirs in [
        ("SoftmaxCentered", B.SoftmaxCentered(), tfb.SoftmaxCentered()),
        ("Cumsum", B.Cumsum(), tfb.Cumsum()),
    ]:
        forward = jp.asarray(np.asarray(theirs(domain)).copy())
        assert_close(f"{name}.forward", ours(domain), theirs(domain))
        assert_close(f"{name}.inverse", ours.inverse(forward),
                     theirs.inverse(forward))
        assert_close(
            f"{name}.fldj", ours.forward_log_det_jacobian(domain, 1),
            theirs.forward_log_det_jacobian(domain, 1))


def test_chain_round_trip():
    bijector = B.Chain([B.Shift(-2.0), B.Scale(4.0), B.Sigmoid()])
    inverse = jp.array([-3.0, 0.0, 2.0])
    recovered = bijector.inverse(bijector(inverse))
    assert jp.allclose(recovered, inverse, atol=1e-5)
