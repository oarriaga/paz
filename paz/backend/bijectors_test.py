from pathlib import Path

import jax
import jax.numpy as jp
import pytest

import paz
from paz.backend import bijectors as B

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


# Snapshots hold TensorFlow Probability outputs for the same inputs.
def assert_snapshot(outputs, filename, atol=1e-9, rtol=1e-7):
    array = jp.concatenate([jp.ravel(output) for output in outputs])
    path = str(Path(__file__).parent / "snapshots" / filename)
    paz.assert_snapshot(array, path, rtol=rtol, atol=atol)


def compute_transforms(bijector, domain, event_ndims):
    forward = bijector(domain)
    forward_log_det = bijector.forward_log_det_jacobian(domain, event_ndims)
    inverse_log_det = bijector.inverse_log_det_jacobian(forward, event_ndims)
    inverse = bijector.inverse(forward)
    return [forward, inverse, forward_log_det, inverse_log_det]


def build_bijectors(key):
    scale = sample(key[1], (), 0.05, 5)
    low = sample(key[2], (), -4, -0.3)
    high = sample(key[3], (), 0.3, 4)
    power = sample(key[4], (), 0.3, 3.0)
    return [
        B.Shift(scale),
        B.Scale(scale),
        B.Sigmoid(),
        B.Sigmoid(low, high),
        B.Exp(),
        B.Log(),
        B.Softplus(),
        B.Tanh(),
        B.NormalCDF(),
        B.Square(),
        B.Reciprocal(),
        B.Power(power),
        B.Identity(),
        B.Softplus(hinge_softness=scale),
        B.Invert(B.Exp()),
        B.Invert(B.Softplus()),
        B.Chain([B.Shift(low), B.Scale(high - low), B.Sigmoid()]),
    ]


def test_scalar_bijector_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        domain = sample(key[0], (5,), 0.05, 4)
        for bijector in build_bijectors(key):
            outputs += compute_transforms(bijector, domain, 0)
    assert_snapshot(outputs, "bijectors_scalar.npy")


def build_vector_bijectors(key):
    scale = sample(key[1], (), 0.3, 4)
    return [
        B.SoftmaxCentered(),
        B.Cumsum(),
        B.Chain([B.SoftmaxCentered(), B.Scale(scale)]),
        B.Chain([B.Scale(scale), B.SoftmaxCentered()]),
    ]


def test_vector_bijector_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 500 + round_index), 3)
        domain = sample(key[0], (4,), -3, 3)
        for bijector in build_vector_bijectors(key):
            outputs += compute_transforms(bijector, domain, 1)
    assert_snapshot(outputs, "bijectors_vector.npy")


def test_vector_chain_defaults_to_event_log_det():
    # Called without event_ndims, a chain sums over the event of its members.
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 900 + round_index), 3)
        domain = sample(key[0], (4,), -3, 3)
        for bijector in build_vector_bijectors(key)[2:]:
            outputs.append(bijector.forward_log_det_jacobian(domain))
            forward = bijector(domain)
            outputs.append(bijector.inverse_log_det_jacobian(forward))
    assert_snapshot(outputs, "bijectors_vector_chain_default.npy")


def test_invert_defaults_to_event_log_det():
    values = jp.ones((4, 3))
    log_det = B.Invert(B.Cumsum()).forward_log_det_jacobian(values)
    assert jp.array_equal(log_det, jp.zeros(4))
    simplex = jp.full((4, 3), 1.0 / 3.0)
    forward = B.SoftmaxCentered().inverse(simplex)
    expected = B.SoftmaxCentered().inverse_log_det_jacobian(simplex)
    inverted = B.Invert(B.SoftmaxCentered())
    assert inverted(simplex).shape == forward.shape
    log_det = inverted.forward_log_det_jacobian(simplex)
    assert jp.allclose(log_det, expected, rtol=1e-12)


def test_chain_round_trip():
    bijector = B.Chain([B.Shift(-2.0), B.Scale(4.0), B.Sigmoid()])
    inverse = jp.array([-3.0, 0.0, 2.0])
    recovered = bijector.inverse(bijector(inverse))
    assert jp.allclose(recovered, inverse, atol=1e-5)
