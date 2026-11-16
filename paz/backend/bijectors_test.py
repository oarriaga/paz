import jax
import jax.numpy as jp
import pytest
from jax.scipy.special import ndtr

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


# A constant log-determinant is returned as a scalar, so shapes broadcast.
def assert_close(actual, expected, rtol=1e-7, atol=1e-9):
    assert jp.allclose(actual, expected, rtol, atol)


# The forward value is checked against a formula written by hand, the
# inverse by a round trip, and both log-determinants against the Jacobian
# that autodiff gives for the forward map.
def assert_transforms(bijector, forward, domain, event_ndims, log_det):
    codomain = bijector(domain)
    assert_close(codomain, forward(domain))
    assert_close(bijector.inverse(codomain), domain)
    args = (domain, event_ndims)
    assert_close(bijector.forward_log_det_jacobian(*args), log_det)
    args = (codomain, event_ndims)
    assert_close(bijector.inverse_log_det_jacobian(*args), -log_det)


def build_bijectors(key):
    scale = sample(key[1], (), 0.05, 5)
    low = sample(key[2], (), -4, -0.3)
    high = sample(key[3], (), 0.3, 4)
    power = sample(key[4], (), 0.3, 3.0)
    bounded_sigmoid = lambda x: low + (high - low) * jax.nn.sigmoid(x)
    soft_softplus = lambda x: scale * jp.log1p(jp.exp(x / scale))
    chain = B.Chain([B.Shift(low), B.Scale(high - low), B.Sigmoid()])
    return [
        (B.Shift(scale), lambda x: x + scale),
        (B.Scale(scale), lambda x: x * scale),
        (B.Sigmoid(), lambda x: 1.0 / (1.0 + jp.exp(-x))),
        (B.Sigmoid(low, high), bounded_sigmoid),
        (B.Exp(), jp.exp),
        (B.Log(), jp.log),
        (B.Softplus(), lambda x: jp.log1p(jp.exp(x))),
        (B.Tanh(), jp.tanh),
        (B.NormalCDF(), ndtr),
        (B.Square(), lambda x: x * x),
        (B.Reciprocal(), lambda x: 1.0 / x),
        (B.Power(power), lambda x: x**power),
        (B.Identity(), lambda x: x),
        (B.Softplus(hinge_softness=scale), soft_softplus),
        (B.Invert(B.Exp()), jp.log),
        (B.Invert(B.Softplus()), lambda x: jp.log(jp.expm1(x))),
        (chain, bounded_sigmoid),
    ]


def test_scalar_bijector_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        domain = sample(key[0], (5,), 0.05, 4)
        for bijector, forward in build_bijectors(key):
            slopes = jax.vmap(jax.grad(forward))(domain)
            log_det = jp.log(jp.abs(slopes))
            assert_transforms(bijector, forward, domain, 0, log_det)


def build_vector_bijectors(key):
    scale = sample(key[1], (), 0.3, 4)
    scale_first = B.Chain([B.SoftmaxCentered(), B.Scale(scale)])
    scale_last = B.Chain([B.Scale(scale), B.SoftmaxCentered()])
    scaled_softmax = lambda x: softmax_centered(scale * x)
    # Scale counts all 5 outputs of SoftmaxCentered, as TensorFlow
    # Probability does, although they span only 4 dimensions.
    scale_log_det = 5.0 * jp.log(scale)
    softmax_log_det = lambda x: compute_log_det(softmax_centered, x)
    return [
        (B.SoftmaxCentered(), softmax_centered, softmax_log_det),
        (B.Cumsum(), jp.cumsum, lambda x: compute_log_det(jp.cumsum, x)),
        (scale_first, scaled_softmax,
         lambda x: compute_log_det(scaled_softmax, x)),
        (scale_last, lambda x: scale * softmax_centered(x),
         lambda x: softmax_log_det(x) + scale_log_det),
    ]


def softmax_centered(x):
    return jax.nn.softmax(jp.concatenate([x, jp.zeros(1)]))


# SoftmaxCentered maps 4 numbers to 5, so its Jacobian J is not square and
# the volume change is sqrt(det(J^T J)).
def compute_log_det(forward, domain):
    jacobian = jax.jacfwd(forward)(domain)
    return 0.5 * jp.linalg.slogdet(jacobian.T @ jacobian)[1]


def test_vector_bijector_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 500 + round_index), 3)
        domain = sample(key[0], (4,), -3, 3)
        for bijector, forward, log_det in build_vector_bijectors(key):
            args = (bijector, forward, domain, 1, log_det(domain))
            assert_transforms(*args)


def test_vector_chain_defaults_to_event_log_det():
    # Called without event_ndims, a chain sums over the event of its members.
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 900 + round_index), 3)
        domain = sample(key[0], (4,), -3, 3)
        for bijector, _, log_det in build_vector_bijectors(key)[2:]:
            expected = log_det(domain)
            assert_close(bijector.forward_log_det_jacobian(domain), expected)
            codomain = bijector(domain)
            actual = bijector.inverse_log_det_jacobian(codomain)
            assert_close(actual, -expected)


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
