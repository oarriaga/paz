import jax
import jax.numpy as jp

from paz.optimization import apply_cauchy
from paz.optimization import apply_huber
from paz.optimization import cauchy_weights
from paz.optimization import huber_weights


def test_apply_huber_matches_hand_computed_values():
    squared_errors = jp.array([0.0, 0.25, 4.0])
    costs = apply_huber(squared_errors, 1.0)
    assert jp.allclose(costs, jp.array([0.0, 0.125, 1.5]))


def test_huber_weights_match_hand_computed_values():
    residual_norms = jp.array([0.0, 0.5, 2.0])
    weights = huber_weights(residual_norms, 1.0)
    assert jp.allclose(weights, jp.array([1.0, 1.0, 0.5]))


def test_apply_cauchy_matches_hand_computed_values():
    squared_errors = jp.array([0.0, 4.0])
    costs = apply_cauchy(squared_errors, 2.0)
    expected = jp.array([0.0, 2.0 * jp.log(2.0)])
    assert jp.allclose(costs, expected)


def test_cauchy_weights_match_hand_computed_values():
    residual_norms = jp.array([0.0, 2.0])
    weights = cauchy_weights(residual_norms, 2.0)
    assert jp.allclose(weights, jp.array([1.0, 0.5]))


def test_robust_costs_have_finite_gradients_at_zero():
    huber_gradient = jax.grad(apply_huber)(0.0, 1.0)
    cauchy_gradient = jax.grad(apply_cauchy)(0.0, 2.0)
    assert jp.isfinite(huber_gradient)
    assert jp.isfinite(cauchy_gradient)


def test_robust_weights_have_finite_gradients_at_zero():
    huber_gradient = jax.grad(huber_weights)(0.0, 1.0)
    cauchy_gradient = jax.grad(cauchy_weights)(0.0, 2.0)
    assert jp.isfinite(huber_gradient)
    assert jp.isfinite(cauchy_gradient)


def test_robust_functions_preserve_dtype():
    values = jp.array([0.0, 2.0], dtype=jp.float16)
    assert apply_huber(values, 1.0).dtype == jp.float16
    assert huber_weights(values, 1.0).dtype == jp.float16
    assert apply_cauchy(values, 1.0).dtype == jp.float16
    assert cauchy_weights(values, 1.0).dtype == jp.float16
