import jax
import jax.numpy as jp

from paz.backend import algebra


def test_safe_norm_gradient_is_finite_for_a_zero_row():
    x = jp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    sum_norms = lambda x: jp.sum(algebra.safe_norm(x, axis=-1))
    expected_gradient = jp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]) / 3.0
    assert jp.allclose(algebra.safe_norm(x, axis=-1), jp.array([0.0, 3.0]))
    assert jp.allclose(jax.grad(sum_norms)(x), expected_gradient)


def test_safe_norm_without_axis_matches_norm():
    x = jp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    assert jp.isclose(algebra.safe_norm(x), 3.0)
    assert algebra.safe_norm(jp.zeros(3)) == 0.0
    assert jp.all(jax.grad(algebra.safe_norm)(jp.zeros(3)) == 0.0)
