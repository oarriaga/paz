import pytest
import numpy as np
import jax
import jax.numpy as jp

from paz.backend import algebra

QUADRATICS = [(1.0, -3.0, 2.0), (2.0, 5.0, -1.0), (-1.5, 0.5, 4.0)]


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


def solve_with_numpy(a, b, c):
    discriminator = np.sqrt(b**2 - 4.0 * a * c)
    return (-b - discriminator) / (2.0 * a), (-b + discriminator) / (2.0 * a)


@pytest.mark.parametrize("a, b, c", QUADRATICS)
def test_solve_quadratic_matches_float64_formula(a, b, c):
    solution_A, solution_B, is_valid = algebra.solve_quadratic(a, b, c)
    expected_A, expected_B = solve_with_numpy(a, b, c)
    assert is_valid
    assert jp.isclose(solution_A, expected_A, rtol=1e-5)
    assert jp.isclose(solution_B, expected_B, rtol=1e-5)


def test_solve_quadratic_keeps_the_small_root_under_cancellation():
    solution_A, solution_B, is_valid = algebra.solve_quadratic(1.0, -1e4, 1.0)
    assert is_valid
    assert jp.isclose(solution_A, 1e-4, rtol=1e-5)
    assert jp.isclose(solution_B, 1e4, rtol=1e-5)


def test_solve_quadratic_returns_the_double_root():
    solution_A, solution_B, _ = algebra.solve_quadratic(1.0, -2.0, 1.0)
    assert solution_A == 1.0
    assert solution_B == 1.0


def test_solve_quadratic_solves_the_linear_equation():
    solution_A, solution_B, is_valid = algebra.solve_quadratic(0.0, 2.0, -4.0)
    assert is_valid
    assert solution_A == 2.0
    assert solution_B == 2.0


def test_solve_quadratic_gradient_is_finite_without_real_roots():
    solve = lambda c: algebra.solve_quadratic(1.0, -2.0, c)[0]
    assert jp.isfinite(jax.grad(solve)(1.0))
    assert jp.isfinite(jax.grad(solve)(3.0))
    assert jp.isclose(jax.grad(solve)(0.0), 0.5)


def test_normalize_returns_unit_vectors_for_small_norms():
    vectors = jp.array([[3.0, 4.0, 0.0], [1e-5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    norms = jp.linalg.norm(algebra.normalize(vectors), axis=-1)
    assert jp.allclose(norms, jp.array([1.0, 1.0, 0.0]))
