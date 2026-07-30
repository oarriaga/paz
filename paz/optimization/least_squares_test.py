from functools import partial

import jax
import jax.numpy as jp
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from paz.backend import pinhole
from paz.backend.lie import SE3
from paz.optimization import gauss_newton
from paz.optimization import gauss_newton_on_manifold
from paz.optimization import levenberg_marquardt
from paz.optimization import solve_normal_equations


def build_linear_problem():
    generator = np.random.default_rng(0)
    left_basis = np.linalg.qr(generator.normal(size=(20, 5)))[0]
    right_basis = np.linalg.qr(generator.normal(size=(5, 5)))[0]
    singular_values = np.linspace(1.0, 2.0, 5)
    A = jp.asarray(left_basis * singular_values @ right_basis)
    b = jp.asarray(generator.normal(size=20))
    return A, b


def build_pose_problem():
    generator = np.random.default_rng(2)
    low, high = [-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]
    points3D = jp.asarray(generator.uniform(low, high, size=(100, 3)))
    intrinsics = jp.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    true_pose = SE3.Exp(jp.array([0.03, -0.02, 0.01, 0.1, -0.2, 0.15]))
    camera_matrix = pinhole.make_camera_matrix(intrinsics, true_pose)
    project = jax.vmap(partial(pinhole.project_to_2D, camera_matrix))
    return points3D, intrinsics, true_pose, project(points3D)


def rosenbrock_residual(parameters):
    x, y = parameters
    return jp.array([10.0 * (y - x**2), 1.0 - x])


def test_gauss_newton_solves_linear_problem_in_one_iteration():
    A, b = build_linear_problem()
    residual_fn = lambda parameters: A @ parameters - b
    result = gauss_newton(residual_fn, jp.zeros(5), 1)
    expected = jp.linalg.lstsq(A, b)[0]
    assert bool(result.valid)
    assert jp.allclose(result.parameters, expected, atol=1e-5)


def test_levenberg_marquardt_solves_linear_problem():
    A, b = build_linear_problem()
    residual_fn = lambda parameters: A @ parameters - b
    result = levenberg_marquardt(residual_fn, jp.zeros(5), 20, 1e-3)
    expected = jp.linalg.lstsq(A, b)[0]
    expected_cost = 0.5 * jp.sum((A @ expected - b) ** 2)
    assert bool(result.valid)
    assert jp.allclose(result.parameters, expected, atol=1e-3)
    assert result.cost <= 1.01 * expected_cost


def test_solvers_match_scipy_on_exponential_fit():
    generator = np.random.default_rng(1)
    times = np.linspace(0.0, 2.0, 40)
    observed = 2.0 * np.exp(-1.3 * times) + 0.5
    observed = observed + generator.normal(0.0, 0.01, size=40)
    times_jax, observed_jax = jp.asarray(times), jp.asarray(observed)

    def residual_fn(parameters):
        a, b, c = parameters
        return a * jp.exp(-b * times_jax) + c - observed_jax

    def numpy_residual(parameters):
        a, b, c = parameters
        return a * np.exp(-b * times) + c - observed

    start = jp.array([1.0, 1.0, 0.0])
    reference = scipy_least_squares(numpy_residual, np.asarray(start))
    expected = jp.asarray(reference.x)
    fitted_gn = gauss_newton(residual_fn, start, 30)
    fitted_lm = levenberg_marquardt(residual_fn, start, 50, 1e-3)
    assert jp.allclose(fitted_gn.parameters, expected, atol=1e-3)
    assert jp.allclose(fitted_lm.parameters, expected, atol=1e-3)
    assert fitted_gn.cost <= 1.01 * reference.cost
    assert fitted_lm.cost <= 1.01 * reference.cost


def test_levenberg_marquardt_converges_where_gauss_newton_stalls():
    start = jp.array([-1.2, 1.0])
    stalled = gauss_newton(rosenbrock_residual, start, 30)
    fitted = levenberg_marquardt(rosenbrock_residual, start, 100, 1.0)
    assert jp.allclose(stalled.parameters, start)
    assert jp.allclose(fitted.parameters, jp.array([1.0, 1.0]), atol=1e-3)
    assert fitted.cost < 1e-6


def test_gauss_newton_on_manifold_recovers_pose():
    points3D, intrinsics, true_pose, observed = build_pose_problem()

    def residual_fn(pose):
        camera_matrix = pinhole.make_camera_matrix(intrinsics, pose)
        project = jax.vmap(partial(pinhole.project_to_2D, camera_matrix))
        return (project(points3D) - observed).ravel()

    perturbation = jp.array([0.02, -0.03, 0.01, 0.05, -0.04, 0.03])
    start_pose = SE3.retract(true_pose, perturbation)
    args = (residual_fn, start_pose, SE3.retract, 6, 10)
    result = gauss_newton_on_manifold(*args)
    error = SE3.local_coordinates(result.parameters, true_pose)
    rotation_error = jp.linalg.norm(error[:3])
    translation_error = jp.linalg.norm(error[3:])
    assert bool(result.valid)
    assert rotation_error < jp.deg2rad(0.01)
    assert translation_error < 1e-4
    assert jp.all(jp.diff(result.cost_trace) <= 0.0)


def test_levenberg_marquardt_stays_finite_on_rank_deficient_jacobian():
    generator = np.random.default_rng(3)
    columns = jp.asarray(generator.normal(size=(10, 2)))
    A = jp.concatenate([columns, jp.zeros((10, 1))], axis=1)
    b = jp.asarray(generator.normal(size=10))
    residual_fn = lambda parameters: A @ parameters - b
    result = levenberg_marquardt(residual_fn, jp.zeros(3), 20, 1e-3)
    assert bool(result.valid)
    assert jp.all(jp.isfinite(result.parameters))
    assert jp.isfinite(result.cost)
    assert jp.all(jp.isfinite(result.cost_trace))
    assert jp.all(jp.isfinite(result.damping_trace))
    assert result.cost <= 0.5 * jp.sum(b**2)


def test_solve_normal_equations_reports_singularity_as_nonfinite():
    JtJ = jp.array([[1.0, 1.0], [1.0, 1.0]])
    delta = solve_normal_equations(JtJ, jp.array([1.0, 1.0]), 0.0)
    assert not bool(jp.all(jp.isfinite(delta)))


def test_gauss_newton_jits_without_recompiling():
    A, b = build_linear_problem()
    residual_fn = lambda parameters: A @ parameters - b

    @jax.jit
    def solve(parameters):
        return gauss_newton(residual_fn, parameters, 5)

    solve(jp.zeros(5))
    solve(jp.ones(5))
    assert solve._cache_size() == 1


def test_levenberg_marquardt_jits_without_recompiling():
    A, b = build_linear_problem()
    residual_fn = lambda parameters: A @ parameters - b

    @jax.jit
    def solve(parameters):
        return levenberg_marquardt(residual_fn, parameters, 5, 1e-3)

    solve(jp.zeros(5))
    solve(jp.ones(5))
    assert solve._cache_size() == 1
