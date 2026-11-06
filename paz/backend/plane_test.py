import jax
import jax.numpy as jp
import numpy as np
import pytest

import paz


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def plane_points():
    xz_key, y_key = jax.random.split(jax.random.PRNGKey(1))
    xz = jax.random.uniform(xz_key, (4096, 2), minval=-1.0, maxval=1.0)
    y = -0.5 + 0.002 * jax.random.normal(y_key, (4096,))
    return jp.stack([xz[:, 0], y, xz[:, 1]], axis=1)


@pytest.fixture
def points_with_outliers(plane_points):
    num_outliers = int(0.3 * len(plane_points) / 0.7)
    shape = (num_outliers, 3)
    outlier_key = jax.random.PRNGKey(2)
    outliers = jax.random.uniform(outlier_key, shape, minval=-1.0, maxval=1.0)
    return jp.concatenate([plane_points, outliers], axis=0)


def test_fit_RANSAC_recovers_noisy_plane(key, plane_points):
    normal, offset, _ = paz.plane.fit_RANSAC(key, plane_points, threshold=0.01)
    assert np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-3)
    assert np.allclose(offset, 0.5, atol=1e-3)


# The mask is measured from the best three-point plane, not the true one,
# so points near the threshold may fall on either side.
def test_fit_RANSAC_ignores_outliers(key, points_with_outliers):
    args = (key, points_with_outliers)
    normal, offset, inliers = paz.plane.fit_RANSAC(*args, threshold=0.01)
    distances = jp.abs(points_with_outliers[:, 1] + 0.5)
    assert np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-3)
    assert np.allclose(offset, 0.5, atol=1e-3)
    assert jp.all(inliers[distances < 0.005])
    assert not jp.any(inliers[distances > 0.015])


def test_fit_RANSAC_jit_matches_eager(key, plane_points):
    fit = jax.jit(paz.plane.fit_RANSAC, static_argnames=["steps"])
    jitted = fit(key, plane_points, steps=50)
    eager = paz.plane.fit_RANSAC(key, plane_points, steps=50)
    assert np.allclose(jitted[0], eager[0], atol=1e-5)
    assert np.allclose(jitted[1], eager[1], atol=1e-5)
    assert np.array_equal(jitted[2], eager[2])


def test_fit_RANSAC_compiles_once(key, plane_points):
    fit = jax.jit(paz.plane.fit_RANSAC, static_argnames=["steps"])
    cache_sizes = []
    for shift in range(3):
        fit(key, plane_points + shift)
        cache_sizes.append(fit._cache_size())
    assert cache_sizes[0] == cache_sizes[-1]


def test_weighted_least_squares_with_zero_weights(points_with_outliers):
    selected = jp.abs(points_with_outliers[:, 1] + 0.5) < 0.01
    weights = selected.astype(points_with_outliers.dtype)
    args = (points_with_outliers, weights)
    weighted = paz.plane.fit_weighted_least_squares(*args)
    expected = paz.plane.fit_least_squares(points_with_outliers[selected])
    for value, expected_value in zip(weighted, expected):
        assert np.allclose(value, expected_value, atol=1e-5)
