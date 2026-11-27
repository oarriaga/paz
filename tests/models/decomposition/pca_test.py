import numpy as np
import jax
import jax.numpy as jp
import pytest
import paz


@pytest.fixture
def x():
    scales = np.array([5.0, 2.0, 0.5, 0.1])
    return np.random.default_rng(0).normal(size=(50, 4)) * scales + 3.0


def compute_covariance_eigenvalues(x, num_components):
    eigenvalues = np.linalg.eigvalsh(np.cov(x.T))
    return eigenvalues[::-1][:num_components]


def test_fit_variances_are_covariance_eigenvalues(x):
    state = paz.PCA.fit(jp.array(x), 2)
    expected = compute_covariance_eigenvalues(x, 2)
    assert np.allclose(state.variances, expected, rtol=1e-4)


def test_fit_randomized_variances_are_covariance_eigenvalues(x):
    state = paz.PCA.fit_randomized(jp.array(x), 2, jax.random.PRNGKey(0))
    expected = compute_covariance_eigenvalues(x, 2)
    assert np.allclose(state.variances, expected, rtol=1e-3)


def test_fit_and_fit_randomized_return_same_mean(x):
    state = paz.PCA.fit(jp.array(x), 2)
    randomized = paz.PCA.fit_randomized(jp.array(x), 2, jax.random.PRNGKey(0))
    assert state.mean.shape == randomized.mean.shape == (4,)
    assert np.allclose(state.mean, x.mean(axis=0), atol=1e-5)
    assert np.allclose(randomized.mean, x.mean(axis=0), atol=1e-5)


def test_transform_variance_matches_fit_variances(x):
    state = paz.PCA.fit(jp.array(x), 2)
    projected = paz.PCA.transform(jp.array(x), state)
    assert projected.shape == (50, 2)
    variances = np.var(projected, axis=0, ddof=1)
    assert np.allclose(variances, state.variances, rtol=1e-4)


def test_fit_randomized_jits_with_static_num_components(x):
    fit = jax.jit(paz.PCA.fit_randomized, static_argnums=1)
    state = fit(jp.array(x), 2, jax.random.PRNGKey(0))
    expected = compute_covariance_eigenvalues(x, 2)
    assert np.allclose(state.variances, expected, rtol=1e-3)


def test_fit_randomized_with_fewer_samples_than_twice_components():
    x = np.random.default_rng(0).normal(size=(5, 20))
    state = paz.PCA.fit_randomized(jp.array(x), 3, jax.random.PRNGKey(0))
    expected = compute_covariance_eigenvalues(x, 3)
    assert state.base.shape == (3, 20)
    assert np.allclose(state.variances, expected, rtol=1e-3)
