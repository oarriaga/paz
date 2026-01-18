import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.gmm.em import fit_gmm_em


tfd = tfp.distributions


def _sample_1d_mixture(key, num_samples):
    weights = jp.array([0.6, 0.4])
    means = jp.array([-2.0, 2.0])
    stdvs = jp.array([0.5, 0.3])
    distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.Normal(loc=means, scale=stdvs),
    )
    samples = distribution.sample(num_samples, seed=key)[:, None]
    return samples, weights, means


def _sample_2d_mixture(key, num_samples):
    weights = jp.array([0.5, 0.5])
    means = jp.array([[-3.0, -0.5], [3.0, 0.5]])
    covariances = jp.array([
        [[0.6, 0.2], [0.2, 0.4]],
        [[0.5, -0.1], [-0.1, 0.3]],
    ])
    distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=covariances
        ),
    )
    samples = distribution.sample(num_samples, seed=key)
    return samples, weights, means


def test_fit_gmm_em_requires_key():
    samples = jp.zeros((4, 1))
    with pytest.raises(ValueError):
        fit_gmm_em(None, samples, k=2)


def test_fit_gmm_em_weights_sum_to_one():
    key = jax.random.PRNGKey(0)
    sample_key, fit_key = jax.random.split(key)
    samples, _, _ = _sample_1d_mixture(sample_key, 200)
    parameters = fit_gmm_em(fit_key, samples, k=2, num_iters=10)
    total = jp.abs(parameters.weights.sum() - 1.0)
    assert float(total) < 1e-5


def test_fit_gmm_em_recovers_means_1d():
    key = jax.random.PRNGKey(1)
    sample_key, fit_key = jax.random.split(key)
    samples, _, means = _sample_1d_mixture(sample_key, 400)
    parameters = fit_gmm_em(fit_key, samples, k=2, num_iters=40)
    fit_means = parameters.means[:, 0]
    order = jp.argsort(fit_means)
    max_diff = jp.max(jp.abs(fit_means[order] - jp.sort(means)))
    assert float(max_diff) < 0.5


def test_fit_gmm_em_recovers_weights_1d():
    key = jax.random.PRNGKey(2)
    sample_key, fit_key = jax.random.split(key)
    samples, weights, _ = _sample_1d_mixture(sample_key, 400)
    parameters = fit_gmm_em(fit_key, samples, k=2, num_iters=40)
    fit_means = parameters.means[:, 0]
    order = jp.argsort(fit_means)
    max_diff = jp.max(jp.abs(parameters.weights[order] - weights))
    assert float(max_diff) < 0.2


def test_fit_gmm_em_recovers_means_full_covariance():
    key = jax.random.PRNGKey(3)
    sample_key, fit_key = jax.random.split(key)
    samples, _, means = _sample_2d_mixture(sample_key, 600)
    parameters = fit_gmm_em(fit_key, samples, k=2, covariance="full", num_iters=50)
    order = jp.argsort(parameters.means[:, 0])
    sorted_means = parameters.means[order]
    max_dist = jp.max(jp.linalg.norm(sorted_means - means, axis=1))
    assert float(max_dist) < 0.7
