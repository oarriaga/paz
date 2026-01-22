import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.gmm.model import GMM


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
    return samples, means


def _initial_parameters():
    weights = jp.array([0.5, 0.5])
    means = jp.array([[-1.0], [1.0]])
    covariances = jp.array([[1.0], [1.0]])
    return weights, means, covariances


def test_gmm_fit_returns_distribution():
    weights, means, covariances = _initial_parameters()
    model = GMM(weights, means, covariances)
    key = jax.random.PRNGKey(0)
    sample_key, fit_key = jax.random.split(key)
    samples, _ = _sample_1d_mixture(sample_key, 200)
    fitted = model.fit(fit_key, samples, method="em", num_iters=10)
    assert isinstance(fitted, tfd.MixtureSameFamily)


def test_gmm_pgm_fit_recovers_means():
    weights, means, covariances = _initial_parameters()
    model = GMM(weights, means, covariances)
    key = jax.random.PRNGKey(1)
    sample_key, fit_key = jax.random.split(key)
    samples, means = _sample_1d_mixture(sample_key, 400)
    fitted = model.fit(fit_key, samples, method="em", num_iters=40)
    components = fitted.components_distribution
    fitted_means = components.loc[:, 0]
    order = jp.argsort(fitted_means)
    max_diff = jp.max(jp.abs(fitted_means[order] - jp.sort(means)))
    assert float(max_diff) < 0.6
