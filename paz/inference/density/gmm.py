import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.core import (
    _build_distribution_density,
    _compute_covariance,
    _flatten_samples,
)

tfd = tfp.distributions


def build_gmm_density(
    key,
    samples,
    latent_space,
    k,
    covariance="diag",
    num_iters=50,
    regularization=1e-6,
):
    if key is None:
        raise ValueError("key is required for GMM fitting")
    flat_samples, unravel = _flatten_samples(samples, latent_space)
    weights, means, covariances = _fit_gmm(
        key, flat_samples, k, covariance, num_iters, regularization
    )
    if covariance == "diag":
        components = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jp.sqrt(covariances)
        )
    else:
        components = tfd.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=covariances
        )
    distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights), components
    )
    return _build_distribution_density(distribution, latent_space, unravel)


def _fit_gmm(key, flat_samples, k, covariance, num_iters, regularization):
    num_samples = flat_samples.shape[0]
    key, subkey = jax.random.split(key)
    init_indices = jax.random.choice(
        subkey, num_samples, shape=(k,), replace=False
    )
    means = flat_samples[init_indices]
    weights = jp.full((k,), 1.0 / k)
    if covariance == "diag":
        base_var = jp.var(flat_samples, axis=0) + regularization
        covariances = jp.tile(base_var, (k, 1))
    else:
        base_cov = _compute_covariance(flat_samples, regularization)
        covariances = jp.tile(base_cov, (k, 1, 1))

    for _ in range(num_iters):
        log_probs = _component_log_prob(
            flat_samples, means, covariances, covariance
        )
        log_joint = log_probs + jp.log(weights)[:, None]
        log_norm = logsumexp(log_joint, axis=0)
        responsibilities = jp.exp(log_joint - log_norm)
        nk = responsibilities.sum(axis=1) + 1e-8
        weights = nk / num_samples
        means = (responsibilities @ flat_samples) / nk[:, None]
        if covariance == "diag":
            diff = flat_samples[None, :, :] - means[:, None, :]
            var = (responsibilities[:, :, None] * diff ** 2).sum(axis=1)
            covariances = var / nk[:, None] + regularization
        else:
            diff = flat_samples[None, :, :] - means[:, None, :]
            cov = jp.einsum("kn,kni,knj->kij", responsibilities, diff, diff)
            covariances = cov / nk[:, None, None]
            covariances = covariances + regularization * jp.eye(
                covariances.shape[-1]
            )
    return weights, means, covariances


def _component_log_prob(flat_samples, means, covariances, covariance):
    if covariance == "diag":
        def log_prob(mean, variance):
            distribution = tfd.MultivariateNormalDiag(
                loc=mean, scale_diag=jp.sqrt(variance)
            )
            return distribution.log_prob(flat_samples)
        return jax.vmap(log_prob)(means, covariances)

    def log_prob(mean, cov):
        distribution = tfd.MultivariateNormalFullCovariance(
            loc=mean, covariance_matrix=cov
        )
        return distribution.log_prob(flat_samples)

    return jax.vmap(log_prob)(means, covariances)
