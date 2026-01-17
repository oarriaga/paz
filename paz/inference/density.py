import jax
import jax.numpy as jp
from jax import flatten_util
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent_space import (
    as_latent_samples,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.types import Density
from paz.inference.utils import (
    get_leading_batch_size,
    squeeze_pytree,
    validate_space,
)

tfd = tfp.distributions


def build_gaussian_density(
    samples,
    latent_space,
    covariance="diag",
    rank=None,
    regularization=1e-6,
):
    flat_samples, unravel = _flatten_samples(samples, latent_space)
    mean = jp.mean(flat_samples, axis=0)
    if covariance == "diag":
        var = jp.var(flat_samples, axis=0) + regularization
        distribution = tfd.MultivariateNormalDiag(mean, jp.sqrt(var))
    else:
        cov = _compute_covariance(flat_samples, regularization)
        if covariance == "lowrank":
            cov = _lowrank_covariance(cov, rank, regularization)
        distribution = tfd.MultivariateNormalFullCovariance(mean, cov)
    return _build_distribution_density(distribution, latent_space, unravel)


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


def build_kde_density(samples, latent_space, bw_method="scott"):
    from jax.scipy.stats import gaussian_kde

    flat_samples, unravel = _flatten_samples(samples, latent_space)
    kde = gaussian_kde(flat_samples.T, bw_method=bw_method)
    return _build_kde_density(kde, latent_space, unravel)


def _build_distribution_density(distribution, latent_space, unravel):
    def sample_flat(key, num_samples):
        return distribution.sample(num_samples, seed=key)

    def log_prob_flat(flat):
        return distribution.log_prob(flat)

    return _build_density_from_flat(
        sample_flat,
        log_prob_flat,
        latent_space,
        unravel,
        {"distribution": distribution},
    )


def _build_kde_density(kde, latent_space, unravel):
    def sample_flat(key, num_samples):
        flat = kde.resample(key, (num_samples,))
        return jp.swapaxes(flat, 0, 1)

    def log_prob_flat(flat):
        return kde.logpdf(flat) if flat.ndim == 1 else kde.logpdf(flat.T)

    return _build_density_from_flat(
        sample_flat,
        log_prob_flat,
        latent_space,
        unravel,
        {"kde": kde},
    )


def _build_density_from_flat(
    sample_flat, log_prob_flat, latent_space, unravel, metadata
):
    def sample(key, num_samples=1, space="inv"):
        validate_space(space)
        flat = sample_flat(key, num_samples)
        structured = _unflatten_samples(unravel, flat)
        if num_samples == 1:
            structured = squeeze_pytree(structured)
        if space == "fwd":
            structured = to_forward_samples(latent_space, structured)
        return structured

    def log_prob(samples, space="inv"):
        validate_space(space)
        if space == "fwd":
            samples = to_inverse_samples(latent_space, samples)
        samples = as_latent_samples(latent_space, samples)
        flat = _flatten_for_log_prob(samples)
        return log_prob_flat(flat)

    def prob(samples, space="inv"):
        return jp.exp(log_prob(samples, space=space))

    return Density(sample, log_prob, prob, latent_space, metadata)


def _fit_gmm(key, flat_samples, k, covariance, num_iters, regularization):
    num_samples, num_dims = flat_samples.shape
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


def _flatten_samples(samples, latent_space):
    samples = as_latent_samples(latent_space, samples)
    sample = jax.tree.map(lambda x: x[0], samples)
    _, unravel = flatten_util.ravel_pytree(sample)
    flat_samples = jax.vmap(lambda s: flatten_util.ravel_pytree(s)[0])(samples)
    return flat_samples, unravel


def _flatten_for_log_prob(samples):
    batch_size = get_leading_batch_size(samples)
    if batch_size is None:
        flat, _ = flatten_util.ravel_pytree(samples)
        return flat
    return jax.vmap(lambda s: flatten_util.ravel_pytree(s)[0])(samples)


def _unflatten_samples(unravel, flat):
    if flat.ndim == 1:
        return unravel(flat)
    return jax.vmap(unravel)(flat)


def _compute_covariance(flat_samples, regularization):
    centered = flat_samples - jp.mean(flat_samples, axis=0)
    cov = centered.T @ centered / max(flat_samples.shape[0] - 1, 1)
    return cov + regularization * jp.eye(flat_samples.shape[1])


def _lowrank_covariance(covariance, rank, regularization):
    num_dims = covariance.shape[0]
    if rank is None:
        rank = min(5, num_dims)
    eigvals, eigvecs = jp.linalg.eigh(covariance)
    top = eigvecs[:, -rank:]
    vals = eigvals[-rank:]
    approx = (top * vals) @ top.T
    return approx + regularization * jp.eye(num_dims)
