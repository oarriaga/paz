import jax
import jax.numpy as jp
from jax import flatten_util
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
