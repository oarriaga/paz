import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.core import (
    _build_distribution_density,
    _compute_covariance,
    _flatten_samples,
    _lowrank_covariance,
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
