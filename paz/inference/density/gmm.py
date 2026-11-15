import jax.numpy as jp

from paz.backend import distributions
from paz.inference.density.core import (
    _build_distribution_density,
    _flatten_samples,
)
from paz.inference.gmm.em import fit_gmm_em


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
    parameters = fit_gmm_em(
        key, flat_samples, k, covariance, num_iters, regularization
    )
    weights, means, covariances = (
        parameters.weights,
        parameters.means,
        parameters.covariances,
    )
    if covariance == "diag":
        components = distributions.MultivariateNormalDiag(
            loc=means, scale_diag=jp.sqrt(covariances)
        )
    else:
        components = distributions.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=covariances
        )
    distribution = distributions.MixtureSameFamily(
        distributions.Categorical(probs=weights), components
    )
    return _build_distribution_density(distribution, latent_space, unravel)
