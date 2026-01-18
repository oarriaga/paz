import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.core import (
    _build_distribution_density,
    _flatten_samples,
)
from paz.inference.gmm.em import fit_gmm_em

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
    parameters = fit_gmm_em(
        key, flat_samples, k, covariance, num_iters, regularization
    )
    weights, means, covariances = (
        parameters.weights,
        parameters.means,
        parameters.covariances,
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
