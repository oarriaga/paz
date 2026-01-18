from jax.scipy.stats import gaussian_kde

from paz.inference.density.core import (
    _build_kde_density,
    _flatten_samples,
)


def build_kde_density(samples, latent_space, bw_method="scott"):
    flat_samples, unravel = _flatten_samples(samples, latent_space)
    kde = gaussian_kde(flat_samples.T, bw_method=bw_method)
    return _build_kde_density(kde, latent_space, unravel)
