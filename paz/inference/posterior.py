import jax
import jax.numpy as jp

from paz.inference.density import (
    build_gaussian_density,
    build_gmm_density,
    build_kde_density,
)
from paz.inference.latent_space import as_latent_samples, to_forward_samples
from paz.inference.types import MCMCPosteriorType
from paz.inference.utils import squeeze_pytree


def MCMCPosterior(inverse_samples, infos, config, latent_space):
    def sample_inverse(key, num_samples=1):
        positions = inverse_samples.position
        num_samples_chain, num_chains = _get_chain_shapes(positions)
        num_draws = num_samples_chain * num_chains
        indices = jax.random.choice(
            key, num_draws, shape=(num_samples,), replace=True
        )
        sample_indices = indices // num_chains
        chain_indices = indices % num_chains

        draws = jax.tree.map(
            lambda leaf: leaf[sample_indices, chain_indices], positions
        )
        draws = as_latent_samples(latent_space, draws)
        if num_samples == 1:
            draws = squeeze_pytree(draws)
        return draws

    def sample(key, num_samples=1):
        draws = sample_inverse(key, num_samples)
        return to_forward_samples(latent_space, draws)

    forward_samples = to_forward_samples(
        latent_space, inverse_samples.position
    )

    def diagnostics():
        acceptance_rate = jp.mean(infos.is_accepted, axis=0)
        return {
            "acceptance_rate": acceptance_rate,
            "mean_acceptance_rate": acceptance_rate.mean(),
        }

    def as_density(key=None, method="gaussian", **kwargs):
        method = method.lower()
        draws = _flatten_chain_samples(inverse_samples.position)
        if method == "gaussian":
            return build_gaussian_density(draws, latent_space, **kwargs)
        if method == "gmm":
            k = kwargs.pop("k", None)
            if k is None:
                raise ValueError("k is required for GMM")
            return build_gmm_density(key, draws, latent_space, k, **kwargs)
        if method == "kde":
            return build_kde_density(draws, latent_space, **kwargs)
        raise ValueError(f"Unknown density method '{method}'")

    def to_empirical():
        # TODO we should most likely remove this for now.
        raise NotImplementedError("Empirical posterior is not implemented yet.")

    def update(key, new_data, **kwargs):
        raise NotImplementedError("MCMC posterior updates are not supported.")

    return MCMCPosteriorType(
        sample,
        sample_inverse,
        diagnostics,
        as_density,
        to_empirical,
        update,
        forward_samples,
        inverse_samples,
        infos,
        config,
        latent_space,
    )


def _flatten_chain_samples(samples):
    num_samples, num_chains = _get_chain_shapes(samples)

    def reshape(leaf):
        if not hasattr(leaf, "shape"):
            return leaf
        if leaf.ndim < 2:
            raise ValueError("Expected chain dimension for MCMC samples.")
        return jp.reshape(leaf, (num_samples * num_chains,) + leaf.shape[2:])

    return jax.tree.map(reshape, samples)


def _get_chain_shapes(samples):
    leaves = jax.tree_util.tree_leaves(samples)
    if len(leaves) == 0:
        raise ValueError("Empty samples.")
    leaf = leaves[0]
    if not hasattr(leaf, "shape") or len(leaf.shape) < 2:
        raise ValueError(
            "MCMC samples require shape (num_samples, num_chains, ...)."
        )
    return leaf.shape[0], leaf.shape[1]


__all__ = ["MCMCPosterior"]
