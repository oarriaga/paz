import jax
import jax.numpy as jp
from jax import flatten_util
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent_space import as_latent_samples, to_forward_samples, to_inverse_samples
from paz.inference.types import Density

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
    return _build_density(distribution, latent_space, unravel)


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
    return _build_density(distribution, latent_space, unravel)


def build_kde_density(samples, latent_space, bw_method="scott"):
    from jax.scipy.stats import gaussian_kde

    flat_samples, unravel = _flatten_samples(samples, latent_space)
    kde = gaussian_kde(flat_samples.T, bw_method=bw_method)
    return _build_kde_density(kde, latent_space, unravel)


def _build_density(distribution, latent_space, unravel):
    def sample(key, num_samples=1, space="inv"):
        _validate_space(space)
        flat = distribution.sample(num_samples, seed=key)
        structured = _unflatten_samples(unravel, flat)
        if num_samples == 1:
            structured = _squeeze_pytree(structured)
        if space == "fwd":
            structured = to_forward_samples(latent_space, structured)
        return structured

    def log_prob(samples, space="inv"):
        _validate_space(space)
        if space == "fwd":
            samples = to_inverse_samples(latent_space, samples)
        samples = as_latent_samples(latent_space, samples)
        flat = _flatten_for_log_prob(samples)
        return distribution.log_prob(flat)

    def prob(samples, space="inv"):
        return jp.exp(log_prob(samples, space=space))

    return Density(sample, log_prob, prob, latent_space, {"distribution": distribution})


def _build_kde_density(kde, latent_space, unravel):
    def sample(key, num_samples=1, space="inv"):
        _validate_space(space)
        flat = kde.resample(key, (num_samples,))
        flat = jp.swapaxes(flat, 0, 1)
        structured = _unflatten_samples(unravel, flat)
        if num_samples == 1:
            structured = _squeeze_pytree(structured)
        if space == "fwd":
            structured = to_forward_samples(latent_space, structured)
        return structured

    def log_prob(samples, space="inv"):
        _validate_space(space)
        if space == "fwd":
            samples = to_inverse_samples(latent_space, samples)
        samples = as_latent_samples(latent_space, samples)
        flat = _flatten_for_log_prob(samples)
        if flat.ndim == 1:
            return kde.logpdf(flat)
        return kde.logpdf(flat.T)

    def prob(samples, space="inv"):
        return jp.exp(log_prob(samples, space=space))

    return Density(sample, log_prob, prob, latent_space, {"kde": kde})


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
            covariances = _regularize_covariance(covariances, regularization)
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
    batch_size = _get_leading_batch_size(samples)
    if batch_size is None:
        flat, _ = flatten_util.ravel_pytree(samples)
        return flat
    return jax.vmap(lambda s: flatten_util.ravel_pytree(s)[0])(samples)


def _get_leading_batch_size(samples):
    leaves = jax.tree_util.tree_leaves(samples)
    shaped = [leaf for leaf in leaves if hasattr(leaf, "shape")]
    if len(shaped) == 0:
        return None
    if any(len(leaf.shape) == 0 for leaf in shaped):
        return None
    first_dim = shaped[0].shape[0]
    if any(leaf.shape[0] != first_dim for leaf in shaped):
        return None
    return first_dim


def _unflatten_samples(unravel, flat):
    if flat.ndim == 1:
        return unravel(flat)
    return jax.vmap(unravel)(flat)


def _compute_covariance(flat_samples, regularization):
    centered = flat_samples - jp.mean(flat_samples, axis=0)
    cov = centered.T @ centered / max(flat_samples.shape[0] - 1, 1)
    return cov + regularization * jp.eye(flat_samples.shape[1])


def _regularize_covariance(covariances, regularization):
    num_components, num_dims, _ = covariances.shape
    identity = regularization * jp.eye(num_dims)
    return covariances + identity[None, :, :]


def _lowrank_covariance(covariance, rank, regularization):
    num_dims = covariance.shape[0]
    if rank is None:
        rank = min(5, num_dims)
    eigvals, eigvecs = jp.linalg.eigh(covariance)
    top = eigvecs[:, -rank:]
    vals = eigvals[-rank:]
    approx = (top * vals) @ top.T
    return approx + regularization * jp.eye(num_dims)


def _squeeze_pytree(pytree):
    def squeeze_leaf(leaf):
        if hasattr(leaf, "shape") and len(leaf.shape) > 0 and leaf.shape[0] == 1:
            return jp.squeeze(leaf, axis=0)
        return leaf

    return jax.tree.map(squeeze_leaf, pytree)


def _validate_space(space):
    if space not in ("inv", "fwd"):
        raise ValueError("space must be 'inv' or 'fwd'")
