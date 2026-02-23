from collections import namedtuple

import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.core import _compute_covariance


tfd = tfp.distributions

GMMParameters = namedtuple("GMMParameters", ["weights", "means", "covariances"])


def build_gmm_parameters(
    key, samples, num_components, covariance="diag", regularization=1e-6
):
    _validate_inputs(key, covariance)
    flat_samples = jp.asarray(samples)
    if flat_samples.ndim == 1:
        flat_samples = flat_samples[:, None]
    args = (key, flat_samples, num_components, covariance, regularization)
    return GMMParameters(*_initialize_parameters(*args))


def fit_gmm_em(key, flat_samples, k, covariance="diag", *args, **kwargs):
    num_iters, regularization = _parse_em_args(args, kwargs)
    _validate_inputs(key, covariance)
    num_samples = flat_samples.shape[0]
    init_args = (key, flat_samples, k, covariance, regularization)
    weights, means, covariances = _initialize_parameters(*init_args)
    previous_ll = -jp.inf
    for _ in range(num_iters):
        e_step_args = (flat_samples, weights, means, covariances, covariance)
        responsibilities = _e_step(*e_step_args)
        log_ll = _compute_log_likelihood(
            flat_samples, weights, means, covariances, covariance
        )
        if jp.abs(log_ll - previous_ll) < 1e-6:
            break
        previous_ll = log_ll
        m_step_args = (flat_samples, responsibilities, num_samples)
        weights, means, nk = _m_step_means_weights(*m_step_args)
        m_step_inputs = (flat_samples, responsibilities, means, nk)
        covariance_settings = (covariance, regularization)
        covariances = _m_step_covariances(m_step_inputs, *covariance_settings)
    return GMMParameters(weights, means, covariances)


def _parse_em_args(args, kwargs):
    if len(args) > 2:
        raise ValueError("Expected at most 2 positional args.")
    num_iters = kwargs.pop("num_iters", None)
    regularization = kwargs.pop("regularization", None)
    if kwargs:
        extra = ", ".join(sorted(kwargs))
        raise ValueError("Unknown fit args: " + extra)
    if len(args) >= 1:
        if num_iters is not None:
            raise ValueError("num_iters specified twice.")
        num_iters = args[0]
    if len(args) == 2:
        if regularization is not None:
            raise ValueError("regularization specified twice.")
        regularization = args[1]
    num_iters = 50 if num_iters is None else num_iters
    regularization = 1e-6 if regularization is None else regularization
    return num_iters, regularization


def _validate_inputs(key, covariance):
    if key is None:
        raise ValueError("key is required for GMM fitting")
    if covariance not in ("diag", "full"):
        raise ValueError("covariance must be 'diag' or 'full'")


def _initialize_parameters(key, flat_samples, k, covariance, regularization):
    key, subkey = jax.random.split(key)
    means = _kmeans_plus_plus(subkey, flat_samples, k)
    weights = jp.full((k,), 1.0 / k)
    cov_args = (flat_samples, k, covariance, regularization)
    covariances = _initialize_covariances(*cov_args)
    return weights, means, covariances


def _kmeans_plus_plus(key, flat_samples, k):
    num_samples = flat_samples.shape[0]
    key, subkey = jax.random.split(key)
    first_index = jax.random.choice(subkey, num_samples)
    centers = [flat_samples[first_index]]
    for _ in range(1, k):
        center_stack = jp.stack(centers)
        diffs = flat_samples[:, None, :] - center_stack[None, :, :]
        distances_sq = jp.sum(diffs**2, axis=-1)
        min_distances_sq = jp.min(distances_sq, axis=1)
        probabilities = min_distances_sq / (jp.sum(min_distances_sq) + 1e-10)
        key, subkey = jax.random.split(key)
        next_index = jax.random.choice(subkey, num_samples, p=probabilities)
        centers.append(flat_samples[next_index])
    return jp.stack(centers)


def _initialize_covariances(flat_samples, k, covariance, regularization):
    if covariance == "diag":
        base_var = jp.var(flat_samples, axis=0) + regularization
        return jp.tile(base_var, (k, 1))
    base_cov = _compute_covariance(flat_samples, regularization)
    return jp.tile(base_cov, (k, 1, 1))


def _compute_log_likelihood(flat_samples, weights, means, covariances, covariance):
    log_probs = _component_log_prob(flat_samples, means, covariances, covariance)
    log_joint = log_probs + jp.log(weights)[:, None]
    return jp.sum(logsumexp(log_joint, axis=0))


def _e_step(flat_samples, weights, means, covariances, covariance):
    log_prob_args = (flat_samples, means, covariances, covariance)
    log_probs = _component_log_prob(*log_prob_args)
    log_joint = log_probs + jp.log(weights)[:, None]
    log_norm = logsumexp(log_joint, axis=0)
    return jp.exp(log_joint - log_norm)


def _m_step_means_weights(flat_samples, responsibilities, num_samples):
    nk = responsibilities.sum(axis=1) + 1e-8
    weights = nk / num_samples
    means = (responsibilities @ flat_samples) / nk[:, None]
    return weights, means, nk


def _m_step_covariances(m_step_inputs, covariance, regularization):
    flat_samples, responsibilities, means, nk = m_step_inputs
    diff = flat_samples[None, :, :] - means[:, None, :]
    if covariance == "diag":
        var = (responsibilities[:, :, None] * diff ** 2).sum(axis=1)
        return var / nk[:, None] + regularization
    cov = jp.einsum("kn,kni,knj->kij", responsibilities, diff, diff)
    cov = cov / nk[:, None, None]
    return cov + regularization * jp.eye(cov.shape[-1])


def _component_log_prob(flat_samples, means, covariances, covariance):
    if covariance == "diag":
        def log_prob(mean, variance):
            parameters = dict(loc=mean, scale_diag=jp.sqrt(variance))
            distribution = tfd.MultivariateNormalDiag(**parameters)
            return distribution.log_prob(flat_samples)
        return jax.vmap(log_prob)(means, covariances)

    def log_prob(mean, cov):
        parameters = dict(loc=mean, covariance_matrix=cov)
        distribution = tfd.MultivariateNormalFullCovariance(**parameters)
        return distribution.log_prob(flat_samples)

    return jax.vmap(log_prob)(means, covariances)
