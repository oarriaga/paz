import jax.numpy as jp

from paz.backend import bijectors
from paz.inference.bijector_fitting import fit_bijector as fit_bijector_module
from paz.inference.naming import build_prior_name
from paz.inference.types import Distribution, NodeState, SampleType, Variable
from paz.inference.utils import squeeze_pytree


def Prior(distribution, bijector=None, name=None):
    if not isinstance(distribution, Distribution):
        raise ValueError("Invalid distribution type")
    node_name = name if name is not None else build_prior_name(distribution)

    Sample = SampleType([node_name])
    bijector = bijectors.Identity() if bijector is None else bijector

    def log_prob(forward_sample):
        log_prob = distribution.log_prob(forward_sample)
        log_prob_sum = log_prob.sum()
        return NodeState(Sample(forward_sample), log_prob, log_prob_sum)

    def log_prob_inverse(inverse_sample):
        forward_sample = bijector(inverse_sample)
        log_prob = distribution.log_prob(forward_sample)
        event_ndims = len(distribution.event_shape)
        log_det_args = (inverse_sample, event_ndims)
        log_prob = log_prob + bijector.forward_log_det_jacobian(*log_det_args)
        log_prob_sum = log_prob.sum()
        return NodeState(Sample(forward_sample), log_prob, log_prob_sum)

    def sample_inverse(key, num_samples=1):
        forward_sample = distribution.sample(num_samples, seed=key)
        inverse_sample = bijector.inverse(forward_sample)
        if num_samples == 1:
            inverse_sample = jp.squeeze(inverse_sample, axis=0)
        return inverse_sample

    def sample(key, num_samples=1):
        forward_samples = distribution.sample(num_samples, seed=key)
        if num_samples == 1:
            forward_samples = squeeze_pytree(forward_samples)
        return forward_samples

    def fit_bijector(
        key,
        target_distribution,
        num_samples=10_000,
        optimizer=None,
        num_steps=1000,
        print=True,
        return_losses=False,
    ):
        optimized_bijector, losses = fit_bijector_module(
            distribution,
            target_distribution,
            bijector,
            key,
            num_samples=num_samples,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=print,
        )
        new_prior = Prior(
            target_distribution, bijector=optimized_bijector, name=node_name
        )
        return (new_prior, losses) if return_losses else new_prior

    return Variable(
        log_prob,
        log_prob_inverse,
        sample,
        sample_inverse,
        node_name,
        [],
        distribution,
        None,
        bijector,
        fit=fit_bijector,
        fit_bijector=fit_bijector,
    )
