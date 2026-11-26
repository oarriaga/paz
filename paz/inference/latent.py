import jax.numpy as jp

from paz.backend import bijectors
from paz.inference.naming import build_latent_name
from paz.inference.types import NodeState, SampleType, Variable
from paz.inference.utils import compute_inverse_state


def Latent(distribution_fn, bijector=None, name=None):
    if not callable(distribution_fn):
        raise ValueError(f"Input {distribution_fn} must be a callable")
    node_name = name if name is not None else build_latent_name(distribution_fn)

    Sample = SampleType([node_name])
    bijector = bijectors.Identity() if bijector is None else bijector

    def log_prob(forward_sample, *args):
        distribution = distribution_fn(*args)
        log_prob = distribution.log_prob(forward_sample)
        log_prob_sum = log_prob.sum()
        return NodeState(Sample(forward_sample), log_prob, log_prob_sum)

    def log_prob_inverse(inverse_sample, *args):
        distribution = distribution_fn(*args)
        state_args = (Sample, distribution, bijector, inverse_sample)
        return compute_inverse_state(*state_args)

    def sample_inverse(key, num_samples, *args):
        distribution = distribution_fn(*args)
        sample = distribution.sample(num_samples, seed=key)
        sample = bijector.inverse(sample)
        sample = jp.squeeze(sample, axis=0) if num_samples == 1 else sample
        return sample

    def sample(key, num_samples, *args):
        distribution = distribution_fn(*args)
        sample = distribution.sample(num_samples, seed=key)
        sample = jp.squeeze(sample, axis=0) if num_samples == 1 else sample
        return sample

    def call(*args):
        edges = list(args)
        return Variable(
            log_prob,
            log_prob_inverse,
            sample,
            sample_inverse,
            node_name,
            edges,
            None,
            distribution_fn,
            bijector,
        )

    return call
