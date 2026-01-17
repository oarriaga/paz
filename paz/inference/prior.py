import jax
import jax.numpy as jp

from paz.inference.naming import build_prior_name
from paz.inference.types import (
    Distribution,
    NodeMetadata,
    NodeState,
    SampleType,
    Variable,
)
from paz.inference.utils import squeeze_pytree
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def Prior(distribution, bijector=None, name=None):
    if not isinstance(distribution, Distribution):
        raise ValueError("Invalid distribution type")
    node_name = name if name is not None else build_prior_name(distribution)

    Sample = SampleType([node_name])
    bijector = tfb.Identity() if bijector is None else bijector
    metadata = NodeMetadata(None, bijector)

    def apply(inverse_sample):
        forward_sample = bijector(inverse_sample)
        log_prob = distribution.log_prob(forward_sample)
        log_prob = log_prob + bijector.forward_log_det_jacobian(inverse_sample)
        log_prob_sum = log_prob.sum()
        return NodeState(Sample(forward_sample), log_prob_sum, log_prob_sum)

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

    return Variable(
        apply, sample, sample_inverse, node_name, [], distribution, metadata
    )
