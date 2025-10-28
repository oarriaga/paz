import jax.numpy as jp
import jax
from paz.inference.types import Distribution, NodeState, Variable, SampleType
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def squeeze_pytree(pytree):
    def squeeze_leaf(leaf):  # Squeeze if array has leading dimension of 1
        has_shape = hasattr(leaf, "shape")
        if has_shape and len(leaf.shape) > 0 and leaf.shape[0] == 1:
            return jp.squeeze(leaf, axis=0)
        return leaf

    return jax.tree.map(squeeze_leaf, pytree)


def Prior(name, distribution, bijector=None):
    if not isinstance(distribution, Distribution):
        raise ValueError("Invalid distribution type")

    Sample = SampleType([name])
    bijector = tfb.Identity() if bijector is None else bijector

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

    return Variable(apply, sample, sample_inverse, name, [], distribution)
