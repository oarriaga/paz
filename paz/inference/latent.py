from typing import Callable
import jax.numpy as jp
from paz.inference import NodeState, SampleType, Variable
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def Latent(name, distribution_fn, bijector=None):
    if not isinstance(distribution_fn, Callable):
        raise ValueError(f"Input {distribution_fn} must be a callable")

    Sample = SampleType([name])
    bijector = tfb.Identity() if bijector is None else bijector
    edges = []

    def apply(inverse_sample, *args):
        forward_sample = bijector(inverse_sample)
        distribution = distribution_fn(*args)
        log_prob = distribution.log_prob(forward_sample)
        log_prob = log_prob + bijector.forward_log_det_jacobian(inverse_sample)
        log_prob_sum = log_prob.sum()
        return NodeState(Sample(forward_sample), log_prob_sum, log_prob_sum)

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
        for arg in args:
            edges.append(arg)
        return Variable(apply, sample, sample_inverse, name, edges, None)

    return call
