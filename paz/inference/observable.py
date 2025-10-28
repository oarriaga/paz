import jax.numpy as jp
from paz.inference.types import NodeState, Variable


def Observable(name, distribution_fn, observation):
    edges = []

    def apply(sample_inverse, *args):
        distribution = distribution_fn(*args)
        log_prob = distribution.log_prob(observation)
        log_prob_sum = log_prob.sum()
        return NodeState(distribution, log_prob_sum, log_prob_sum)

    def sample(key, num_samples, *args):
        distribution = distribution_fn(*args)
        sample = distribution.sample(num_samples, seed=key)
        sample = jp.squeeze(sample, axis=0) if num_samples == 1 else sample
        return sample

    def call(*args):
        for arg in args:
            edges.append(arg)
        return Variable(apply, sample, None, name, edges, None)

    return call
