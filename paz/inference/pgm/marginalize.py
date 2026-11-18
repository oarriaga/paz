from functools import partial

import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp

from paz.backend import bijectors
from paz.backend import distributions
from paz.inference.infer import build_log_density
from paz.inference.latent_space import (
    LatentSpace,
    as_latent_samples,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.pgm.ops import (
    build_configure,
    build_fit,
    build_infer,
    build_prior_prob,
)
from paz.inference.types import (
    PGM,
    Density,
    DiscretePosterior,
    Likelihood,
    NodeState,
    SampleType,
)
from paz.inference.utils import get_leading_batch_size


def marginalize(pgm, names):
    node = find_discrete_node(pgm, names)
    latent_space = remove_latent(pgm.latent_space, node.name)
    prior = build_prior(pgm, node, latent_space)
    likelihood = build_likelihood(pgm, node, latent_space)
    return build_marginal(pgm, node.name, prior, likelihood)


def find_discrete_node(pgm, names):
    if len(names) != 1:
        raise NotImplementedError("Only one name can be marginalized.")
    nodes = [node for node in pgm.latent_nodes if node.name == names[0]]
    if len(nodes) == 0:
        raise ValueError(f"Latent node {names[0]} not found.")
    node = nodes[0]
    if not isinstance(node.bijector, bijectors.Identity):
        message = "Discrete latent variables must use the identity bijector."
        raise NotImplementedError(message)
    # A prior has a fixed distribution, so a bad support can fail right here.
    if is_prior(node):
        compute_support(node.distribution)
    return node


def is_prior(node):
    return node.distribution is not None


def compute_support(distribution):
    shapes = (distribution.batch_shape, distribution.event_shape)
    if any(len(shape) != 0 for shape in shapes):
        message = "Only scalar discrete variables are supported."
        raise NotImplementedError(message)
    if isinstance(distribution, distributions.Bernoulli):
        num_values = 2
    elif isinstance(distribution, distributions.Categorical):
        num_values = distribution.probs_parameter().shape[-1]
    else:
        message = "Only Bernoulli and Categorical are supported."
        raise NotImplementedError(message)
    # Latent samples are floats, so integer supports are cast to float.
    dtype = distribution.dtype
    is_discrete = jp.issubdtype(dtype, jp.integer) or dtype == jp.bool_
    return jp.arange(num_values, dtype=jp.float32 if is_discrete else dtype)


def remove_latent(latent_space, name):
    names = [other for other in latent_space.names if other != name]
    kept = {other: latent_space.bijectors[other] for other in names}
    return LatentSpace(names, kept, SampleType(names))


def build_prior(pgm, node, latent_space):
    def sample(key, num_samples=1):
        samples = pgm.prior.sample(key, num_samples)
        return as_latent_samples(latent_space, samples)

    def sample_inverse(key, num_samples=1):
        samples = pgm.prior.sample_inverse(key, num_samples)
        return as_latent_samples(latent_space, samples)

    def log_prob(samples):
        samples = as_latent_samples(latent_space, samples)
        return sum_prior(pgm.prior.log_prob, node, samples, samples)

    def log_prob_inverse(samples):
        samples = as_latent_samples(latent_space, samples)
        forward_samples = to_forward_samples(latent_space, samples)
        args = (pgm.prior.log_prob_inverse, node, forward_samples, samples)
        return sum_prior(*args)

    probs = build_prior_prob(log_prob, log_prob_inverse)
    functions = (sample, sample_inverse, log_prob, log_prob_inverse)
    return Density(*functions, *probs, latent_space, None)


def sum_prior(log_prob, node, forward_samples, samples):
    args = (log_prob, node, forward_samples, samples)
    _, log_priors = enumerate_log_probs(*args)
    return build_state(forward_samples, logsumexp(log_priors, axis=-1))


def enumerate_log_probs(log_prob, node, forward_samples, samples):
    enumerate_one = partial(enumerate_sample, log_prob, node)
    if get_leading_batch_size(samples) is None:
        result = enumerate_one(forward_samples, samples)
    else:
        enumerate_batch = jax.vmap(enumerate_one, out_axes=(None, 0))
        result = enumerate_batch(forward_samples, samples)
    return result


def enumerate_sample(log_prob, node, forward_sample, sample):
    support = compute_support(build_distribution(node, forward_sample))

    def compute_log_prob(value):
        values = {**sample._asdict(), node.name: value}
        return log_prob(values).log_prob_sum

    return support, jax.vmap(compute_log_prob)(support)


def build_distribution(node, forward_sample):
    if is_prior(node):
        distribution = node.distribution
    else:
        parents = [getattr(forward_sample, edge.name) for edge in node.edges]
        distribution = node.distribution_fn(*parents)
    return distribution


def build_state(forward_samples, log_prob_sum):
    log_prob = {"_marginal": log_prob_sum}
    return NodeState(forward_samples, log_prob, log_prob_sum)


def build_likelihood(pgm, node, latent_space):
    def log_prob(samples, data):
        samples = to_inverse_samples(latent_space, samples)
        return log_prob_inverse(samples, data)

    def log_prob_inverse(samples, data):
        samples = as_latent_samples(latent_space, samples)
        forward_samples = to_forward_samples(latent_space, samples)
        args = (pgm, node, forward_samples, samples, data)
        _, log_priors, log_joints = compute_log_joints(*args)
        log_prior = logsumexp(log_priors, axis=-1)
        log_prob_sum = logsumexp(log_joints, axis=-1) - log_prior
        return build_state(forward_samples, log_prob_sum)

    return Likelihood(log_prob, log_prob_inverse, latent_space, None)


def compute_log_joints(pgm, node, forward_samples, inverse_samples, data):
    def log_likelihood(samples):
        return pgm.likelihood.log_prob_inverse(samples, data)

    args = (node, forward_samples, inverse_samples)
    log_prior = pgm.prior.log_prob_inverse
    support, log_priors = enumerate_log_probs(log_prior, *args)
    _, log_likelihoods = enumerate_log_probs(log_likelihood, *args)
    return support, log_priors, log_priors + log_likelihoods


def build_marginal(pgm, name, prior, likelihood):
    def sample(key, num_samples=1):
        return remove_field(pgm.sample(key, num_samples), name)

    defaults = {}
    configure = build_configure(defaults, lambda: marginal)
    infer = build_infer(prior, likelihood, defaults)
    samplers = (sample, prior.sample_inverse)
    marginal_name = f"{pgm.name}_marg_{name}"
    node_groups = remove_node(pgm, name)
    densities = (prior.latent_space, prior, likelihood)
    inference = (configure, configure, infer, build_fit(), defaults)
    joint = build_log_density(prior, likelihood)
    args = (*samplers, marginal_name, *node_groups, *densities, *inference)
    marginal = PGM(*args, *joint)
    return marginal


def remove_field(samples, name):
    names = [field for field in samples._fields if field != name]
    return SampleType(names)(*[getattr(samples, field) for field in names])


def remove_node(pgm, name):
    groups = (pgm.nodes, pgm.inputs, pgm.non_priors, pgm.latent_nodes)
    groups = groups + (pgm.output_nodes, pgm.observable_nodes)
    return [[node for node in nodes if node.name != name] for nodes in groups]


def recover_discrete_posterior(pgm, name, inverse_samples, data=None):
    node = find_discrete_node(pgm, [name])
    latent_space = remove_latent(pgm.latent_space, name)
    inverse_samples = as_latent_samples(latent_space, inverse_samples)
    forward_samples = to_forward_samples(latent_space, inverse_samples)
    args = (pgm, node, forward_samples, inverse_samples, data)
    support, _, log_joints = compute_log_joints(*args)
    log_evidence = logsumexp(log_joints, axis=-1, keepdims=True)
    log_posterior = log_joints - log_evidence
    z_map = jp.argmax(log_posterior, axis=-1)
    z_map_value = jp.take(support, z_map)
    posterior = jp.exp(log_posterior)
    args = (support, log_posterior, posterior, z_map, z_map_value)
    return DiscretePosterior(*args)
