from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp

from paz.backend import bijectors
from paz.backend import distributions
from paz.inference.latent_space import LatentSpace, to_forward_samples
from paz.inference.observable import Observable
from paz.inference.pgm.ops import build_observations
from paz.inference.pgm.pgm import PGM
from paz.inference.types import DiscretePosterior, SampleType

Marginal = namedtuple("Marginal", ["log_prob", "sample"])


def marginalize(pgm, names):
    """Builds a `PGM` with a discrete prior summed out of its observed child.

    # Arguments
    - `pgm`: `PGM` with a discrete prior that has an observed child.
    - `names`: List with the name of the one discrete prior to sum out.

    # Returns
    `PGM` named `{pgm.name}_marg_{node.name}` without the discrete prior
    in its inputs.
    """
    node = find_discrete_node(pgm, names)
    child = find_observed_child(pgm, node)
    parents = merge_parents(node, child)
    parent_names = [parent.name for parent in parents]
    distribution_fn = MarginalFn(node, child, parent_names)
    marginal_child = Observable(distribution_fn, name=child.name)(*parents)
    inputs = [prior for prior in pgm.inputs if prior.name != node.name]
    outputs = []
    for output in pgm.output_nodes:
        is_child = output.name == child.name
        outputs.append(marginal_child if is_child else output)
    return PGM(inputs, outputs, f"{pgm.name}_marg_{node.name}")


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


def find_observed_child(pgm, node):
    children = find_children(pgm, node)
    is_observed = len(children) == 1 and children[0].sample_inverse is None
    if not is_observed or len(find_children(pgm, children[0])) != 0:
        message = f"{node.name} must feed exactly one observed node, "
        message += "and no other node may depend on that node."
        raise NotImplementedError(message)
    return children[0]


def find_children(pgm, node):
    children = []
    for other in pgm.nodes:
        if node.name in [edge.name for edge in other.edges]:
            children.append(other)
    return children


# The child takes the parents of the discrete node in its place.
def merge_parents(node, child):
    parents = {}
    for edge in child.edges:
        replacements = node.edges if edge.name == node.name else [edge]
        parents.update({parent.name: parent for parent in replacements})
    return list(parents.values())


def MarginalFn(node, child, parent_names):
    def apply(*parent_values):
        values = dict(zip(parent_names, parent_values))
        discrete = build_discrete_distribution(node, values)
        build_child = partial(build_child_distribution, node, child, values)

        def log_prob(observation):
            args = (discrete, build_child, observation)
            _, log_joints = compute_log_joints(*args)
            return logsumexp(log_joints)

        def sample(num_samples, seed):
            return sample_child(discrete, build_child, num_samples, seed)

        return Marginal(log_prob, sample)

    return apply


def build_discrete_distribution(node, values):
    if is_prior(node):
        distribution = node.distribution
    else:
        parents = [values[edge.name] for edge in node.edges]
        distribution = node.distribution_fn(*parents)
    return distribution


def build_child_distribution(node, child, values, discrete_value):
    values = {**values, node.name: discrete_value}
    parents = [values[edge.name] for edge in child.edges]
    return child.distribution_fn(*parents)


def compute_log_joints(discrete, build_child, observation):
    def compute_log_likelihood(discrete_value):
        return build_child(discrete_value).log_prob(observation).sum()

    support = compute_support(discrete)
    log_likelihoods = jax.vmap(compute_log_likelihood)(support)
    return support, discrete.log_prob(support) + log_likelihoods


def sample_child(discrete, build_child, num_samples, seed):
    def sample_once(key, discrete_value):
        return build_child(discrete_value).sample(1, seed=key)[0]

    key_discrete, key_child = jax.random.split(seed)
    discrete_values = discrete.sample(num_samples, seed=key_discrete)
    # The child sees the same dtype here as in the sum over the support.
    dtype = compute_support(discrete).dtype
    keys = jax.random.split(key_child, num_samples)
    return jax.vmap(sample_once)(keys, discrete_values.astype(dtype))


def recover_discrete_posterior(pgm, name, inverse_samples, data):
    node = find_discrete_node(pgm, [name])
    child = find_observed_child(pgm, node)
    latent_space = remove_latent(pgm.latent_space, name)
    values = to_forward_samples(latent_space, inverse_samples)._asdict()
    nodes = (pgm.output_nodes, pgm.observable_nodes)
    observation = build_observations(data, *nodes)[child.name]
    discrete = build_discrete_distribution(node, values)
    build_child = partial(build_child_distribution, node, child, values)
    args = (discrete, build_child, observation)
    support, log_joints = compute_log_joints(*args)
    log_posterior = log_joints - logsumexp(log_joints)
    z_map = jp.argmax(log_posterior)
    posterior = jp.exp(log_posterior)
    args = (support, log_posterior, posterior, z_map, support[z_map])
    return DiscretePosterior(*args)


def remove_latent(latent_space, name):
    names = [other for other in latent_space.names if other != name]
    kept = {other: latent_space.bijectors[other] for other in names}
    return LatentSpace(names, kept, SampleType(names))
