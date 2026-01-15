import time

import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference import pgm as pgm_module
from paz.inference.types import NodeState, SampleType, Variable

tfd = tfp.distributions
tfb = tfp.bijectors
DiscreteUniform = getattr(tfd, "DiscreteUniform", None)


def _get_freevars(function):
    if function.__closure__ is None:
        return {}
    names = function.__code__.co_freevars
    values = [cell.cell_contents for cell in function.__closure__]
    return dict(zip(names, values))


def _get_pgm_latent_nodes(pgm):
    freevars = _get_freevars(pgm.sample_inverse)
    latent_nodes = freevars.get("latent_nodes")
    if latent_nodes is None:
        raise ValueError("Expected a PGM Variable with latent_nodes.")
    return latent_nodes


def _get_pgm_inputs_and_non_priors(pgm):
    freevars = _get_freevars(pgm.apply)
    inputs = freevars.get("inputs")
    non_priors = freevars.get("non_priors")
    if inputs is None or non_priors is None:
        raise ValueError("Expected a PGM Variable with inputs and non_priors.")
    return inputs, non_priors


def _get_node_by_name(nodes, name):
    for node in nodes:
        if node.name == name:
            return node
    return None


def _get_bijector(node):
    freevars = _get_freevars(node.apply)
    return freevars.get("bijector")


def _check_identity_bijector(node):
    bijector = _get_bijector(node)
    if bijector is None or not isinstance(bijector, tfb.Identity):
        raise NotImplementedError(
            "Discrete latent variables must use the identity bijector."
        )


def _shape_length(value):
    if hasattr(value, "ndim"):
        return value.ndim
    if hasattr(value, "ndims"):
        return value.ndims
    if hasattr(value, "shape"):
        return len(value.shape)
    if isinstance(value, tuple):
        return len(value)
    return 0


def _ensure_scalar_distribution(distribution):
    if _shape_length(distribution.batch_shape) != 0:
        raise NotImplementedError("Only scalar discrete variables are supported.")
    if _shape_length(distribution.event_shape) != 0:
        raise NotImplementedError("Only scalar discrete variables are supported.")


def finite_support(distribution):
    _ensure_scalar_distribution(distribution)
    support_dtype = distribution.dtype
    if jp.issubdtype(support_dtype, jp.integer) or support_dtype == jp.bool_:
        support_dtype = jp.float32
    if isinstance(distribution, tfd.Bernoulli):
        return jp.array([0, 1], dtype=support_dtype)
    if isinstance(distribution, tfd.Categorical):
        logits = distribution.logits_parameter()
        probs = distribution.probs_parameter()
        if logits is None and probs is None:
            raise NotImplementedError("Categorical distribution needs logits or probs.")
        num_categories = probs.shape[-1] if logits is None else logits.shape[-1]
        return jp.arange(num_categories, dtype=support_dtype)
    if DiscreteUniform is not None and isinstance(distribution, DiscreteUniform):
        low = distribution.low
        high = distribution.high
        if hasattr(low, "shape") and low.shape != ():
            raise NotImplementedError("Only scalar discrete variables are supported.")
        if hasattr(high, "shape") and high.shape != ():
            raise NotImplementedError("Only scalar discrete variables are supported.")
        return jp.arange(low, high + 1, dtype=support_dtype)
    raise NotImplementedError("Only Bernoulli, Categorical, DiscreteUniform supported.")


def _inject_inverse_samples(inverse_samples, name, value):
    if isinstance(inverse_samples, dict):
        updated = dict(inverse_samples)
        updated[name] = value
        return updated
    if hasattr(inverse_samples, "_asdict"):
        updated = inverse_samples._asdict()
        updated[name] = value
        Sample = SampleType(list(updated.keys()))
        return Sample(**updated)
    raise TypeError("inverse_samples must be a dict or namedtuple.")


def _remove_sample_field(samples, name):
    if samples is None:
        return None
    if isinstance(samples, dict):
        updated = dict(samples)
        updated.pop(name, None)
        return updated
    if hasattr(samples, "_asdict"):
        updated = samples._asdict()
        updated.pop(name, None)
        Sample = SampleType(list(updated.keys()))
        return Sample(**updated)
    raise TypeError("samples must be a dict or namedtuple.")


def _get_parent_samples(pgm, node, inverse_samples):
    inputs, non_priors = _get_pgm_inputs_and_non_priors(pgm)
    samples = {}
    for prior in inputs:
        inverse_sample = pgm_module.get_namedtuple_value(
            prior.name, inverse_samples
        )
        state = prior.apply(inverse_sample)
        samples[prior.name] = pgm_module.get_namedtuple_value(
            prior.name, state.sample
        )
    if _get_node_by_name(inputs, node.name) is not None:
        return [samples[edge.name] for edge in node.edges]
    found = False
    for current in non_priors:
        if current.name == node.name:
            found = True
            break
        node_inputs = [samples[edge.name] for edge in current.edges]
        node_sample = pgm_module.get_namedtuple_value(
            current.name, inverse_samples
        )
        state = current.apply(node_sample, *node_inputs)
        samples[current.name] = pgm_module.get_namedtuple_value(
            current.name, state.sample
        )
    if not found:
        raise ValueError(f"Node {node.name} not found in PGM.")
    return [samples[edge.name] for edge in node.edges]


def _build_distribution(pgm, node, inverse_samples):
    if node.distribution is not None:
        return node.distribution
    freevars = _get_freevars(node.apply)
    distribution_fn = freevars.get("distribution_fn")
    if distribution_fn is None:
        raise ValueError("Latent node missing distribution_fn.")
    parent_samples = _get_parent_samples(pgm, node, inverse_samples)
    return distribution_fn(*parent_samples)


def _get_batch_size(inverse_samples):
    leaves = jax.tree_util.tree_leaves(inverse_samples)
    shaped = [leaf for leaf in leaves if hasattr(leaf, "shape")]
    if len(shaped) == 0:
        return None
    if any(len(leaf.shape) == 0 for leaf in shaped):
        return None
    first_dim = shaped[0].shape[0]
    if any(leaf.shape[0] != first_dim for leaf in shaped):
        return None
    return first_dim


def _slice_chain(inverse_samples, chain_index):
    return jax.tree_util.tree_map(
        lambda value: value[chain_index], inverse_samples
    )


def _get_support_for_chain(pgm, node, inverse_samples):
    distribution = _build_distribution(pgm, node, inverse_samples)
    return finite_support(distribution)


def _validate_supports(pgm, node, inverse_samples):
    num_chains = _get_batch_size(inverse_samples)
    if num_chains is None:
        return _get_support_for_chain(pgm, node, inverse_samples)
    first_chain = _slice_chain(inverse_samples, 0)
    distribution = _build_distribution(pgm, node, first_chain)
    support = finite_support(distribution)
    if DiscreteUniform is None or not isinstance(distribution, DiscreteUniform):
        return support
    for chain_index in range(1, num_chains):
        chain_samples = _slice_chain(inverse_samples, chain_index)
        chain_support = _get_support_for_chain(pgm, node, chain_samples)
        if support.shape != chain_support.shape:
            raise NotImplementedError("Discrete support must be consistent.")
        if not bool(jp.all(chain_support == support)):
            raise NotImplementedError("Discrete support must be consistent.")
    return support


def _compute_log_joint(pgm, z_name, support, theta_inverse_samples):
    log_joint = []
    for value in support:
        full_inverse = _inject_inverse_samples(
            theta_inverse_samples, z_name, value
        )
        state = pgm.apply(full_inverse)
        log_joint.append(state.log_prob_sum)
    return jp.stack(log_joint)


def marginalize(pgm, names):
    # v1 constraints: single scalar discrete latent with finite support.
    if len(names) != 1:
        raise NotImplementedError("v1 supports marginalizing exactly one name.")
    z_name = names[0]
    latent_nodes = _get_pgm_latent_nodes(pgm)
    z_node = _get_node_by_name(latent_nodes, z_name)
    if z_node is None:
        raise ValueError(f"Latent node {z_name} not found.")
    if z_node.sample_inverse is None:
        raise NotImplementedError("Only latent/prior nodes can be marginalized.")
    _check_identity_bijector(z_node)
    if z_node.distribution is not None:
        finite_support(z_node.distribution)

    def apply(theta_inverse_samples):
        support = _validate_supports(pgm, z_node, theta_inverse_samples)
        num_chains = _get_batch_size(theta_inverse_samples)

        def chain_log_prob(chain_theta):
            log_joint = _compute_log_joint(pgm, z_name, support, chain_theta)
            return logsumexp(log_joint)

        if num_chains is None:
            log_prob_sum = chain_log_prob(theta_inverse_samples)
        else:
            log_prob_sum = jax.vmap(chain_log_prob)(theta_inverse_samples)

        return NodeState(None, {"marginalized": log_prob_sum}, log_prob_sum)

    def sample_inverse(key, num_samples=1):
        samples = pgm.sample_inverse(key, num_samples)
        return _remove_sample_field(samples, z_name)

    def sample(key, num_samples=1):
        samples = pgm.sample(key, num_samples)
        return _remove_sample_field(samples, z_name)

    apply._marginalize_base_pgm = pgm
    apply._marginalize_z_node = z_node
    apply._marginalize_z_name = z_name

    name = f"{pgm.name}_marg_{z_name}"
    return Variable(apply, sample, sample_inverse, name, [], None)


def _get_marginalize_metadata(pgm):
    apply = pgm.apply
    base_pgm = getattr(apply, "_marginalize_base_pgm", None)
    z_node = getattr(apply, "_marginalize_z_node", None)
    z_name = getattr(apply, "_marginalize_z_name", None)
    if base_pgm is None or z_node is None or z_name is None:
        raise ValueError("PGM does not appear to be marginalized.")
    return base_pgm, z_node, z_name


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def recover_discrete_posterior(
    pgm_marg, z_name, theta_inverse_samples, timed=False
):
    if timed:
        start_total = time.perf_counter()
    base_pgm, z_node, stored_name = _get_marginalize_metadata(pgm_marg)
    if z_name != stored_name:
        raise ValueError("z_name does not match marginalized PGM.")
    if timed:
        start_support = time.perf_counter()
    support = _validate_supports(base_pgm, z_node, theta_inverse_samples)
    if timed:
        support_seconds = time.perf_counter() - start_support
    num_chains = _get_batch_size(theta_inverse_samples)

    def chain_log_joint(chain_theta):
        return _compute_log_joint(base_pgm, z_name, support, chain_theta)

    if timed:
        start_log_joint = time.perf_counter()
    if num_chains is None:
        log_joint = chain_log_joint(theta_inverse_samples)
    else:
        log_joint = jax.vmap(chain_log_joint)(theta_inverse_samples)
    if timed:
        _block_until_ready(log_joint)
        log_joint_seconds = time.perf_counter() - start_log_joint

    log_posterior = log_joint - logsumexp(log_joint, axis=-1, keepdims=True)
    posterior = jp.exp(log_posterior)
    z_map = jp.argmax(log_posterior, axis=-1)
    z_map_value = jp.take(support, z_map)
    if timed:
        _block_until_ready(posterior)
        total_seconds = time.perf_counter() - start_total
        print("=" * 60)
        print("recover_discrete_posterior timing")
        print("=" * 60)
        print(f"support seconds={support_seconds:.4f}")
        print(f"log_joint seconds={log_joint_seconds:.4f}")
        print(f"total seconds={total_seconds:.4f}")

    return {
        "support": support,
        "log_posterior": log_posterior,
        "posterior": posterior,
        "z_map": z_map,
        "z_map_value": z_map_value,
    }
