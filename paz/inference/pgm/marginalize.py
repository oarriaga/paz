import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.metadata import (
    get_bijector,
    get_distribution_fn,
    get_inputs,
    get_latent_nodes,
    get_non_priors,
    get_output_nodes,
)
from paz.inference.latent_space import (
    LatentSpace,
    as_latent_samples,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.types import (
    DiscretePosterior,
    Density,
    Likelihood,
    NodeState,
    PGM,
    SampleType,
)
from paz.inference.pgm.ops import (
    build_configure,
    build_data_mapping,
    build_fit,
    build_infer,
    build_prior_prob,
    get_namedtuple_value,
)
from paz.inference.utils import (
    get_leading_batch_size,
    slice_batch,
)

tfd = tfp.distributions
tfb = tfp.bijectors
DiscreteUniform = getattr(tfd, "DiscreteUniform", None)
def finite_support(distribution):
    if len(distribution.batch_shape) != 0 or len(distribution.event_shape) != 0:
        raise NotImplementedError("Only scalar discrete variables are supported.")
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

_MISSING = object()

def _update_samples(samples, name, value=_MISSING):
    if samples is None:
        if value is _MISSING:
            return None
        raise TypeError("samples must be a dict or namedtuple.")
    if isinstance(samples, dict):
        updated = dict(samples)
        if value is _MISSING:
            updated.pop(name, None)
        else:
            updated[name] = value
        return updated
    if hasattr(samples, "_asdict"):
        updated = samples._asdict()
        if value is _MISSING:
            updated.pop(name, None)
        else:
            updated[name] = value
        Sample = SampleType(list(updated.keys()))
        return Sample(**updated)
    raise TypeError("samples must be a dict or namedtuple.")


def _map_batches(theta_inverse_samples, data_mapping, batch_log_prob):
    num_batches = get_leading_batch_size(theta_inverse_samples)
    data_batch = get_leading_batch_size(data_mapping)
    if num_batches is None:
        return batch_log_prob(theta_inverse_samples, data_mapping)
    if data_batch == num_batches:
        return jax.vmap(batch_log_prob)(theta_inverse_samples, data_mapping)
    return jax.vmap(
        lambda batch_theta: batch_log_prob(batch_theta, data_mapping)
    )(theta_inverse_samples)


def _build_marginal_state(latent_space, theta_inverse_samples, log_prob_sum):
    forward_samples = to_forward_samples(latent_space, theta_inverse_samples)
    log_prob = {"_marginal": log_prob_sum}
    return NodeState(forward_samples, log_prob, log_prob_sum)


def _log_joint_values(pgm, z_name, support, batch_theta, batch_data):
    return jp.stack(
        [
            pgm.prior.log_prob_inverse(
                _update_samples(batch_theta, z_name, value)
            ).log_prob_sum
            + pgm.likelihood.log_prob_inverse(
                _update_samples(batch_theta, z_name, value), batch_data
            ).log_prob_sum
            for value in support
        ]
    )


def _log_prior_values(pgm, z_name, support, batch_theta):
    return jp.stack(
        [
            pgm.prior.log_prob_inverse(
                _update_samples(batch_theta, z_name, value)
            ).log_prob_sum
            for value in support
        ]
    )


def _validate_supports(pgm, node, inverse_samples):
    inputs = get_inputs(pgm)
    non_priors = get_non_priors(pgm)
    input_names = {prior.name for prior in inputs}

    def build_distribution(samples):
        if node.distribution is not None:
            return node.distribution
        distribution_fn = get_distribution_fn(node)
        if distribution_fn is None:
            raise ValueError("Latent node missing distribution_fn.")
        sample_map = {}
        for prior in inputs:
            inverse_sample = get_namedtuple_value(prior.name, samples)
            state = prior.log_prob_inverse(inverse_sample)
            sample_map[prior.name] = get_namedtuple_value(
                prior.name, state.sample
            )
        if node.name not in input_names:
            for current in non_priors:
                if current.name == node.name:
                    break
                node_inputs = [
                    sample_map[edge.name] for edge in current.edges
                ]
                node_sample = get_namedtuple_value(current.name, samples)
                state = current.log_prob_inverse(node_sample, *node_inputs)
                sample_map[current.name] = get_namedtuple_value(
                    current.name, state.sample
                )
            else:
                raise ValueError(f"Node {node.name} not found in PGM.")
        parent_samples = [sample_map[edge.name] for edge in node.edges]
        return distribution_fn(*parent_samples)

    num_batches = get_leading_batch_size(inverse_samples)
    first_batch = (
        inverse_samples
        if num_batches is None
        else slice_batch(inverse_samples, 0)
    )
    distribution = build_distribution(first_batch)
    support = finite_support(distribution)
    if num_batches is None:
        return support
    if DiscreteUniform is None or not isinstance(distribution, DiscreteUniform):
        return support
    for batch_index in range(1, num_batches):
        batch_samples = slice_batch(inverse_samples, batch_index)
        batch_support = finite_support(build_distribution(batch_samples))
        if support.shape != batch_support.shape:
            raise NotImplementedError("Discrete support must be consistent.")
        if not bool(jp.all(batch_support == support)):
            raise NotImplementedError("Discrete support must be consistent.")
    return support


def marginalize(pgm, names):
    # v1 constraints: single scalar discrete latent with finite support.
    if len(names) != 1:
        raise NotImplementedError("v1 supports marginalizing exactly one name.")
    z_name = names[0]
    latent_nodes = get_latent_nodes(pgm)
    z_node = next(
        (node for node in latent_nodes if node.name == z_name), None
    )
    if z_node is None:
        raise ValueError(f"Latent node {z_name} not found.")
    if z_node.sample_inverse is None:
        raise NotImplementedError("Only latent/prior nodes can be marginalized.")
    bijector = get_bijector(z_node)
    if bijector is None or not isinstance(bijector, tfb.Identity):
        raise NotImplementedError(
            "Discrete latent variables must use the identity bijector."
        )
    if z_node.distribution is not None:
        finite_support(z_node.distribution)

    base_latent_space = pgm.prior.latent_space
    names = [name for name in base_latent_space.names if name != z_name]
    bijectors = {name: base_latent_space.bijectors[name] for name in names}
    Sample = SampleType(names)
    latent_space = LatentSpace(names, bijectors, Sample)
    output_nodes = get_output_nodes(pgm)

    def select_latent(samples):
        if isinstance(samples, dict):
            data = samples
        elif hasattr(samples, "_asdict"):
            data = samples._asdict()
        else:
            raise TypeError("samples must be a dict or namedtuple.")
        return latent_space.Sample(
            **{name: data[name] for name in latent_space.names}
        )

    def sample_inverse(key, num_samples=1):
        samples = pgm.sample_inverse(key, num_samples)
        return select_latent(samples)

    def sample(key, num_samples=1):
        samples = pgm.sample(key, num_samples)
        return _update_samples(samples, z_name)

    def prior_sample_inverse(key, num_samples=1):
        samples = pgm.prior.sample_inverse(key, num_samples)
        return select_latent(samples)

    def prior_sample(key, num_samples=1):
        samples = pgm.prior.sample(key, num_samples)
        return select_latent(samples)

    def prior_log_prob(samples):
        theta_inverse_samples = to_inverse_samples(latent_space, samples)
        return prior_log_prob_inverse(theta_inverse_samples)

    def prior_log_prob_inverse(samples):
        theta_inverse_samples = as_latent_samples(latent_space, samples)
        support = _validate_supports(pgm, z_node, theta_inverse_samples)

        def batch_log_prob(batch_theta, _):
            log_prior = _log_prior_values(pgm, z_name, support, batch_theta)
            return logsumexp(log_prior)

        log_prob_sum = _map_batches(theta_inverse_samples, None, batch_log_prob)
        return _build_marginal_state(
            latent_space, theta_inverse_samples, log_prob_sum
        )

    prior_prob, prior_prob_inverse = build_prior_prob(
        prior_log_prob, prior_log_prob_inverse
    )

    def likelihood_log_prob(samples, data):
        theta_inverse_samples = to_inverse_samples(latent_space, samples)
        return likelihood_log_prob_inverse(theta_inverse_samples, data)

    def likelihood_log_prob_inverse(samples, data):
        data_mapping = build_data_mapping(data, output_nodes)
        theta_inverse_samples = as_latent_samples(latent_space, samples)
        support = _validate_supports(pgm, z_node, theta_inverse_samples)

        def batch_log_prob(batch_theta, batch_data):
            log_joint = _log_joint_values(
                pgm, z_name, support, batch_theta, batch_data
            )
            log_prior = _log_prior_values(pgm, z_name, support, batch_theta)
            return logsumexp(log_joint) - logsumexp(log_prior)
        log_prob_sum = _map_batches(
            theta_inverse_samples, data_mapping, batch_log_prob
        )
        return _build_marginal_state(
            latent_space, theta_inverse_samples, log_prob_sum
        )

    likelihood_log_prob_inverse._marginalize_base_pgm = pgm
    likelihood_log_prob_inverse._marginalize_z_node = z_node
    likelihood_log_prob_inverse._marginalize_z_name = z_name
    likelihood_log_prob_inverse._marginalize_latent_space = latent_space

    name = f"{pgm.name}_marg_{z_name}"
    prior_density = Density(
        prior_sample,
        prior_sample_inverse,
        prior_log_prob,
        prior_log_prob_inverse,
        prior_prob,
        prior_prob_inverse,
        latent_space,
        None,
    )
    likelihood_density = Likelihood(
        likelihood_log_prob,
        likelihood_log_prob_inverse,
        latent_space,
        None,
    )
    inference_defaults = {}
    fit = build_fit()
    def without_z(nodes):
        return [node for node in nodes if node.name != z_name]

    configure = build_configure(inference_defaults, lambda: pgm_marg)
    tune = configure
    infer = build_infer(prior_density, likelihood_density, inference_defaults)
    pgm_marg = PGM(
        sample,
        sample_inverse,
        name,
        without_z(pgm.nodes),
        without_z(pgm.inputs),
        without_z(pgm.non_priors),
        without_z(pgm.latent_nodes),
        without_z(pgm.output_nodes),
        without_z(pgm.observable_nodes),
        latent_space,
        prior_density,
        likelihood_density,
        configure,
        tune,
        infer,
        fit,
        inference_defaults,
    )
    return pgm_marg


def recover_discrete_posterior(
    pgm_marg, z_name, theta_inverse_samples, data=None
):
    log_prob_inverse = pgm_marg.likelihood.log_prob_inverse
    base_pgm = getattr(log_prob_inverse, "_marginalize_base_pgm", None)
    z_node = getattr(log_prob_inverse, "_marginalize_z_node", None)
    stored_name = getattr(log_prob_inverse, "_marginalize_z_name", None)
    if base_pgm is None or z_node is None or stored_name is None:
        raise ValueError("PGM does not appear to be marginalized.")
    if z_name != stored_name:
        raise ValueError("z_name does not match marginalized PGM.")
    support = _validate_supports(base_pgm, z_node, theta_inverse_samples)
    data_mapping = build_data_mapping(data, get_output_nodes(base_pgm))

    def batch_log_joint(batch_theta, batch_data):
        return _log_joint_values(
            base_pgm, z_name, support, batch_theta, batch_data
        )
    log_joint = _map_batches(
        theta_inverse_samples, data_mapping, batch_log_joint
    )

    log_posterior = log_joint - logsumexp(log_joint, axis=-1, keepdims=True)
    posterior = jp.exp(log_posterior)
    z_map = jp.argmax(log_posterior, axis=-1)
    z_map_value = jp.take(support, z_map)

    return DiscretePosterior(
        support, log_posterior, posterior, z_map, z_map_value
    )
