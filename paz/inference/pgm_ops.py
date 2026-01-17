import jax
import jax.numpy as jp

from paz.inference.infer import (
    infer as infer_fn,
    _get_initial_positions,
    _resolve_num_warmup,
    _run_tuner,
)
from paz.inference.metadata import get_distribution_fn
from paz.inference.latent_space import (
    as_latent_samples,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.types import Distribution, NodeState


def build_sample_inverse(context):
    Latent = context.Latent
    inputs = context.inputs
    latent_nodes = context.latent_nodes

    def sample_inverse(key, num_samples=1):
        return Latent(**_sample_inverse(inputs, latent_nodes, key, num_samples))

    return sample_inverse


def build_sample_forward(context):
    Sample = context.Sample
    inputs = context.inputs
    non_priors = context.non_priors

    def sample_forward(key, num_samples=1):
        return Sample(**_sample_forward(inputs, non_priors, key, num_samples))

    return sample_forward


def build_apply(context):
    Sample = context.Sample
    inputs = context.inputs
    non_priors = context.non_priors
    output_nodes = context.output_nodes
    observable_names = context.observable_names

    def apply(inverse_samples, data=None):
        data_mapping = build_data_mapping(data, output_nodes)
        observable_name_set = set(observable_names)
        _validate_observations(data_mapping, observable_name_set)
        sample, log_prob = _apply(
            inputs, non_priors, inverse_samples, data_mapping, observable_name_set
        )
        return NodeState(Sample(**sample), log_prob, sum_log_probs(log_prob))

    return apply


def build_prior_sample(context, sample_inverse):
    latent_space = context.latent_space

    def prior_sample(key, num_samples=1, space="inv"):
        _validate_space(space)
        inverse_samples = sample_inverse(key, num_samples)
        inverse_samples = as_latent_samples(latent_space, inverse_samples)
        if space == "inv":
            return inverse_samples
        return to_forward_samples(latent_space, inverse_samples)

    return prior_sample


def build_prior_log_prob(context):
    latent_nodes_sorted = context.latent_nodes_sorted
    latent_space = context.latent_space

    def prior_log_prob(samples, space="inv"):
        _validate_space(space)
        if space == "inv":
            inverse_samples = as_latent_samples(latent_space, samples)
            forward_samples = to_forward_samples(latent_space, inverse_samples)
            log_prob = _compute_latent_log_probs(
                latent_nodes_sorted,
                forward_samples,
                inverse_samples,
                latent_space.bijectors,
                include_jacobian=True,
            )
        else:
            forward_samples = as_latent_samples(latent_space, samples)
            log_prob = _compute_latent_log_probs(
                latent_nodes_sorted,
                forward_samples,
                None,
                latent_space.bijectors,
                include_jacobian=False,
            )
        return sum(log_prob.values()) if log_prob else jp.array(0.0)

    return prior_log_prob


def build_prior_prob(prior_log_prob):
    def prior_prob(samples, space="inv"):
        return jp.exp(prior_log_prob(samples, space=space))

    return prior_prob


def build_likelihood_log_prob(context, apply):
    output_nodes = context.output_nodes
    observable_names = context.observable_names
    latent_space = context.latent_space

    def likelihood_log_prob(samples, data, space="inv"):
        _validate_space(space)
        data_mapping = build_data_mapping(data, output_nodes)
        _validate_observations(data_mapping, set(observable_names))
        if space == "fwd":
            inverse_samples = to_inverse_samples(latent_space, samples)
        else:
            inverse_samples = as_latent_samples(latent_space, samples)
        state = apply(inverse_samples, data_mapping)
        if len(observable_names) == 0:
            return jp.array(0.0)
        return sum(state.log_prob[name] for name in observable_names)

    return likelihood_log_prob


def build_tune(pgm_prior, pgm_likelihood, inference_defaults):
    def tune(
        key,
        data,
        num_chains=None,
        tuner="adaptive_step",
        progress=True,
        **method_kwargs,
    ):
        method_name = "mh"
        defaults = inference_defaults.setdefault(method_name, {})
        defaults["tuner"] = tuner
        if num_chains is not None:
            defaults["num_chains"] = num_chains
        defaults["progress"] = progress
        defaults.update(method_kwargs)
        inference_defaults["_last_method"] = method_name
        sigma = defaults.get("sigma", 0.1)
        num_chains = defaults.get("num_chains", 1)
        init = defaults.get("init", None)
        space = defaults.get("space", "inv")
        num_samples = defaults.get("num_samples", None)
        warmup = defaults.get("warmup", None)
        if isinstance(warmup, float) and num_samples is None:
            raise ValueError("num_samples is required for warmup fractions.")

        def log_density_fn(position):
            log_prior = pgm_prior.log_prob(position, space="inv")
            log_likelihood = pgm_likelihood.log_prob(
                position, data, space="inv"
            )
            return log_prior + log_likelihood

        key_init, key_tune = jax.random.split(key)
        positions = _get_initial_positions(
            key_init, pgm_prior, init, num_chains, space
        )
        num_warmup = _resolve_num_warmup(num_samples, warmup)
        tuned_sigma = _run_tuner(
            tuner,
            key_tune,
            log_density_fn,
            positions,
            num_chains,
            sigma,
            num_warmup,
            progress,
            defaults,
        )
        defaults["sigma"] = tuned_sigma
        defaults["tuner"] = None
        return tuned_sigma

    return tune


def build_infer(pgm_prior, pgm_likelihood, inference_defaults):
    def infer(
        key,
        data,
        prior=None,
        likelihood=None,
        method=None,
        **overrides,
    ):
        method_name = method
        if method_name is None:
            method_name = inference_defaults.get("_last_method", "mh")
        method_name = method_name.lower()
        resolved = dict(inference_defaults.get(method_name, {}))
        resolved.update(overrides)
        prior = prior if prior is not None else pgm_prior
        likelihood = likelihood if likelihood is not None else pgm_likelihood
        return infer_fn(key, data, prior, likelihood, method_name, **resolved)

    return infer


def build_data_mapping(data, output_nodes):
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "_asdict"):
        return data._asdict()
    if isinstance(data, (list, tuple)):
        if len(data) != len(output_nodes):
            raise ValueError(
                "Data list must match the number of PGM outputs."
            )
        return {node.name: value for node, value in zip(output_nodes, data)}
    if len(output_nodes) == 1:
        return {output_nodes[0].name: data}
    raise TypeError(
        "Data must be a dict, list aligned to outputs, or a single "
        "observation for single-output models."
    )


def get_edge_names(node):
    return [edge.name for edge in node.edges]


def get_values_by_key(keys, dictionary):
    return [dictionary[key] for key in keys]


def get_node_inputs(node, dictionary):
    return get_values_by_key(get_edge_names(node), dictionary)


def get_namedtuple_value(field, named_tuple, default=None):
    if isinstance(named_tuple, Distribution):
        return named_tuple
    return getattr(named_tuple, field, default)


def sum_log_probs(log_prob_dict):
    return sum(log_prob_dict.values())


def _validate_space(space):
    if space not in ("inv", "fwd"):
        raise ValueError("space must be 'inv' or 'fwd'")


def _validate_observations(data_mapping, observable_names):
    if len(observable_names) == 0:
        return
    missing = [name for name in observable_names if name not in data_mapping]
    if missing:
        missing.sort()
        raise ValueError(
            "Missing observations for: " + ", ".join(missing)
        )


def _sample_inverse_priors(prior_nodes, key, num_samples):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample_inverse(key, num_samples)
    return samples


def sample_inverse_latent(node, key, num_samples, node_inputs):
    if num_samples == 1:
        return node.sample_inverse(key, 1, *node_inputs)
    subkeys = jax.random.split(key, num_samples)

    def sample_latent(subkey, *inputs):
        return node.sample_inverse(subkey, 1, *inputs)

    in_axes = (0,) + (0,) * len(node_inputs)
    return jax.vmap(sample_latent, in_axes=in_axes)(subkeys, *node_inputs)


def _sample_inverse(prior_nodes, latent_nodes, key, num_samples):
    key_prior, key_node = jax.random.split(key)
    samples = _sample_inverse_priors(prior_nodes, key_prior, num_samples)
    prior_names = {prior.name for prior in prior_nodes}
    non_prior_latents = [n for n in latent_nodes if n.name not in prior_names]
    keys = jax.random.split(key_node, len(non_prior_latents))
    for key, node in zip(keys, non_prior_latents):
        node_inputs = get_node_inputs(node, samples)
        node_inverse_sample = sample_inverse_latent(
            node, key, num_samples, node_inputs
        )
        samples[node.name] = node_inverse_sample
    return samples


def _sample_forward_priors(prior_nodes, key, num_samples):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample(key, num_samples)
    return samples


def sample_forward_node(node, key, num_samples, node_inputs):
    if num_samples == 1 or len(node_inputs) == 0:
        return node.sample(key, num_samples, *node_inputs)
    subkeys = jax.random.split(key, num_samples)

    def sample_node(subkey, *inputs):
        return node.sample(subkey, 1, *inputs)

    in_axes = (0,) + (0,) * len(node_inputs)
    return jax.vmap(sample_node, in_axes=in_axes)(subkeys, *node_inputs)


def _sample_forward(prior_nodes, non_prior_nodes, key, num_samples):
    key_prior, key_node = jax.random.split(key)
    samples = _sample_forward_priors(prior_nodes, key_prior, num_samples)
    keys = jax.random.split(key_node, len(non_prior_nodes))
    for key, node in zip(keys, non_prior_nodes):
        node_inputs = get_node_inputs(node, samples)
        node_sample = sample_forward_node(node, key, num_samples, node_inputs)
        samples[node.name] = node_sample
    return samples


def _apply_priors(priors, inverse_samples):
    samples, log_prob = {}, {}
    for prior in priors:
        state = prior.apply(get_namedtuple_value(prior.name, inverse_samples))
        samples[prior.name] = get_namedtuple_value(prior.name, state.sample)
        log_prob[prior.name] = state.log_prob.sum()
    return samples, log_prob


def _apply(priors, non_priors, inverse_samples, data_mapping, observable_names):
    samples, log_prob = _apply_priors(priors, inverse_samples)
    for node in non_priors:
        node_inputs = [samples[edge.name] for edge in node.edges]
        if node.name in observable_names:
            observation = data_mapping[node.name]
            state = node.apply(observation, *node_inputs)
            samples[node.name] = observation
        else:
            node_sample = get_namedtuple_value(node.name, inverse_samples)
            state = node.apply(node_sample, *node_inputs)
            samples[node.name] = get_namedtuple_value(node.name, state.sample)
        log_prob[node.name] = state.log_prob.sum()
    return samples, log_prob


def _compute_latent_log_probs(
    latent_nodes, forward_samples, inverse_samples, bijectors, include_jacobian
):
    log_prob = {}
    for node in latent_nodes:
        forward_sample = getattr(forward_samples, node.name)
        if node.distribution is not None:
            distribution = node.distribution
        else:
            distribution_fn = get_distribution_fn(node)
            if distribution_fn is None:
                raise ValueError("Latent node missing distribution_fn.")
            parent_samples = [
                getattr(forward_samples, edge.name) for edge in node.edges
            ]
            distribution = distribution_fn(*parent_samples)
        log_prob_value = distribution.log_prob(forward_sample)
        if include_jacobian:
            if inverse_samples is None:
                raise ValueError("Inverse samples required for Jacobian.")
            bijector = bijectors[node.name]
            inverse_sample = getattr(inverse_samples, node.name)
            log_prob_value = (
                log_prob_value
                + bijector.forward_log_det_jacobian(inverse_sample)
            )
        log_prob[node.name] = log_prob_value.sum()
    return log_prob
